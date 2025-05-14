import json
import os
import re
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import subprocess
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM
                         )
from datasets import load_from_disk
from trl import apply_chat_template, DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer
import wandb
import random
from accelerate import PartialState
from peft import AutoPeftModelForCausalLM, PeftModel
import sys
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/utils'))
from set_seed import *
from util_llama3 import *
def parse_args():
    parser = argparse.ArgumentParser(
        description="RareDxGPT-DPO"
    )
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size for training"
                        )
    
    parser.add_argument("--learning_rate",
                        type=float,
                        default=3e-5,
                        help="Learning rate for training"
                        )
    
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs"
                        )
    
    parser.add_argument("--ratio",
                        type=float,
                        default=0.3,
                        help="Train test split ratio"
                        )

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for train test split")

    return parser.parse_args()



def setup_ddp():
    dist.init_process_group(backend='nccl')#, init_method='env://')

def get_nvidia_smi_output():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print(result.stdout) 
        else:
            print(f"Error running nvidia-smi: {result.stderr}")
    
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure it's installed and in your PATH.")

def cleanup_ddp():
    dist.destroy_process_group()

def print_memory_usage():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def main():
    ###########Model Set Up######################################
    path = "/home/wangz12/projects/RareDxGPT"
    os.chdir(path)
    args = parse_args()
    set_seed(args.seed)
    local_rank = int(os.environ['LOCAL_RANK'])
    print(local_rank)
    torch.cuda.set_device(local_rank)
    setup_ddp()
    peft_model_id = "checkpoints/RareDxGPT-sft-PRIVATE{args.epochs}/final_checkpoint"
    device = torch.device(f'cuda:{local_rank}')
    ###########Model Loading###############################
    model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id,
                                                    torch_dtype=torch.float16,
                                                    # adapter_name='train',
                                                    is_trainable=True,
                                                    # load_in_4bit=True,
                                                    low_cpu_mem_usage=True
                                                    )

    ref_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id,
                                                         torch_dtype=torch.float16,
                                                        #  load_in_4bit=True,
                                                         low_cpu_mem_usage=True,

    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    ###########Model Setup#########################################
    # for param in model.base_model.model.parameters():
    #     param.requires_grad = True
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    model = model.to(device)
    # model.load_adapter(peft_model_id, adapter_name='reference')
    ###########WandB Set Up#####################################
    wandb.init(
        project='RareDxGPT-DPO',
        config={
            "learning_rate": args.learning_rate,
            "architecture": "LLaMA3.2-PRIVATE-Intruct",
            "dataset": "GestaltMML-preference-dataset",
            "epochs": args.epochs,
            "batch_size": args.batch_size
        },
        name=f"lr_{args.learning_rate}_epochs_{args.epochs}", 
    )
    ##########Dataset Set Up######################################
    train_dataset_dict = load_from_disk("datasets/orpo_dpo_dataset_cask10_10")
    train_dataset_dict = train_dataset_dict.remove_columns('image_id')
    train_dataset = train_dataset_dict['train']
    def format_chat_template(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row
    train_dataset = train_dataset.map(format_chat_template)
    ##########Peft Config###########################
    peft_config = LoraConfig(
       task_type="CAUSAL_LM",
       inference_mode=False,
       r=256,
       lora_alpha=128,
       lora_dropout=0.05,
       bias='none',
    #    target_modules=['q_proj', 'v_proj']
       target_modules='all-linear'
    )
    #########Tokenized Length#######################
    prompt_length = 700
    max_seq_length = 1400
    ##########DPO Config Set Up################################33
    training_args = DPOConfig(
        output_dir=f"/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-dpo-PRIVATE{args.epochs}", # directory to save and repository id
        num_train_epochs=args.epochs,                     # number of training epochs
        per_device_train_batch_size=args.batch_size,         # batch size per device during training
        # per_device_eval_batch_size=4,           # batch size for evaluation
        gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        learning_rate=args.learning_rate,                     # 10x higher LR than QLoRA paper
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",             # use cosine learning rate scheduler
        logging_steps=25,                       # log every 25 steps
        save_steps=500,                         # when to save checkpoint
        save_total_limit=2,                     # limit the total amount of checkpoints
        evaluation_strategy="no",            # evaluate every 1000 steps
        do_eval = False,
        # eval_steps=700,                         # when to evaluate
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        push_to_hub=False,                      # push model to hub
        report_to="wandb",                # report metrics to tensorboard
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        model_adapter_name='train',
        ref_adapter_name='reference'
    )

    dpo_args = {
    "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
    "loss_type": "sigmoid"                  # The loss type for DPO.
    }
    ###########Tokenizer Padding For DPO Training###############
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'left' # to prevent errors with FA
    # tokenizer.truncation_side = 'left' # to prevent cutting off last generation
    ###########DPO Trainer###################################################
    trainer = DPOTrainer(
    model=model,
    ref_model=None, # set to none since we use peft
    # peft_config=peft_config,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=prompt_length,
    beta=dpo_args["beta"],
    loss_type=dpo_args["loss_type"],
    )
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        trainer.train()
    trainer.model.save_pretrained(f'/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-dpo-PRIVATE{args.epochs}/final_checkpoint')
    tokenizer.save_pretrained(f"/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-dpo-PRIVATE{args.epochs}/final_checkpoint")
    trainer.save_model()
    wandb.finish()
    cleanup_ddp()
if __name__ == "__main__":
    main()