import json
import os
import re
import random
import argparse
import sys
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
                         AutoModelForCausalLM,
                         AutoModelForVision2Seq,
                         AutoProcessor,
                         MllamaForConditionalGeneration 
                        )
from datasets import load_from_disk
from trl import apply_chat_template, DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer,setup_chat_format
import wandb
from accelerate import PartialState
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/utils'))
from set_seed import *
from util_llama32_vision import *
def parse_args():
    parser = argparse.ArgumentParser(
        description="RareDxGPT-orpo"
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


def cleanup_ddp():
    dist.destroy_process_group()

    
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
    device = torch.device(f'cuda:{local_rank}')
    model, model_path = get_model()
    ###########Tokenizer Set Up################################
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = model.to(device)
    # model, tokenizer = setup_chat_format(model, tokenizer)
    # Set up the chat template
    if model.config.model_type == "idefics2":
        pass  # the processor already has a valid chat template
    elif model.config.model_type == "paligemma":
        processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] if item['type'] == 'text' %}{{ item['text'] }}<|im_end|>{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
    elif model.config.model_type == "llava":
        processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # if script_args.ignore_bias_buffers:
    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]

    ###########WandB Set Up######################################
    wandb.init(
        project='RareDxGPT-ORPO',
        config={
            "learning_rate": args.learning_rate,
            "architecture": "llama3.2-1/2",
            "dataset": "GestaltMML-preference-dataset",
            "epochs": args.epochs,
            "batch_size": args.batch_size
        },
        name=f"lr_{args.learning_rate}_epochs_{args.epochs}",
    )
    ##########Dataset Set Up######################################
    train_dataset_dict = load_from_disk("/home/wangz12/Downloads/pannuke (2)")
    train_dataset = train_dataset_dict['train']
    ########################Collate Function#############################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]
        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    ##########################################################################################
    def format_chat_template(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row
    train_dataset = train_dataset.map(format_chat_template)
    train_dataset = train_dataset.rename_column("image", "images")
    #########Tokenized Length#######################
    prompt_length = 700
    max_seq_length = 1400
    ##########ORPO Config Set Up################################33
    orpo_args = ORPOConfig(
    output_dir=f"/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b{args.learning_rate}", # directory to save and repository id
    num_train_epochs=args.epochs,                     # number of training epochs
    per_device_train_batch_size=args.batch_size,         # batch size per device during training
    # per_device_eval_batch_size=4,           # batch size for evaluation
    gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    learning_rate=args.learning_rate,                     # 10x higher LR than QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use cosine learning rate scheduler
    logging_steps=25,                       # log every 25 steps
    save_steps=500,                         # when to save checkpoint
    save_total_limit=2,                     # limit the total amount of checkpoints
    eval_strategy="no",            # evaluate every 1000 steps
    do_eval = False,
    # eval_steps=700,                         # when to evaluate
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    push_to_hub=False,                      # push model to hub
    report_to="wandb",                # report metrics to tensorboard
    local_rank=local_rank,
    ddp_find_unused_parameters=True,
    beta=0.1,
    max_length=max_seq_length,
    max_prompt_length=prompt_length,
    remove_unused_columns = False,
    gradient_checkpointing_kwargs={'use_reentrant':False} 
    )  


    ###########Tokenizer Padding For ORPO Training###############
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation
    ###########ORPO Trainer###################################################
    trainer = ORPOTrainer(
    model,
    args=orpo_args,
    # data_collator=collate_fn,
    train_dataset=train_dataset,
    # tokenizer=tokenizer,
    processing_class=processor.tokenizer
    
    )
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        trainer.train()
    trainer.model.save_pretrained(f'/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b{args.learning_rate}/final_checkpoint')
    tokenizer.save_pretrained(f"/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b{args.learning_rate}/final_checkpoint")
    trainer.save_model()
    wandb.finish()
    cleanup_ddp()
if __name__ == "__main__":
    main()


