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
from trl import apply_chat_template, DPOConfig, SFTConfig, DPOTrainer, ORPOConfig, ORPOTrainer, SFTTrainer, setup_chat_format
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
    device_string = PartialState().process_index
    model, model_path = get_model(device_string)
    model = model.to(device)
    ###########Tokenizer Set Up################################
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    # model.resize_token_embeddings(len(tokenizer))
    ###########WandB Set Up#####################################
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
    train_dataset = train_dataset.rename_column("image", "images")
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [example["messages"] for example in examples]
        images = [example["images"] for example in examples]
        print(type(texts[0]))
        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch
    def format_data(example):
        example['messages'] = [
            example['prompt'][0],
            example['chosen'][0]
        ]
        example['messages'] = tokenizer.apply_chat_template(example['messages'], tokenize=False)
        return example
    train_dataset = train_dataset.map(format_data)
    train_dataset = train_dataset.remove_columns(['chosen', 'rejected', 'prompt'])
    tokenizer.pad_token = tokenizer.eos_token
    #########Tokenized Length#######################
    prompt_length = 700
    max_seq_length = 1400
    ##########SFT Config Set Up################################
    sft_args = SFTConfig(
    output_dir=f"/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b{args.learning_rate}-SFT", # directory to save and repository id
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
    evaluation_strategy="no",            # evaluate every 1000 steps
    do_eval = False,
    # eval_steps=700,                         # when to evaluate
    bf16=True,                              # use bfloat16 precision
    push_to_hub=False,                      # push model to hub
    report_to="wandb",                # report metrics to tensorboard
    local_rank=local_rank,
    ddp_find_unused_parameters=True,
    use_liger=True,
    remove_unused_columns = False,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    # max_seq_length=max_seq_length,
    # max_prompt_length=prompt_length,
    # dataset_text_field='messages',
    dataset_kwargs = {"skip_prepare_dataset": True}
    )   

    ###########Tokenizer Padding For ORPO Training###############
    
    # tokenizer.padding_side = 'left' # to prevent errors with FA
    # tokenizer.truncation_side = 'left' # to prevent cutting off last generation
    ###########ORPO Trainer###################################################
    trainer = SFTTrainer(
    model,
    data_collator=collate_fn,
    args=sft_args,
    train_dataset=train_dataset,
    processing_class = tokenizer
    )
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        trainer.train()
    trainer.model.save_pretrained(f'/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b{args.learning_rate}-SFT/final_checkpoint')
    tokenizer.save_pretrained(f"/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b{args.learning_rate}-SFT/final_checkpoint")
    trainer.save_model()
    wandb.finish()
    cleanup_ddp()
if __name__ == "__main__":
    main()


