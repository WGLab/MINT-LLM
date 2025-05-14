import json
import os
import re
import random
import copy
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
from datasets import load_from_disk, features
from peft import PeftModel
from trl import apply_chat_template, DPOConfig, DPOTrainer, setup_chat_format
import wandb
from accelerate import PartialState
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/utils'))
from set_seed import *
from util_llama32_vision import *
from PIL import Image
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
    cache_dir = "/mnt/isilon/wang_lab/shared/dpo_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TMPDIR"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_dir}/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{cache_dir}/datasets"
    os.environ["TRITON_CACHE_DIR"] = f"{cache_dir}/triton"
    peft_model_id = f"/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b5e-05-SFT/final_checkpoint"
    device = torch.device(f'cuda:{local_rank}')
    ###########Model Loading###############################
   
    device_string = PartialState().process_index
    base_model, base_model_id = get_model(device_string)
    base_model_id = "HuggingFaceM4/idefics2-8b"
    model = AutoModelForVision2Seq.from_pretrained(base_model_id, 
                                                    torch_dtype=torch.bfloat16, 
                                                    device_map={'':device_string})
    # model = PeftModel.from_pretrained(base_model,
    #                                   peft_model_id,
    #                                   torch_dtype=torch.bfloat16,
    #                                   is_trainable=True,
    #                                   device_map={'':device_string}
    #                                 #   adapter_name="train"
    #                                  ).to(device)
    
    # ref_model = PeftModel.from_pretrained(base_model,
    #                                   peft_model_id,
    #                                   torch_dtype=torch.bfloat16,
    #                                   is_trainable=False,
    #                                 #   adapter_name="train"
    #                                  )
    # model.load_adapter(peft_model_id, adapter_name="reference")

    processor = AutoProcessor.from_pretrained(base_model_id, padding=True, return_tensor='pt', do_image_splitting=False)

    tokenizer = processor.tokenizer

    # ref_model = copy.deepcopy(model).to('cpu')
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
    # model._ddp_params_and_buffers_to_ignore = [
    #     name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    # ]
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer.pad_token = tokenizer.eos_token
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
    ##########################################################################################
    # def format_chat_template(row):
    #     row["chosen"] = "<|image|>" + tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    #     row["rejected"] = "<|image|>" + tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    #     return row
    def format_chat_template(row):
        row['prompt'] = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row['prompt'][0]['content'][1]['text']}]}]
        row['prompt'] = tokenizer.apply_chat_template(row['prompt'], tokenize=False)
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        row['image'] = [row['image'].convert('RGB')]
        return row

    def collate_fn(examples):
        max_length = max(len(example['chosen_input_ids']) for example in examples)
        max_length2 = max(len(example['rejected_input_ids']) for example in examples)
        pad_token_id = 128001
        prompt_input_ids = torch.tensor([example['prompt_input_ids'] for example in examples])
        chosen_input_ids = torch.tensor([example['chosen_input_ids'] + [pad_token_id] * (max_length - len(example['chosen_input_ids'])) 
                     for example in examples])
        rejected_input_ids = torch.tensor([example['rejected_input_ids'] + [pad_token_id] * (max_length2 - len(example['rejected_input_ids'])) 
                     for example in examples])
        prompt_attention_mask = (prompt_input_ids != pad_token_id).long()
        chosen_attention_mask = (chosen_input_ids != pad_token_id).long()
        rejected_attention_mask = (rejected_input_ids != pad_token_id).long()
        images = [example["images"] for example in examples]
        batch = processor(images=images, return_tensors="pt", padding=True)
        batch['aspect_ratio_ids'] = torch.full((len(examples), 1), 1, dtype=torch.long)
        batch['prompt_input_ids'] = prompt_input_ids
        batch['chosen_input_ids'] = chosen_input_ids
        batch['rejected_input_ids'] = rejected_input_ids
        batch['prompt_attention_mask'] = prompt_attention_mask
        batch['chosen_attention_mask'] = chosen_attention_mask
        batch['rejected_attention_mask'] = rejected_attention_mask
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        print(batch.keys())
        return batch
    train_dataset = train_dataset.map(format_chat_template,     
                                    # cache_file_name=None,  # Disable caching to file
                                    # load_from_cache_file=False  # Force recomputation)
                                    )
    train_dataset = train_dataset.rename_column("image", "images")


    print(train_dataset.column_names)
    f = train_dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    train_dataset = train_dataset.cast(f)
    #########Tokenized Length#######################
    prompt_length = 700
    max_seq_length = 1400
    ##########DPO Config Set Up################################33
    training_args = DPOConfig(
        output_dir=f"/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b{args.learning_rate}_dpo", # directory to save and repository id
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
        # tf32=True,                              # use tf32 precision
        push_to_hub=False,                      # push model to hub
        report_to="wandb",                # report metrics to tensorboard
        local_rank=local_rank,
        ddp_find_unused_parameters=True,
        model_adapter_name='train',
        ref_adapter_name='reference',
        remove_unused_columns = False,
        dataset_num_proc=32,  # tokenization will use 32 processes
        dataloader_num_workers=32,  # data loading will use 32 workers
        gradient_checkpointing_kwargs={'use_reentrant':False} 
    )


    dpo_args = {
    "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
    "loss_type": "sigmoid"                  # The loss type for DPO.
    }

    ###########DPO Trainer###################################################
    trainer = DPOTrainer(
    model=model,
    ref_model=None, # set to none since we use peft
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor,
    peft_config=LoraConfig(target_modules='all-linear')
    # data_collator=collate_fn
    )
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        trainer.train()
    trainer.model.save_pretrained(f'/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b{args.learning_rate}_dpo/final_checkpoint')
    tokenizer.save_pretrained(f"/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b{args.learning_rate}_dpo/final_checkpoint")
    trainer.save_model()
    wandb.finish()
    cleanup_ddp()
if __name__ == "__main__":
    main()


