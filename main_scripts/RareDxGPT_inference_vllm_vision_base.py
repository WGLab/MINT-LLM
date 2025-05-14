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
                          AutoModelForCausalLM,
                          pipeline,
                          AutoProcessor,
                          AutoModelForVision2Seq,
                          MllamaForConditionalGeneration
                         )
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from datasets import load_from_disk, load_dataset
from trl import apply_chat_template, DPOConfig, DPOTrainer
import wandb
import random
from accelerate import PartialState
from peft import AutoPeftModelForCausalLM, PeftModel, AutoPeftModel, PeftConfig
import sys
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/utils'))
from set_seed import *
from util_llama32_vision import *
from disease_list_extract import *
from external_analysis_util import *
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/AutoEvaluator'))
from AutoEvaluator import *
from EvaluatorProcessor import *
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(
        description="RareDxGPT-orpo"
    )
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size for training"
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

    parser.add_argument("--disease",
                        type=str,
                        default='bws',
                        help='Disease name for external dataset')

    parser.add_argument("--peft_model_id",
                        type=str,
                        default="x")
    return parser.parse_args()
id2label = {0: 'Adrenal Glands',
            1: 'Bile Duct',
            2: 'Bladder',
            3: 'Breast',
            4: 'Cervix',
            5: 'Colon',
            6: 'Esophagus',
            7: 'Head & Neck',
            8: 'Kidney',
            9: 'Liver',
            10: 'Lung',
            11: 'Ovarian',
            12: 'Pancreatic',
            13: 'Prostate',
            14: 'Skin',
            15: 'Stomach',
            16: 'Testis',
            17: 'Thyroid',
            18: 'Uterus'
            }
label2id = {v:k for k,v in id2label.items()}
class EvaluateTissueMetric:
    def __init__(self, predictions, ground_truth_top_k, ground_truth_bottom_k, lambda_weight=0.5):
        self.predictions = predictions
        self.ground_truth_top_k = ground_truth_top_k
        self.ground_truth_bottom_k = ground_truth_bottom_k
        self.lambda_weight = lambda_weight
        self.coverage_rate = None
        self.avoidance_rate = None

    def calculate_coverage_rate(self):
        if not self.predictions:
            self.coverage_rate = 0.0
            return
            
        intersection = 0
        for pred in self.predictions:
            if pred in self.ground_truth_top_k:
                intersection += 1
        self.coverage_rate = intersection / len(self.predictions)
        return self.coverage_rate

    def calculate_avoidance_rate(self):
        if not self.predictions:
            self.avoidance_rate = 0.0
            return
            
        intersection = 0
        for pred in self.predictions:
            if pred in self.ground_truth_bottom_k:
                intersection += 1
        self.avoidance_rate = 1 - (intersection / len(self.predictions))
        return self.avoidance_rate

    def calculate_optional_metrics(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if self.coverage_rate is None:
            self.calculate_coverage_rate()
        if self.avoidance_rate is None:
            self.calculate_avoidance_rate()
            
        car_score = None
        if self.coverage_rate == 0 and self.avoidance_rate == 0:
            car_score = 0.0
        else:
            denominator = self.lambda_weight * self.coverage_rate + self.avoidance_rate
            if denominator == 0:
                car_score = 0.0
            else:
                car_score = ((1 + self.lambda_weight) * self.coverage_rate * self.avoidance_rate) / denominator
def extract_tissue(text):
    occurrences = [i for i in range(len(text)) if text.startswith("MOST_LIKELY_TISSUE:", i)]
    
    if len(occurrences) >= 2:
        second_occurrence = occurrences[1]
        diseases_section = text[second_occurrence:].split("MOST_LIKELY_TISSUE:")[1].strip()
        
        tissue_types = []
        lines = diseases_section.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(f"{i}." in line for i in range(1, 6)):
                tissue_type = line.split(".", 1)[1].strip()
                tissue_type = tissue_type.strip("'")
                tissue_types.append(label2id[tissue_type])
        
        return tissue_types
    else:
        return []
def setup_ddp():
    dist.init_process_group(backend='nccl')#, init_method='env://')


def cleanup_ddp():
    dist.destroy_process_group()

def main():
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    args = parse_args()
    set_seed(args.seed)
    parent_path = "/home/wangz12/projects"
    child_path = args.peft_model_id
    peft_model_id = os.path.join(parent_path, child_path)
    ###########Model Set Up######################################
   
    local_rank = int(os.environ['LOCAL_RANK'])
    print(local_rank)
    torch.cuda.set_device(local_rank)
    setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    base_model, base_model_id = get_model()
    base_model = AutoModelForVision2Seq.from_pretrained(base_model_id)
    peft_model_id = "/home/wangz12/projects/RareDxGPT/checkpoints/RareDxGPT-llama3.2-11b5e-05-SFT"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(base_model,
                                      peft_model_id,
                                      device_map="auto",
                                      torch_dtype=torch.bfloat16
                                     ).to(device)
    model = base_model.to(device)
    tokenizer= AutoTokenizer.from_pretrained(peft_model_id)
    # tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    processer = AutoProcessor.from_pretrained(base_model_id)
    # tokenizer = processor.tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    # pipe = pipeline("visual-question-answering",
    #                 model=model,
    #                 torch_dtype=torch.bfloat16,
    #                 device_map="auto",
    #                 tokenizer=tokenizer,
    #                 max_new_tokens=512,
    #                 image_processor=processer
    #                 )
    test_dataset = load_from_disk("/home/wangz12/Downloads/pannuke (2)")['test']
    def format_data(example):
        return {"messages": [
        {"role": "system", "content": "You are a professional pathologist analyzing nucleus images. For each image, identify the most likely tissue types based on the nuclear morphology. Follow the output format precisely."},
        {"role": "user", "content": f"""'Based on the morphological features observed in the provided nucleus image, which tissue types could this nucleus originate from?'
        \n\nPlease choose from the following candidates: {list(label2id.keys())}
        \n\nUse EXACTLY this format:
        \n\nMOST_LIKELY_TISSUE:
        \n1. 'tissue type 1'
        \n2. 'tissue type 2'
        \n3. 'tissue type 3'
        \n4. 'tissue type 4'
        \n5. 'tissue type 5'
        \n\nEnsure all tissue types are in single quotes, and List exactly 5 tissue types in decreasing order of likelihood. 
        Do not deviate from this format or add any explanations."""},
        ]}

    test_dataset = test_dataset.map(format_data)
    def format_chat_template(row):
       row["messages"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
       return row
    test_dataset = test_dataset.map(format_chat_template)
    inference_list = []
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        image = example['image']
        question = "<|image|>" + example['messages']
        inputs = processer(text=question, images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_k=10,
                top_p=0.6
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_list.append(generated_text)
        print(generated_text)
    print(inference_list)

    df = pd.DataFrame(inference_list)
    df.to_csv("/home/wangz12/projects/RareDxGPT/output_CoT/plip_sft.csv", index=False)
    inference_list = [extract_tissue(text) for text in inference_list]
    def accuracy(inference_list):
        corr_top1 = 0
        corr_top5 = 0
        for i in range(len(inference_list)):
            top5 = inference_list[i][:5]
            label = test_dataset['label'][i]
            if label == top5[0]:
                corr_top1 += 1
            if label in top5:
                corr_top5 += 1
        return corr_top1/len(inference_list), corr_top5/len(inference_list)
    top1_acc, top5_acc = accuracy(inference_list)

    car_score_total = 0
    for i in range(len(test_dataset)):
        predictions = [pred[:5] for pred in inference_list][i]
        ground_truth_top_k = test_dataset['top5_list'][i]
        ground_truth_bottom_k = test_dataset['bottom5_list'][i]
        evaluator = EvaluateTissueMetric(predictions, ground_truth_top_k, ground_truth_bottom_k)
        coverage_rate, avoidance_rate, car_score = evaluator.calculate_optional_metrics()
        car_score_total += car_score
    car_score_total /= len(test_dataset)
    print(f"""
          Top 1 Accuracy: {top1_acc},
          Top 5 Accuracy: {top5_acc},
          Car Score: {car_score}
          """)            

if __name__ == "__main__":
    main()

