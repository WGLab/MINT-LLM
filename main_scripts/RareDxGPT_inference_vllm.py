import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import safetensors.torch
import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "utils"))
from disease_gene_convert import *
from disease_list_extract import *
from external_analysis_util import *
from set_seed import *
from util_llama3_70b import *

sys.path.append(str(PROJECT_ROOT / "AutoEvaluator"))
from AutoEvaluator import *
from EvaluatorProcessor import *

DISEASE_SYSTEM_PROMPT = (
    "You are a genetic counselor. Your task is to identify potential rare diseases "
    "based on given phenotypes. Follow the output format precisely."
)

DISEASE_USER_PROMPT_TEMPLATE = (
    "{clinical_note}\n\n"
    "Based on this information, provide a numbered list of EXACTLY 10 potential rare diseases.\n\n"
    "Use EXACTLY this format:\n\n"
    "POTENTIAL_DISEASES:\n"
    "1. 'Disease1'\n2. 'Disease2'\n3. 'Disease3'\n4. 'Disease4'\n5. 'Disease5'\n"
    "6. 'Disease6'\n7. 'Disease7'\n8. 'Disease8'\n9. 'Disease9'\n10. 'Disease10'\n\n"
    "Ensure all disease names are in single quotes, and there are exactly 10 in the list. "
    "Do not deviate from this format or add any explanations."
)


def parse_args():
    parser = argparse.ArgumentParser(description="RareDxGPT vLLM inference")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--ratio", type=float, default=0.3, help="Train test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train test split")
    parser.add_argument("--disease", type=str, default="bws", help="Disease name for external dataset")
    parser.add_argument("--peft_model_id", type=str, default="x")
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root path used to resolve datasets and reference files",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/mnt/isilon/wang_lab/shared/LLaMA3.2-3B-Instruct",
        help="Local or HF base model path used by vLLM",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token. If not provided, uses HF_TOKEN env var.",
    )
    return parser.parse_args()


def loading_dataset(project_root: Path):
    reference_dir = project_root / "reference_data"
    datasets_dir = project_root / "datasets"

    total_train = load_dataset("csv", data_files=str(reference_dir / "total_train.csv"))
    disease_name = pd.read_csv(reference_dir / "disease_name_full.csv")
    reference_list = list(disease_name.Name)

    full_dataset = total_train["train"]
    test_dataset_dict = load_from_disk(str(datasets_dir / "orpo_dpo_dataset_cask10_10"))
    test_dataset = test_dataset_dict["test"]

    dataset1_df = full_dataset.to_pandas()
    dataset2_df = test_dataset.to_pandas()
    dataset1_df["image_id"] = dataset1_df["image_id"].astype(str)
    dataset2_df["image_id"] = dataset2_df["image_id"].astype(str)

    merged_df = pd.merge(
        dataset2_df[["image_id"]],
        dataset1_df[["image_id", "Response"]],
        on="image_id",
        how="left",
    )
    ground_truth_list = merged_df["Response"].tolist()
    return test_dataset, ground_truth_list, reference_list


def build_messages(example):
    return {
        "messages": [
            {"role": "system", "content": DISEASE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DISEASE_USER_PROMPT_TEMPLATE.format(clinical_note=example["clinical_note"]),
            },
        ]
    }


def apply_chat_template(row, tokenizer):
    row["messages"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    return row


def filter_lora_weights(peft_model_id: str):
    lora_path = f"{peft_model_id}/adapter_model.safetensors"
    new_path = f"{peft_model_id}x/adapter_model.safetensors"

    tensors = safetensors.torch.load_file(lora_path)
    nonlora_keys = [k for k in list(tensors.keys()) if "lora" not in k]

    for key in nonlora_keys:
        del tensors[key]

    safetensors.torch.save_file(tensors, new_path)
    return f"{peft_model_id}x"


def main():
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    os.environ["HF_HOME"] = "/tmp"
    peft_model_id = str((project_root / args.peft_model_id).resolve())
    set_seed(args.seed)

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("HF token not provided; continuing without explicit login.")

    sampling_params = SamplingParams(temperature=0.8, top_p=0.8, top_k=10, max_tokens=512)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    _, _, reference_list = loading_dataset(project_root)
    test_dataset = load_from_disk(str(project_root / "datasets" / args.disease))
    test_dataset = test_dataset.rename_column("original_text", "clinical_note")
    test_dataset = test_dataset.rename_column("response", "disease")
    ground_truth_list = test_dataset["disease"]

    test_dataset = test_dataset.map(build_messages)
    test_dataset = test_dataset.map(lambda row: apply_chat_template(row, tokenizer))
    prompts = test_dataset["messages"]

    base_model_id = args.base_model_path
    AutoModelForCausalLM.from_pretrained(base_model_id)

    lora_runtime_path = filter_lora_weights(peft_model_id)
    print(f"Using runtime LoRA path: {lora_runtime_path}")

    llm = LLM(
        model=base_model_id,
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=256,
        trust_remote_code=True,
        download_dir="/mnt/isilon/wang_lab/shared/",
    )

    inference_list = []
    for prompt in prompts:
        outputs = llm.generate(prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text
        inference_list.append(generated_text)

    inference_list = [extract_potential_diseases(text) for text in inference_list if text]
    gene_samples = gene_list_convert(inference_list)
    results = evaluation(gene_samples, ground_truth_list)
    print(results)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
