# main_scripts Overview

This folder contains training and inference entrypoints used in MINT experiments.

## Script groups

### Text-only workflows
- `RareDxGPT-sft.py`: Supervised fine-tuning (SFT).
- `RareDxGPT-dpo.py`: Direct Preference Optimization (DPO).
- `RareDxGPT-orpo.py`: ORPO training.
- `RareDxGPT_inference_vllm.py`: vLLM inference for text-based rare disease generation.
- `RareDxGPT_RAG_inference.py`: Retrieval-augmented generation inference.

### Vision / multimodal workflows
- `RareDxGPT_sft_vision.py`: Vision-language SFT.
- `RareDxGPT_dpo_vision.py`: Vision-language DPO.
- `RareDxGPT_orpo_vision.py`: Vision-language ORPO.
- `RareDxGPT_inference_vllm_vision_base.py`: Base-model vision inference.
- `RareDxGPT_inference_vllm_vision_finetuned.py`: Finetuned vision inference.

## Public repository cleanup status

All scripts now resolve local imports with a repository-relative root (`PROJECT_ROOT`) rather than hardcoded absolute paths for `utils/` and `AutoEvaluator/` imports.

## Recommended execution pattern

Run scripts from repository root with explicit path arguments where supported:

```bash
python main_scripts/RareDxGPT_inference_vllm.py \
  --project_root /path/to/MINT-LLM \
  --peft_model_id checkpoints/adapter_x \
  --base_model_path /path/to/base_model
```

For scripts that still include dataset/checkpoint absolute defaults, override those values in the script arguments/variables before execution in your own environment.
