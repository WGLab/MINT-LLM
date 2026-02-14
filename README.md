# MINT-LLM

Official codebase for **MINT (Multimodal Integrated kNowledge Transfer)** from the paper:

> Wu, D., Wang, Z., Nguyen, Q., Xu, Z., Wang, K., *Multimodal Integrated Knowledge Transfer to Large Language Models through Preference Optimization with Biomedical Applications*. arXiv:2505.05736

ArXiv link: https://arxiv.org/abs/2505.05736

---

## Public Repository Notes

- Due to privacy and compliance restrictions, the GMDB dataset used in the paper is **not publicly released**.
- This repository provides training/inference scripts and pretrained checkpoints for reproducibility-oriented experiments.
- Some scripts were originally developed for internal HPC paths and are being progressively refactored into path-configurable public workflows.
- Phenopacket-derived clinical notes path: https://github.com/WGLab/CoT-RAG-LLM-Gene-Prioritization-Disease-Diagnosis/tree/main/dataset
---

## Repository Structure

- `main_scripts/` — main SFT/DPO/ORPO training and inference entry scripts.

- `main_scripts/README.md` — script catalog and execution guidance.
- `utils/` — model loading utilities, seed setup, dataset helpers, and post-processing.
- `AutoEvaluator/` — evaluation pipeline and processors.
- `MINT-ckpt/` — released language-model checkpoint artifacts.
- `MINT-vision-ckpt/` — released vision-language checkpoint artifacts.

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need Hugging Face gated model access, set token via environment variable:

```bash
export HF_TOKEN=your_token_here
```

---

## Example: Inference with vLLM Script

`main_scripts/RareDxGPT_inference_vllm.py` now supports public-friendly path arguments:

```bash
python main_scripts/RareDxGPT_inference_vllm.py \
  --project_root /path/to/MINT-LLM \
  --peft_model_id checkpoints/your_adapter_dir \
  --base_model_path /path/to/base_model \
  --disease bws
```

Notes:
- `--peft_model_id` is resolved relative to `--project_root`.
- `--hf_token` can be passed explicitly, or omitted if `HF_TOKEN` is already set.

---

## Hardware and Runtime

- Original large-scale experiments were run on multi-GPU SLURM clusters (e.g., A100).
- For local inference, adjust model size, batch size, and vLLM settings according to available GPU memory.

---

## License

This project is released under the license in `LICENSE`.
