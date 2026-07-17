# SLM From Scratch — Build & Fine-Tune a Small Language Model

A project-based walkthrough of building a GPT-style model from scratch, then fine-tuning and
aligning it through a full pretrain → SFT → DPO pipeline, and deploying it. Built while working
through *How to Build and Fine-Tune a Small Language Model* (Paul Liu), as interview-prep-grade
documentation: every chapter's folder has runnable code plus a README explaining what was built,
why, and how it maps to production ML systems.

## Structure
Each `chXX_*/` folder corresponds to one chapter/stage of the project. See [`RUNBOOK.md`](./RUNBOOK.md)
for the full plan, architecture diagrams, and a running log of what was covered in each session.

| Folder | Stage |
|---|---|
| `ch02_gpt_from_scratch` | Character-level GPT built from raw PyTorch tensors |
| `ch03_finetune_gpt2` | Fine-tune pretrained GPT-2 with Hugging Face |
| `ch04_dataset_preparation` | Real-world data sourcing, cleaning, custom BPE tokenizer |
| `ch05_architecture_config` | Model sizing / config decisions |
| `ch06_training_loop` | Production-grade training loop (LR schedule, checkpointing, monitoring) |
| `ch07_evaluation` | Perplexity, token accuracy, benchmark evaluation |
| `ch08_minimind_pretrain` | Full pretraining run with RoPE / RMSNorm / SwiGLU / GQA |
| `ch09_minimind_sft` | Supervised fine-tuning (instruction tuning) |
| `ch10_minimind_dpo` | Direct Preference Optimization (alignment) |
| `ch11_deployment` | Quantization + serving (llama.cpp / vLLM) |
| `ch12_capstone` | End-to-end capstone project + responsible-AI notes |
| `diagrams/` | Exported architecture/pipeline diagrams |

## Environment
```bash
conda env create -f environment.yml
conda activate slm
```
or
```bash
pip install -r requirements.txt
```

## Training infra
Developed on an Azure `Standard_NC4as_T4_v3` spot VM (1x T4, 16GB VRAM) — see `setup_azure_vm.sh`
for the provisioning/bootstrap script.
