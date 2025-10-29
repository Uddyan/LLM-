# Model

This directory contains LLM training, fine-tuning, and inference components.

## Structure

- **training/**: Base model training and pre-training
  - Training pipelines
  - Model configuration
  - Training scripts

- **fine-tuning/**: Domain-specific fine-tuning
  - Supervised Fine-Tuning (SFT)
  - RLHF/RLAIF implementation
  - LoRA/QLoRA adapters
  - Fine-tuning datasets

- **inference/**: Model serving and inference
  - vLLM for efficient inference
  - Model deployment scripts
  - API endpoints

## Technologies

- HuggingFace Transformers
- PyTorch
- DeepSpeed
- vLLM
- TRL (Transformer Reinforcement Learning)

## Base Model Options

- GPT-4 / GPT-4 Turbo (via Azure OpenAI)
- Claude 3.5 Sonnet
- Llama 3.1 70B/405B
- Mistral Large 2 / Mixtral 8x7B
