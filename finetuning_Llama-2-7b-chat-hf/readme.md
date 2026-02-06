# Fine-tuning Llama-2-7b-chat-hf with QLoRA

This repository demonstrates fine-tuning the **meta-llama/Llama-2-7b-chat-hf** model using **QLoRA** (Quantized Low-Rank Adaptation) on a custom dataset from the Bhagavad Gita, creating an AI Krishna that responds with spiritual wisdom.

## What This Project Does

**Fine-tuning** = Taking a pre-trained model and teaching it domain-specific knowledge without full retraining

**QLoRA** = Memory-efficient training using:

- 4-bit quantization for the base model
- Low-rank adapters for trainable parameters
- Smart memory management for consumer GPUs

## Core Workflow
![Workflow Diagram](flowdiagram.png)

![Fine-tuned Krishna AI Response](output.png)

**For detailed code explanation**: [Fine-tuning Llama2-7b on Personal Dataset](https://medium.com/@pokhrelankit/fine-tuning-llama2-7b-on-personal-dataset-code-explanation-provided-4bcbfe956b3e?postPublishedType=repub)
