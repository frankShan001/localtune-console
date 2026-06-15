# FAQ

[English](FAQ.md) | [简体中文](FAQ.zh-CN.md)

## What does LocalTune Console do?

LocalTune Console focuses on local LLM fine-tuning: manage local base models, validate training datasets, start short test runs or formal training runs, monitor logs and loss, manage Adapters and checkpoints, and compare Base vs Adapter outputs.

It is not a RAG system, knowledge base, chat client, or model API server.

## Can I train without NVIDIA CUDA?

The verified training path in the current public release is Windows + NVIDIA CUDA + bnb4/NF4 QLoRA. Non-CUDA machines can open the console, manage datasets, and scan models, but should not expect to produce a fine-tuned Adapter in this release.

## How much VRAM does a 27B model need?

The validated 27B test-run setup uses about 24 GB of VRAM. Longer sequences, larger batches, higher LoRA rank, and formal training can change the requirement. The console gives model-size guidance from the current hardware, but you should still start with a 1-step test run.

## Where should model files come from?

Use official model releases, ModelScope, Hugging Face, or another model source you trust. LocalTune needs a Transformers-compatible base model directory, usually with `config.json`, tokenizer files, and `safetensors` or `.bin` weights.

GGUF files are mainly for inference runtimes and cannot be used as the current fine-tuning input.

## What is uv, and why do I need it?

`uv` is a Python environment and dependency manager. LocalTune uses it to create `.venv`, install dependencies, and run scripts. The startup script prepares as much as it can, but the machine still needs a working `uv`.

## Does the startup script access the network?

The first start may download Python dependencies, the project-local Node.js toolchain, or CUDA PyTorch / bitsandbytes packages on NVIDIA CUDA machines. Normal training does not upload your models, datasets, or training outputs.

## Why start with a short test run?

A short test run usually runs only a few steps. It confirms model loading, dataset reading, loss logging, and artifact wiring before a formal run. It is a readiness check, not the final model-quality run.

