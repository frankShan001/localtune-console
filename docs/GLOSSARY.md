# Glossary

[English](GLOSSARY.md) | [简体中文](GLOSSARY.zh-CN.md)

| Term | Meaning |
| --- | --- |
| Fine-tuning | Continuing training from a base model so it adapts more reliably to a style, format, task, or domain. |
| Base model | The original model loaded before fine-tuning, such as a Transformers-format Qwen or Llama directory. |
| LoRA | Low-Rank Adaptation, a method that trains a small set of additional parameters to reduce memory and storage cost. |
| QLoRA | A fine-tuning method that combines quantized loading with LoRA, often used to train larger models on a single GPU. |
| PEFT | Parameter-Efficient Fine-Tuning, a family of methods that includes LoRA. |
| bitsandbytes | A library commonly used for 4-bit / 8-bit quantized loading. The current bnb4/NF4 QLoRA path depends on it. |
| NF4 / bnb4 | A bitsandbytes 4-bit loading mode. LocalTune currently validates bnb4/NF4 QLoRA. |
| Adapter | The small fine-tuned weight delta loaded together with the base model during inference. |
| Checkpoint | A saved training state that can be used to resume training or compare intermediate results. |
| JSONL | A text format with one JSON object per line, suitable for streaming training samples. |
| Dataset Profile | LocalTune's management unit for one train/val/test dataset set. |
| Transformers-compatible model directory | A model directory with `config.json`, tokenizer files, and `safetensors` / `.bin` weights. |
| GGUF | A common inference model format for runtimes such as llama.cpp. It cannot be used as the current fine-tuning input. |
| Short test run | A small-step training run that checks model loading, dataset reading, loss logging, and artifact wiring before a formal run. |

