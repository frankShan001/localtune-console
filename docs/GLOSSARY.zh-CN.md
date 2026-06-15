# 术语表

[English](GLOSSARY.md) | [简体中文](GLOSSARY.zh-CN.md)

| 术语 | 含义 |
| --- | --- |
| 微调 | 在已有基础模型上继续训练，让模型更稳定地适配特定风格、格式、任务或领域。 |
| 基础模型 | 微调开始前加载的原始模型，例如 Transformers 格式的 Qwen、Llama 等模型目录。 |
| LoRA | 低秩适配训练方法，只训练少量新增参数，降低显存和存储成本。 |
| QLoRA | 结合量化加载和 LoRA 的微调方法，常用于在单机 GPU 上训练更大的模型。 |
| PEFT | Parameter-Efficient Fine-Tuning，参数高效微调方法集合，LoRA 是其中一种。 |
| bitsandbytes | 常用于 4-bit / 8-bit 量化加载的库，当前 bnb4/NF4 QLoRA 路线依赖它。 |
| NF4 / bnb4 | bitsandbytes 的 4-bit 量化加载方式。LocalTune 当前已验证的是 bnb4/NF4 QLoRA。 |
| Adapter | 微调后生成的小型权重增量，推理时与基础模型一起加载。 |
| Checkpoint | 训练过程中的阶段性保存点，可用于恢复训练或对比不同阶段效果。 |
| JSONL | 每行一个 JSON 对象的文本格式，适合大规模训练样本逐行读取。 |
| 语料档案 | LocalTune 对一组 train/val/test 训练文件的管理单位。 |
| Transformers 兼容模型目录 | 包含 `config.json`、tokenizer 和 `safetensors` / `.bin` 权重的模型目录。 |
| GGUF | 常见的推理模型格式，适合 llama.cpp 等推理引擎；当前不能作为微调输入。 |
| 试运行训练 | 用少量 step 检查模型加载、语料读取、loss 和产物链路是否正常，不用于追求最终效果。 |

