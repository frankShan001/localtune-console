# FAQ

[English](FAQ.md) | [简体中文](FAQ.zh-CN.md)

## LocalTune Console 能做什么？

LocalTune Console 聚焦本地大模型微调：管理本地基础模型、检查训练语料、启动试运行训练或正式训练、监控日志与 loss、管理 Adapter 和 Checkpoint，并做 Base / Adapter 推理对比。

它不是 RAG、知识库、聊天客户端，也不是模型 API 服务。

## 没有 NVIDIA CUDA 能训练吗？

当前公开版本已经验证的训练路线是 Windows + NVIDIA CUDA + bnb4/NF4 QLoRA。非 CUDA 设备可以打开管理台、管理语料和扫描模型，但不要期待当前版本能跑出微调 Adapter。

## 27B 模型需要多少显存？

项目验证过的 27B 试运行配置大约使用 24 GB 显存。更长序列、更大 batch、更高 LoRA 秩或正式训练都会改变显存需求。管理台会根据当前硬件给出模型规模提示，但最终仍建议先跑 1 step 试运行。

## 模型文件从哪里来？

你可以从模型官方发布渠道、ModelScope、Hugging Face 或其他可信模型仓库下载。LocalTune 需要的是 Transformers 兼容的基础模型目录，通常包含 `config.json`、tokenizer 文件和 `safetensors` 或 `.bin` 权重。

GGUF 文件主要用于推理引擎，不适合作为当前微调输入。

## uv 是什么，为什么需要它？

`uv` 是 Python 项目环境和依赖管理工具。LocalTune 用它创建 `.venv`、安装依赖并运行脚本。首次启动脚本会尽量自动准备项目环境，但机器上仍需要能运行 `uv`。

## 启动脚本会联网吗？

首次启动可能会联网安装 Python 依赖、项目内 Node.js 工具链，或在 NVIDIA CUDA 环境下补齐 CUDA PyTorch / bitsandbytes 依赖。正常训练不会上传模型、语料或训练产物。

## 为什么要先跑试运行训练？

试运行训练通常只跑很少的 step，用来确认模型能加载、语料能读取、loss 能记录、产物链路能生成。它不是为了得到最终效果，而是为了在正式训练前尽早发现环境、语料或显存问题。

