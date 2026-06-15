# 故障排查

[English](TROUBLESHOOTING.md) | [简体中文](TROUBLESHOOTING.zh-CN.md)

## `uv sync` 很慢

首次同步会安装 PyTorch、Transformers 等较大的依赖。项目不在 `pyproject.toml` 中硬编码镜像；如果你需要国内镜像，请在自己的 uv 配置或环境变量中设置。

## 启动时报 Node.js 或 npm 版本不满足

LocalTune 会优先使用本机已安装且版本合格的 Node.js / npm。如果本机没有或版本过低，启动脚本会把 Node.js 22 安装到项目 `.venv` 中，尽量不占用系统盘全局环境。

## 模型目录扫描不到模型

请选择基础模型目录或它的上级目录。一个可训练的 Transformers 模型目录通常包含 `config.json`、tokenizer 文件和 `safetensors` 或 `.bin` 权重。只有 GGUF 文件的目录不会被当前微调入口接受。

## 训练报 CUDA out of memory

先降低 `max_seq_length`、batch size、gradient accumulation 或 LoRA rank，并确认没有其他进程占用显存。正式训练前先跑 1 step 试运行，能更快定位显存问题。

## loss 曲线或日志没有更新

确认训练任务仍在运行，并检查 `logging_steps`。如果任务已经结束，训练页只展示当前任务数据；历史日志请到任务中心查看对应任务。

## 管理台没有自动打开

启动脚本会等后端服务健康检查通过后再打开浏览器。如果浏览器没有打开，可以手动访问默认地址：

```text
http://127.0.0.1:6543
```

如果修改过端口，请以启动脚本输出为准。

