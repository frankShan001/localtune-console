# Troubleshooting

[English](TROUBLESHOOTING.md) | [简体中文](TROUBLESHOOTING.zh-CN.md)

## `uv sync` is slow

The first sync installs large dependencies such as PyTorch and Transformers. The project does not hardcode a PyPI mirror in `pyproject.toml`; configure a mirror in your own uv settings or environment variables if you need one.

## Startup reports an invalid Node.js or npm version

LocalTune first uses a system Node.js / npm when the versions are high enough. If they are missing or too old, the startup script installs Node.js 22 into the project `.venv` so it does not require a global system install.

## Model scanning finds no model

Select a base model directory or one of its parent directories. A trainable Transformers model directory usually contains `config.json`, tokenizer files, and `safetensors` or `.bin` weights. A directory that only contains GGUF files is not accepted by the current fine-tuning entry.

## Training fails with CUDA out of memory

Lower `max_seq_length`, batch size, gradient accumulation, or LoRA rank, and check whether another process is using VRAM. Run a 1-step test run before formal training to catch memory issues quickly.

## Loss or logs do not update

Confirm the training task is still running and check `logging_steps`. The Training page focuses on the current task; historical logs are available from Task Center.

## The dashboard does not open automatically

The startup script waits for the backend health check before opening the browser. If the browser does not open, visit:

```text
http://127.0.0.1:6543
```

If you changed the port, use the URL printed by the startup script.

