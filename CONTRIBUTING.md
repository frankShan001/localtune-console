# Contributing

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

Thanks for improving LocalTune Console.

## Development Principles

- Pull requests must not include model weights, local datasets, logs, checkpoints, or adapters.
- Do not add data that you are not authorized to redistribute or that contains sensitive information.
- Prefer small, testable changes.
- Keep configuration examples portable. Avoid machine-specific absolute paths.
- Update both README languages when user-facing behavior changes.
- Treat Unsloth and NVFP4 as optional/experimental paths unless a change is verified locally.

## Local Checks

Run before opening a pull request:

```powershell
.\.venv\Scripts\python.exe scripts\release_check.py --skip-data
```

For training-related changes, run a smoke training job from the web console or CLI:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --branch bnb4 --no-fallback
```

## Pull Request Checklist

- No local data/model/output files are included.
- `README.md` and `README.zh-CN.md` are updated when user-facing behavior changes.
- `configs/model_config.example.yaml` stays generic.
- New runtime files are ignored by `.gitignore`.
- The dashboard still starts and the route list is valid.
- The English and Simplified Chinese documents remain aligned.
