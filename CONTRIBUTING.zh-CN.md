# 参与贡献

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

感谢你帮助改进 LocalTune Console。

## 开发原则

- Pull Request 不应包含模型权重、本机语料、日志、Checkpoint 或 Adapter。
- 不得向项目仓库加入无权公开分发或包含敏感信息的数据。
- 优先提交范围小、可验证的修改。
- 示例配置必须可移植，不能包含本机绝对路径。
- 用户可见行为变化应同步更新中英文 README。
- 未经独立验证的 Unsloth 和 NVFP4 路线保持实验状态。
- 帮助和说明类文档必须同时更新英文与简体中文版本。

## 本地检查

提交 Pull Request 前运行：

```powershell
.\.venv\Scripts\python.exe scripts\release_check.py --skip-data
```

训练相关修改还应通过管理台或 CLI 运行一次冒烟训练：

```powershell
.\.venv\Scripts\python.exe scripts\train.py --branch bnb4 --no-fallback
```

## Pull Request 清单

- 不包含本机语料、模型和训练产物。
- 用户可见变更已同步更新 `README.md` 和 `README.zh-CN.md`。
- `configs/model_config.example.yaml` 保持通用。
- 新生成文件已被 `.gitignore` 排除。
- 管理台可以启动，Flask 路由检查通过。
