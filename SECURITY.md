# Security Policy

[English](SECURITY.md) | [简体中文](SECURITY.zh-CN.md)

## Supported Versions

The `main` branch is the supported development line.

## Reporting a Vulnerability

Please open a private security advisory or contact the maintainers directly if repository hosting supports it.

Do not post secrets, tokens, private dataset samples, or proprietary model paths in public issues.

## Data and Model Safety

This project is designed for local fine-tuning. Users are responsible for:

- Verifying the license of model weights they download.
- Verifying the license and provenance of training data.
- Keeping private data out of version control and public sharing channels.
- Keeping the unauthenticated dashboard bound to localhost or placing it behind access controls.
- Loading `trust_remote_code` models only from trusted sources.
- Reviewing generated model outputs before redistribution.
