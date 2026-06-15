# -*- coding: utf-8 -*-
"""Download a model snapshot into a local directory."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_model(model_name: str, cache_dir: str = None, provider: str = "modelscope"):
    """Download a model snapshot from ModelScope or Hugging Face."""
    provider = (provider or "modelscope").strip().lower()
    print(f"[INFO] Provider: {provider}")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Target directory: {cache_dir or 'default cache'}")

    if provider == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "ModelScope is not installed. Run `uv sync --extra modelscope` first."
            ) from exc
        model_dir = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            revision="master",
        )
    elif provider in {"huggingface", "hf"}:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Run the LocalTune launcher again."
            ) from exc
        model_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_dir=Path(cache_dir) / model_name if cache_dir else None,
            local_dir_use_symlinks=False,
        )
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

    print(f"[YES] Download completed: {model_dir}")
    return model_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a model from ModelScope or Hugging Face")
    parser.add_argument(
        "--provider",
        choices=["modelscope", "huggingface", "hf"],
        default="modelscope",
        help="Model provider (default: modelscope)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier, for example Qwen/Qwen3-14B",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "models"),
        help="Output directory (default: ./models)",
    )

    args = parser.parse_args()
    model_dir = download_model(args.model, args.output, args.provider)

    print("\n" + "=" * 60)
    print("[YES] Download completed")
    print(f"Model path: {model_dir}")
    print("=" * 60)
