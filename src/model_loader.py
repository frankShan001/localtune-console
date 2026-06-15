"""Model loading and LoRA setup for the supported LocalTune training paths.

The first public release intentionally supports the NVIDIA CUDA +
bitsandbytes NF4 QLoRA path, with an optional Unsloth loader when explicitly
selected. Experimental NVFP4 loaders were removed from this module because
LocalTune does not expose NVFP4 as a fine-tuning entry point.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

from src.config import RuntimeConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config: RuntimeConfig) -> Tuple:
    """Load the base model and tokenizer for the selected supported branch."""
    branch = config.branch
    logger.info("Loading model: branch=%s load_mode=%s", branch.name, branch.load_mode)

    try:
        if branch.load_mode == "bnb_qlora":
            model, tokenizer = _load_bnb_qlora(config)
        elif branch.load_mode == "unsloth_qlora":
            model, tokenizer = _load_unsloth_qlora(config)
        else:
            raise ValueError(f"Unsupported load mode for LocalTune training: {branch.load_mode}")
        return model, tokenizer, branch.name
    except Exception as exc:
        logger.error("Branch %s failed to load: %s", branch.name, exc)
        return _fallback_load(config, str(exc))


def _load_unsloth_qlora(config: RuntimeConfig) -> Tuple:
    """Load a 4-bit QLoRA model with Unsloth for an explicit Unsloth branch."""
    from unsloth import FastLanguageModel

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    logger.info("Loading model with Unsloth: %s", config.model_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_path,
        max_seq_length=config.max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
        device_map="auto",
    )
    logger.info("Model loaded with Unsloth")
    return model, tokenizer


def _load_bnb_qlora(config: RuntimeConfig) -> Tuple:
    """Load a model with Transformers + bitsandbytes NF4 for PEFT QLoRA."""
    if config.framework == "unsloth":
        return _load_unsloth_qlora(config)

    logger.info("Loading model with Transformers + bitsandbytes NF4: %s", config.model_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=config.trust_remote_code,
        )
    except ValueError as exc:
        if "Some modules are dispatched on the CPU or the disk" in str(exc):
            raise RuntimeError(
                "BNB4 strict GPU load failed: the model does not fit fully in "
                "available VRAM after 4-bit quantization. CPU/disk offload can "
                "load parts of a model for inference, but it is not a reliable "
                "local QLoRA training path on this machine."
            ) from exc
        raise

    logger.info("Model loaded with Transformers + bitsandbytes NF4")
    return model, tokenizer


def apply_lora(model, config: RuntimeConfig):
    """Attach LoRA adapters using the configured training framework."""
    if config.framework == "unsloth":
        try:
            return _apply_lora_unsloth(model, config)
        except (ImportError, ModuleNotFoundError):
            logger.warning("Unsloth is unavailable; falling back to PEFT LoRA")
    return _apply_lora_peft(model, config)


def _apply_lora_unsloth(model, config: RuntimeConfig):
    """Apply LoRA with Unsloth."""
    from unsloth import FastLanguageModel

    logger.info("Applying LoRA with Unsloth: r=%s alpha=%s", config.lora_r, config.lora_alpha)
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=config.lora_task_type,
        target_modules=config.lora_target_modules,
        use_rslora=config.lora_use_rslora,
        use_dora=config.lora_use_dora,
    )
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %s / %s (%.2f%%)", trainable, total, trainable / total * 100)
    return model


def _apply_lora_peft(model, config: RuntimeConfig):
    """Apply LoRA with PEFT."""
    from peft import LoraConfig, TaskType, get_peft_model

    logger.info("Applying LoRA with PEFT: r=%s alpha=%s", config.lora_r, config.lora_alpha)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _fallback_load(config: RuntimeConfig, error_msg: str) -> Tuple:
    """Try configured fallback branches after the requested branch fails."""
    from pathlib import Path

    from src.config import load_config, resolve_branch
    from src.env_detect import check_branch_compatibility, detect_environment

    chain = config.fallback_chain
    current = config.branch_name
    start_idx = chain.index(current) + 1 if current in chain else 0
    env_info = detect_environment()

    for candidate in chain[start_idx:]:
        ok, reason = check_branch_compatibility(candidate, env_info)
        if not ok:
            logger.warning("Fallback branch %s is not compatible: %s", candidate, reason)
            continue

        logger.info("Trying fallback branch %s (%s)", candidate, reason)
        try:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "model_config.yaml"
            raw_config = load_config(config_path)
            new_config = resolve_branch(raw_config, env_info, override_branch=candidate)

            if new_config.branch.load_mode == "bnb_qlora":
                model, tokenizer = _load_bnb_qlora(new_config)
            elif new_config.branch.load_mode == "unsloth_qlora":
                model, tokenizer = _load_unsloth_qlora(new_config)
            else:
                logger.warning("Skipping unsupported fallback load mode: %s", new_config.branch.load_mode)
                continue

            logger.info("Fallback succeeded with branch %s", candidate)
            return model, tokenizer, candidate
        except Exception as exc:
            logger.warning("Fallback branch %s failed: %s", candidate, exc)

    raise RuntimeError(
        f"All model loading branches failed. Original error: {error_msg}. "
        f"Tried fallbacks: {chain[start_idx:]}"
    )
