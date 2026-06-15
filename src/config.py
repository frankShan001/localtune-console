# ============================================================
# 配置管理模块
# 加载 YAML, 解析量化分支, 校验配置一致性, 提供运行时配置对象
# ============================================================

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List

import yaml

from src.constants import DEFAULT_OUTPUT_DIR

from src.env_detect import EnvInfo, check_branch_compatibility

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_local_path(value: str) -> str:
    """Resolve explicit local paths relative to the project root."""
    if not value:
        return value
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if expanded.startswith(("./", ".\\")) or Path(expanded).is_absolute():
        path = Path(expanded)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)
    return expanded


def _resolve_model_path(model_cfg: Dict, branch_raw: Dict, target_branch: str) -> str:
    """Resolve the selected model path for a quantization branch."""
    catalog = model_cfg.get("models") or model_cfg.get("catalog") or {}
    selected_model = model_cfg.get("active_model")
    if catalog:
        if not selected_model or selected_model not in catalog:
            selected_model = next(iter(catalog), "")
        item = catalog.get(selected_model, {}) or {}
        branch_paths = item.get("paths") or item.get("branch_paths") or item.get("model_paths") or {}
        model_path = branch_paths.get(target_branch) or item.get("path") or item.get("model_path")
        if model_path:
            return model_path
    return branch_raw.get("model_path", model_cfg.get("name", ""))


def _resolve_model_name(model_cfg: Dict) -> str:
    catalog = model_cfg.get("models") or model_cfg.get("catalog") or {}
    selected_model = model_cfg.get("active_model")
    if catalog:
        if not selected_model or selected_model not in catalog:
            selected_model = next(iter(catalog), "")
        item = catalog.get(selected_model, {}) or {}
        return item.get("name") or selected_model
    return model_cfg.get("name", "")


# ============================================================
# 数据类
# ============================================================

@dataclass
class QuantBranchConfig:
    """单个量化分支的配置"""
    name: str                               # "bnb4" | optional experimental branches
    description: str = ""
    model_path: str = ""
    framework: str = "unsloth"              # "unsloth" | "peft"
    load_mode: str = ""                     # "unsloth_qlora" | "bnb_qlora"
    quant_type: str = ""                    # "nf4"
    load_in_4bit: bool = True
    bnb_config: Optional[Dict] = None
    lora_dtype: str = "bfloat16"
    vram_estimate_gb: float = 17.5
    requires: Dict = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    """运行时配置（经过分支解析后的最终配置）"""
    # 量化分支
    branch_name: str = ""
    branch: Optional[QuantBranchConfig] = None

    # 模型
    model_name: str = ""
    model_path: str = ""                    # 最终使用的模型路径
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    lora_target_modules: List[str] = field(default_factory=list)
    lora_use_rslora: bool = True
    lora_use_dora: bool = False

    # 数据
    train_file: str = ""
    val_file: str = ""
    test_file: str = ""
    max_seq_length: int = 512
    min_seq_length: int = 128

    # 训练
    output_dir: str = ""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_steps: int = -1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict = field(default_factory=lambda: {"use_reentrant": False})
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    optimizer: str = "adamw_torch"
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True

    # 评估
    do_eval: bool = True

    # 框架
    framework: str = "unsloth"

    # 环境
    auto_detect: bool = True
    fallback_chain: List[str] = field(default_factory=lambda: ["bnb4"])


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: Path) -> dict:
    """加载原始 YAML 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_branch(raw_config: dict, env_info: EnvInfo = None, override_branch: str = None) -> RuntimeConfig:
    """
    解析量化分支，合并分支配置与基础配置，生成 RuntimeConfig

    Args:
        raw_config: 原始 YAML 字典
        env_info: 环境检测结果（可选，用于自动回退）
        override_branch: 命令行覆盖的分支名（可选）

    Returns:
        RuntimeConfig: 合并后的运行时配置
    """
    quant_cfg = raw_config.get("quantization", {})
    model_cfg = raw_config.get("model", {})
    data_cfg = raw_config.get("data", {})
    train_cfg = raw_config.get("training", {})
    eval_cfg = raw_config.get("evaluation", {})

    # 确定目标分支
    target_branch = override_branch or quant_cfg.get("active_branch", "bnb4")
    fallback_chain = quant_cfg.get("fallback_chain", ["bnb4"])
    auto_detect = quant_cfg.get("auto_detect", True)

    # 环境检测 + 自动回退
    if auto_detect and env_info is not None:
        ok, reason = check_branch_compatibility(target_branch, env_info)
        if not ok:
            logger.warning(f"分支 {target_branch} 不兼容: {reason}")
            # 沿 fallback_chain 回退
            start_idx = fallback_chain.index(target_branch) + 1 if target_branch in fallback_chain else 0
            for candidate in fallback_chain[start_idx:]:
                ok2, reason2 = check_branch_compatibility(candidate, env_info)
                if ok2:
                    logger.info(f"回退到分支: {candidate} ({reason2})")
                    target_branch = candidate
                    break
            else:
                logger.error("无可用的量化分支!")
                # 仍然使用目标分支，让后续加载时自然报错

    # 获取分支配置
    branches = quant_cfg.get("branches", {})
    branch_raw = branches.get(target_branch, {})

    # 构建分支配置对象
    branch_config = QuantBranchConfig(
        name=target_branch,
        description=branch_raw.get("description", ""),
        model_path=_resolve_local_path(_resolve_model_path(model_cfg, branch_raw, target_branch)),
        framework=branch_raw.get("framework", "unsloth"),
        load_mode=branch_raw.get("load_mode", "bnb_qlora"),
        quant_type=branch_raw.get("quant_type", "nf4"),
        load_in_4bit=branch_raw.get("load_in_4bit", True),
        bnb_config=branch_raw.get("bnb_config"),
        lora_dtype=branch_raw.get("lora_dtype", "bfloat16"),
        vram_estimate_gb=branch_raw.get("vram_estimate_gb", 17.5),
        requires=branch_raw.get("requires", {}),
    )

    # LoRA 配置
    lora_cfg = model_cfg.get("lora", {})

    # 构建运行时配置
    config = RuntimeConfig(
        branch_name=target_branch,
        branch=branch_config,
        model_name=_resolve_model_name(model_cfg),
        model_path=branch_config.model_path,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        lora_r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        lora_bias=lora_cfg.get("bias", "none"),
        lora_task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        lora_target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_use_rslora=lora_cfg.get("use_rslora", True),
        lora_use_dora=lora_cfg.get("use_dora", False),
        train_file=_resolve_local_path(data_cfg.get("train_file", "")),
        val_file=_resolve_local_path(data_cfg.get("val_file", "")),
        test_file=_resolve_local_path(data_cfg.get("test_file", "")),
        max_seq_length=train_cfg.get("max_seq_length", 512),
        min_seq_length=data_cfg.get("min_seq_length", 128),
        output_dir=_resolve_local_path(train_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        max_steps=train_cfg.get("max_steps", -1),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=train_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        tf32=train_cfg.get("tf32", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 500),
        eval_steps=train_cfg.get("eval_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        optimizer=train_cfg.get("optimizer", "adamw_torch"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
        do_eval=eval_cfg.get("do_eval", True),
        framework=branch_config.framework,
        auto_detect=auto_detect,
        fallback_chain=fallback_chain,
    )

    logger.info(f"激活分支: {config.branch_name} ({config.branch.description})")
    logger.info(f"框架: {config.framework}")
    logger.info(f"模型路径: {config.model_path}")
    logger.info(f"加载模式: {config.branch.load_mode}")
    logger.info(f"量化类型: {config.branch.quant_type}")
    logger.info(f"预估VRAM: {config.branch.vram_estimate_gb}GB")

    return config


def print_config_summary(config: RuntimeConfig):
    """打印配置摘要"""
    logger.info("=" * 60)
    logger.info("训练配置摘要")
    logger.info("=" * 60)
    logger.info(f"  量化分支:    {config.branch_name}")
    logger.info(f"  模型路径:    {config.model_path}")
    logger.info(f"  框架:        {config.framework}")
    logger.info(f"  LoRA r:      {config.lora_r}")
    logger.info(f"  LoRA alpha:  {config.lora_alpha}")
    logger.info(f"  最大序列长:  {config.max_seq_length}")
    logger.info(f"  学习率:      {config.learning_rate}")
    logger.info(f"  训练轮数:    {config.num_train_epochs}")
    logger.info(f"  最大步数:    {config.max_steps}")
    logger.info(f"  批大小:      {config.per_device_train_batch_size}")
    logger.info(f"  梯度累积:    {config.gradient_accumulation_steps}")
    logger.info(f"  预估VRAM:    {config.branch.vram_estimate_gb}GB")
    logger.info("=" * 60)
