#!/usr/bin/env python
# ============================================================
# 统一 QLoRA 训练入口 (v2.0)
#
# 支持 BNB4 分支化训练
# 自动环境检测 + 回退机制
#
# 用法:
#   python scripts/train.py                    # 使用配置文件中的 active_branch
#   python scripts/train.py --branch bnb4      # 指定 BNB4
#   python scripts/train.py --branch auto      # 自动检测+回退
#   python scripts/train.py --no-fallback      # 禁用回退
# ============================================================

import argparse
import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import load_config, resolve_branch, RuntimeConfig, print_config_summary
from src.env_detect import detect_environment, print_env_summary, check_branch_compatibility
from src.model_loader import load_model_and_tokenizer, apply_lora
from src.data_utils import load_and_prepare_dataset
from src.logging_utils import setup_logging, get_logger


def log_training_runtime_summary(logger, config, dataset, output_dir):
    logger.info("[run] -------- training runtime summary --------")
    logger.info(f"[run] branch={config.branch_name}")
    logger.info(f"[run] model_path={config.model_path}")
    logger.info(f"[run] output_dir={output_dir}")
    logger.info(f"[run] train_file={config.train_file}")
    logger.info(f"[run] val_file={config.val_file}")
    logger.info(f"[run] train_rows={len(dataset['train'])}, val_rows={len(dataset['val'])}")
    logger.info(
        "[run] training_args="
        f"max_steps={config.max_steps}, "
        f"epochs={config.num_train_epochs}, "
        f"seq={config.max_seq_length}, "
        f"batch={config.per_device_train_batch_size}, "
        f"grad_accum={config.gradient_accumulation_steps}, "
        f"logging_steps={config.logging_steps}, "
        f"save_steps={config.save_steps}, "
        f"lr={config.learning_rate}, "
        f"optimizer={config.optimizer}"
    )
    logger.info(
        "[run] lora="
        f"r={config.lora_r}, alpha={config.lora_alpha}, "
        f"dropout={config.lora_dropout}, targets={','.join(config.lora_target_modules)}"
    )
    logger.info("[run] ----------------------------------------")


class ConsoleMetricsCallback:
    def __init__(self, logger=None):
        self.logger = logger
        self.metrics_path = None

    def _write_metrics_record(self, state, logs):
        metrics_path = getattr(self, "metrics_path", None)
        if not metrics_path:
            return
        numeric_metrics = {
            key: value
            for key, value in logs.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if not numeric_metrics:
            return
        record = {
            "step": int(getattr(state, "global_step", 0) or 0),
            "epoch": logs.get("epoch"),
            "wall_time": time.time(),
            "metrics": numeric_metrics,
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        fields = []
        for key in ["loss", "eval_loss", "learning_rate", "grad_norm", "mean_token_accuracy", "epoch"]:
            if key in logs:
                value = logs[key]
                if isinstance(value, float):
                    fields.append(f"{key}={value:.6g}")
                else:
                    fields.append(f"{key}={value}")
        if fields:
            self.logger.info(f"[metrics] step={state.global_step} " + " ".join(fields))
        self._write_metrics_record(state, logs)


def main():
    parser = argparse.ArgumentParser(
        description="统一 QLoRA 训练入口 (BNB4/NF4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/train.py                    # 使用配置文件中的 active_branch
  python scripts/train.py --branch unsloth   # Unsloth 4-bit QLoRA (if import works)
  python scripts/train.py --branch bnb4      # BNB4 NF4 QLoRA (稳定)
  python scripts/train.py --branch auto      # 自动检测+回退
  python scripts/train.py --no-fallback      # 禁用自动回退
        """
    )
    parser.add_argument(
        "--config", type=str, default="configs/model_config.yaml",
        help="配置文件路径 (default: configs/model_config.yaml)"
    )
    parser.add_argument(
        "--branch", type=str, choices=["unsloth", "bnb4", "auto"],
        default=None, help="量化分支 (覆盖配置文件)"
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="禁用自动回退，加载失败直接报错"
    )
    parser.add_argument(
        "--resume-from-checkpoint", type=str,
        help="从指定 checkpoint 目录恢复训练"
    )

    args = parser.parse_args()

    # 1. 配置日志
    logger = setup_logging(prefix="train")
    logger.info("=" * 60)
    logger.info("BNB4/NF4 QLoRA 训练")
    logger.info("=" * 60)

    # 2. 环境检测
    logger.info("检测运行环境...")
    env_info = detect_environment()
    print_env_summary(env_info)

    # 3. 加载配置
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    raw_config = load_config(config_path)

    # 4. 确定量化分支
    override_branch = args.branch
    if override_branch == "auto":
        # 自动模式: 沿 fallback_chain 尝试
        chain = raw_config.get("quantization", {}).get("fallback_chain", ["bnb4"])
        selected = None
        for candidate in chain:
            ok, reason = check_branch_compatibility(candidate, env_info)
            if ok:
                selected = candidate
                logger.info(f"自动选择分支: {candidate} ({reason})")
                break
            else:
                logger.warning(f"分支 {candidate} 不兼容: {reason}")
        if selected is None:
            logger.error("无可用的量化分支!")
            sys.exit(1)
        override_branch = selected
    elif override_branch and args.no_fallback:
        # 指定分支 + 禁用回退
        ok, reason = check_branch_compatibility(override_branch, env_info)
        if not ok:
            logger.error(f"分支 {override_branch} 不兼容: {reason} (--no-fallback 已禁用回退)")
            sys.exit(1)

    # 5. 解析分支配置
    config = resolve_branch(raw_config, env_info, override_branch=override_branch)
    if args.no_fallback:
        config.fallback_chain = [config.branch_name]
    print_config_summary(config)

    # 6. VRAM 预警
    if config.branch.vram_estimate_gb > env_info.gpu_vram_gb:
        logger.warning(
            f"[WARN] 预估VRAM ({config.branch.vram_estimate_gb}GB) > "
            f"实际VRAM ({env_info.gpu_vram_gb}GB)，可能OOM!"
        )
        logger.warning("建议: 减小 max_seq_length 或 lora_r")

    # 7. 加载模型 (核心)
    logger.info("加载模型...")
    try:
        model, tokenizer, actual_branch = load_model_and_tokenizer(config)
        if actual_branch != config.branch_name:
            logger.info(f"实际使用分支: {actual_branch} (从 {config.branch_name} 回退)")
            config.branch_name = actual_branch
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        sys.exit(1)

    # 8. 应用 LoRA
    logger.info("应用 LoRA 适配器...")
    model = apply_lora(model, config)

    # 9. 加载数据
    logger.info("加载训练数据...")
    dataset = load_and_prepare_dataset(
        train_file=config.train_file,
        val_file=config.val_file,
        max_seq_length=config.max_seq_length,
        min_seq_length=config.min_seq_length,
        num_proc=max(1, config.dataloader_num_workers),
    )

    # 10. 创建 Trainer
    logger.info("创建训练器...")
    from transformers import TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(42)

    # 输出目录包含分支名
    output_dir = Path(config.output_dir) / config.branch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_training_runtime_summary(logger, config, dataset, output_dir)
    dashboard_job_id = os.environ.get("LOCALTUNE_JOB_ID", "").strip()
    run_name = dashboard_job_id or f"{config.branch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging_dir = output_dir / "runs" / run_name
    metrics_path = logging_dir / "metrics.jsonl"

    training_args = SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if config.do_eval else "no",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        bf16=config.bf16,
        fp16=config.fp16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs,
        optim=config.optimizer,
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        report_to=[],
        run_name=run_name,
        dataset_text_field="text",
        max_length=config.max_seq_length,
        dataset_num_proc=max(1, config.dataloader_num_workers),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if config.do_eval else None,
        args=training_args,
    )
    metrics_callback = type("MetricsLoggingCallback", (TrainerCallback, ConsoleMetricsCallback), {})()
    metrics_callback.logger = logger
    metrics_callback.metrics_path = metrics_path
    trainer.add_callback(metrics_callback)

    # 11. 训练
    logger.info("=" * 60)
    logger.info("开始训练!")
    logger.info(f"分支: {config.branch_name}")
    logger.info(f"输出: {output_dir}")
    logger.info(f"[run] metrics_file={metrics_path}")
    logger.info("=" * 60)

    resume_from_checkpoint = args.resume_from_checkpoint or raw_config.get("training", {}).get("resume_from_checkpoint")
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        if not checkpoint_path.is_dir():
            logger.error(f"训练检查点不存在: {checkpoint_path}")
            sys.exit(1)
        logger.info(f"从训练检查点恢复: {checkpoint_path}")
        train_result = trainer.train(resume_from_checkpoint=str(checkpoint_path))
    else:
        train_result = trainer.train()

    # 12. 保存
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"总步数: {train_result.global_step}")
    logger.info(f"最终损失: {train_result.training_loss:.4f}")
    logger.info(f"模型保存: {final_path}")
    logger.info("=" * 60)

    print(f"\n训练完成! 分支: {config.branch_name}")
    print(f"模型保存: {final_path}")
    print("\n下一步:")
    print("1. 在管理台的推理验证页面选择这个 Adapter 进行测试")
    print(f"2. 如需合并权重: python scripts/merge_lora.py --lora_path {final_path} --base_model <原始模型目录> --output_path <完整模型输出目录>")


if __name__ == "__main__":
    main()
