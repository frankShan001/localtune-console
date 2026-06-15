# ============================================================
# 数据工具模块
# 从 train_unsloth.py / train_peft.py 抽取公共数据逻辑
# ============================================================

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _load_datasets_package():
    try:
        from datasets import load_dataset as hf_load_dataset
    except Exception as exc:
        raise RuntimeError(
            "The Hugging Face datasets package is unavailable. "
            "Check the Python training environment before starting a run."
        ) from exc
    return hf_load_dataset


def format_chatml(example: dict) -> dict:
    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "")
            if role and content:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return {"text": "".join(parts)}

    if example.get("instruction") or example.get("output"):
        system = str(example.get("system") or "You are a helpful AI assistant.")
        instruction = str(example.get("instruction") or "")
        input_text = str(example.get("input") or "")
        user = f"{instruction}\n\n{input_text}".strip()
        assistant = str(example.get("output") or "")
        return {"text": (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
        )}

    if example.get("source") or example.get("target"):
        system = str(example.get("system") or "You are a helpful AI assistant.")
        instruction = str(example.get("instruction") or "Rewrite the following text.")
        user = f"{instruction}\n\n{example.get('source') or ''}".strip()
        assistant = str(example.get("target") or "")
        return {"text": (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
        )}

    """
    ChatML 格式转换 (Qwen官方推荐格式)

    输入示例:
    {
        "system": "你是猫腻，...",
        "user": "描述一个修仙世界",
        "assistant": "在浩瀚的宇宙中..."
    }

    输出格式 (ChatML):
    <|im_start|>system
    你是猫腻，...
    <|im_end|>
    <|im_start|>user
    描述一个修仙世界
    <|im_end|>
    <|im_start|>assistant
    在浩瀚的宇宙中...
    <|im_end|>
    """
    system = example.get("system", "你是一个有帮助的AI助手。")
    user = example.get("user", "")
    assistant = example.get("assistant", "")

    text = f"<|im_start|>system\n{system}<|im_end|>\n"
    text += f"<|im_start|>user\n{user}<|im_end|>\n"
    text += f"<|im_start|>assistant\n{assistant}<|im_end|>\n"

    return {"text": text}


def load_and_prepare_dataset(
    train_file: str,
    val_file: str,
    max_seq_length: int = 512,
    min_seq_length: int = 128,
    num_proc: int = 4,
    val_key: str = "val",
) -> Any:
    """
    加载并预处理数据集（统一版本）

    Args:
        train_file: 训练集 JSONL 文件路径
        val_file: 验证集 JSONL 文件路径
        max_seq_length: 最大序列长度
        min_seq_length: 最小序列长度
        num_proc: 并行处理进程数
        val_key: 验证集键名 ("val" 或 "validation")

    Returns:
        DatasetDict: {"train": ..., "val": ...}
    """
    logger.info(f"加载训练数据: {train_file}")
    logger.info(f"加载验证数据: {val_file}")

    hf_load_dataset = _load_datasets_package()
    dataset = hf_load_dataset("json", data_files={
        "train": train_file,
        val_key: val_file
    })

    # 格式化为 ChatML
    dataset = dataset.map(
        format_chatml,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
        desc="格式化为ChatML"
    )

    # Tokenize 数据
    def tokenize_function(examples, tokenizer):
        # 注意：这里需要从配置中获取 tokenizer，但为了简化，使用全局 tokenizer
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )

    # 暂时跳过 tokenization，让 Trainer 处理
    # 我们需要在外部传入 tokenizer
    logger.info("数据已格式化为 ChatML，将在训练时进行 tokenization")

    # 过滤过短/过长的序列 (按字符长度)
    # 注意: min_seq_length 和 max_seq_length 在这里是字符数
    # 对于验证模式，使用更小的 min_seq_length
    actual_min = min(50, min_seq_length)  # 至少 50 字符

    def filter_by_length(example):
        text_len = len(example["text"])
        return actual_min <= text_len <= max_seq_length * 10  # 宽松的上限

    dataset = dataset.filter(
        filter_by_length,
        num_proc=num_proc,
        desc="按长度过滤"
    )

    logger.info(f"训练样本数: {len(dataset['train'])}")
    logger.info(f"验证样本数: {len(dataset[val_key])}")

    return dataset
