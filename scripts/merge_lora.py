# ============================================================
# 合并 LoRA 权重脚本
#
# 将 QLoRA 适配器合并回基础模型
# 输出可用于推理的完整模型
#
# 支持格式: HuggingFace .safetensors / GGUF (llama.cpp)
# ============================================================

"""
合并策略:

1. Unsloth 训练:
   - 使用 unsloth.merge_weights() 或手动合并

2. PEFT 训练:
   - 显式加载基础模型
   - 使用 peft.PeftModel.from_pretrained() 加载 LoRA Adapter
   - model.merge_and_unload()

3. 导出 GGUF:
   - 使用 llama.cpp 的 convert.py 转换
   - 量化后生成 .gguf 文件
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

import torch


def setup_logging():
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return log_file


def merge_unsloth(lora_path: Path, output_path: Path, dtype=torch.bfloat16):
    """合并 Unsloth LoRA 权重"""
    import logging
    logger = logging.getLogger(__name__)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth 未安装")
        return False

    logger.info(f"加载 Unsloth LoRA: {lora_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_path),
        dtype=dtype,
    )

    # 合并权重
    logger.info("合并 LoRA 权重...")
    model = FastLanguageModel.merge_weights(model)

    # 保存
    logger.info(f"保存合并后的模型: {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("✅ Unsloth 合并完成！")
    return True


def merge_peft(
    lora_path: Path,
    base_model_name: str,
    output_path: Path,
    dtype=torch.bfloat16,
    trust_remote_code: bool = False,
):
    """合并 PEFT LoRA 权重"""
    import logging
    logger = logging.getLogger(__name__)

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"加载 PEFT LoRA: {lora_path}")
    logger.info(f"基础模型: {base_model_name}")

    # 先加载原始基础模型，再叠加 LoRA Adapter。这样可以确保合并结果来自用户指定的原模型目录。
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, str(lora_path))

    # 合并到基础模型
    logger.info("合并 LoRA 权重到基础模型...")
    model = model.merge_and_unload()

    # 保存
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"保存合并后的模型: {output_path}")

    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="5GB",  # 分片保存，避免大文件问题
    )

    # 保存 tokenizer
    try:
        tokenizer_source = str(lora_path) if (lora_path / "tokenizer_config.json").exists() else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
        tokenizer.save_pretrained(str(output_path))
    except Exception as e:
        logger.warning(f"Tokenizer 保存失败: {e}")

    logger.info("✅ PEFT 合并完成！")
    return True


def export_to_gguf(merged_model_path: Path, gguf_output_path: Path,
                   quant_type: str = "q4_k_m"):
    """
    将合并后的模型导出为 GGUF 格式

    需要 llama.cpp
    pip install llama-cpp-python
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"导出 GGUF 格式...")
    logger.info(f"输入模型: {merged_model_path}")
    logger.info(f"输出路径: {gguf_output_path}")

    # 检查 llama.cpp
    try:
        import llama_cpp
        logger.info(f"llama.cpp 版本: {llama_cpp.__version__}")
    except ImportError:
        logger.warning("llama-cpp-python 未安装，GGUF 导出将不可用")
        logger.info("安装: pip install llama-cpp-python")
        return False

    # 使用 HuggingFace 上专门的 GGUF 转换工具
    # llama.cpp 提供了 convert.py
    llama_cpp_dir = Path(__file__).parent.parent / "llama.cpp"
    convert_script = llama_cpp_dir / "convert.py"

    if not convert_script.exists():
        logger.info("下载 llama.cpp...")
        import subprocess
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git",
                      str(llama_cpp_dir)], check=True)

    # 转换命令
    import subprocess
    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_model_path),
        "--outfile", str(gguf_output_path),
        "--outtype", quant_type,
    ]

    logger.info(f"执行: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        logger.info(f"✅ GGUF 导出完成: {gguf_output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"GGUF 导出失败: {e.stderr}")
        return False


def get_model_size(path: Path) -> str:
    """获取模型大小"""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return f"{total / 1024**3:.2f} GB"


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重")

    parser.add_argument("--lora_path", type=str, required=True,
                       help="LoRA 权重目录")
    parser.add_argument("--base_model", type=str, required=True,
                       help="原始基础模型目录或模型名称")
    parser.add_argument("--output_path", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"],
                       default="bf16", help="输出精度")
    parser.add_argument("--export_gguf", action="store_true",
                       help="同时导出 GGUF 格式")
    parser.add_argument("--gguf_quant", type=str,
                       choices=["q4_k_m", "q5_k_m", "q6_k", "q8_0"],
                       default="q4_k_m",
                       help="GGUF 量化级别")
    parser.add_argument("--framework", type=str,
                       choices=["unsloth", "peft", "auto"],
                       default="auto",
                       help="训练框架 (auto=自动检测)")

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repositories to execute custom Python code",
    )

    args = parser.parse_args()

    logger = setup_logging()

    lora_path = Path(args.lora_path)
    output_path = Path(args.output_path)

    if not lora_path.exists():
        logger.error(f"LoRA 路径不存在: {lora_path}")
        return

    logger.info(f"LoRA 路径: {lora_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"模型大小: {get_model_size(lora_path)}")

    # dtype
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # 自动检测框架
    if args.framework == "auto":
        # 检测是否有 Unsloth 特征
        if (lora_path / "unsloth_config.json").exists():
            framework = "unsloth"
        else:
            framework = "peft"
    else:
        framework = args.framework

    logger.info(f"检测到的框架: {framework}")

    # 合并
    if framework == "unsloth":
        success = merge_unsloth(lora_path, output_path, dtype)
    else:
        success = merge_peft(
            lora_path,
            args.base_model,
            output_path,
            dtype,
            trust_remote_code=args.trust_remote_code,
        )

    if success:
        # GGUF 导出
        if args.export_gguf:
            gguf_path = output_path.parent / f"{output_path.name}.{args.gguf_quant}.gguf"
            export_to_gguf(output_path, gguf_path, args.gguf_quant)

        # 输出摘要
        logger.info("=" * 60)
        logger.info("✅ 合并完成！")
        logger.info(f"输出路径: {output_path}")
        logger.info(f"大小: {get_model_size(output_path)}")
        logger.info("\n合并后的模型可由兼容的 Transformers 推理工具加载:")
        logger.info(f"  {output_path}")
        logger.info(f"\nLM Studio 导入:")
        logger.info(f"  1. 打开 LM Studio")
        logger.info(f"  2. 加载模型 -> 选择 {output_path}")
        logger.info(f"  3. 开始对话!")
        logger.info("=" * 60)
    else:
        logger.error("❌ 合并失败！")


if __name__ == "__main__":
    main()
