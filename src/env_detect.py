# ============================================================
# 环境检测模块
# Detect accelerator backend and training dependencies.
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EnvInfo:
    """运行环境信息"""
    compute_backend: str = "cpu"
    accelerator_available: bool = False
    cuda_version: Optional[str] = None
    cuda_major: int = 0
    cuda_minor: int = 0
    gpu_name: Optional[str] = None
    gpu_arch: Optional[str] = None          # "blackwell", "ada_lovelace", "ampere" etc
    gpu_vram_gb: Optional[float] = None
    compute_capability: Optional[str] = None  # e.g. "12.0" for RTX 5090
    has_unsloth: bool = False
    has_bitsandbytes: bool = False
    has_bf16: bool = False
    pytorch_version: Optional[str] = None
    python_version: Optional[str] = None


def detect_environment() -> EnvInfo:
    """
    自动检测运行环境

    Returns:
        EnvInfo: 完整环境信息
    """
    env = EnvInfo()

    # Python 版本
    import sys
    env.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # PyTorch 版本
    try:
        import torch
        env.pytorch_version = torch.__version__
    except ImportError:
        logger.error("PyTorch 未安装!")
        return env

    # Accelerator backend
    if torch.cuda.is_available():
        env.compute_backend = "cuda"
        env.accelerator_available = True
        env.cuda_version = torch.version.cuda
        if env.cuda_version:
            parts = env.cuda_version.split(".")
            env.cuda_major = int(parts[0])
            env.cuda_minor = int(parts[1]) if len(parts) > 1 else 0

        # GPU 信息
        try:
            props = torch.cuda.get_device_properties(0)
            env.gpu_name = props.name
            env.gpu_vram_gb = round(props.total_memory / (1024**3), 1)  # total_memory单位是bytes
            cc = props.major, props.minor
            env.compute_capability = f"{cc[0]}.{cc[1]}"
            env.gpu_arch = _detect_gpu_arch(cc)
        except Exception as e:
            logger.warning(f"无法获取GPU属性: {e}")

        # bf16 支持
        env.has_bf16 = torch.cuda.is_bf16_supported()
    else:
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        xpu_backend = getattr(torch, "xpu", None)
        if mps_backend and mps_backend.is_available():
            env.compute_backend = "mps"
            env.accelerator_available = True
            env.gpu_name = "Apple Metal / MPS"
            env.gpu_arch = "mps"
        elif xpu_backend and xpu_backend.is_available():
            env.compute_backend = "xpu"
            env.accelerator_available = True
            env.gpu_name = xpu_backend.get_device_name(0)
            env.gpu_arch = "xpu"
        else:
            logger.warning("No accelerator backend is available")

    # 依赖检测
    try:
        import importlib.util
        import importlib.metadata

        if importlib.util.find_spec("unsloth") is not None:
            env.has_unsloth = True
            logger.info(f"Unsloth: {importlib.metadata.version('unsloth')}")
        else:
            env.has_unsloth = False
            logger.warning("Unsloth 未安装")
    except Exception as e:
        env.has_unsloth = False
        logger.warning(f"Unsloth 不可用: {e}")

    try:
        import bitsandbytes
        env.has_bitsandbytes = True
        logger.info(f"bitsandbytes: {getattr(bitsandbytes, '__version__', 'unknown')}")
    except ImportError:
        env.has_bitsandbytes = False
        logger.warning("bitsandbytes 未安装")

    return env


def _detect_gpu_arch(compute_capability: Tuple[int, int]) -> str:
    """
    通过 compute_capability 识别 GPU 架构

    RTX 5090 = sm_120 (compute_capability 12.0)
    RTX 4090 = sm_89 (compute_capability 8.9)
    A100 = sm_80 (compute_capability 8.0)
    H100 = sm_90 (compute_capability 9.0)
    """
    major, minor = compute_capability
    if major >= 12:
        return "blackwell"
    elif major == 10:
        return "blackwell_datacenter"  # B100/B200
    elif major == 9:
        return "hopper"
    elif major == 8:
        if minor >= 6:
            return "ada_lovelace"  # RTX 4090
        else:
            return "ampere"  # A100
    else:
        return "older"


def check_branch_compatibility(branch_name: str, env: EnvInfo) -> Tuple[bool, str]:
    """
    检查当前环境是否兼容指定的量化分支

    Args:
        branch_name: "bnb4" | optional experimental branch
        env: 环境信息

    Returns:
        (is_compatible, reason)
    """
    if branch_name == "unsloth":
        if not env.has_unsloth:
            return False, "Unsloth 未安装或导入失败"
        if env.compute_backend != "cuda":
            return False, f"Unsloth QLoRA currently requires CUDA; detected backend: {env.compute_backend}"
        if env.cuda_major < 12:
            return False, f"CUDA版本过低: {env.cuda_version}"
        return True, "兼容 (Unsloth 4-bit QLoRA)"
    elif branch_name == "nvfp4":
        return False, "NVFP4 is not a supported training branch in LocalTune Console"
    elif branch_name == "bnb4":
        if not env.has_bitsandbytes:
            return False, "bitsandbytes未安装"
        if env.compute_backend != "cuda":
            return False, f"bitsandbytes QLoRA currently requires CUDA; detected backend: {env.compute_backend}"
        if env.cuda_major < 12:
            return False, f"CUDA版本过低: {env.cuda_version}"
        return True, "兼容 (标准NF4 QLoRA)"
    else:
        return False, f"未知分支: {branch_name}"


def print_env_summary(env: EnvInfo):
    """打印环境摘要"""
    logger.info("=" * 60)
    logger.info("环境检测摘要")
    logger.info("=" * 60)
    logger.info(f"  Python:         {env.python_version}")
    logger.info(f"  PyTorch:        {env.pytorch_version}")
    logger.info(f"  Backend:        {env.compute_backend}")
    logger.info(f"  CUDA:           {env.cuda_version}")
    logger.info(f"  GPU:            {env.gpu_name}")
    logger.info(f"  GPU架构:        {env.gpu_arch}")
    logger.info(f"  Compute Cap:    {env.compute_capability}")
    logger.info(f"  VRAM:           {env.gpu_vram_gb}GB")
    logger.info(f"  BF16支持:       {env.has_bf16}")
    logger.info(f"  Unsloth:        {env.has_unsloth}")
    logger.info(f"  bitsandbytes:   {env.has_bitsandbytes}")
    logger.info("=" * 60)

