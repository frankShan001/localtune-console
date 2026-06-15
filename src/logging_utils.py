# ============================================================
# 增强日志工具模块
# 提供结构化日志、日志查询、训练指标记录功能
# ============================================================

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class EnhancedLogger:
    """
    增强日志类

    功能:
    - 结构化日志 (JSON 格式)
    - 训练指标记录
    - 日志查询
    - 日志统计
    """

    def __init__(
        self,
        project_name: str = "localtune-console",
        log_dir: str = None,
        level: str = "INFO"
    ):
        """
        初始化增强日志器

        Args:
            project_name: 项目名称
            log_dir: 日志目录
            level: 日志级别
        """
        self.project_name = project_name
        self.log_dir = Path(log_dir or Path(__file__).parent.parent / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.level = getattr(logging, level.upper())

        # 日志文件
        timestamp = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"{project_name}_{timestamp}.log"
        self.json_log_file = self.log_dir / f"{project_name}_{timestamp}.jsonl"

        # 初始化日志
        self._init_logging()

        self.logger = logging.getLogger(project_name)

    def _init_logging(self):
        """初始化日志系统"""
        # 清除已有 handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 配置日志
        logging.basicConfig(
            level=self.level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def get_logger(self, name: str = None):
        """获取 logger"""
        return logging.getLogger(name or self.project_name)

    def log_structured(self, level: str, message: str, **kwargs):
        """
        记录结构化日志

        Args:
            level: 日志级别
            message: 消息
            **kwargs: 额外字段
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            **kwargs
        }

        # 写入 JSONL 文件
        with open(self.json_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # 写入标准日志
        log_func = getattr(self.logger, level.lower())
        log_func(message)

    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        self.log_structured(
            "INFO",
            "训练开始",
            event_type="training_start",
            config=config
        )

    def log_training_end(self, metrics: Dict[str, Any]):
        """记录训练结束"""
        self.log_structured(
            "INFO",
            "训练完成",
            event_type="training_end",
            metrics=metrics
        )

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        记录训练指标

        Args:
            metrics: 指标字典
            step: 步数
        """
        self.log_structured(
            "INFO",
            f"训练步 {step}",
            event_type="training_step",
            step=step,
            **metrics
        )

    def query_logs(
        self,
        level: str = None,
        limit: int = 100
    ):
        """
        查询日志

        Args:
            level: 日志级别过滤
            limit: 返回条数

        Returns:
            日志列表
        """
        if not self.json_log_file.exists():
            return []

        results = []
        with open(self.json_log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)

                    # 级别过滤
                    if level and entry.get("level") != level.upper():
                        continue

                    results.append(entry)

                    if len(results) >= limit:
                        break
                except Exception:
                    continue

        return results

    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计"""
        stats = {
            "log_file": str(self.log_file),
            "json_log_file": str(self.json_log_file),
            "log_count": {}
        }

        if self.json_log_file.exists():
            with open(self.json_log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        level = entry.get("level", "UNKNOWN")
                        stats["log_count"][level] = stats["log_count"].get(level, 0) + 1
                    except Exception:
                        continue

        return stats


# 全局 logger 实例
_global_logger = None


def get_enhanced_logger(project_name: str = "localtune-console", **kwargs) -> EnhancedLogger:
    """获取全局增强日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = EnhancedLogger(project_name=project_name, **kwargs)
    return _global_logger


# 兼容旧接口
def setup_logging(prefix: str = "train", log_dir: str = None) -> logging.Logger:
    """兼容旧接口"""
    logger = EnhancedLogger(project_name=prefix, log_dir=log_dir)
    return logger.get_logger()


def get_logger(name: str = None) -> logging.Logger:
    """获取命名 logger"""
    return logging.getLogger(name)
