"""config.py
统一配置和日志初始化模块。

使用方式（在脚本顶部引入）：
    from config import setup_logging
    logger = setup_logging(__name__)

默认日志格式：
    2024-01-01 12:00:00 [INFO] module_name: message
"""

import logging
import os
import sys
from typing import Optional


# -------------------------
# 日志配置
# -------------------------

DEFAULT_LOG_FORMAT  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    初始化并返回一个带有统一格式的 Logger。

    Args:
        name:     Logger 名称（通常传入 __name__）
        level:    日志级别（默认 INFO）
        log_file: 若指定，则同时输出到文件

    Returns:
        配置好的 logging.Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件 handler（可选）
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # 禁止向 root logger 传播，避免在根 logger 也有 handler 时出现重复输出
    logger.propagate = False

    return logger


def get_root_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    配置根 Logger（影响全局所有 logger）。
    通常在脚本 main() 入口调用一次即可。
    """
    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    root = logging.getLogger()

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
        root.addHandler(fh)

    return root
