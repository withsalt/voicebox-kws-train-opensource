"""路径/文件名工具（纯标准库）。"""

from __future__ import annotations

import re

# 路径分隔符、通配符、引号、尖括号、竖线与空白都视为不安全字符。
_UNSAFE = re.compile(r'[\\/:*?"<>|\s]+')


def safe_filename(text: str, max_len: int = 80) -> str:
    """把任意文本（唤醒词等）清洗为安全的文件/目录名。

    保留中文与常见字符，替换路径分隔符与空白为下划线。
    """
    cleaned = _UNSAFE.sub("_", text.strip()).strip("_.")
    cleaned = cleaned or "untitled"
    return cleaned[:max_len]
