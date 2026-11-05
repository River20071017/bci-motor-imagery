"""处理链版本：模型包的 ``pipeline_version`` 唯一出处。

改动 ``filters.py`` / ``quality.py`` / ``windows.py`` / ``features.py`` 的**行为**时，
必须同时改这里；否则旧模型能在配置哈希不变的情况下被静默加载，训练/推理的特征语义
可能已经对不上。

首版是人工版本号，不自动哈希源码：打包成 exe 后源码文件不一定还在，静态常量更可靠。
"""

from __future__ import annotations

__all__ = ["PIPELINE_VERSION"]

PIPELINE_VERSION = "mi-lda-v1"
"""首版：A 的 8–13/13–30 Hz 因果滤波 + B 的 2 s 窗 18 维对数方差 + 三状态 LDA。"""
