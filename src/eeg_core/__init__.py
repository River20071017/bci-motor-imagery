"""底层核心：契约、处理、解码。

只允许标准库 + numpy/scipy/sklearn；**不依赖 ``eeg_app`` / ``training`` / 文件 / 线程 / UI**。
分层规则见 ``docs/00_架构总览.md``，并由 ``tests/unit/core/test_layering.py`` 自动检查。
"""
