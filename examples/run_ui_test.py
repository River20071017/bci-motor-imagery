"""界面测试启动入口（C）。

运行方式（在仓库根目录，使用已安装 PySide6 的虚拟环境）：

    python examples/run_ui_test.py

本脚本把 ``src`` 加入 ``sys.path``，然后构造 C 的主窗口，并注入界面测试用的
模拟数据源与模拟解码器（均在 ``examples/`` 下）。接入 A 的真实数据源与 B 的真实
解码器后，仅需替换这里的注入对象，界面与其余模块无需改动。

注意：本入口与注入的模拟器仅用于 C 开发桌面界面，不代表任何真实采集 / 算法输出。
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PySide6.QtWidgets import QApplication

from eeg_app.ui.main_window import MainWindow
from mock_source import MockEegSource
from mock_decoder import MockDecoder


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("EEG MI 连续解码 · 界面测试 (C)")

    source = MockEegSource()
    decoder = MockDecoder(sample_rate_hz=source.info.sample_rate_hz)
    window = MainWindow(source=source, decoder=decoder, project_dir=_ROOT)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
