"""界面测试启动入口（C）。

运行方式（在仓库根目录，使用已安装 PySide6 的虚拟环境）：

    python examples/run_ui_test.py

本脚本把 ``src`` 加入 ``sys.path``，然后构造 C 的主窗口，并注入界面测试用的
模拟数据源、模拟解码器与**模拟结果存档**（均在 ``examples/`` 下）。接入 A 的真实
数据源、B 的真实解码器、D 的真实存档后，仅需替换这里的注入对象，界面与其余模块
无需改动。

存档部分说明（与 A 的交接）：
- 真实链路由 A 的组装入口在工作线程构造 ``JsonlResultSink``，并调用
  ``MainWindow.attach_result_sink(sink)`` 交给界面；
- 界面只读 ``sink.status`` 显示状态，不写文件、不调用 ``close()``；
- 本脚本用 ``MockResultSink`` 模拟状态变化，因此窗口顶部与存档状态条都标注了
  “界面测试 / 模拟”，**不代表真实存档链路已接通**。

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

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from eeg_app.ui.main_window import MainWindow
from mock_source import MockEegSource
from mock_decoder import MockDecoder
from mock_sink import MockResultSink

# 模拟会话目录（真实链路由 A/D 的 SessionArchive 创建，C 不创建目录）
DEMO_SESSION_DIR = os.path.join(_ROOT, "results_ui_test", "session_demo-ui-test")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("EEG MI 连续解码 · 界面测试 (C)")

    source = MockEegSource()
    decoder = MockDecoder(sample_rate_hz=source.info.sample_rate_hz)
    sink = MockResultSink(state="writing", directory=DEMO_SESSION_DIR)

    window = MainWindow(
        source=source,
        decoder=decoder,
        project_dir=_ROOT,
        result_sink=sink,
        test_mode=True,          # 明确标注：存档为模拟，非真实链路
    )

    # 模拟生命周期（仅用于界面走查，真实节奏由 A/D 的 sink 决定）：
    #   停止 → closing → 0.6 s 后 closed(completed=True)
    def _simulate_close() -> None:
        sink.set_state("closing", directory=DEMO_SESSION_DIR)
        QTimer.singleShot(
            600,
            lambda: sink.set_state(
                "closed", directory=DEMO_SESSION_DIR, completed=True
            ),
        )

    window.stop_btn.clicked.connect(_simulate_close)

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
