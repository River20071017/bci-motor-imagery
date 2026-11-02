"""C 维护的主窗口：控制区 + 波形 + 时间轴 + 火柴人 + 状态。

布局对齐 PRD 第 3 节的界面结构：
- 顶栏：数据回放模式 / 数据文件 / 个人模型 / 输出目录 / 运行状态
- 主体：脑电滚动波形（左）｜ 火柴人与状态文字（右）
- 底部：模型预测轨 + 独立参考轨
- 控制条：开始 / 暂停 / 继续 / 停止 / 已处理时间 / 保存状态 / 打开结果目录

顶部醒目标注"界面测试模式"，避免把模拟数据误认为真实算法输出。

本窗口只渲染 ``DecisionFrame``（硬边界③：界面不改写结果，动画不得由参考标签触发）。
数据源与解码器通过构造参数注入，因此"接入 A 的数据"与"匹配 B 的算法"对界面透明。
"""
from __future__ import annotations

import os
import time
import uuid

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QSplitter, QFrame, QCheckBox,
)

from eeg_core.contracts import DEFAULT_CONFIG, MODEL_CHANNELS, DecisionState
from eeg_app.ui.playback import PlaybackWorker, DecoderProtocol, SourceProtocol
from eeg_app.ui.waveform import WaveformWidget
from eeg_app.ui.timeline import TimelineWidget
from eeg_app.ui.stick_figure import StickmanWidget
from eeg_app.ui.state_text import StatusPanel

# 默认显示的 8 个通道（PRD F06：FC3、FCz、FC4、C3、Cz、C4、CP3、CPz）
DEFAULT_DISPLAY_CHANNELS = list(MODEL_CHANNELS[:8])


def _new_session_id(prefix: str = "ui-test") -> str:
    """生成唯一会话标识，用于会话目录与决策编号。"""
    return f"{prefix}-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"


class MainWindow(QMainWindow):
    def __init__(
        self,
        source: SourceProtocol,
        decoder: DecoderProtocol,
        project_dir: str = "",
    ) -> None:
        super().__init__()
        if source is None or decoder is None:
            raise ValueError("MainWindow 需要注入 source 与 decoder")
        self.source = source
        self.decoder = decoder
        self.project_dir = project_dir or os.getcwd()
        self.worker = None
        self._expiry = QTimer(self)
        self._expiry.setSingleShot(True)
        self._expiry.timeout.connect(self._on_result_expired)
        self._running = False
        self._paused = False

        self.setWindowTitle("EEG 运动想象连续解码 · 界面测试框架 (C)")
        self.resize(1180, 760)
        self._build_ui()
        self._set_control_enabled(start=True, pause=False, resume=False, stop=False)

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 顶部警示横幅
        banner = QLabel(
            "⚠ 界面测试模式：当前使用模拟数据（数据源与解码器均为模拟），"
            "非真实采集 / 算法输出。仅供 C 开发桌面界面使用。"
        )
        banner.setStyleSheet(
            "background:#fff3cd; color:#856404; padding:6px; font-weight:bold;"
        )
        banner.setWordWrap(True)
        root.addWidget(banner)

        # 控制栏
        top = QHBoxLayout()
        self.data_edit = QLineEdit("模拟数据 (界面测试)")
        self.data_edit.setReadOnly(True)
        self.model_edit = QLineEdit("模拟模型 MOCK-lda-v1 (界面测试)")
        self.model_edit.setReadOnly(True)
        self.out_edit = QLineEdit(os.path.join(self.project_dir, "results_ui_test"))
        self.out_edit.setReadOnly(True)
        out_btn = QPushButton("选择输出目录")
        out_btn.clicked.connect(self._choose_output)
        top.addWidget(QLabel("数据文件"))
        top.addWidget(self.data_edit)
        top.addWidget(QLabel("个人模型"))
        top.addWidget(self.model_edit)
        top.addWidget(QLabel("输出目录"))
        top.addWidget(self.out_edit)
        top.addWidget(out_btn)
        root.addLayout(top)

        # 主体：波形 + 右栏
        splitter = QSplitter(Qt.Horizontal)
        disp = DEFAULT_CONFIG.display
        self.waveform = WaveformWidget(
            DEFAULT_DISPLAY_CHANNELS,
            sample_rate_hz=self.source.info.sample_rate_hz,
            window_sec=disp.waveform_span_sec,
        )
        right = QWidget()
        rcol = QVBoxLayout(right)
        self.stickman = StickmanWidget()
        self.status_panel = StatusPanel()
        rcol.addWidget(self.stickman, stretch=3)
        rcol.addWidget(self.status_panel, stretch=1)
        splitter.addWidget(self.waveform)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=3)

        # 时间轴（界面测试模式展示独立参考轨；接入真实模块后设 reference_mode=False）
        self.timeline = TimelineWidget(reference_mode=True)
        root.addWidget(self.timeline, stretch=1)

        # 控制条
        bottom = QHBoxLayout()
        self.start_btn = QPushButton("开始分析")
        self.pause_btn = QPushButton("暂停")
        self.resume_btn = QPushButton("继续")
        self.stop_btn = QPushButton("停止")
        self.processed_label = QLabel("已处理: 0.0 s")
        self.save_label = QLabel("保存状态: 存档模块待接入 (界面测试)")
        self.open_dir_btn = QPushButton("打开结果目录")
        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn.clicked.connect(self._on_pause)
        self.resume_btn.clicked.connect(self._on_resume)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_dir_btn.clicked.connect(self._on_open_dir)
        bottom.addWidget(self.start_btn)
        bottom.addWidget(self.pause_btn)
        bottom.addWidget(self.resume_btn)
        bottom.addWidget(self.stop_btn)
        bottom.addWidget(self.processed_label)
        bottom.addStretch(1)
        bottom.addWidget(self.save_label)
        bottom.addWidget(self.open_dir_btn)
        # N08 演示开关：勾选后只断决策、不断数据，可现场观察 750ms 自动静止
        self.n08_demo_cb = QCheckBox("模拟决策中断（演示 N08）")
        self.n08_demo_cb.setToolTip(
            "勾选后数据源继续流式供数（波形不停），但解码器暂停发布决策；\n"
            "超过 750 ms 无新有效决策，火柴人动作反馈自动静止（N08）。"
        )
        self.n08_demo_cb.stateChanged.connect(self._on_n08_demo_toggled)
        self.n08_demo_cb.setEnabled(False)
        bottom.addWidget(self.n08_demo_cb)
        root.addLayout(bottom)

        # 状态条
        self.statusBar().showMessage("就绪 · 界面测试模式")

    # ---------------- 控制 ----------------
    def _set_control_enabled(self, start, pause, resume, stop) -> None:
        self.start_btn.setEnabled(start)
        self.pause_btn.setEnabled(pause)
        self.resume_btn.setEnabled(resume)
        self.stop_btn.setEnabled(stop)

    def _choose_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    def _on_start(self) -> None:
        if self.worker and self._running:
            return
        self.waveform.clear()
        self.timeline.reset()
        session = _new_session_id("ui-test")
        self.worker = PlaybackWorker(source=self.source, decoder=self.decoder)
        self.worker.set_session(session)
        self.worker.chunk_ready.connect(self.waveform.append_chunk)
        self.worker.decision_ready.connect(self._on_decision)
        self.worker.playback_stopped.connect(lambda: self._finalize("已停止 · 结果过期"))
        self.worker.playback_finished.connect(lambda: self._finalize("文件结束，会话已完成"))
        self.worker.start()
        self._running = True
        self._paused = False
        self.n08_demo_cb.setEnabled(True)
        self._set_control_enabled(start=False, pause=True, resume=False, stop=True)
        self.statusBar().showMessage(f"运行中 · 会话 {session}")

    def _on_pause(self) -> None:
        if self.worker and self._running and not self._paused:
            self.worker.pause()
            self._paused = True
            self._expiry.stop()            # 停止结果有效期计时
            self.stickman.set_paused()     # N07：暂停后立即停止动作反馈（<250ms）
            self._set_control_enabled(start=False, pause=False, resume=True, stop=True)
            self.statusBar().showMessage("已暂停")

    def _on_resume(self) -> None:
        if self.worker and self._running and self._paused:
            self.worker.resume()
            self._paused = False
            self.stickman.clear_pause()   # 取消暂停冻结，等待下一个有效决策恢复动作反馈
            self._set_control_enabled(start=False, pause=True, resume=False, stop=True)
            self.statusBar().showMessage("继续运行")

    def _on_stop(self) -> None:
        if self.worker and self._running:
            self._expiry.stop()
            self.stickman.mark_stale()     # N07：停止后立即停止动作反馈（<250ms）
            self.worker.stop()

    def _finalize(self, message: str) -> None:
        if self.worker is None:
            return
        self.worker.quit()
        self.worker.wait()
        self.worker = None
        self._running = False
        self._paused = False
        self._expiry.stop()
        self.stickman.mark_stale()
        self.n08_demo_cb.setEnabled(False)
        self.n08_demo_cb.setChecked(False)
        self.statusBar().showMessage(message)
        self._set_control_enabled(start=True, pause=False, resume=False, stop=False)

    def _on_decision(self, df: DecisionFrame) -> None:
        self.timeline.add_decision(df)
        self.stickman.set_state(df.decision_state)
        self.status_panel.update_decision(df)
        self.processed_label.setText(f"已处理: {df.source_t_sec:.1f} s")
        # 结果有效期：750 ms 内无新有效决策则动画回到静止（PRD N08 / 6.3）
        self._expiry.start(df.valid_for_ms)

    def _on_result_expired(self) -> None:
        self.stickman.mark_stale()

    def _on_open_dir(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_edit.text()))

    def _on_n08_demo_toggled(self, state: int) -> None:
        # N08 演示：暂停/恢复决策发布（数据仍继续流式供数）
        if self.worker is not None:
            self.worker.set_suppress_decisions(state == Qt.Checked)
        if state == Qt.Checked:
            self.statusBar().showMessage("N08 演示：决策中断，750 ms 后动作反馈自动静止")
        else:
            self.statusBar().showMessage("N08 演示：决策恢复发布")

    def closeEvent(self, event) -> None:
        if self.worker and self._running:
            self.worker.stop()
            self.worker.quit()
            self.worker.wait()
        super().closeEvent(event)
