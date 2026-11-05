"""运行过程可视化（C 新增，QTabWidget 第二屏）：展现系统的"思考/运行过程"
与"波形被绘制出的过程"，面向非专家，便于观察数据如何从原始块变成决策。

组成：
- :class:`PipelineWidget`：概念管线（数据块→滤波→加窗→特征→解码→决策）随数据流动高亮，
  决策到达时高亮末端；明确标注为"概念示意"，真实算法由 B 提供、界面不改写结果。
- :class:`ProcessLog`：逐块接收与逐次决策的文本轨迹（"思考"痕迹）。
- :class:`ChunkDrawer`：把最新数据块逐样本绘制出来，直观展示波形"被画出来"的过程。
- :class:`ProcessView`：把以上组装为第二屏，统一接收 worker 的 chunk/decision 信号。

不改动 eeg_core.contracts（硬边界①）；只读取契约数据盒做展示。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame, QLabel, QPlainTextEdit, QVBoxLayout, QWidget,
)

from eeg_core.contracts import DecisionFrame, EEGChunk

# 概念管线阶段（真实算法由 B 提供，这里只做"流动"示意）
_PIPELINE = ["① 数据块", "② 滤波", "③ 加窗", "④ 特征", "⑤ 解码", "⑥ 决策"]


class PipelineWidget(QWidget):
    """概念管线：数据块到达时各阶段依次高亮，决策到达时末端闪烁。

    仅示意"系统在处理"，不表示真实内部步骤；真实算法对界面不可见（硬边界③）。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(64)
        self._pulse = 0
        self._decision_flash = 0
        self._last_state = None
        self._running = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_running(self, on: bool) -> None:
        self._running = bool(on)
        if self._running:
            if not self._timer.isActive():
                self._timer.start(70)
        else:
            self._timer.stop()
            self._decision_flash = 0
            self.update()

    def on_chunk(self, _chunk: EEGChunk) -> None:
        # 数据流动感由定时器循环高亮体现；此处仅标记"有数据在流"
        pass

    def on_decision(self, df: DecisionFrame) -> None:
        self._last_state = df.decision_state
        self._decision_flash = 4  # 约 4×70ms = 280ms 内锁定在末端
        if not self._timer.isActive() and self._running:
            self._timer.start(70)
        self.update()

    def _tick(self) -> None:
        if self._decision_flash > 0:
            self._decision_flash -= 1
            self._pulse = len(_PIPELINE) - 1
        else:
            self._pulse = (self._pulse + 1) % len(_PIPELINE)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        n = len(_PIPELINE)
        gap = 10
        bw = (w - gap * (n - 1)) / n
        top, bh = 10, h - 34
        for i, name in enumerate(_PIPELINE):
            x = i * (bw + gap)
            active = (i == self._pulse)
            fill = QColor("#1565c0") if active else QColor("#e3e3e3")
            painter.setBrush(fill)
            painter.setPen(QPen(QColor("#9e9e9e")))
            painter.drawRoundedRect(int(x), top, int(bw), bh, 6, 6)
            painter.setPen(QColor("#ffffff") if active else QColor("#333333"))
            painter.setFont(QFont("Sans", 10, QFont.Bold))
            painter.drawText(int(x) + 4, top + bh // 2 + 4, name)
            if i < n - 1:
                painter.setPen(QColor("#9e9e9e"))
                painter.drawText(int(x + bw) + 1, top + bh // 2 + 4, "→")
        if self._last_state is not None:
            painter.setPen(QColor("#333333"))
            painter.setFont(QFont("Sans", 10))
            painter.drawText(8, h - 6, f"最新决策：{self._last_state.name}")


class ProcessLog(QPlainTextEdit):
    """逐块接收 / 逐次决策 的文本轨迹，便于观察"思考"过程。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(600)

    def on_chunk(self, chunk: EEGChunk) -> None:
        t = chunk.start_sample / chunk.sample_rate_hz if chunk.sample_rate_hz else 0.0
        self.appendPlainText(
            f"[{t:7.2f}s] 收到数据块  起始样本={chunk.start_sample} "
            f"形状={tuple(chunk.samples.shape)}"
        )

    def on_decision(self, df: DecisionFrame) -> None:
        sc = df.scores
        sc_str = (
            f"idle={sc.idle:.2f} left={sc.left_hand:.2f} right={sc.right_hand:.2f}"
            if sc is not None else "—"
        )
        win = df.window
        win_str = f"样本[{win.start_sample},{win.end_sample}]" if win is not None else "—"
        self.appendPlainText(
            f"[{df.source_t_sec:7.2f}s] 决策={df.decision_state.name}  "
            f"评分[{sc_str}]  质量={'OK' if df.quality.valid else 'FAIL'}  {win_str}"
        )


class ChunkDrawer(QWidget):
    """把最新数据块逐样本绘制出来，展示波形"被画出来"的过程。"""

    def __init__(self, display_channels, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(170)
        self._channels = list(display_channels)
        self._chunk = None
        self._n = 0
        self._drawn = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def on_chunk(self, chunk: EEGChunk) -> None:
        self._chunk = chunk.samples
        self._n = int(chunk.samples.shape[1])
        self._drawn = 0
        self._timer.stop()
        self._timer.start(16)

    def _tick(self) -> None:
        step = max(1, self._n // 16)
        self._drawn = min(self._n, self._drawn + step)
        self.update()
        if self._drawn >= self._n:
            self._timer.stop()

    def _y_of(self, ci: int, idx: int, row_h: float, pad: float) -> float:
        row_h2 = row_h * 0.40
        y0 = pad + ci * row_h + row_h / 2.0
        v = float(self._chunk[ci, idx])
        # ±50 μV 满刻度
        return y0 - (v / 50e-6) * row_h2

    def _draw_trace(self, painter, ci, upto, w, pad, plot_w, row_h) -> None:
        if upto < 2:
            return
        n = self._chunk.shape[1]
        prev = None
        for k in range(upto):
            x = pad + k / n * plot_w
            y = self._y_of(ci, k, row_h, pad)
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.setBrush(QColor("#0b0b0b"))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, w, h)
        pad = 10
        if self._chunk is None:
            painter.setPen(QColor("#888888"))
            painter.setFont(QFont("Sans", 11))
            painter.drawText(12, h // 2, "等待数据块…（开始分析后，这里会逐样本绘制最新波形）")
            return
        n_ch, n = self._chunk.shape
        plot_w = w - 2 * pad
        plot_h = h - 2 * pad - 16
        row_h = plot_h / max(1, n_ch)
        for ci in range(n_ch):
            y0 = pad + ci * row_h
            # 完整块的淡色轮廓
            painter.setPen(QPen(QColor("#2a2a2a"), 1))
            self._draw_trace(painter, ci, n, w, pad, plot_w, row_h)
            # 已绘制部分高亮（"正在画"）
            painter.setPen(QPen(QColor("#4fc3f7"), 1.6))
            self._draw_trace(painter, ci, self._drawn, w, pad, plot_w, row_h)
            # 笔头
            if 0 < self._drawn <= n:
                x = pad + (self._drawn - 1) / n * plot_w
                y = self._y_of(ci, self._drawn - 1, row_h, pad)
                painter.setBrush(QColor("#ffeb3b"))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)
            # 通道名
            painter.setPen(QColor("#9e9e9e"))
            painter.setFont(QFont("Sans", 8))
            painter.drawText(2, int(y0) + int(row_h) - 2, self._channels[ci])
        if self._drawn < n:
            painter.setPen(QColor("#4fc3f7"))
            painter.setFont(QFont("Sans", 9))
            painter.drawText(12, h - 6, f"正在绘制最新块：{self._drawn}/{self._n} 样本")
        else:
            painter.setPen(QColor("#cccccc"))
            painter.setFont(QFont("Sans", 9))
            painter.drawText(
                12, h - 6, f"最新块绘制完成：{self._n} 样本 / {n_ch} 通道（等待下一块）"
            )


class ProcessView(QWidget):
    """运行过程屏：管线 + 日志 + 波形绘制，统一接收 chunk/decision 信号。"""

    def __init__(self, display_channels, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        pipe_frame = QFrame()
        pipe_frame.setFrameStyle(QFrame.StyledPanel)
        pl = QVBoxLayout(pipe_frame)
        pl.addWidget(QLabel(
            "处理管线（概念示意：原始块→滤波→加窗→特征→解码→决策；"
            "真实算法由 B 提供，界面不改写结果）"
        ))
        self.pipeline = PipelineWidget()
        pl.addWidget(self.pipeline)
        root.addWidget(pipe_frame)

        root.addWidget(QLabel("处理日志（每块的接收与每次决策，便于观察“思考”过程）"))
        self.log = ProcessLog()
        root.addWidget(self.log, stretch=1)

        root.addWidget(QLabel("波形绘制过程（最新数据块被逐样本绘制）"))
        self.drawer = ChunkDrawer(display_channels)
        root.addWidget(self.drawer, stretch=1)

    def on_chunk(self, chunk: EEGChunk) -> None:
        self.pipeline.on_chunk(chunk)
        self.log.on_chunk(chunk)
        self.drawer.on_chunk(chunk)

    def on_decision(self, df: DecisionFrame) -> None:
        self.pipeline.on_decision(df)
        self.log.on_decision(df)

    def reset(self) -> None:
        self.log.clear()
        self.drawer._chunk = None
        self.drawer._drawn = 0
        self.drawer._n = 0
        self.drawer.update()
        self.pipeline._last_state = None
        self.pipeline._decision_flash = 0
        self.pipeline._pulse = 0
        self.pipeline.update()

    def set_running(self, on: bool) -> None:
        self.pipeline.set_running(on)
