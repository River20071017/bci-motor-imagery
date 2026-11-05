"""F06 脑电波形组件（滚动显示，可选通道与幅度，绘图不阻塞解码）。

使用 PyQtGraph 绘制（PRD 5：波形与时间轴用 PyQtGraph）。
- 维护固定长度环形缓冲（默认最近 ``DisplayConfig.waveform_span_sec`` 秒），控制绘图负载（PRD 6.5）。
- 支持通道选择（PRD 3：默认显示 8 通道、最近 10 秒）。
- 支持幅度（µV）缩放，便于观察。
- 绘图刷新由回放节奏驱动，目标 ≥15 次/秒（PRD 9 N04）。
- 显示选择不改变模型通道或计算输入。

数据契约直接来自 ``eeg_core.contracts.EEGChunk``（单位伏，绘图时乘 1e6 显示为 µV）。
"""
from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QSlider,
)
from collections import deque

from eeg_core.contracts import EEGChunk

# 通道显示颜色循环
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#393b79",
]


class WaveformWidget(QWidget):
    def __init__(
        self,
        channels,
        sample_rate_hz: float = 160.0,
        window_sec: float = 10.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.channels = list(channels)
        self.sample_rate_hz = sample_rate_hz
        self.window_sec = window_sec
        self.max_len = int(window_sec * sample_rate_hz)
        self._cum = 0  # 跨块连续样本计数，用于对齐时间轴
        self._x = deque(maxlen=self.max_len)
        self._y = {ch: deque(maxlen=self.max_len) for ch in self.channels}
        self._selected = set(self.channels)
        self._ymax = 50.0  # µV

        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        # 通道选择
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(110)
        for ch in self.channels:
            item = QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        right = QVBoxLayout()
        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "µV")
        self.plot.setLabel("bottom", "秒")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setYRange(-self._ymax, self._ymax)
        right.addWidget(self.plot, stretch=1)

        # 幅度控制
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("幅度:"))
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(5, 200)
        self.amp_slider.setValue(int(self._ymax))
        self.amp_slider.valueChanged.connect(self._on_amp)
        self.amp_label = QLabel(f"±{self._ymax:.0f} µV")
        ctrl.addWidget(self.amp_slider, stretch=1)
        ctrl.addWidget(self.amp_label)
        right.addLayout(ctrl)
        layout.addLayout(right, stretch=1)

        # 曲线
        self.curves = {}
        for i, ch in enumerate(self.channels):
            color = _PALETTE[i % len(_PALETTE)]
            self.curves[ch] = self.plot.plot(pen=color, name=ch)

    def _on_item_changed(self, item: QListWidgetItem):
        ch = item.text()
        if item.checkState() == Qt.Checked:
            self._selected.add(ch)
        else:
            self._selected.discard(ch)
        self.curves[ch].setVisible(item.checkState() == Qt.Checked)

    def _on_amp(self, v: int):
        self._ymax = float(v)
        self.amp_label.setText(f"±{self._ymax:.0f} µV")
        self.plot.setYRange(-self._ymax, self._ymax)

    def append_chunk(self, chunk: EEGChunk) -> None:
        samples = chunk.samples
        if samples is None:
            return
        chs = chunk.channel_names
        n = samples.shape[1]
        start = self._cum
        # 降采样绘制：点数过多时抽稀，控制绘图负载（PRD 6.5）
        decim = 1
        total = self.max_len
        if total > 1500:
            decim = max(1, total // 1500)
        for i, ch in enumerate(chs):
            if ch not in self._y:
                continue
            arr = (samples[i] * 1e6).astype(np.float32)  # V -> µV
            for j in range(n):
                self._y[ch].append(float(arr[j]))
        # 时间轴每个数据块只推进一次（与通道数无关）
        for j in range(n):
            self._x.append((start + j) / self.sample_rate_hz)
        self._cum += n

        # 统一时间轴（秒）
        xs = np.asarray(self._x, dtype=np.float32)
        # 仅绘制选中通道；x 所有通道共享
        for ch in self.channels:
            if ch not in self._selected:
                continue
            ys = np.asarray(self._y[ch], dtype=np.float32)
            if decim > 1:
                self.curves[ch].setData(xs[::decim], ys[::decim])
            else:
                self.curves[ch].setData(xs, ys)

    def clear(self) -> None:
        self._x.clear()
        for ch in self._y:
            self._y[ch].clear()
        self._cum = 0
        for ch in self.channels:
            self.curves[ch].setData([], [])
