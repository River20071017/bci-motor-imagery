"""F07 双轨时间轴组件（模型预测轨 + 独立参考轨）。

按 PRD 3 / 7 的要求：
- 预测轨与参考轨分别绘制、互不驱动（决策只驱动预测轨；参考轨由独立
  时钟/标注显示，模型不可访问，见 PRD 5.3 与 7.3）。
- 时间轴不能只靠颜色区分类别，需同时提供文字或图例（PRD 3）。
- 显示检测时刻与证据窗口信息（PRD F07）。

本组件用 QPainter 自绘，便于精确控制"两轨独立"与图例。
参考轨在界面测试阶段为模拟脚本（明确标注"参考(模拟)"），
接入真实任务标注后由 A/B 提供独立参考对象（硬边界④：在线模式传 reference_mode=False）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget

from eeg_core.contracts import DecisionFrame, DecisionState

_WINDOW_SEC = 30.0  # 可视时间窗宽度（秒），自动滚动

# 状态显示颜色（同时配文字图例，不靠颜色单独区分类别，见 PRD 3 / 10）
STATE_COLORS = {
    DecisionState.WARMING_UP: "#9e9e9e",  # 灰
    DecisionState.IDLE: "#2e7d32",        # 绿
    DecisionState.LEFT_HAND: "#1565c0",   # 蓝
    DecisionState.RIGHT_HAND: "#ef6c00",  # 橙
    DecisionState.UNCERTAIN: "#f9a825",   # 黄
    DecisionState.INVALID: "#c62828",     # 红
}

STATE_LABELS_ZH = {
    DecisionState.WARMING_UP: "预热",
    DecisionState.IDLE: "空闲",
    DecisionState.LEFT_HAND: "左手",
    DecisionState.RIGHT_HAND: "右手",
    DecisionState.UNCERTAIN: "不确定",
    DecisionState.INVALID: "异常",
}


class TimelineWidget(QWidget):
    def __init__(self, parent=None, reference_mode: bool = True) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        # 在线模式必须传 False（硬边界④：在线路径不见标签）；两轨互不驱动
        self._reference_mode = reference_mode
        self.pred_segments = []          # [(t0, t1, state)]
        self._pred_open = None           # (t0, state)
        self.cursor_t = 0.0
        # 参考轨：仅界面测试 / 离线可用；在线模式为空，且绝不由标签驱动
        self.ref_segments = self._mock_reference() if reference_mode else []

    def _mock_reference(self):
        # 一段与预测脚本不同的参考模式，仅用于演示"两轨独立"
        return [
            (0.0, 3.0, DecisionState.IDLE),
            (3.0, 7.0, DecisionState.LEFT_HAND),
            (7.0, 10.0, DecisionState.IDLE),
            (10.0, 13.0, DecisionState.RIGHT_HAND),
            (13.0, 16.0, DecisionState.IDLE),
            (16.0, 18.0, DecisionState.UNCERTAIN),
            (18.0, 19.5, DecisionState.IDLE),
            (19.5, 21.0, DecisionState.INVALID),
            (21.0, 24.0, DecisionState.IDLE),
            (24.0, 27.0, DecisionState.LEFT_HAND),
            (27.0, 30.0, DecisionState.RIGHT_HAND),
            (30.0, 34.0, DecisionState.IDLE),
        ]

    def add_decision(self, df: DecisionFrame) -> None:
        t = df.source_t_sec
        s = df.decision_state
        if self._pred_open is not None:
            self.pred_segments.append((self._pred_open[0], t, self._pred_open[1]))
        self._pred_open = (t, s)
        self.cursor_t = max(self.cursor_t, t)
        self.update()

    def reset(self) -> None:
        self.pred_segments = []
        self._pred_open = None
        self.cursor_t = 0.0
        self.ref_segments = self._mock_reference() if self._reference_mode else []
        self.update()

    def set_reference_mode(self, on: bool) -> None:
        """切换参考轨可用性；在线模式必须传 False（硬边界④）。"""
        self._reference_mode = on
        self.ref_segments = self._mock_reference() if on else []
        self.update()

    def _draw_track(self, painter, x0, y0, h, segments, title, is_ref):
        # 背景
        painter.setBrush(QBrush(QColor("#f5f5f5")))
        painter.setPen(QPen(QColor("#cccccc")))
        painter.drawRect(x0, y0, self.width() - 2 * x0, h)
        # 标题
        painter.setPen(QColor("#333333"))
        painter.setFont(QFont("Sans", 10, QFont.Bold))
        painter.drawText(x0 + 4, y0 + 14, title)
        # 时间窗
        t_max = max(_WINDOW_SEC, self.cursor_t)
        t_min = max(0.0, t_max - _WINDOW_SEC)
        track_x = x0 + 90
        track_w = (self.width() - 2 * x0) - 100
        # 段
        for (t0, t1, st) in segments:
            if t1 < t_min or t0 > t_max:
                continue
            a = max(t0, t_min)
            b = min(t1, t_max)
            if b <= a:
                continue
            px = track_x + (a - t_min) / _WINDOW_SEC * track_w
            pw = (b - a) / _WINDOW_SEC * track_w
            color = QColor(STATE_COLORS.get(st, "#999999"))
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRect(int(px), y0 + 18, max(1, int(pw)), h - 24)
            # 文字标签（不靠颜色单独区分，PRD 3）
            if pw > 26:
                painter.setPen(QColor("#ffffff"))
                painter.setFont(QFont("Sans", 9))
                painter.drawText(int(px) + 3, y0 + 18 + (h - 24) // 2 + 3,
                                STATE_LABELS_ZH.get(st, str(st)))
        # 当前游标
        if self.cursor_t >= t_min and self.cursor_t <= t_max:
            cx = track_x + (self.cursor_t - t_min) / _WINDOW_SEC * track_w
            painter.setPen(QPen(QColor("#000000"), 1))
            painter.drawLine(int(cx), y0 + 16, int(cx), y0 + h - 4)
        # 时间刻度
        painter.setPen(QColor("#888888"))
        painter.setFont(QFont("Sans", 8))
        for k in range(int(_WINDOW_SEC) + 1):
            tt = t_min + k
            if tt > t_max:
                break
            px = track_x + (tt - t_min) / _WINDOW_SEC * track_w
            painter.drawText(int(px) - 8, y0 + h + 10, f"{tt:.0f}s")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        x0 = 10
        track_h = (self.height() - 40) // 2
        pred_y = 6
        ref_y = pred_y + track_h + 6
        self._draw_track(
            painter, x0, pred_y, track_h,
            self._closed_pred() + ([(self._pred_open[0], self.cursor_t, self._pred_open[1])]
                                   if self._pred_open else []),
            "模型预测轨（由决策驱动）", is_ref=False,
        )
        ref_title = (
            "参考任务标注轨（界面测试模拟，独立，模型不可访问）"
            if self._reference_mode else
            "参考轨（在线模式不可用）"
        )
        self._draw_track(
            painter, x0, ref_y, track_h, self.ref_segments, ref_title, is_ref=True,
        )
        # 图例（文字 + 颜色块）
        self._draw_legend(painter, x0, self.height() - 16)

    def _closed_pred(self):
        return self.pred_segments

    def _draw_legend(self, painter, x0, y):
        painter.setFont(QFont("Sans", 9))
        x = x0
        for st, lab in STATE_LABELS_ZH.items():
            painter.setBrush(QBrush(QColor(STATE_COLORS[st])))
            painter.setPen(Qt.NoPen)
            painter.drawRect(x, y - 10, 12, 12)
            painter.setPen(QColor("#333333"))
            painter.drawText(x + 16, y + 1, lab)
            x += 70
