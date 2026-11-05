"""状态面板：当前状态文字、模型评分、模型与配置标识、已处理时长。

把 PRD 第 3 节的状态文字、DecisionFrame 分数与运行信息集中展示，
便于评审"看懂当前数据和模型状态"（提案 2）。

数据契约直接来自 ``eeg_core.contracts``（DecisionFrame / DecisionState / Scores）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QFormLayout, QLabel

from eeg_core.contracts import DecisionFrame, DecisionState

STATE_COLORS = {
    DecisionState.WARMING_UP: "#9e9e9e",
    DecisionState.IDLE: "#2e7d32",
    DecisionState.LEFT_HAND: "#1565c0",
    DecisionState.RIGHT_HAND: "#ef6c00",
    DecisionState.UNCERTAIN: "#f9a825",
    DecisionState.INVALID: "#c62828",
}

STATE_USER_TEXT = {
    DecisionState.WARMING_UP: "正在准备分析",
    DecisionState.IDLE: "当前判断为空闲",
    DecisionState.LEFT_HAND: "检测到左手运动想象",
    DecisionState.RIGHT_HAND: "检测到右手运动想象",
    DecisionState.UNCERTAIN: "暂未得到稳定判断",
    DecisionState.INVALID: "当前数据异常，分析暂不可用",
}


class StatusPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        self.state_label = QLabel("—")
        self.state_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.score_label = QLabel("—")
        self.model_label = QLabel("—")
        self.config_label = QLabel("—")
        self.time_label = QLabel("0.0 s")
        layout.addRow("当前判断:", self.state_label)
        layout.addRow("模型评分:", self.score_label)
        layout.addRow("模型:", self.model_label)
        layout.addRow("配置:", self.config_label)
        layout.addRow("已处理:", self.time_label)

    def update_decision(self, df: DecisionFrame) -> None:
        text = STATE_USER_TEXT.get(df.decision_state, str(df.decision_state))
        color = STATE_COLORS.get(df.decision_state, "#333333")
        self.state_label.setText(f'<font color="{color}">{text}</font>')
        if df.scores is not None:
            parts = ", ".join(f"{k}={v:.2f}" for k, v in df.scores.as_dict().items())
            self.score_label.setText(parts)
        else:
            self.score_label.setText("无（异常/预热）")
        self.model_label.setText(df.model_id or "—")
        self.config_label.setText(df.config_hash or "—")

    def set_time(self, sec: float) -> None:
        self.time_label.setText(f"{sec:.1f} s")
