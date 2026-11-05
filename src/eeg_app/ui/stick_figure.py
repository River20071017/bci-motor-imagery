"""F08 火柴人动画组件（Qt 自绘）。

按 PRD 3 / 8 与分工方案 2.3：
- 火柴人仅响应"有效模型决策"；动画只由模型决策驱动，不读取参考标签。
- 固定左右手含义与动作映射，固定视角并显示"左手/右手"文字，避免镜像歧义
  （PRD 3：火柴人的左手指火柴人自身解剖学左侧）。
- 暂停、异常、停止或结果过期后，火柴人回到静止（PRD 6.4 / N07 / N08）。
- 动画是类别示意，不表示重建了真实手势、关节轨迹或用力程度。

数据契约直接来自 ``eeg_core.contracts.DecisionState``。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QBrush
from PySide6.QtWidgets import QWidget

from eeg_core.contracts import DecisionState

# 状态显示颜色（同时配文字图例，不靠颜色单独区分类别，见 PRD 3 / 10）
STATE_COLORS = {
    DecisionState.WARMING_UP: "#9e9e9e",  # 灰
    DecisionState.IDLE: "#2e7d32",        # 绿
    DecisionState.LEFT_HAND: "#1565c0",   # 蓝
    DecisionState.RIGHT_HAND: "#ef6c00",  # 橙
    DecisionState.UNCERTAIN: "#f9a825",   # 黄
    DecisionState.INVALID: "#c62828",     # 红
}

# 用户可见状态文案（PRD 第 3 节界面状态文字示例）
STATE_USER_TEXT = {
    DecisionState.WARMING_UP: "正在准备分析",
    DecisionState.IDLE: "当前判断为空闲",
    DecisionState.LEFT_HAND: "检测到左手运动想象",
    DecisionState.RIGHT_HAND: "检测到右手运动想象",
    DecisionState.UNCERTAIN: "暂未得到稳定判断",
    DecisionState.INVALID: "当前数据异常，分析暂不可用",
}


class StickmanWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 300)
        self.state = DecisionState.IDLE
        self.stale = False       # 结果过期（750 ms 内无新有效决策）
        self._feedback = False   # 动作反馈是否激活（暂停/过期/停止时关闭，N07/N08）
        self._paused = False     # 是否处于暂停冻结

    def set_state(self, state: DecisionState) -> None:
        # 收到有效决策：激活动作反馈（手臂抬起由 paintEvent 依据 _feedback 决定）
        self.state = state
        self.stale = False
        self._paused = False
        self._feedback = True
        self.update()

    def mark_stale(self) -> None:
        # N08：结果过期 → 关闭动作反馈，手臂回落到静止
        self.stale = True
        self._feedback = False
        self.update()

    def set_paused(self) -> None:
        # N07：暂停 → 立即关闭动作反馈（<250ms 手臂回落）
        self._paused = True
        self._feedback = False
        self.update()

    def clear_pause(self) -> None:
        # 继续运行：取消暂停冻结，但等待下一个有效决策才恢复动作反馈
        self._paused = False
        self._feedback = False
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx = w // 2

        # 几何
        head_r = min(w, h) * 0.10
        top = h * 0.10
        head_cy = top + head_r
        neck = head_cy + head_r
        shoulder_y = neck + h * 0.06
        hip_y = shoulder_y + h * 0.26
        foot_y = hip_y + h * 0.26
        shoulder_dx = w * 0.10
        hip_dx = w * 0.06

        # 颜色：异常态常显红色；动作反馈激活且为动作状态时用状态色；否则中性灰
        if self.state == DecisionState.INVALID:
            limb_color = QColor(STATE_COLORS[DecisionState.INVALID])
        elif self._feedback and self.state in (DecisionState.LEFT_HAND, DecisionState.RIGHT_HAND):
            limb_color = QColor(STATE_COLORS[self.state])
        else:
            limb_color = QColor("#555555")

        pen = QPen(limb_color, 4, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)

        # 头
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(int(cx - head_r), int(head_cy - head_r),
                           int(head_r * 2), int(head_r * 2))
        # 躯干
        painter.drawLine(int(cx), int(neck), int(cx), int(hip_y))
        # 腿
        painter.drawLine(int(cx), int(hip_y), int(cx - hip_dx), int(foot_y))
        painter.drawLine(int(cx), int(hip_y), int(cx + hip_dx), int(foot_y))

        # 手臂：默认下垂；动作反馈激活时抬起"对应"手臂；暂停/过期/停止时回落静止（N07/N08）
        # 火柴人自身左侧 = 观看者右侧（cx + dx）；自身右侧 = 观看者左侧（cx - dx）
        left_arm_up = self._feedback and self.state == DecisionState.LEFT_HAND     # 抬起自身左侧（观看者右侧）
        right_arm_up = self._feedback and self.state == DecisionState.RIGHT_HAND  # 抬起自身右侧（观看者左侧）

        # 左肩（观看者左侧, cx - dx）
        if right_arm_up:
            painter.drawLine(int(cx - shoulder_dx), int(shoulder_y),
                            int(cx - shoulder_dx * 1.6), int(shoulder_y - h * 0.14))
        else:
            painter.drawLine(int(cx - shoulder_dx), int(shoulder_y),
                            int(cx - shoulder_dx), int(shoulder_y + h * 0.16))
        # 右肩（观看者右侧, cx + dx）
        if left_arm_up:
            painter.drawLine(int(cx + shoulder_dx), int(shoulder_y),
                            int(cx + shoulder_dx * 1.6), int(shoulder_y - h * 0.14))
        else:
            painter.drawLine(int(cx + shoulder_dx), int(shoulder_y),
                            int(cx + shoulder_dx), int(shoulder_y + h * 0.16))

        # 文案
        painter.setPen(QColor("#222222"))
        painter.setFont(QFont("Sans", 12, QFont.Bold))
        if self._paused:
            text = "已暂停（动作反馈静止）"
        elif self.stale:
            text = "（结果过期，动画静止）"
        else:
            text = STATE_USER_TEXT.get(self.state, str(self.state))
        painter.drawText(8, h - 30, text)

        if self._feedback and self.state == DecisionState.LEFT_HAND:
            painter.setPen(QColor(STATE_COLORS[DecisionState.LEFT_HAND]))
            painter.drawText(int(cx + shoulder_dx), int(shoulder_y - h * 0.16), "左手")
        elif self._feedback and self.state == DecisionState.RIGHT_HAND:
            painter.setPen(QColor(STATE_COLORS[DecisionState.RIGHT_HAND]))
            painter.drawText(int(cx - shoulder_dx - 30), int(shoulder_y - h * 0.16), "右手")

        # 模拟水印（明确非真实算法输出）
        painter.setPen(QColor("#bbbbbb"))
        painter.setFont(QFont("Sans", 9))
        painter.drawText(8, 16, "界面测试 · 模拟决策")
