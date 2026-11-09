"""存档状态显示（C）：把 D 的 ``JsonlResultSink.status`` 翻译成用户可见文案。

职责边界（与 A/D 的交接）：
- 本模块只**读** sink 的 ``status`` 字典；不构造 sink、不调用 ``close()``、不写文件。
- 构造 sink 与 ``close()`` 都可能等待 IO，必须由 A 的组装入口在工作线程完成；
  C 只在 GUI 主线程定时轮询 ``status``（一次字典拷贝，代价很小）。
- 界面只展示状态，不改写预测结果、不使用参考标签驱动动作（硬边界③/④）。
- ``submit()`` 成功只表示入队，不能据此显示“保存完成”；以 ``status`` 为准。

状态映射（D 的 status 字段：state / directory / error / queued / completed）：

===========  ==========================================
state        界面显示
===========  ==========================================
opening      正在准备保存目录…
writing      正在保存…
closing      正在写完剩余记录…
closed       保存完成（completed=true）／保存未完成（completed=false）
failed       保存失败：<error>
===========  ==========================================
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


@runtime_checkable
class ResultSinkLike(Protocol):
    """C 只需要 sink 的 ``status``（D 的 JsonlResultSink 即满足，无需新增契约）。"""

    @property
    def status(self) -> Mapping[str, object]:
        ...


@runtime_checkable
class SessionControl(Protocol):
    """需要 A 提供的会话控制接口（存档失败时由 C 调用）。

    A 的运行时控制器实现其中之一即可；C 在 ``failed`` 时调用 :meth:`request_stop`。
    也可以改为连接 ``MainWindow.sink_failed`` 信号，二选一。
    """

    def request_pause(self, reason: str) -> None:
        ...

    def request_stop(self, reason: str) -> None:
        ...


class StaticSink:
    """状态恒定的 sink，只用于把"启动前就失败"的情况交给界面显示。

    D 的 ``JsonlResultSink`` 在**构造阶段**创建目录失败时会直接抛
    ``ArchiveWriteError``（而不是返回一个 ``state="failed"`` 的对象），
    此时 A 还没有 sink 可注入。用本类包一层，界面就能走同一条失败路径
    （显示原因 → 停动作反馈 → 通知 A），避免"启动前失败"在界面上无迹可寻。

    用法（A 侧）::

        try:
            sink = JsonlResultSink(root, session_id, metadata)
        except ArchiveWriteError as exc:
            window.attach_result_sink(failed_sink(str(exc)))
        else:
            window.attach_result_sink(sink)
    """

    def __init__(
        self,
        *,
        state: str = "failed",
        error: str | None = None,
        directory: str | None = None,
    ) -> None:
        self._status: dict[str, object] = {
            "state": state,
            "directory": directory,
            "error": error,
            "queued": 0,
            "completed": False,
        }

    @property
    def status(self) -> dict[str, object]:
        return dict(self._status)


def failed_sink(reason: str) -> StaticSink:
    """构造一个"已失败"的 sink（A 在存档初始化失败时注入）。"""
    return StaticSink(state="failed", error=reason or "存档初始化失败")


@dataclass(frozen=True)
class SaveStatus:
    """一次状态映射的结果（纯数据，便于单测）。"""

    text: str
    level: str = "muted"  # info | ok | warn | error | muted
    directory: str | None = None
    error: str | None = None
    state: str | None = None
    queued: int | None = None


_LEVEL_COLORS = {
    "info": "#1565c0",   # 蓝
    "ok": "#2e7d32",     # 绿
    "warn": "#ef6c00",   # 橙
    "error": "#c62828",  # 红
    "muted": "#757575",  # 灰
}


def format_sink_status(status: Mapping[str, object] | None) -> SaveStatus:
    """把 D 的 status 字典映射为界面文案（纯函数，可单测）。"""
    if status is None:
        return SaveStatus("存档模块未接入（界面测试）", "muted")

    state = status.get("state")
    directory = status.get("directory")
    directory = str(directory) if directory else None
    error = status.get("error")
    error = str(error) if error else None
    queued = status.get("queued")
    try:
        queued = int(queued) if queued is not None else None
    except (TypeError, ValueError):
        queued = None
    completed = bool(status.get("completed"))

    if state == "opening":
        return SaveStatus("正在准备保存目录…", "info", directory, state=state, queued=queued)
    if state == "writing":
        return SaveStatus("正在保存…", "info", directory, state=state, queued=queued)
    if state == "closing":
        return SaveStatus("正在写完剩余记录…", "info", directory, state=state, queued=queued)
    if state == "closed":
        if completed:
            return SaveStatus("保存完成", "ok", directory, state=state, queued=queued)
        return SaveStatus("保存未完成", "warn", directory, state=state, queued=queued)
    if state == "failed":
        detail = error or "未知错误"
        return SaveStatus(
            f"保存失败：{detail}", "error", directory, error=error, state=state, queued=queued
        )
    return SaveStatus(
        f"存档状态未知：{state!r}", "warn", directory,
        state=str(state) if state is not None else None, queued=queued,
    )


class _ElidedLabel(QLabel):
    """单行自动省略的标签：宽度不足时用 … 截断，完整文本进 tooltip。

    控制条高度固定，长目录 / 长错误原因不能把按钮挤变形或撑高整栏。
    ``text()`` 仍返回完整文本（便于测试与无障碍读取）。
    """

    def __init__(self, color: str = "#333333", *, bold: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self._text = ""
        font = self.font()
        font.setBold(bold)
        self.setFont(font)
        # Ignored：允许被布局压缩到任意宽度（由省略号负责可读性）
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)

    def set_text(self, text: str) -> None:
        self._text = text
        super().setText(text)          # 供 text() / 无障碍 / 测试读取
        self.setToolTip(text)
        self.update()

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802  Qt 命名
        painter = QPainter(self)
        painter.setPen(self._color)
        width = max(self.width() - 2, 10)
        elided = self.fontMetrics().elidedText(self._text, Qt.ElideRight, width)
        painter.drawText(self.rect(), Qt.AlignLeft | Qt.AlignVCenter, elided)


class SaveStatusWidget(QWidget):
    """紧凑的存档状态条：状态文字 + 会话目录（可选中复制）。"""

    def __init__(self, parent=None, *, test_mode: bool = False) -> None:
        super().__init__(parent)
        self._test_mode = bool(test_mode)
        self.current_status: SaveStatus | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._text_label = _ElidedLabel(bold=True)
        self._dir_label = _ElidedLabel("#666666")
        small = self._dir_label.font()
        small.setPointSizeF(max(small.pointSizeF() - 1.0, 7.0))
        self._dir_label.setFont(small)
        layout.addWidget(self._text_label)
        layout.addWidget(self._dir_label)
        self.setMinimumWidth(360)      # 保证状态文字有可读宽度（不足时省略号截断）
        self.setMaximumWidth(460)      # 防止长目录/长错误独占控制条

        self.set_status(SaveStatus("存档模块未接入（界面测试）", "muted"))

    def set_status(self, status: SaveStatus) -> None:
        self.current_status = status
        color = _LEVEL_COLORS.get(status.level, "#333333")
        prefix = "存档（界面测试·模拟）" if self._test_mode else "存档"
        text = f"{prefix}：{status.text}"
        if status.queued:
            text += f"（队列 {status.queued}）"
        self._text_label.set_color(color)
        self._text_label.set_text(text)
        if status.directory:
            self._dir_label.set_text(f"目录：{status.directory}")
            self._dir_label.setVisible(True)
            self.setToolTip(f"{text}\n{status.directory}")
        else:
            self._dir_label.set_text("")
            self._dir_label.setVisible(False)
            self.setToolTip(text)

    def set_test_mode(self, on: bool) -> None:
        """切换"界面测试·模拟存档"标注（不改变状态判定）。"""
        self._test_mode = bool(on)
        if self.current_status is not None:
            self.set_status(self.current_status)
