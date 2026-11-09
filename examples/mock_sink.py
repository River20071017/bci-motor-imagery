"""界面测试用模拟存档 sink（非真实磁盘 IO）。

只实现 C 需要的 ``status`` 形状，字段与 D 的 ``JsonlResultSink.status`` 对齐：
``state`` / ``directory`` / ``error`` / ``queued`` / ``completed``。

用途：在没有 D 模块（PR #11 尚未合并）时，验证界面在
writing / closing / closed(completed=True/False) / failed 下的显示与失败处理。
**不代表真实存档已接通**；真实 sink 由 A 的组装入口在工作线程构造并注入。
"""
from __future__ import annotations


class MockResultSink:
    def __init__(
        self,
        *,
        state: str = "writing",
        directory: str | None = None,
        completed: bool = False,
    ) -> None:
        self._status: dict[str, object] = {
            "state": state,
            "directory": directory,
            "error": None,
            "queued": 0,
            "completed": completed,
        }

    @property
    def status(self) -> dict[str, object]:
        return dict(self._status)

    def submit(self, item) -> None:  # 仅模拟“入队”
        self._status["queued"] = int(self._status["queued"]) + 1

    def set_state(
        self,
        state: str,
        *,
        directory: str | None = None,
        error: str | None = None,
        completed: bool = False,
        queued: int | None = None,
    ) -> None:
        """测试驱动：切换 sink 状态（模拟 A 生命周期/后台写入结果）。"""
        self._status["state"] = state
        if directory is not None:
            self._status["directory"] = directory
        self._status["error"] = error
        self._status["completed"] = completed
        if queued is not None:
            self._status["queued"] = queued
