"""D 的同步存档实现；线上通过 sinks.py 的单一后台写入线程使用。"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from eeg_core.contracts import (
    CONTRACT_VERSION,
    ContractError,
    DecisionFrame,
    IntentEvent,
    RuntimeEvent,
    utc_now_iso,
)

EVENT_FIELDS = (
    "event_id",
    "session_id",
    "recording_id",
    "state",
    "start_sample",
    "start_decision_seq",
    "end_sample",
    "end_decision_seq",
    "end_reason",
)
_SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    """先写同目录临时文件再替换；失败时保留原文件。"""
    text = json.dumps(dict(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SessionArchive:
    """五文件账本。会话目录独占创建，永不覆盖已有会话。

    metadata 必须包含 recording_id/model_id/pipeline_version/config_hash。
    配置、依赖版本和源文件 SHA 由组装方提供，未知值不伪造。
    这里只写上游传入的已结束 IntentEvent，不自行猜动作边界。
    """

    def __init__(self, root: str | Path, session_id: str, metadata: Mapping[str, object]):
        if not isinstance(session_id, str) or _SESSION_NAME.fullmatch(session_id) is None:
            raise ContractError("session_id 只允许字母、数字、下划线和短横线，长度 1–96")
        self._metadata = json.loads(json.dumps(dict(metadata), allow_nan=False))
        for key in ("recording_id", "model_id", "pipeline_version", "config_hash"):
            if not isinstance(self._metadata.get(key), str) or not self._metadata[key].strip():
                raise ContractError(f"metadata 缺少非空字符串 {key}")
        self.session_id = session_id
        self.directory = Path(root) / f"session_{session_id}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._stack = ExitStack()
        self._closed = False
        self._next_seq = 0
        self._event_ids: set[str] = set()
        self._counts: Counter[str] = Counter()
        self._states: Counter[str] = Counter()
        self._last_source_time = 0.0
        try:
            atomic_json(
                self.directory / "manifest.json",
                {
                    "archive_format_version": "1.0",
                    "schema_version": CONTRACT_VERSION,
                    "session_id": session_id,
                    "created_at_utc": utc_now_iso(),
                    "metadata": self._metadata,
                },
            )
            # 即使进程突然消失，原始 summary 也不会声称成功。
            self._summary(False, "in_progress")
            self._decisions = self._open("decisions.jsonl")
            self._runtime = self._open("runtime.jsonl")
            self._events = self._open("events.csv")
            self._csv = csv.DictWriter(self._events, fieldnames=EVENT_FIELDS)
            self._csv.writeheader()
            self.flush()
        except Exception:
            self._stack.close()
            raise

    def _open(self, name: str):
        return self._stack.enter_context(
            (self.directory / name).open("x", encoding="utf-8", newline="")
        )

    def _check_session(self, session_id: str) -> None:
        if self._closed:
            raise RuntimeError("存档已经关闭")
        if session_id != self.session_id:
            raise ContractError("不能把其他会话的结果写进当前目录")

    def append_decision(self, frame: DecisionFrame) -> None:
        self._check_session(frame.session_id)
        if frame.seq != self._next_seq:
            raise ContractError(f"决策必须从 0 连续递增：期望 {self._next_seq}，收到 {frame.seq}")
        for name in ("recording_id", "model_id", "pipeline_version", "config_hash"):
            if getattr(frame, name) != self._metadata[name]:
                raise ContractError(f"决策的 {name} 与 manifest 不一致")
        if frame.source_t_sec < self._last_source_time:
            raise ContractError("决策数据时间不能倒退；新记录必须新建会话")
        self._decisions.write(frame.to_json() + "\n")
        self._next_seq += 1
        self._last_source_time = frame.source_t_sec
        self._counts["decisions"] += 1
        self._states[frame.decision_state.value] += 1

    def append_event(self, event: IntentEvent) -> None:
        self._check_session(event.session_id)
        if event.recording_id != self._metadata["recording_id"]:
            raise ContractError("动作事件的 recording_id 与 manifest 不一致")
        if event.is_open:
            raise ContractError("events.csv 只接受已结束事件；未结束事件由运行时保管")
        if event.event_id in self._event_ids:
            raise ContractError(f"重复 event_id：{event.event_id}")
        if event.end_decision_seq >= self._next_seq:
            raise ContractError("动作事件必须引用已经提交的决策")
        self._csv.writerow(event.to_dict())
        self._event_ids.add(event.event_id)
        self._counts["events"] += 1

    def append_runtime(self, event: RuntimeEvent) -> None:
        self._check_session(event.session_id)
        self._runtime.write(
            json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        self._counts["runtime"] += 1

    def flush(self) -> None:
        for stream in (self._decisions, self._events, self._runtime):
            stream.flush()
            os.fsync(stream.fileno())

    def _summary(self, completed: bool, reason: str) -> None:
        atomic_json(
            self.directory / "summary.json",
            {
                "archive_format_version": "1.0",
                "session_id": self.session_id,
                "completed": completed,
                "end_reason": reason,
                "updated_at_utc": utc_now_iso(),
                "counts": {name: self._counts[name] for name in ("decisions", "events", "runtime")},
                "decision_state_counts": dict(self._states),
                "last_source_t_sec": self._last_source_time,
            },
        )

    def close(self, *, completed: bool, reason: str) -> None:
        if self._closed:
            return
        try:
            self.flush()
            self._stack.close()
            self._summary(completed, reason)
        finally:
            self._closed = True
            self._stack.close()
