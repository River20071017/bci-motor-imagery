"""独立读取存档：只用标准库，不导入 writer、算法或 frozen 契约。

python examples/read_session.py sessions/session_d-demo-...
--recover 只容忍 JSONL 最后一个未写完的物理行；不修改任何文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

EVENT_FIELDS = [
    "event_id",
    "session_id",
    "recording_id",
    "state",
    "start_sample",
    "start_decision_seq",
    "end_sample",
    "end_decision_seq",
    "end_reason",
]
STATES = {"idle", "left_hand", "right_hand", "uncertain", "invalid", "warming_up"}


def _reject_constant(value):
    raise ValueError(f"JSON 不允许 {value}")


def _json(text):
    value = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("每条记录必须为 JSON 对象")
    return value


def _jsonl(path: Path, recover: bool, warnings: list[str]):
    with path.open(encoding="utf-8", newline="") as stream:
        line = stream.readline()
        index = 0
        while line:
            index += 1
            following = stream.readline()
            if not line.endswith("\n"):
                if recover and not following:
                    warnings.append(f"{path.name}:{index} 尾部未完成行被忽略")
                    break
                raise ValueError(f"{path.name}:{index} 缺少行结束符")
            try:
                yield _json(line)
            except ValueError as error:
                raise ValueError(f"{path.name}:{index}: {error}") from error
            line = following


def inspect_session(directory: str | Path, *, recover: bool = False) -> dict:
    """检查五文件、版本、会话、序号、事件关联与完成时的计数。"""
    directory = Path(directory)
    manifest = _json((directory / "manifest.json").read_text(encoding="utf-8"))
    summary = _json((directory / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("archive_format_version") != "1.0" or manifest.get("schema_version") != "1.0":
        raise ValueError("不支持的存档/契约版本")
    session_id = manifest["session_id"]
    if summary.get("session_id") != session_id:
        raise ValueError("summary 会话不一致")
    warnings = []
    if summary.get("completed") is not True:
        warnings.append("会话未正常完成；以实际可解析记录重建计数")
    metadata = manifest["metadata"]
    decisions = list(_jsonl(directory / "decisions.jsonl", recover, warnings))
    for seq, frame in enumerate(decisions):
        if (
            frame.get("session_id") != session_id
            or frame.get("seq") != seq
            or frame.get("decision_id") != f"{session_id}/{seq:06d}"
        ):
            raise ValueError("决策会话、序号或 ID 不一致")
        if frame.get("schema_version") != "1.0" or frame.get("decision_state") not in STATES:
            raise ValueError("决策版本或状态不合法")
        for name in ("recording_id", "model_id", "pipeline_version", "config_hash"):
            if frame.get(name) != metadata[name]:
                raise ValueError(f"决策 {name} 与 manifest 不一致")
    with (directory / "events.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EVENT_FIELDS:
            raise ValueError("events.csv 字段不完整或顺序错误")
        events = list(reader)
    seen = set()
    for event in events:
        if None in event or any(value is None for value in event.values()):
            raise ValueError("events.csv 存在不完整行")
        if event["event_id"] in seen:
            raise ValueError("重复事件 ID")
        seen.add(event["event_id"])
        if event["session_id"] != session_id or event["recording_id"] != metadata["recording_id"]:
            raise ValueError("事件会话/记录不一致")
        start, end = int(event["start_decision_seq"]), int(event["end_decision_seq"])
        if not 0 <= start <= end < len(decisions):
            raise ValueError("事件引用的决策不存在")
        if event["state"] not in {"left_hand", "right_hand"}:
            raise ValueError("事件必须为左右手动作")
        if decisions[start]["decision_state"] != event["state"]:
            raise ValueError("事件起点与对应决策类别不一致")
        if not 0 <= int(event["start_sample"]) < int(event["end_sample"]):
            raise ValueError("事件样本范围不合法")
        if event["end_reason"] not in {
            "state_changed",
            "paused",
            "stopped",
            "finished",
            "invalid",
            "expired",
        }:
            raise ValueError("事件结束原因不合法")
    runtime = list(_jsonl(directory / "runtime.jsonl", recover, warnings))
    for item in runtime:
        if item.get("session_id") != session_id:
            raise ValueError("运行日志会话不一致")
        timing = item.get("at_monotonic_ms")
        if not isinstance(timing, (int, float)) or not math.isfinite(timing) or timing < 0:
            raise ValueError("运行日志单调时间不合法")
    counts = {"decisions": len(decisions), "events": len(events), "runtime": len(runtime)}
    if summary.get("completed") is True and (warnings or summary.get("counts") != counts):
        raise ValueError("声称完成的会话存在尾部损坏或计数不匹配")
    return {
        "session_id": session_id,
        "completed": summary.get("completed") is True,
        "counts": counts,
        "warnings": warnings,
        "evidence": metadata.get("evidence", "unspecified"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="独立核对会话存档，不加载 EEG 或算法")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = inspect_session(args.directory, recover=args.recover)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"存档核对失败：{error}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 不完整存档虽可抢救，也不应被批处理脚本误当成通过验收。
    return 0 if result["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
