"""D 的存档验收：真实契约对象 → 后台 IO → 独立标准库读取。"""

from __future__ import annotations

import importlib.util
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from eeg_app.outputs import ArchiveWriteError, JsonlResultSink
from eeg_app.outputs.archive import SessionArchive
from eeg_core.contracts import (
    AppConfig,
    ContractError,
    DecisionFrame,
    DecisionState,
    IntentEvent,
    RuntimeEvent,
    RuntimeEventKind,
    Scores,
    WindowRange,
)
from eeg_core.version import PIPELINE_VERSION

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "independent_reader", ROOT / "examples/read_session.py"
)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def metadata():
    return {
        "recording_id": "format-only",
        "model_id": "test-model",
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": AppConfig().config_hash,
        "evidence": "synthetic_test_only",
    }


def frame(seq=0):
    end = 480 + seq * 40
    return DecisionFrame(
        session_id="test",
        recording_id="format-only",
        seq=seq,
        decision_state=DecisionState.LEFT_HAND,
        intent=DecisionState.LEFT_HAND,
        sample_rate_hz=160,
        source_t_sec=end / 160,
        generated_at_utc="2026-11-09T00:00:00.000Z",
        emitted_monotonic_ms=seq * 250,
        model_id="test-model",
        pipeline_version=PIPELINE_VERSION,
        config_hash=AppConfig().config_hash,
        window=WindowRange(160, end - 320, end),
        scores=Scores(0.1, 0.8, 0.1),
    )


def event():
    return IntentEvent(
        event_id="evt-0",
        session_id="test",
        recording_id="format-only",
        state=DecisionState.LEFT_HAND,
        start_sample=480,
        start_decision_seq=0,
        end_sample=520,
        end_decision_seq=1,
        end_reason="finished",
    )


def test_five_files_round_trip_and_exact_decision(tmp_path):
    sink = JsonlResultSink(tmp_path, "test", metadata())
    decisions = [frame(i) for i in range(40)]
    for item in decisions:
        sink.submit(item)
    sink.submit(event())
    sink.submit(RuntimeEvent("test", 10000, RuntimeEventKind.FINISH, "完成，包含中文"))
    assert sink.status["completed"] is False
    sink.close()
    directory = tmp_path / "session_test"
    assert {p.name for p in directory.iterdir()} == {
        "manifest.json",
        "decisions.jsonl",
        "events.csv",
        "runtime.jsonl",
        "summary.json",
    }
    saved = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines()]
    assert saved == [item.to_dict() for item in decisions]
    assert reader.inspect_session(directory)["counts"] == {
        "decisions": 40,
        "events": 1,
        "runtime": 1,
    }
    assert sink.status["completed"] is True
    sink.close()  # 幂等
    with pytest.raises(ArchiveWriteError):
        sink.submit(frame(40))


@pytest.mark.parametrize("name", ["../outside", "bad/name", "bad\\name", "", "CON:"])
def test_reject_unsafe_session_directory(tmp_path, name):
    with pytest.raises(ArchiveWriteError):
        JsonlResultSink(tmp_path, name, metadata())
    assert list(tmp_path.iterdir()) == []


def test_never_overwrite_existing_session(tmp_path):
    sink = JsonlResultSink(tmp_path, "test", metadata())
    sink.submit(frame())
    sink.close()
    before = (tmp_path / "session_test/decisions.jsonl").read_bytes()
    with pytest.raises(ArchiveWriteError):
        JsonlResultSink(tmp_path, "test", metadata())
    assert (tmp_path / "session_test/decisions.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "changed",
    [
        {"session_id": "other"},
        {"recording_id": "other"},
        {"model_id": "other"},
        {"pipeline_version": "other"},
        {"config_hash": "other"},
        {"seq": 1},
    ],
)
def test_wrong_identity_or_skipped_decision_fails_session(tmp_path, changed):
    sink = JsonlResultSink(tmp_path, "test", metadata())
    sink.submit(replace(frame(), **changed))
    with pytest.raises(ArchiveWriteError):
        sink.close()
    assert sink.status["state"] == "failed"
    summary = json.loads((tmp_path / "session_test/summary.json").read_text())
    assert summary["completed"] is False


def test_duplicate_decision_rejected_not_silently_written(tmp_path):
    sink = JsonlResultSink(tmp_path, "test", metadata())
    sink.submit(frame())
    sink.submit(frame())
    with pytest.raises(ArchiveWriteError):
        sink.close()
    report = reader.inspect_session(tmp_path / "session_test")
    assert report["completed"] is False
    assert report["counts"]["decisions"] == 1


def test_event_must_be_closed_unique_and_link_existing_decisions(tmp_path):
    archive = SessionArchive(tmp_path, "test", metadata())
    archive.append_decision(frame())
    with pytest.raises(ContractError):
        archive.append_event(event())  # end seq=1 还没写入
    archive.append_decision(frame(1))
    with pytest.raises(ContractError):
        archive.append_event(
            replace(event(), end_sample=None, end_decision_seq=None, end_reason=None)
        )
    archive.append_event(event())
    with pytest.raises(ContractError):
        archive.append_event(event())
    archive.close(completed=True, reason="finished")
    assert reader.inspect_session(archive.directory)["counts"]["events"] == 1


def test_no_scores_for_invalid_and_uncertain_preserved(tmp_path):
    from eeg_core.contracts import QualityReport

    sink = JsonlResultSink(tmp_path, "test", metadata())
    sink.submit(
        replace(
            frame(),
            decision_state=DecisionState.INVALID,
            intent=None,
            scores=None,
            quality=QualityReport.failed(["non_finite"]),
        )
    )
    sink.submit(replace(frame(1), decision_state=DecisionState.UNCERTAIN, intent=None))
    sink.close()
    content = (tmp_path / "session_test/decisions.jsonl").read_text()
    rows = [json.loads(line) for line in content.splitlines()]
    assert rows[0]["scores"] is None and "NaN" not in content
    assert rows[1]["decision_state"] == "uncertain"


def test_disk_error_latched_and_incomplete(tmp_path, monkeypatch):
    sink = JsonlResultSink(tmp_path, "test", metadata())

    def disk_full(self, item):
        raise OSError("模拟磁盘已满")

    monkeypatch.setattr(SessionArchive, "append_decision", disk_full)
    sink.submit(frame())
    with pytest.raises(ArchiveWriteError, match="磁盘已满"):
        sink.close()
    with pytest.raises(ArchiveWriteError, match="磁盘已满"):
        sink.submit(frame())
    assert reader.inspect_session(tmp_path / "session_test")["completed"] is False


def test_unusable_output_location_reports_failure(tmp_path):
    location = tmp_path / "regular-file"
    location.write_text("不是目录")
    with pytest.raises(ArchiveWriteError):
        JsonlResultSink(location, "test", metadata())


def test_queue_full_is_explicit_failure(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = SessionArchive.append_decision

    def blocked(self, item):
        entered.set()
        assert release.wait(3)
        original(self, item)

    monkeypatch.setattr(SessionArchive, "append_decision", blocked)
    sink = JsonlResultSink(tmp_path, "test", metadata(), queue_size=1)
    try:
        sink.submit(frame())
        assert entered.wait(3)
        sink.submit(frame(1))
        with pytest.raises(ArchiveWriteError, match="队列已满"):
            sink.submit(frame(2))
    finally:
        release.set()
        with pytest.raises(ArchiveWriteError):
            sink.close()
    assert sink.status["completed"] is False


def test_periodic_flush_without_more_incoming_results(tmp_path, monkeypatch):
    flushed = threading.Event()
    original = SessionArchive.flush

    def watched(self):
        original(self)
        if self._counts["decisions"]:
            flushed.set()

    monkeypatch.setattr(SessionArchive, "flush", watched)
    sink = JsonlResultSink(tmp_path, "test", metadata(), flush_interval_sec=0.05)
    try:
        sink.submit(frame())
        assert flushed.wait(3)
        assert json.loads((tmp_path / "session_test/decisions.jsonl").read_text())["seq"] == 0
    finally:
        sink.close()


def test_recovery_ignores_only_unterminated_tail_and_does_not_modify(tmp_path):
    sink = JsonlResultSink(tmp_path, "test", metadata())
    sink.submit(frame())
    sink.close(completed=False, reason="interrupted")
    directory = tmp_path / "session_test"
    path = directory / "decisions.jsonl"
    with path.open("a") as stream:
        stream.write('{"seq":1')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        reader.inspect_session(directory)
    report = reader.inspect_session(directory, recover=True)
    assert report["counts"]["decisions"] == 1
    assert report["warnings"] and report["completed"] is False
    assert path.read_bytes() == before
    path.write_text("{broken}\n" + before.decode())
    with pytest.raises(ValueError):
        reader.inspect_session(directory, recover=True)


def test_reader_detects_summary_tampering(tmp_path):
    sink = JsonlResultSink(tmp_path, "test", metadata())
    sink.submit(frame())
    sink.close()
    path = tmp_path / "session_test/summary.json"
    summary = json.loads(path.read_text())
    summary["counts"]["decisions"] = 100
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="计数"):
        reader.inspect_session(path.parent)
