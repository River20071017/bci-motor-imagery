"""D 的格式自检样例，全部为合成决策，不是 EEG 解码成绩。

python examples/demo_archive.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_app.outputs import JsonlResultSink  # noqa: E402
from eeg_core.contracts import (  # noqa: E402
    AppConfig,
    DecisionFrame,
    DecisionState,
    IntentEvent,
    RuntimeEvent,
    RuntimeEventKind,
    Scores,
    WindowRange,
    utc_now_iso,
)
from eeg_core.version import PIPELINE_VERSION  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="D 存档格式自检（合成结果，不评价算法）")
    parser.add_argument("--output", type=Path, default=Path("sessions"))
    args = parser.parse_args(argv)
    config = AppConfig()
    session_id = f"d-demo-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    sink = JsonlResultSink(
        args.output,
        session_id,
        {
            "recording_id": "synthetic-format-only",
            "model_id": "synthetic-no-trained-model",
            "pipeline_version": PIPELINE_VERSION,
            "config_hash": config.config_hash,
            "config": config.to_dict(),
            "app_version": "0.1.0",
            "dependencies": {"python": sys.version.split()[0]},
            "source_files": [],
            "evidence": "synthetic_archive_format_check_only",
        },
    )
    try:
        sink.submit(RuntimeEvent(session_id, 0, RuntimeEventKind.SESSION_START))
        for seq, state in enumerate(
            (
                DecisionState.WARMING_UP,
                DecisionState.LEFT_HAND,
                DecisionState.LEFT_HAND,
                DecisionState.IDLE,
            )
        ):
            end = 440 + seq * 40
            frame = DecisionFrame(
                session_id=session_id,
                recording_id="synthetic-format-only",
                seq=seq,
                decision_state=state,
                sample_rate_hz=160,
                source_t_sec=end / 160,
                generated_at_utc=utc_now_iso(),
                emitted_monotonic_ms=end / 160 * 1000,
                model_id="synthetic-no-trained-model",
                pipeline_version=PIPELINE_VERSION,
                config_hash=config.config_hash,
                window=None if seq == 0 else WindowRange(160, end - 320, end),
                scores=None if seq == 0 else Scores(0.1, 0.8, 0.1),
                intent=state if state.is_action else None,
            )
            sink.submit(frame)
        sink.submit(
            IntentEvent(
                event_id=f"{session_id}/evt-0000",
                session_id=session_id,
                recording_id="synthetic-format-only",
                state=DecisionState.LEFT_HAND,
                start_sample=480,
                start_decision_seq=1,
                end_sample=560,
                end_decision_seq=3,
                end_reason="state_changed",
            )
        )
        sink.submit(RuntimeEvent(session_id, 3500, RuntimeEventKind.FINISH))
        sink.close()
    except Exception:
        sink.close(completed=False, reason="demo_failed")
        raise
    print("合成格式自检完成；这些不是实验结果。")
    print(sink.status["directory"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
