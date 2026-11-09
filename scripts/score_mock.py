"""用 mock（合成）数据给整条算法链打分。

这不是真实 EEG 的结论，只是“在已知答案下，链路能不能学对”的体检：训练和测试都由
``training/mock.py`` 生成，T1/T2 在 8–13 / 13–30 Hz 上做左右不对称，正确答案已知；
分数偏低只可能是链路（质量 → 滤波 → 分窗 → 特征 → LDA → 决策）出了问题。

两个层级分别打分，不要混：

* **窗口级**：走 ``build_feature_dataset``——和训练/评估同一条路径；每个 2 s 窗一个预测，
  只算整窗落在同一标注区间、且过了 3 s 预热的窗。
* **决策级**：模拟运行时每 250 ms 一个更新点，把质量、预热、分数喂给 ``DecisionEngine``。
  ``uncertain``（没到门限或连续确认不够）、``invalid``、``warming_up`` 都算“没答对”，
  同时另报“有答案的决策点里答对多少”；两个数一起看才知道损失在特征还是在确认逻辑。

场景：

* ``clean``：正常合成信号（类别可分）；
* ``zero_effect``：测试信号没有类别差异——对照，分数应回到 1/3 随机水平；
* ``spike`` / ``flatline`` / ``nan``：测试记录注入伪迹，检查质量门拦不拦、拦完还能答多少。

用法::

    .\\.venv\\Scripts\\python.exe scripts\\score_mock.py
    .\\.venv\\Scripts\\python.exe scripts\\score_mock.py --test-sec 240 --json score.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eeg_core.contracts import (  # noqa: E402
    AppConfig,
    DecisionState,
    EEGChunk,
    QualityReport,
    ReferenceMarker,
    SourceInfo,
    WindowRange,
)
from eeg_core.decoding import DecisionEngine, ModelPackage, load_model_package  # noqa: E402
from eeg_core.processing import (  # noqa: E402
    FilterBank,
    hop_samples,
    log_variance_features,
    required_warmup_samples,
    window_samples,
)
from training.evaluate import window_metrics  # noqa: E402
from training.mock import make_mock_chunks  # noqa: E402
from training.pipeline import LABEL_BY_MARKER, build_feature_dataset, label_of_window  # noqa: E402
from training.train import train_subject  # noqa: E402

SCENARIOS = (
    ("clean", {}),
    ("zero_effect", {"class_effect": 0.0}),
    ("spike", {"artifact": "spike"}),
    ("flatline", {"artifact": "flatline"}),
    ("nan", {"artifact": "nan"}),
)
"""场景名 → 传给 ``make_mock_chunks`` 的覆盖参数。"""

CLASSES = ("idle", "left_hand", "right_hand")


@dataclass(frozen=True, slots=True)
class _Recording:
    """``RecordingLike`` + ``sha256``；``MockRecording`` 没有 sha 字段，训练入口要读。"""

    info: SourceInfo
    samples: np.ndarray
    markers: tuple[ReferenceMarker, ...]
    sha256: str | None = None

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[1])


@dataclass(frozen=True, slots=True)
class _Stream:
    """一次流式处理的缓存：两路频带、逐样本有效性、连续段起点、无效原因。"""

    band_low: np.ndarray
    band_high: np.ndarray
    valid: np.ndarray
    segment_start: np.ndarray
    reasons: tuple[tuple[str, ...] | None, ...]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 mock 数据给算法链打分")
    parser.add_argument("--train-sec", type=float, default=160.0, help="训练 mock 时长（秒）")
    parser.add_argument("--test-sec", type=float, default=120.0, help="测试 mock 时长（秒）")
    parser.add_argument("--trial-sec", type=float, default=4.0, help="每个试次时长（秒）")
    parser.add_argument("--train-seed", type=int, default=11)
    parser.add_argument("--test-seed", type=int, default=22)
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="模型临时目录（默认仓库内隐藏目录）"
    )
    parser.add_argument("--json", type=Path, default=None, help="可选：把结果写成 JSON")
    return parser.parse_args(argv)


def _make_recording(
    config: AppConfig,
    *,
    duration_sec: float,
    trial_sec: float,
    seed: int,
    **overrides: object,
) -> _Recording:
    info, markers, chunks = make_mock_chunks(
        duration_sec=duration_sec,
        trial_sec=trial_sec,
        sample_rate_hz=config.source.sample_rate_hz,
        seed=seed,
        **overrides,
    )
    samples = np.concatenate([chunk.samples for chunk in chunks], axis=1)
    return _Recording(info=info, samples=samples, markers=markers)


def _stream(recording: _Recording, config: AppConfig) -> _Stream:
    """按 8 样本块喂 ``FilterBank``，缓存频带与逐样本有效性（与 training/pipeline 同口径）。"""
    source, processing = config.source, config.processing
    chunk_samples = max(1, int(round(source.target_chunk_ms / 1000.0 * source.sample_rate_hz)))
    n_samples = recording.n_samples
    n_channels = len(recording.info.channel_names)
    band_low = np.full((n_channels, n_samples), np.nan, dtype=np.float64)
    band_high = np.full((n_channels, n_samples), np.nan, dtype=np.float64)
    valid = np.zeros(n_samples, dtype=bool)
    segment_start = np.full(n_samples, -1, dtype=np.int64)
    reasons: list[tuple[str, ...] | None] = [None] * n_samples

    bank = FilterBank(
        amplitude_limit_v=processing.amplitude_limit_uv * 1e-6,
        flatline_sec=processing.flatline_sec,
        flatline_tolerance_v=processing.flatline_tolerance_uv * 1e-6,
        config=config,
    )
    segment: int | None = None
    for seq, start in enumerate(range(0, n_samples, chunk_samples)):
        stop = min(start + chunk_samples, n_samples)
        chunk = EEGChunk(
            source_id=recording.info.source_id,
            recording_id=recording.info.recording_id,
            seq=seq,
            start_sample=start,
            sample_rate_hz=recording.info.sample_rate_hz,
            channel_names=recording.info.channel_names,
            samples=recording.samples[:, start:stop],
            released_monotonic_ms=float(seq * source.target_chunk_ms),
            discontinuity=(seq == 0),
        )
        result = bank.process(chunk)
        quality = result["quality"]
        if not quality.valid:
            for index in range(start, stop):
                reasons[index] = quality.reasons
            segment = None
            continue
        if chunk.discontinuity or segment is None:
            segment = start
        band_low[:, start:stop] = result["band_8_13_hz"]
        band_high[:, start:stop] = result["band_13_30_hz"]
        valid[start:stop] = True
        segment_start[start:stop] = segment
    return _Stream(
        band_low=band_low,
        band_high=band_high,
        valid=valid,
        segment_start=segment_start,
        reasons=tuple(reasons),
    )


def _decision_frames(
    package: ModelPackage, recording: _Recording, config: AppConfig, stream: _Stream
) -> list[tuple[str | None, object]]:
    """每 250 ms 一个决策点，返回 ``(窗口标签或 None, DecisionFrame)``。"""
    processing = config.processing
    span_samples = window_samples(config)
    step = hop_samples(config)
    warmup = required_warmup_samples(config)
    rate = recording.info.sample_rate_hz
    indices = [recording.info.channel_names.index(name) for name in config.source.model_channels]
    engine = DecisionEngine(
        config=config,
        spec=package.spec,
        session_id="mock-score",
        recording_id=recording.info.recording_id,
    )
    frames: list[tuple[str | None, object]] = []

    for end in range(step, recording.n_samples + 1, step):
        start = end - span_samples
        source_t = end / rate
        if start < 0:
            frame = engine.update(
                quality=QualityReport.ok(),
                ready=False,
                scores=None,
                window=None,
                emitted_monotonic_ms=end / rate * 1000.0,
                source_t_sec=source_t,
            )
            frames.append((None, frame))
            continue

        window = WindowRange(sample_rate_hz=rate, start_sample=start, end_sample=end)
        # 只把“整窗在 3 s 预热之后”的点计分；伪迹造成的重新预热、invalid 都算没答对。
        label = label_of_window(recording.markers, start, end) if start >= warmup else None
        if not bool(stream.valid[start:end].all()):
            codes: set[str] = set()
            for index in range(start, end):
                codes.update(stream.reasons[index] or ())
            frame = engine.update(
                quality=QualityReport.failed(tuple(codes) or ("amplitude",), window=window),
                ready=False,
                scores=None,
                window=window,
                emitted_monotonic_ms=end / rate * 1000.0,
                source_t_sec=source_t,
            )
            frames.append((label, frame))
            continue

        segment = int(stream.segment_start[start])
        ready = segment >= 0 and segment == int(stream.segment_start[end - 1])
        ready = ready and start >= segment + warmup
        if not ready:
            frame = engine.update(
                quality=QualityReport.ok(window=window),
                ready=False,
                scores=None,
                window=None,
                emitted_monotonic_ms=end / rate * 1000.0,
                source_t_sec=source_t,
            )
            frames.append((label, frame))
            continue

        features = log_variance_features(
            stream.band_low,
            stream.band_high,
            indices,
            start,
            end,
            floor=processing.log_variance_floor,
        )
        frame = engine.update(
            quality=QualityReport.ok(window=window),
            ready=True,
            scores=package.scores(features),
            window=window,
            emitted_monotonic_ms=end / rate * 1000.0,
            source_t_sec=source_t,
        )
        frames.append((label, frame))
    return frames


def _decision_metrics(
    frames: list[tuple[str | None, object]],
    markers: tuple[ReferenceMarker, ...],
    sample_rate: float,
) -> dict[str, object]:
    """决策级指标：严格准确率、覆盖/答案内准确率、错误动作率、动作确认延迟。"""
    labelled = [(label, frame) for label, frame in frames if label is not None]
    if not labelled:
        return {"n_labeled": 0}

    states = [frame.decision_state for _, frame in labelled]
    counts = Counter(state.value for state in states)
    total = len(labelled)
    correct = sum(1 for label, frame in labelled if frame.decision_state.value == label)
    answered = sum(counts[name] for name in CLASSES)
    wrong_action = sum(
        1
        for label, frame in labelled
        if frame.decision_state.is_action and frame.decision_state.value != label
    )
    confusion = {true: {pred: 0 for pred in CLASSES} for true in CLASSES}
    for label, frame in labelled:
        state = frame.decision_state
        if state is DecisionState.IDLE or state.is_action:
            confusion[label][state.value] += 1

    # 动作确认延迟：每个 T1/T2 试次里第一次给出正确动作的时刻（窗口末端）减试次起点。
    latencies: list[float] = []
    missed_trials = 0
    for marker in markers:
        if marker.label not in LABEL_BY_MARKER or marker.duration_samples is None:
            continue
        expected = LABEL_BY_MARKER[marker.label]
        if expected not in ("left_hand", "right_hand"):
            continue
        onset = marker.onset_sample / sample_rate
        stop = (marker.onset_sample + marker.duration_samples) / sample_rate
        first = None
        for label, frame in labelled:
            if label != expected or frame.window is None:
                continue
            if not (onset <= frame.window.start_sample / sample_rate):
                continue
            if not (frame.window.end_sample / sample_rate <= stop):
                continue
            if frame.decision_state.value == expected:
                first = frame.source_t_sec - onset
                break
        if first is None:
            missed_trials += 1
        else:
            latencies.append(first)

    return {
        "n_labeled": total,
        "counts": dict(counts),
        "strict_accuracy": correct / total,
        "coverage": answered / total,
        "accuracy_when_answered": correct / answered if answered else 0.0,
        "wrong_action_rate": wrong_action / total,
        "confusion": confusion,
        "action_latency_sec": {
            "median": float(np.median(latencies)) if latencies else None,
            "p90": float(np.percentile(latencies, 90)) if latencies else None,
            "scored_trials": len(latencies),
            "missed_trials": missed_trials,
        },
    }


def _window_scores(
    package: ModelPackage, recording: _Recording, config: AppConfig
) -> dict[str, object]:
    samples, stats = build_feature_dataset(recording, config)
    y_true = [item.label for item in samples]
    y_pred = [package.predict_label(item.features) for item in samples]
    metrics = window_metrics(y_true, y_pred) if y_true else {}
    return {
        "chunks": stats.n_chunks,
        "invalid_chunks": stats.invalid_chunks,
        "total_windows": stats.total_windows,
        "usable_windows": stats.usable_windows,
        "metrics": metrics,
    }


def _print_scenario(name: str, window: dict[str, object], decision: dict[str, object]) -> None:
    metrics = window["metrics"]
    print(f"\n-- {name} --")
    print(
        f"  窗口级：可用 {window['usable_windows']}/{window['total_windows']} 窗，"
        f"invalid 块 {window['invalid_chunks']}/{window['chunks']}"
    )
    if metrics:
        print(
            f"    准确率 {metrics['accuracy']:.3f}  宏 F1 {metrics['macro_f1']:.3f}  "
            f"左右手平衡召回 {metrics['left_right_balanced_recall']:.3f}"
        )
        support = "、".join(
            f"{label} {metrics['per_class'][label]['support']}" for label in CLASSES
        )
        f1 = "、".join(f"{label} {metrics['per_class'][label]['f1']:.2f}" for label in CLASSES)
        print(f"    支持数：{support}")
        print(f"    每类 F1：{f1}")
    if not decision.get("n_labeled"):
        print("  决策级：没有带标签的决策点")
        return
    counts = decision["counts"]
    print(
        f"  决策级：有标签 {decision['n_labeled']} 点 → "
        f"严格准确率 {decision['strict_accuracy']:.3f}，"
        f"有答案覆盖 {decision['coverage']:.3f}，"
        f"答案内准确率 {decision['accuracy_when_answered']:.3f}，"
        f"错误动作率 {decision['wrong_action_rate']:.3f}"
    )
    detail = "  ".join(
        f"{key} {counts.get(key, 0)}"
        for key in ("idle", "left_hand", "right_hand", "uncertain", "invalid", "warming_up")
    )
    print(f"    状态分布：{detail}")
    latency = decision.get("action_latency_sec") or {}
    if latency.get("median") is not None:
        trials = latency["scored_trials"] + latency["missed_trials"]
        print(
            f"    动作确认延迟：中位 {latency['median']:.2f} s，p90 {latency['p90']:.2f} s，"
            f"漏答 {latency['missed_trials']}/{trials} 个动作试次"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = AppConfig()
    train_recording = _make_recording(
        config, duration_sec=args.train_sec, trial_sec=args.trial_sec, seed=args.train_seed
    )
    print(
        f"训练 mock：{args.train_sec:g} s / 试次 {args.trial_sec:g} s / seed {args.train_seed}；"
        f"测试 mock：{args.test_sec:g} s / seed {args.test_seed}"
    )

    work_dir = args.work_dir if args.work_dir is not None else ROOT / ".mock-score-models"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        spec, model_json = train_subject(
            "S999",
            recording=train_recording,
            train_file="mock",
            config=config,
            out_dir=work_dir,
            model_id="mock-lda",
        )
        package = load_model_package(model_json)
        print(f"模型：{spec.model_id}（{spec.pipeline_version}，18 维，三类）")

        results: dict[str, object] = {}
        for name, overrides in SCENARIOS:
            recording = _make_recording(
                config,
                duration_sec=args.test_sec,
                trial_sec=args.trial_sec,
                seed=args.test_seed,
                **overrides,
            )
            package.spec.check_compatible(recording.info, config)
            window_result = _window_scores(package, recording, config)
            stream = _stream(recording, config)
            frames = _decision_frames(package, recording, config, stream)
            decision_result = _decision_metrics(
                frames, recording.markers, recording.info.sample_rate_hz
            )
            _print_scenario(name, window_result, decision_result)
            results[name] = {"window": window_result, "decision": decision_result}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if args.json is not None:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.json}")

    _print_verdict(results)
    return 0


def _print_verdict(results: dict[str, object]) -> None:
    """把 clean 与 zero_effect 对照放在一起，说明分数该怎么读。"""

    def metrics(name: str) -> dict[str, object]:
        return results[name]["window"]["metrics"]  # type: ignore[index]

    def decision(name: str) -> dict[str, object]:
        return results[name]["decision"]  # type: ignore[index]

    clean, control = metrics("clean"), metrics("zero_effect")
    print("\n== 判读 ==")
    print(
        f"clean 窗口级宏 F1 {clean['macro_f1']:.3f}；"
        f"zero_effect 对照 {control['macro_f1']:.3f}（三类随机水平 0.333）"
    )
    if control["macro_f1"] < 0.45:
        print("→ 对照接近随机，说明分数确实来自类别差异，不是标签泄漏或过拟合。")
    else:
        print("→ 对照明显高于随机，先查标签/特征是否串了，再谈分数。")
    print(
        f"clean 决策级：严格准确率 {decision('clean')['strict_accuracy']:.3f}，"
        f"答案内准确率 {decision('clean')['accuracy_when_answered']:.3f}，"
        f"错误动作率 {decision('clean')['wrong_action_rate']:.3f}"
    )
    print(
        "→ 窗级高而严格准确率低的那部分，损失在确认逻辑（uncertain/预热），不是特征。"
        "\n→ 伪迹场景的 invalid 决策点占比，就是质量门实际付出的代价。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
