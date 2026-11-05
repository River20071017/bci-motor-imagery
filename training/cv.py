"""2a 的「训练 / 验证 / 测试」划分与评估。

协议（2026-11-05 定）：

* **训练 + 验证**：session ``T`` 的 6 个 run 做 **按 run 分组的 6 折交叉验证**；
  每折留出 1 个 run 验证、其余 5 个 run 训练。不按 trial/窗随机分——重叠窗会泄漏。
* **最终模型**：超参/特征定完后，用 ``T`` 的全部 6 个 run 重训。
* **测试**：session ``E`` 封存，只在冻结后跑一次（加载层 ``allow_sealed=True``）。

2a 的一个 trial 产生两个 ``ReferenceMarker``（闲置 T0 + 想象 T1/T2/T3/T4），所以
用「窗口所在 marker 之前的想象段个数」就是 trial 序号，``trial_index // 48`` 就是 run 序号
（每 run 48 trial）。这样即使某个 trial 的闲置段被跳过，序号也不会错位。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold

from eeg_core.contracts import AppConfig, ContractError
from eeg_core.decoding import load_model_package
from training.evaluate import window_metrics
from training.pipeline import build_feature_dataset

__all__ = ["evaluate_recording", "run_grouped_cv", "window_trial_and_run"]

RUNS_PER_SESSION = 6
TRIALS_PER_RUN = 48


def _marker_index(recording: object, start: int, end: int) -> int:
    """窗口完整落在哪个 marker 内；找不到返回 -1。"""
    for index, marker in enumerate(recording.markers):
        if marker.duration_samples is None:
            continue
        stop = marker.onset_sample + marker.duration_samples
        if marker.onset_sample <= start and end <= stop:
            return index
    return -1


def window_trial_and_run(
    recording: object, windows: list[object], *, trials_per_run: int = TRIALS_PER_RUN
) -> list[tuple[int, int]]:
    """每个窗的 ``(trial, run)``；trial 按“该窗之前的想象段个数”计算。"""
    mi_before: list[int] = []
    count = 0
    for marker in recording.markers:
        if marker.label != "T0":
            count += 1
        mi_before.append(count)

    pairs: list[tuple[int, int]] = []
    for item in windows:
        span = item.window if hasattr(item, "window") else item
        index = _marker_index(recording, span.start_sample, span.end_sample)
        if index < 0:
            pairs.append((-1, -1))
            continue
        # 闲置段属于它后面那个 trial 之前的空档，归到前一个 trial；首个闲置段归 trial 0。
        trial = max(0, mi_before[index] - 1)
        pairs.append((trial, trial // trials_per_run))
    return pairs


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray) -> list[str]:
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(x_train, y_train)
    return list(model.predict(x_val))


def run_grouped_cv(
    recording: object, config: AppConfig, *, n_splits: int = 6, trials_per_run: int = TRIALS_PER_RUN
) -> dict[str, object]:
    """在 session T 上做按 run 分组的交叉验证，返回每折与汇总指标。"""
    samples, stats = build_feature_dataset(recording, config)
    if not samples:
        raise ContractError(f"{recording.info.recording_id} 没有可用窗，检查质量门与预热")

    features = np.stack([item.features for item in samples])
    labels = np.asarray([item.label for item in samples], dtype=str)
    pairs = window_trial_and_run(recording, list(samples), trials_per_run=trials_per_run)
    runs = np.asarray([run for _, run in pairs], dtype=np.int64)
    if (runs < 0).any():
        raise ContractError("有窗不属于任何 marker，无法确定 run 分组")
    n_runs = int(runs.max()) + 1
    if n_runs < n_splits:
        raise ContractError(f"只有 {n_runs} 个 run，不够 {n_splits} 折")

    folds: list[dict[str, object]] = []
    for fold, (train_idx, val_idx) in enumerate(
        GroupKFold(n_splits=n_splits).split(features, labels, runs)
    ):
        predictions = _fit_predict(features[train_idx], labels[train_idx], features[val_idx])
        metrics = window_metrics(list(labels[val_idx]), predictions)
        folds.append(
            {
                "fold": fold,
                "run": int(runs[val_idx][0]),
                "n_val": len(val_idx),
                "macro_f1": metrics["macro_f1"],
                "left_right_balanced_recall": metrics["left_right_balanced_recall"],
                "accuracy": metrics["accuracy"],
            }
        )

    def _mean(key: str) -> float:
        return float(np.mean([fold[key] for fold in folds]))

    return {
        "recording_id": recording.info.recording_id,
        "n_windows": len(samples),
        "n_runs": n_runs,
        "windows_per_run": {str(run): int((runs == run).sum()) for run in range(n_runs)},
        "folds": folds,
        "mean": {
            "macro_f1": _mean("macro_f1"),
            "left_right_balanced_recall": _mean("left_right_balanced_recall"),
            "accuracy": _mean("accuracy"),
        },
        "std": {
            "macro_f1": float(np.std([fold["macro_f1"] for fold in folds])),
            "left_right_balanced_recall": float(
                np.std([fold["left_right_balanced_recall"] for fold in folds])
            ),
            "accuracy": float(np.std([fold["accuracy"] for fold in folds])),
        },
        "dataset": {
            "n_chunks": stats.n_chunks,
            "invalid_chunks": stats.invalid_chunks,
            "total_windows": stats.total_windows,
            "usable_windows": stats.usable_windows,
        },
    }


def evaluate_recording(
    model_json: str | Path, recording: object, config: AppConfig
) -> dict[str, object]:
    """在给定记录（通常是封存的 E session）上评估一个已训练的模型包。"""
    package = load_model_package(model_json)
    package.spec.check_compatible(recording.info, config)
    samples, stats = build_feature_dataset(recording, config)
    if not samples:
        raise ContractError(f"{recording.info.recording_id} 没有可用窗")
    metrics = window_metrics(
        [item.label for item in samples],
        [package.predict_label(item.features) for item in samples],
    )
    return {
        "recording_id": recording.info.recording_id,
        "model_id": package.model_id,
        "metrics": metrics,
        "dataset": {
            "n_chunks": stats.n_chunks,
            "invalid_chunks": stats.invalid_chunks,
            "total_windows": stats.total_windows,
            "usable_windows": stats.usable_windows,
        },
    }
