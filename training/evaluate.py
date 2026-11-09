"""窗口级评估：先把训练/推理链路跑通，再上 PRD §10 的决策点口径。

这里的 ``macro_f1`` 是**窗口级**的：每个 2 s 窗一个预测，不按 250 ms 决策点计分，
也不排除标签转换区。它适合开发阶段比较特征/模型，不可直接当作封存测试报告。

正式的决策点评估（排除转换前后 1 s、起始 3 s 预热、uncertain/invalid 计入遗漏）
在 PRD §10，后续在 ``evaluation/`` 里实现。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from eeg_core.contracts import AppConfig, ContractError
from eeg_core.decoding import CLASSES, load_model_package
from training.pipeline import RecordingLike, build_feature_dataset, load_split

__all__ = ["evaluate_subject", "main", "part_file_of", "window_metrics"]


def window_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    """窗口级混淆矩阵、每类 P/R/F1、宏 F1、左右手平衡召回。"""
    if not y_true:
        raise ContractError("没有可评估的窗口")
    if len(y_true) != len(y_pred):
        raise ContractError("y_true 与 y_pred 长度不一致")

    labels = list(CLASSES)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    left_index = labels.index("left_hand")
    right_index = labels.index("right_hand")
    return {
        "n_windows": len(y_true),
        "accuracy": float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)])),
        "macro_f1": float(np.mean(f1)),
        "left_right_balanced_recall": float((recall[left_index] + recall[right_index]) / 2.0),
        "per_class": per_class,
        "confusion": {
            true_label: {
                pred_label: int(matrix[true_index, pred_index])
                for pred_index, pred_label in enumerate(labels)
            }
            for true_index, true_label in enumerate(labels)
        },
    }


def part_file_of(split: Mapping[str, object], subject: str, part: str) -> Path:
    """从划分文件取某人的 validation/test 文件；只做解析，不读文件。"""
    if part not in {"validation", "test"}:
        raise ContractError(f"part 只能是 validation 或 test，收到 {part!r}")
    subjects = split.get("subjects")
    if not isinstance(subjects, dict) or subject not in subjects:
        raise ContractError(f"划分文件里没有受试者 {subject!r}")
    entry = subjects[subject]
    if not isinstance(entry, dict):
        raise ContractError(f"{subject} 的划分条目不是对象")
    part_entry = entry.get(part)
    if not isinstance(part_entry, dict) or "file" not in part_entry:
        raise ContractError(f"{subject} 缺少 {part}.file")
    return Path(str(part_entry["file"]))


def evaluate_subject(
    model_json: str | Path,
    *,
    subject: str,
    part: str,
    recording: RecordingLike,
    config: AppConfig | None = None,
) -> dict[str, object]:
    """加载模型，在一名受试者的 ``validation`` / ``test`` 记录上算窗口级指标。

    本函数只依赖 ``RecordingLike`` 和 ``eeg_core``，不读 EDF、不依赖 ``eeg_app``；
    “把文件变成 Recording” 由入口 ``main`` 负责。
    """
    if part not in {"validation", "test"}:
        raise ContractError(f"part 只能是 validation 或 test，收到 {part!r}")

    cfg = config or AppConfig()
    package = load_model_package(model_json)
    package.spec.check_compatible(recording.info, cfg)

    samples, stats = build_feature_dataset(recording, cfg)
    if not samples:
        raise ContractError(
            f"{subject} {part} 没有可用窗：invalid={stats.invalid_chunks}/{stats.n_chunks}"
        )
    features = np.stack([item.features for item in samples])
    probabilities = package.predict_proba(features)
    y_pred = [CLASSES[int(index)] for index in np.argmax(probabilities, axis=1)]
    y_true = [item.label for item in samples]

    return {
        "subject": subject,
        "part": part,
        "recording_id": recording.info.recording_id,
        "model_id": package.model_id,
        "pipeline_version": package.spec.pipeline_version,
        "dataset": stats.as_dict(),
        "metrics": window_metrics(y_true, y_pred),
    }


def _load_recording(data_root: Path, part_file: Path, *, allow_sealed: bool) -> RecordingLike:
    """组合根：只有这里接触应用层的 EDF 读取。"""
    from eeg_app.sources.input import load_edf

    return load_edf(data_root / part_file, with_sha256=False, allow_sealed=allow_sealed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="窗口级评估一个个人 LDA 模型")
    parser.add_argument("--model", type=Path, required=True, help="模型 .model.json 路径")
    parser.add_argument("--subject", required=True, help="如 S001")
    parser.add_argument("--part", choices=["validation", "test"], default="validation")
    parser.add_argument("--data-root", type=Path, default=Path("data/eegmmidb"))
    parser.add_argument("--splits", type=Path, default=Path("training/splits.json"))
    parser.add_argument("--out", type=Path, default=None, help="可选：把结果写成 JSON")
    parser.add_argument(
        "--allow-sealed",
        action="store_true",
        help="确认一次性使用封存 run12（--part test）；默认拒绝，跑完不要再改任何东西",
    )
    args = parser.parse_args(argv)

    split = load_split(args.splits)
    part_file = part_file_of(split, args.subject, args.part)
    recording = _load_recording(args.data_root, part_file, allow_sealed=args.allow_sealed)
    result = evaluate_subject(
        args.model,
        subject=args.subject,
        part=args.part,
        recording=recording,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
