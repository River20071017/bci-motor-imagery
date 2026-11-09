"""按 ``training/splits.json`` 训练个人三状态 LDA，并导出模型包。

用法（在仓库根目录）：

.. code-block:: powershell

    .\\.venv\\Scripts\\python.exe -m training.train --subjects S001,S002

输出（默认 ``training/models/``）：

* ``<model_id>.weights.npz``：``lda_npz_v1`` 权重；
* ``<model_id>.model.json``：``ModelInputSpec`` + ``DecisionConfig`` + 训练数据统计。

每人一个模型：run 4 拟合，run 8 选门限，run 12 封存。本脚本只碰 run 4。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eeg_core.contracts import (
    AppConfig,
    ContractError,
    ModelInputSpec,
    utc_now_iso,
)
from eeg_core.decoding import fit_lda, save_weights
from eeg_core.processing import FEATURE_DIM
from eeg_core.version import PIPELINE_VERSION
from training.pipeline import RecordingLike, build_feature_dataset, load_split

__all__ = ["main", "train_file_of", "train_subject"]

DEFAULT_DATA_ROOT = Path("data/eegmmidb")
DEFAULT_SPLITS = Path("training/splits.json")
DEFAULT_OUT = Path("training/models")


def _commit_short() -> str:
    """取当前提交短号；不在 git 仓库里时返回 ``nogit``。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "nogit"


def _model_id(subject: str, *, now: datetime | None = None) -> str:
    moment = now or datetime.now(UTC)
    return f"mi-{subject}-lda-{moment:%Y%m%d}-{_commit_short()}"


def train_file_of(split: Mapping[str, object], subject: str) -> Path:
    """从划分文件取某人的 run4 训练文件；只做解析，不读文件。"""
    subjects = split.get("subjects")
    if not isinstance(subjects, dict) or subject not in subjects:
        raise ContractError(f"划分文件里没有受试者 {subject!r}")
    entry = subjects[subject]
    if not isinstance(entry, dict):
        raise ContractError(f"{subject} 的划分条目不是对象")
    train_entry = entry.get("train")
    if not isinstance(train_entry, dict) or "file" not in train_entry:
        raise ContractError(f"{subject} 缺少 train.file")
    return Path(str(train_entry["file"]))


def train_subject(
    subject: str,
    *,
    recording: RecordingLike,
    train_file: str,
    config: AppConfig,
    out_dir: Path = DEFAULT_OUT,
    model_id: str | None = None,
) -> tuple[ModelInputSpec, Path]:
    """训练一个受试者并写模型包；返回 ``(spec, model_json)``。

    本函数只依赖 ``RecordingLike`` 和 ``eeg_core``，不读 EDF、不依赖 ``eeg_app``；
    “把文件变成 Recording” 由入口 ``main`` 负责。
    """
    samples, stats = build_feature_dataset(recording, config)
    if not samples:
        raise ContractError(
            f"{subject} 没有可用窗：invalid={stats.invalid_chunks}/"
            f"{stats.n_chunks}，先检查 A 的质量门与预热"
        )

    features = np.stack([item.features for item in samples])
    labels = np.asarray([item.label for item in samples], dtype=str)
    if features.shape[1] != FEATURE_DIM:
        raise ContractError(
            f"特征维数应为 {FEATURE_DIM}，收到 {features.shape[1]}；检查配置与特征布局"
        )

    arrays, fit_info = fit_lda(
        features,
        labels,
        solver=config.decision.lda_solver,
        shrinkage=config.decision.lda_shrinkage,
    )
    identifier = model_id or _model_id(subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{identifier}.weights.npz"
    weights_sha = save_weights(weights_path, arrays)

    training_files = None
    if recording.sha256 is not None:
        training_files = {train_file: recording.sha256}

    spec = ModelInputSpec(
        model_id=identifier,
        pipeline_version=PIPELINE_VERSION,
        channels=config.source.model_channels,
        sample_rate_hz=recording.info.sample_rate_hz,
        reference_policy=config.source.reference_policy,
        bands_hz=config.processing.bands_hz,
        window_sec=config.processing.window_sec,
        step_sec=config.processing.step_sec,
        feature_layout="channel_band",
        feature_floor=config.processing.log_variance_floor,
        classes=("idle", "left_hand", "right_hand"),
        decision=config.decision,
        weights_sha256=weights_sha,
        weights_format="lda_npz_v1",
        preprocessing_hash=config.preprocessing_hash,
        channel_positions=recording.info.channel_positions,
        training_files=training_files,
    )

    payload: dict[str, object] = {
        "spec": spec.to_dict(),
        "decision": asdict(spec.decision),
        "weights_file": weights_path.name,
        "training": {
            "subject": subject,
            "train_file": train_file,
            "created_at_utc": utc_now_iso(),
            "dataset": stats.as_dict(),
            "model": fit_info,
        },
    }
    model_json = out_dir / f"{identifier}.model.json"
    model_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return spec, model_json


def _parse_subjects(split: Mapping[str, object], raw: str | None) -> list[str]:
    if raw:
        subjects = [item.strip() for item in raw.split(",") if item.strip()]
        if not subjects:
            raise ContractError("--subjects 不能为空")
        return subjects
    protocol = split.get("protocol")
    if isinstance(protocol, dict):
        development = protocol.get("development_subjects")
        if isinstance(development, list) and development:
            return [str(item) for item in development]
    subjects = split.get("subjects")
    if isinstance(subjects, dict):
        return [str(item) for item in subjects]
    raise ContractError("划分文件既没有 protocol.development_subjects，也没有 subjects")


def _load_recording(data_root: Path, train_file: Path) -> RecordingLike:
    """组合根：只有这里接触应用层的 EDF 读取。

    ``train_subject`` 只吃 ``RecordingLike``；命令行入口负责把“文件”变成“记录”。
    这样单元测试可以直接注入内存 mock，不需要 mne/EDF。
    """
    from eeg_app.sources.input import load_edf

    return load_edf(data_root / train_file, with_sha256=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练 EEGMMIDB 个人三状态 LDA 模型")
    parser.add_argument(
        "--subjects",
        default=None,
        help="逗号分隔的受试者，如 S001,S002；缺省用 splits.json 的 development_subjects",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    split = load_split(args.splits)
    subjects = _parse_subjects(split, args.subjects)
    config = AppConfig()
    failures = 0
    for subject in subjects:
        try:
            train_file = train_file_of(split, subject)
            recording = _load_recording(args.data_root, train_file)
            spec, model_json = train_subject(
                subject,
                recording=recording,
                train_file=str(train_file),
                config=config,
                out_dir=args.out,
            )
        except Exception as exc:  # noqa: BLE001 - CLI 逐人汇报，不因一个人中断全批
            failures += 1
            print(f"[FAIL] {subject}: {exc}", file=sys.stderr)
            continue
        print(f"[OK] {subject}: {spec.model_id} -> {model_json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
