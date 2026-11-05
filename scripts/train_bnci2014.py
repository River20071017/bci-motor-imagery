"""在 BCI Competition IV 2a 上按协议训练与评估。

协议（见 ``training/splits_bnci2014.json`` 与 ``docs/05``）：

* session **T**：6 折按 run 分组的交叉验证（开发/选参）；
* 超参定完后用 **T 全部 6 个 run** 重训最终模型；
* session **E**：封存测试，只有显式 ``--allow-sealed`` 才加载，且只应跑一次。

示例::

    .\\.venv\\Scripts\\python.exe scripts\\train_bnci2014.py              # 全部 9 人，CV + 重训
    .\\.venv\\Scripts\\python.exe scripts\\train_bnci2014.py --subjects A01
    .\\.venv\\Scripts\\python.exe scripts\\train_bnci2014.py --subjects A01 --allow-sealed
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eeg_app.sources.bnci2014 import load_bnci2014_session  # noqa: E402
from eeg_core.contracts import DEFAULT_CONFIG, AppConfig  # noqa: E402
from training.cv import evaluate_recording, run_grouped_cv  # noqa: E402
from training.pipeline import load_split  # noqa: E402
from training.train import train_subject  # noqa: E402

DEFAULT_DATA_ROOT = ROOT / "data" / "bnci2014"
DEFAULT_SPLITS = ROOT / "training" / "splits_bnci2014.json"
DEFAULT_OUT = ROOT / "training" / "models" / "bnci2014"


def _subject_number(name: str) -> int:
    text = name.strip().upper().lstrip("A")
    if not text.isdigit():
        raise SystemExit(f"受试者应形如 A01，收到 {name!r}")
    return int(text)


def _config_2a() -> AppConfig:
    """2a 的算法配置：原生 250 Hz，其余与默认一致。"""
    return AppConfig(source=replace(DEFAULT_CONFIG.source, sample_rate_hz=250.0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2a 训练/验证/测试（T 内 run CV + E 封存）")
    parser.add_argument("--subjects", default=None, help="逗号分隔，如 A01,A02；缺省为全部")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--allow-sealed", action="store_true", help="加载 session E（封存测试）")
    parser.add_argument("--json", type=Path, default=None, help="可选：把结果写成 JSON")
    args = parser.parse_args(argv)

    split = load_split(args.splits)
    subjects = (
        [name.strip().upper() for name in args.subjects.split(",") if name.strip()]
        if args.subjects
        else list(split["subjects"])
    )
    config = _config_2a()

    results: dict[str, object] = {}
    failures = 0
    for subject in subjects:
        entry = split["subjects"].get(subject)
        if not isinstance(entry, dict):
            print(f"[FAIL] {subject} 不在划分文件里", file=sys.stderr)
            failures += 1
            continue
        number = _subject_number(subject)
        train_path = args.data_root / str(entry["train"])
        try:
            recording = load_bnci2014_session(train_path, subject=number, session="T")
            cv = run_grouped_cv(recording, config, n_splits=args.folds)
            print(
                f"[CV ] {subject}: {cv['n_windows']} 窗 / {cv['n_runs']} run · "
                f"宏 F1 {cv['mean']['macro_f1']:.3f}±{cv['std']['macro_f1']:.3f} · "
                f"平衡召回 {cv['mean']['left_right_balanced_recall']:.3f} · "
                f"准确率 {cv['mean']['accuracy']:.3f}"
            )
            spec, model_json = train_subject(
                subject,
                recording=recording,
                train_file=str(entry["train"]),
                config=config,
                out_dir=args.out,
            )
            print(f"[OK ] {subject}: {spec.model_id} -> {model_json}")
            item: dict[str, object] = {"cv": cv, "model_id": spec.model_id}

            if args.allow_sealed:
                test_path = args.data_root / str(entry["test"])
                test_recording = load_bnci2014_session(
                    test_path, subject=number, session="E", allow_sealed=True
                )
                test = evaluate_recording(model_json, test_recording, config)
                item["test"] = test
                print(
                    f"[TEST] {subject}: 宏 F1 {test['metrics']['macro_f1']:.3f} · "
                    f"平衡召回 {test['metrics']['left_right_balanced_recall']:.3f} · "
                    f"准确率 {test['metrics']['accuracy']:.3f}"
                )
            results[subject] = item
        except Exception as exc:  # noqa: BLE001 - 逐人汇报，不中断全批
            failures += 1
            print(f"[FAIL] {subject}: {exc}", file=sys.stderr)

    if args.json is not None:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
