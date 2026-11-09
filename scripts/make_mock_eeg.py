"""生成一份符合契约的 mock ``EEGChunk`` 流。

示例：

.. code-block:: powershell

    .\\.venv\\Scripts\\python.exe scripts\\make_mock_eeg.py `
        --out data\\mock\\S999R04.mock.npz --duration-sec 24

输出是一个 ``.npz``：固定块长的 ``samples`` + ``seq`` / ``start_sample`` /
``released_monotonic_ms`` / ``discontinuity``，外加 ``ReferenceMarker``。
不是 EDF；训练需要整条记录时用 ``load_mock_stream(path).as_recording()``。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from training.mock import create_mock_chunk_file, load_mock_stream  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 mock EEGChunk 流（.npz）")
    parser.add_argument("--out", type=Path, default=Path("data/mock/S999R04.mock.npz"))
    parser.add_argument("--subject", default="S999")
    parser.add_argument("--recording-id", default=None)
    parser.add_argument("--duration-sec", type=float, default=24.0)
    parser.add_argument("--trial-sec", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=float, default=160.0)
    parser.add_argument("--chunk-ms", type=float, default=50.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--noise-uv", type=float, default=8.0)
    parser.add_argument("--class-effect", type=float, default=0.35)
    parser.add_argument(
        "--artifact",
        choices=("none", "spike", "flatline", "nan"),
        default="none",
        help="注入一种伪迹，用来测质量门",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在文件")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.out.exists() and not args.overwrite:
        print(f"[FAIL] 文件已存在：{args.out}；加 --overwrite 覆盖", file=sys.stderr)
        return 1
    recording_id = args.recording_id or f"{args.subject}R04"
    stream = create_mock_chunk_file(
        args.out,
        duration_sec=args.duration_sec,
        trial_sec=args.trial_sec,
        sample_rate_hz=args.sample_rate,
        chunk_ms=args.chunk_ms,
        channels=args.channels,
        subject=args.subject,
        recording_id=recording_id,
        seed=args.seed,
        noise_uv=args.noise_uv,
        class_effect=args.class_effect,
        artifact=args.artifact,
    )
    # 重新读一遍，确认落盘 file 能被 load_mock_stream 消费。
    loaded = load_mock_stream(stream.path)
    label_counts = Counter(marker.label for marker in loaded.markers)
    print(f"[OK] {loaded.path}")
    print(
        f"  {len(loaded.info.channel_names)} ch × {loaded.info.n_samples} samples"
        f" · {loaded.n_chunks} chunks × {loaded.chunk_samples} samples"
        f" · {loaded.info.sample_rate_hz:g} Hz"
    )
    print(
        "  markers: "
        + " / ".join(f"{name}={label_counts[name]}" for name in ("T0", "T1", "T2"))
        + f" · artifact={args.artifact}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
