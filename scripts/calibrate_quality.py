"""标定质量阈值：用真实记录回答“这三个数取多少”。

三个阈值各自要回答的问题不同，不要用一个“看起来合理”的数字糊过去：

1. **量化步长**（相邻样本最小非零差）→ ``flatline_tolerance_uv`` 的下限。
   容差小于一步量化时，“未变化”退化成“严格相等”；死通道只要在相邻两个码值之间
   抖动（±1 LSB），就永远不会被判常量段。
2. **幅值分位与阈值扫描** → ``amplitude_limit_uv``。看每块（默认 8 样本）内模型通道
   最大 ``|x|`` 的分位数，以及各阈值下有多少块会被判 ``invalid``。
3. **常量段游程** → ``flatline_sec``。各容差下最长连续“未变化”段有多长；
   生理信号本身不会连续 1 s 不动，能跑到这个长度的只有死通道或文件尾补齐。

``--windows`` 再把阈值真正代进训练管线，量“可用窗”损失：无效块本身很少，但每次
``invalid`` 会清空滤波状态并重新预热 3 s，真实代价要用窗数看，不能只看无效块数。

只读 EDF、不改配置。结论写进 ``docs/05_变更记录.md``；改数值会改
``preprocessing_hash``，必须重训模型。

**只用开发集**：默认 ``--runs 4,8``（训练/验证 run）。``run12`` 是封存测试集，不参与标定；
``--runs all`` 只用于事后复核“换成全部数据结论是否变”，不要在定值前跑。

示例::

    .\\.venv\\Scripts\\python.exe scripts\\calibrate_quality.py
    .\\.venv\\Scripts\\python.exe scripts\\calibrate_quality.py --windows
    .\\.venv\\Scripts\\python.exe scripts\\calibrate_quality.py --runs all   # 复核用
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eeg_app.sources.input import load_edf  # noqa: E402
from eeg_core.contracts import DEFAULT_CONFIG, MODEL_CHANNELS  # noqa: E402

DEFAULT_DATA_ROOT = ROOT / "data" / "eegmmidb"
AMPLITUDE_SWEEP_UV = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0)
TOLERANCE_SWEEP_UV = (0.0, 0.5, 1.0, 2.0)
RUN_SECONDS = (0.5, 1.0, 2.0)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="标定质量阈值（幅值 / 常量段）")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--runs",
        default="4,8",
        help="参与标定的 run（逗号分隔，默认开发集 4,8）；all 表示全部 run",
    )
    parser.add_argument(
        "--windows",
        action="store_true",
        help="额外把阈值代进训练管线，统计可用窗损失（较慢）",
    )
    parser.add_argument(
        "--allow-sealed",
        action="store_true",
        help="允许扫 run12（封存测试集）；只在 --runs all 的事后复核时用",
    )
    return parser.parse_args(argv)


def _parse_runs(text: str) -> set[str]:
    """``"4,8"`` → ``{"04", "08"}``；``all`` → 空集合（表示不过滤）。"""
    value = text.strip().lower()
    if value == "all":
        return set()
    runs: set[str] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            raise SystemExit(f"--runs 只接受 run 编号（如 4,8）或 all，收到 {item!r}")
        runs.add(item.zfill(2))
    if not runs:
        raise SystemExit("--runs 不能为空；要全部 run 请写 all")
    return runs


def _recordings(data_root: Path, runs: set[str]) -> list[Path]:
    paths = sorted(
        path for path in data_root.glob("S*/S*R*.edf") if not runs or path.stem[-2:] in runs
    )
    if not paths:
        label = "全部 run" if not runs else "、".join(sorted(runs))
        raise SystemExit(
            f"在 {data_root} 下找不到 {label} 的 S*/S*R*.edf，先跑 scripts/download_eegmmidb.py"
        )
    return paths


def _grid() -> tuple[int, float]:
    """与运行时一致的分块长度（样本）与采样率。"""
    source = DEFAULT_CONFIG.source
    chunk = max(1, int(round(source.target_chunk_ms / 1000.0 * source.sample_rate_hz)))
    return chunk, source.sample_rate_hz


def _model_samples_uv(path: Path, *, allow_sealed: bool = False) -> np.ndarray:
    """读一条记录，只取模型通道，换算成微伏；形状 (通道, 样本)。"""
    recording = load_edf(path, with_sha256=False, allow_sealed=allow_sealed)
    indices = [recording.info.channel_names.index(name) for name in MODEL_CHANNELS]
    return np.asarray(recording.samples[indices], dtype=np.float64) * 1e6


def _block_maxima(samples_uv: np.ndarray, chunk: int) -> np.ndarray:
    """每块内每通道的最大 |x|；形状 (通道, 块数)。"""
    n_blocks = samples_uv.shape[1] // chunk
    blocks = samples_uv[:, : n_blocks * chunk].reshape(samples_uv.shape[0], n_blocks, chunk)
    return np.abs(blocks).max(axis=2)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """一维布尔序列的 True 游程（长度, 起点）。"""
    padded = np.concatenate(([False], mask, [False]))
    flips = np.flatnonzero(padded[1:] != padded[:-1])
    return flips[1::2] - flips[0::2], flips[0::2]


def scan(
    data_root: Path, chunk: int, runs: set[str], *, allow_sealed: bool = False
) -> dict[str, object]:
    """扫 ``runs`` 指定的记录（默认开发集 4/8），返回幅值/量化/常量段的原始统计。"""
    paths = _recordings(data_root, runs)
    block_peaks: list[np.ndarray] = []
    amplitude_hits = {limit: 0 for limit in AMPLITUDE_SWEEP_UV}
    n_blocks = 0
    min_nonzero_diff: list[float] = []
    runs_by_tolerance: dict[float, list[int]] = {tol: [] for tol in TOLERANCE_SWEEP_UV}

    for path in paths:
        samples = _model_samples_uv(path, allow_sealed=allow_sealed)
        peaks = _block_maxima(samples, chunk)
        n_blocks += peaks.shape[1]
        block_peaks.append(peaks.ravel())
        for limit in AMPLITUDE_SWEEP_UV:
            amplitude_hits[limit] += int((peaks > limit).any(axis=0).sum())

        diffs = np.abs(np.diff(samples, axis=1))
        nonzero = diffs[diffs > 0]
        if nonzero.size:
            min_nonzero_diff.append(float(nonzero.min()))
        for tolerance in TOLERANCE_SWEEP_UV:
            unchanged = diffs <= tolerance
            for row in range(unchanged.shape[0]):
                lengths, _ = _runs(unchanged[row])
                if lengths.size:
                    runs_by_tolerance[tolerance].extend(lengths.tolist())

    return {
        "n_records": len(paths),
        "runs": "all" if not runs else "、".join(f"run{run}" for run in sorted(runs)),
        "n_blocks": n_blocks,
        "peaks": np.concatenate(block_peaks),
        "amplitude_hits": amplitude_hits,
        "min_nonzero_diff": np.asarray(min_nonzero_diff),
        "runs_lengths": runs_by_tolerance,
    }


def window_cost(
    data_root: Path, limits: tuple[float, ...], runs: set[str], *, allow_sealed: bool = False
) -> list[tuple[float, int, int]]:
    """把幅值阈值代进训练管线，返回 (阈值, 无效块, 可用窗)。"""
    from eeg_core.processing import FilterBank
    from training.pipeline import build_feature_dataset

    rows: list[tuple[float, int, int]] = []
    for limit in limits:
        invalid = usable = 0
        for path in _recordings(data_root, runs):
            recording = load_edf(path, with_sha256=False, allow_sealed=allow_sealed)
            bank = FilterBank(
                amplitude_limit_v=limit * 1e-6,
                flatline_sec=DEFAULT_CONFIG.processing.flatline_sec,
                flatline_tolerance_v=DEFAULT_CONFIG.processing.flatline_tolerance_uv * 1e-6,
                config=DEFAULT_CONFIG,
            )
            _, stats = build_feature_dataset(recording, DEFAULT_CONFIG, filter_bank=bank)
            invalid += stats.invalid_chunks
            usable += stats.usable_windows
        rows.append((limit, invalid, usable))
    return rows


def report(stats: dict[str, object], chunk: int, sample_rate: float) -> None:
    peaks = np.asarray(stats["peaks"])
    n_blocks = int(stats["n_blocks"])
    print(
        f"记录 {stats['n_records']} 条（{stats['runs']}）· 模型通道 {len(MODEL_CHANNELS)} 个 · "
        f"每块 {chunk} 样本 · 共 {n_blocks} 块"
    )
    print(f"模型通道：{'、'.join(MODEL_CHANNELS)}")

    steps = np.asarray(stats["min_nonzero_diff"])
    print("\n== 量化步长 ==")
    print(f"相邻样本最小非零差：min {steps.min():.4f} µV / median {np.median(steps):.4f} µV")
    print("→ flatline_tolerance_uv 应 ≥ 一个量化步长，否则 ±1 LSB 抖动的死通道不算“未变化”。")

    print("\n== 幅值：每块内模型通道的最大 |x| ==")
    for q in (50, 90, 99, 99.9, 99.99):
        print(f"  p{q:<6}: {np.percentile(peaks, q):9.2f} µV")
    print(f"  max   : {peaks.max():9.2f} µV")
    print("\n阈值(µV)  判无效块数  占比")
    for limit in AMPLITUDE_SWEEP_UV:
        hits = int(stats["amplitude_hits"][limit])
        print(f"  {limit:6.0f}  {hits:10d}  {hits / n_blocks:7.2%}")

    print("\n== 常量段：连续“相邻差 ≤ 容差”的游程 ==")
    limits_txt = "  ".join(f"≥{sec:g}s" for sec in RUN_SECONDS)
    print(f"容差(µV)  最长游程  {limits_txt}")
    for tolerance in TOLERANCE_SWEEP_UV:
        lengths = np.asarray(stats["runs_lengths"][tolerance])
        if lengths.size == 0:
            print(f"  {tolerance:5.1f}  {'—':>7}  " + "  ".join(f"{0:>6d}" for _ in RUN_SECONDS))
            continue
        counts = "  ".join(
            f"{int((lengths >= sec * sample_rate).sum()):>6d}" for sec in RUN_SECONDS
        )
        print(f"  {tolerance:5.1f}  {int(lengths.max()):7d}  {counts}")
    print("→ 生理信号不会连续不动；能跑满 1 s 的只有死通道或文件尾补齐。")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs = _parse_runs(args.runs)
    if not runs and not args.allow_sealed:
        raise SystemExit(
            "--runs all 会扫到封存 run12；确认只做一次性复核时，同时加 --allow-sealed。"
        )
    chunk, sample_rate = _grid()
    stats = scan(args.data_root, chunk=chunk, runs=runs, allow_sealed=args.allow_sealed)
    report(stats, chunk, sample_rate)
    if args.windows:
        print("\n== 幅值阈值 → 可用窗（阈值代进 training/pipeline.py）==")
        rows = window_cost(args.data_root, AMPLITUDE_SWEEP_UV, runs, allow_sealed=args.allow_sealed)
        baseline_limit = DEFAULT_CONFIG.processing.amplitude_limit_uv
        baseline = next((usable for limit, _, usable in rows if limit == baseline_limit), None)
        print(f"阈值(µV)  无效块  可用窗  相对 {baseline_limit:g} µV 的窗差")
        for limit, invalid, usable in rows:
            delta = f"{(usable - baseline) / baseline:+7.1%}" if baseline else "   —"
            print(f"  {limit:6.0f}  {invalid:6d}  {usable:6d}  {delta:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
