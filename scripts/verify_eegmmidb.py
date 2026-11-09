r"""Verify local EEGMMIDB EDF files against the repository manifest.

Default checks: file exists, byte size, SHA-256.  With ``--mne`` it also reads
each EDF header and checks 64 channels / 160 Hz and the T0/T1/T2 annotation
counts.  The script uses only the standard library unless ``--mne`` is given.

Examples
--------
    .\.venv\Scripts\python.exe scripts\verify_eegmmidb.py
    .\.venv\Scripts\python.exe scripts\verify_eegmmidb.py --mne
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "eegmmidb"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sums_file(manifest_path: Path, by_path: dict[str, dict]) -> list[str]:
    sums_path = manifest_path.with_name("SHA256SUMS.txt")
    if not sums_path.exists():
        return [f"缺少校验文件：{sums_path}"]
    problems: list[str] = []
    seen: set[str] = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            problems.append(f"SHA256SUMS.txt 行格式错误：{line!r}")
            continue
        digest, rel = parts[0], parts[1].lstrip("*")
        seen.add(rel)
        rec = by_path.get(rel)
        if rec is None:
            problems.append(f"SHA256SUMS.txt 多出清单外的文件：{rel}")
        elif rec["sha256"] != digest:
            problems.append(f"SHA256SUMS.txt 与清单不一致：{rel}")
    missing = set(by_path) - seen
    if missing:
        problems.append("SHA256SUMS.txt 缺少：" + "、".join(sorted(missing)))
    return problems


def check_mne(path: Path) -> list[str]:
    try:
        import mne
    except Exception as exc:  # noqa: BLE001 - optional dependency, any failure is reported
        return [f"{path.name}: 无法导入 mne（{exc}）"]

    problems: list[str] = []
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001 - MNE raises several exception types
        return [f"{path.name}: mne 读取失败（{exc}）"]

    if raw.info["nchan"] != 64:
        problems.append(f"{path.name}: 通道数 {raw.info['nchan']} != 64")
    if abs(float(raw.info["sfreq"]) - 160.0) > 1e-6:
        problems.append(f"{path.name}: 采样率 {raw.info['sfreq']} != 160")
    counts = {label: 0 for label in ("T0", "T1", "T2")}
    for description in raw.annotations.description:
        if description in counts:
            counts[description] += 1
    if counts["T0"] != 15:
        problems.append(f"{path.name}: T0 数量 {counts['T0']} != 15")
    if counts["T1"] not in (7, 8):
        problems.append(f"{path.name}: T1 数量 {counts['T1']} 不在 7–8")
    if counts["T2"] not in (7, 8):
        problems.append(f"{path.name}: T2 数量 {counts['T2']} 不在 7–8")
    return problems


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--mne", action="store_true", help="额外读取 EDF 头并检查标注")
    parser.add_argument("--quiet", action="store_true", help="只输出汇总和错误")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    manifest_path = args.manifest or (data_dir / "download_manifest.json")
    if not manifest_path.exists():
        print(f"找不到清单：{manifest_path}", file=sys.stderr)
        return 2

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    files = manifest.get("files", [])
    if not files:
        print("清单里没有文件条目", file=sys.stderr)
        return 2

    problems: list[str] = []
    total_bytes = 0
    by_path: dict[str, dict] = {}
    for record in files:
        rel = record["path"]
        by_path[rel] = record
        path = data_dir / rel
        if not path.exists():
            problems.append(f"{rel}: 文件不存在")
            continue
        size = path.stat().st_size
        total_bytes += size
        if size != int(record["size"]):
            problems.append(f"{rel}: 大小 {size} != {record['size']}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            problems.append(f"{rel}: SHA-256 不符")
        if args.mne:
            problems.extend(check_mne(path))

    problems.extend(check_sums_file(manifest_path, by_path))

    print(f"清单：{manifest_path}")
    print(f"文件：{len(files)} 个，本地总字节：{total_bytes:,}（清单 {int(manifest.get('total_bytes', 0)):,}）")
    print(f"结果：{'通过' if not problems else '未通过'}，问题 {len(problems)} 项")
    if problems:
        for problem in problems:
            print(f"  - {problem}")
        return 1
    if not args.quiet:
        print("所有 EDF 均存在、大小与 SHA-256 匹配" + ("，EDF 头/标注检查通过" if args.mne else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
