r"""Download PhysioNet EEGMMIDB EDF files and keep a verifiable manifest.

Only the EDF files are downloaded.  The T0/T1/T2 labels are read from the EDF+
embedded annotation channel, so the companion ``.edf.event`` files are not needed.

Examples
--------
    .\.venv\Scripts\python.exe scripts\download_eegmmidb.py --subjects 1-10 --runs 4,8,12
    .\.venv\Scripts\python.exe scripts\download_eegmmidb.py --subjects 1-109 --runs 4,8,12

The script uses only the Python standard library.  It tries the public AWS mirror
first, then physionet.org.  Existing files with a matching SHA-256 are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "eegmmidb"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "download_manifest.json"

MIRRORS = (
    "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/",
    "https://physionet.org/files/eegmmidb/1.0.0/",
)
USER_AGENT = "bci-motor-imagery/1.0 (course project; data download)"


def parse_int_set(spec: str, *, low: int, high: int, label: str) -> list[int]:
    """Parse ``1-10`` or ``1,2,5`` into a sorted list of ints."""
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
        else:
            start = end = int(part)
        for value in range(start, end + 1):
            if not low <= value <= high:
                raise ValueError(f"{label} 超出范围 {low}-{high}: {value}")
            values.add(value)
    if not values:
        raise ValueError(f"{label} 不能为空")
    return sorted(values)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {
            "dataset": "PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb)",
            "version": "1.0.0",
            "doi": "10.13026/C28G6P",
            "license": "ODC-By 1.0",
            "source_url": "https://physionet.org/content/eegmmidb/1.0.0/",
            "mirrors": list(MIRRORS),
            "subjects": [],
            "runs": [4, 8, 12],
            "channel_count": 64,
            "sample_rate_hz": 160.0,
            "annotation_source": "EDF+ embedded annotation channel (T0/T1/T2)",
            "files": [],
        }
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def download_one(url: str, dest: Path, *, timeout: float, retries: int) -> None:
    """Download ``url`` to ``dest`` via a .part file, with a few retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                expected = response.headers.get("Content-Length")
                with part.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                if expected is not None and part.stat().st_size != int(expected):
                    raise OSError(
                        f"大小不一致：服务器 {expected}，实际 {part.stat().st_size}"
                    )
            os.replace(part, dest)
            return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"下载失败：{url}（{last_error}）")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        default="1-10",
        help="受试者编号，如 1-10 或 1,3,5（默认 1-10）",
    )
    parser.add_argument(
        "--runs",
        default="4,8,12",
        help="run 编号，逗号分隔（默认 4,8,12；本项目只用这三个）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"EDF 输出目录（默认 {DEFAULT_DATA_DIR}）",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="清单路径（默认 <out>/download_manifest.json）",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="每个镜像的重试次数")
    parser.add_argument("--force", action="store_true", help="即使 SHA-256 匹配也重新下载")
    args = parser.parse_args(argv)

    try:
        subjects = parse_int_set(args.subjects, low=1, high=109, label="subjects")
        runs = parse_int_set(args.runs, low=1, high=14, label="runs")
    except ValueError as exc:
        parser.error(str(exc))

    out_dir: Path = args.out
    manifest_path = args.manifest or (out_dir / "download_manifest.json")
    manifest = load_manifest(manifest_path)
    known = {entry["path"]: entry for entry in manifest.get("files", [])}

    failures: list[str] = []
    downloaded = 0
    skipped = 0

    for subject in subjects:
        sid = f"S{subject:03d}"
        for run in runs:
            rel = f"{sid}/{sid}R{run:02d}.edf"
            dest = out_dir / rel
            record = known.get(rel)
            expected_size = int(record["size"]) if record else None
            expected_sha = record.get("sha256") if record else None

            if dest.exists() and not args.force:
                actual_size = dest.stat().st_size
                if expected_size is not None and actual_size != expected_size:
                    print(f"[redo] {rel}: 大小不符，重新下载")
                elif expected_sha:
                    if sha256_file(dest) != expected_sha:
                        print(f"[redo] {rel}: SHA-256 不符，重新下载")
                    else:
                        print(f"[skip] {rel}")
                        skipped += 1
                        continue
                else:
                    # No reference hash yet (first run without a manifest):
                    # record what is already on disk instead of re-downloading.
                    known[rel] = {
                        "path": rel,
                        "subject": sid,
                        "run": run,
                        "size": actual_size,
                        "sha256": sha256_file(dest),
                        "url": f"https://physionet.org/files/eegmmidb/1.0.0/{rel}",
                    }
                    print(f"[skip] {rel}（已登记）")
                    skipped += 1
                    continue

            last_error: Exception | None = None
            for base in MIRRORS:
                url = base + rel
                try:
                    print(f"[get ] {rel} <- {base}")
                    download_one(url, dest, timeout=args.timeout, retries=args.retries)
                    actual_size = dest.stat().st_size
                    actual_sha = sha256_file(dest)
                    if expected_size is not None and actual_size != expected_size:
                        raise RuntimeError(
                            f"大小不符：期望 {expected_size}，实际 {actual_size}"
                        )
                    if expected_sha and actual_sha != expected_sha:
                        raise RuntimeError(
                            f"SHA-256 不符：期望 {expected_sha}，实际 {actual_sha}"
                        )
                    record = {
                        "path": rel,
                        "subject": sid,
                        "run": run,
                        "size": actual_size,
                        "sha256": actual_sha,
                        "url": f"https://physionet.org/files/eegmmidb/1.0.0/{rel}",
                    }
                    known[rel] = record
                    downloaded += 1
                    last_error = None
                    break
                except RuntimeError as exc:
                    last_error = exc
                    print(f"[fail] {rel}: {exc}")
                    dest.unlink(missing_ok=True)

            if last_error is not None:
                failures.append(f"{rel}: {last_error}")

    files = sorted(known.values(), key=lambda item: (item["subject"], item["run"]))
    manifest["files"] = files
    manifest["file_count"] = len(files)
    manifest["total_bytes"] = sum(int(item["size"]) for item in files)
    manifest["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["subjects"] = sorted(
        set(manifest.get("subjects", [])) | {int(item["subject"][1:]) for item in files}
    )
    manifest["runs"] = sorted(set(manifest.get("runs", [])) | {int(item["run"]) for item in files})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sums_path = manifest_path.with_name("SHA256SUMS.txt")
    sums_path.write_text(
        "\n".join(f"{item['sha256']}  {item['path']}" for item in files) + "\n",
        encoding="utf-8",
    )

    print(
        f"\n完成：下载 {downloaded}，跳过 {skipped}，失败 {len(failures)}；"
        f"清单 {manifest_path}（{len(files)} 个文件），校验 {sums_path}"
    )
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
