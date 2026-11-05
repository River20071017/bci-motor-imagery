"""下载 BCI Competition IV 2a（BNCI 2014-001）的 .mat session 文件。

每个 session 一个 ``.mat``（6 run × 48 trial），文件名 ``A01T.mat`` / ``A01E.mat``。
脚本只做下载、校验与清单，不解析内容；解析在 ``eeg_app.sources.bnci2014``。

镜像（按顺序尝试）：

1. ``https://lampx.tugraz.at/~bci/database/001-2014/``（MOABB 当前使用的 TU Graz 镜像）
2. ``https://bnci-horizon-2020.eu/database/data-sets/001-2014/``（BNCI Horizon 原站，可能不可用）

示例::

    .\\.venv\\Scripts\\python.exe scripts\\download_bnci2014.py --subjects 1-9 --sessions T,E
    .\\.venv\\Scripts\\python.exe scripts\\download_bnci2014.py --subjects 1 --insecure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "bnci2014"
MIRRORS = (
    "https://lampx.tugraz.at/~bci/database/001-2014/",
    "https://bnci-horizon-2020.eu/database/data-sets/001-2014/",
)


def _parse_int_set(text: str, *, low: int, high: int, label: str) -> list[int]:
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, stop = part.partition("-")
            if not start.isdigit() or not stop.isdigit():
                raise SystemExit(f"{label} 区间格式应为 a-b，收到 {part!r}")
            values.update(range(int(start), int(stop) + 1))
        elif part.isdigit():
            values.add(int(part))
        else:
            raise SystemExit(f"{label} 只接受数字或 a-b，收到 {part!r}")
    if not values or min(values) < low or max(values) > high:
        raise SystemExit(f"{label} 必须在 {low}..{high} 之间")
    return sorted(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, target: Path, *, timeout: float, insecure: bool) -> None:
    context = None
    if insecure:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    request = urllib.request.Request(url, headers={"User-Agent": "bci-motor-imagery/0.1"})
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        total = int(response.headers.get("Content-Length") or 0)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        written = 0
        with temporary.open("wb") as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                written += len(block)
                if total:
                    print(f"\r  {target.name}: {written / total:5.1%}", end="", flush=True)
        print()
        if total and written != total:
            temporary.unlink(missing_ok=True)
            raise OSError(f"下载不完整：{written}/{total} 字节")
        temporary.replace(target)


def _validate(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "文件不存在"
    if path.stat().st_size < 1_000_000:
        return False, f"文件过小：{path.stat().st_size} 字节"
    try:
        import scipy.io
    except ImportError:  # pragma: no cover
        return True, "未安装 scipy，跳过结构校验"
    try:
        mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception as exc:  # noqa: BLE001 - 损坏文件
        return False, f"MAT 读取失败：{exc}"
    if "data" not in mat:
        return False, "缺少 data 结构"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 BNCI 2014-001（BCI IV 2a）session 文件")
    parser.add_argument("--subjects", default="1-9", help="如 1-9 或 1,3,5")
    parser.add_argument("--sessions", default="T,E", help="逗号分隔，只允许 T/E")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2, help="每个镜像的重试次数")
    parser.add_argument(
        "--insecure", action="store_true", help="跳过 TLS 证书校验（镜像证书链不全时）"
    )
    parser.add_argument("--force", action="store_true", help="即使本地文件有效也重新下载")
    args = parser.parse_args(argv)

    subjects = _parse_int_set(args.subjects, low=1, high=9, label="--subjects")
    sessions = [item.strip().upper() for item in args.sessions.split(",") if item.strip()]
    if not sessions or any(item not in ("T", "E") for item in sessions):
        raise SystemExit("--sessions 只接受 T,E")

    manifest_path = args.out / "download_manifest.json"
    manifest: dict[str, object] = {
        "dataset": "bnci2014-001",
        "version": "1.0.0",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": [],
    }
    entries: list[dict[str, object]] = []
    failures = 0

    for subject in subjects:
        for session in sessions:
            name = f"A{subject:02d}{session}.mat"
            target = args.out / name
            if target.is_file() and not args.force:
                ok, reason = _validate(target)
                if ok:
                    print(f"[SKIP] {name}（已存在，{reason}）")
                    entries.append(
                        {"name": name, "size": target.stat().st_size, "sha256": _sha256(target)}
                    )
                    continue
                print(f"[WARN] {name} 校验失败（{reason}），重新下载")

            downloaded = False
            for mirror in MIRRORS:
                url = mirror + name
                for attempt in range(1, args.retries + 1):
                    try:
                        print(f"[GET ] {url}（第 {attempt} 次）")
                        _download(url, target, timeout=args.timeout, insecure=args.insecure)
                        downloaded = True
                        break
                    except (urllib.error.URLError, OSError, ssl.SSLError) as exc:
                        print(f"[FAIL] {type(exc).__name__}: {exc}")
                        time.sleep(1.0)
                if downloaded:
                    break
            if not downloaded:
                failures += 1
                print(f"[SKIP] {name} 所有镜像都失败")
                continue

            ok, reason = _validate(target)
            if not ok:
                failures += 1
                print(f"[BAD ] {name}: {reason}")
                continue
            entries.append({"name": name, "size": target.stat().st_size, "sha256": _sha256(target)})
            print(f"[OK  ] {name}（{target.stat().st_size / 1e6:.1f} MB）")

    manifest["files"] = entries
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sums = args.out / "SHA256SUMS.txt"
    sums.write_text(
        "".join(f"{entry['sha256']}  {entry['name']}\n" for entry in entries),
        encoding="utf-8",
    )
    print(f"\n清单：{manifest_path}（{len(entries)} 个文件）")
    if failures:
        print(f"有文件没下载成功。可手动从 MOABB/BNCI 获取后放到 {args.out}，再重跑本脚本校验。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
