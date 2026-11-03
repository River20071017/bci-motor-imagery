"""把一条 EDF+ 记录读成内存里的数据：样本、档案卡、参考标注。

本模块只管「文件 → 数组」，不管节奏；1 倍速按块供数在 :mod:`eeg_app.sources.replay`。
两条纪律在这里落地：

* **单位**：读出后立刻是伏（``EEG_UNIT``），下游不再换算；界面显示时自己乘 ``1e6``。
* **标签隔离**：EEG 样本进 :attr:`Recording.samples`，原始标注单独进
  :attr:`Recording.markers`。前者喂在线路径，后者只给参考轨与评估用（PRD §5.3）。

通道名按 MNE ``mne.datasets.eegbci.standardize`` 的同一规则规范成 10-20 写法
（``Fc5.`` → ``FC5``、``Fcz.`` → ``FCz``），保证 ``FC3/FCz/FC4/C3/Cz/C4/CP3/CPz/CP4``
这九个模型通道能被 ``SourceInfo.check_compatible`` 认出来。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eeg_core.contracts import ContractError, ReferenceMarker, SourceInfo

__all__ = [
    "SOURCE_ID",
    "Recording",
    "load_edf",
    "load_recording",
    "recording_id_of",
    "recording_path",
    "standardize_channel_names",
]

SOURCE_ID = "physionet-eegmmidb"
"""``EEGChunk.source_id`` 的取值，对应 README §7.1 的数据集。"""

_RECORDING_ID = re.compile(r"^(?P<subject>S\d{3})R(?P<run>\d{2})$")
"""PhysioNet 的文件名约定：``S001R04`` → 受试者 ``S001``、run ``4``。"""


def standardize_channel_names(names: object) -> tuple[str, ...]:
    """按 PhysioNet→10-20 的规则规范化通道名，返回新元组。

    规则与 MNE 的 ``eegbci.standardize`` 一致：去掉首尾空白与结尾的点、转大写、
    结尾 ``Z`` 收回小写 ``z``、``FP`` 开头的写成 ``Fp``。规则只改写法，不改顺序、不增删通道。
    规范化后若出现重名（例如 ``Cz.`` 与 ``CZ.`` 同时存在），直接抛 :class:`ContractError`：
    这种记录该停下来核对，不能猜哪一行才是模型要的通道。
    """
    if isinstance(names, (str, bytes)):
        raise ContractError("通道名必须是序列，不能是单个字符串")
    try:
        items = list(names)  # type: ignore[arg-type]
    except TypeError:
        raise ContractError("通道名必须是字符串序列") from None
    if not items:
        raise ContractError("记录没有任何通道")

    standardized: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ContractError(f"通道名[{index}] 必须是字符串")
        name = item.strip().rstrip(".").strip().upper()
        if not name:
            raise ContractError(f"通道名[{index}] 规范化后为空")
        if name.endswith("Z"):
            name = name[:-1] + "z"
        if name.startswith("FP"):
            name = "Fp" + name[2:]
        standardized.append(name)

    duplicates = sorted({name for name in standardized if standardized.count(name) > 1})
    if duplicates:
        raise ContractError("规范化后通道名重复：" + "、".join(duplicates))
    return tuple(standardized)


def recording_id_of(path: str | Path) -> str:
    """取记录编号（``S001R04.edf`` → ``S001R04``）；文件名不合约定时抛 :class:`ContractError`。"""
    stem = Path(path).stem
    if not _RECORDING_ID.match(stem):
        raise ContractError(f"文件名不符合 S###R## 约定，无法确定记录编号：{Path(path).name}")
    return stem


@dataclass(frozen=True, slots=True)
class Recording:
    """一条读进内存的记录：档案卡 + 样本 + 参考标注。

    ``samples`` 形状 ``(通道数, 样本数)``、单位伏、只读 ``float64``；行序就是
    ``info.channel_names`` 的顺序。``markers`` 与样本完全分开，任何块里都不会出现标签。
    ``path`` / ``sha256`` 是给 manifest 用的溯源信息，不属于契约里的 ``SourceInfo``。
    """

    info: SourceInfo
    samples: np.ndarray
    markers: tuple[ReferenceMarker, ...] = ()
    path: Path | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 2:
            raise ContractError(
                f"samples 必须是二维数组（通道数 × 样本数），收到 {samples.ndim} 维"
            )
        if samples.shape[0] != len(self.info.channel_names):
            raise ContractError(
                f"samples 第 0 维应为通道数 {len(self.info.channel_names)}，收到 {samples.shape[0]}"
            )
        if samples.shape[1] < 1:
            raise ContractError("记录至少要有 1 个样本")
        if not np.issubdtype(samples.dtype, np.floating):
            raise ContractError(f"samples 必须是浮点数组（单位伏），收到 dtype={samples.dtype}")

        stored = np.array(samples, dtype=np.float64, order="C", copy=True)
        stored.flags.writeable = False
        object.__setattr__(self, "samples", stored)

        for marker in self.markers:
            if marker.recording_id != self.info.recording_id:
                raise ContractError(
                    f"标注属于记录 {marker.recording_id}，与档案卡 {self.info.recording_id} 不一致"
                )

        if self.info.n_samples is not None and self.info.n_samples != stored.shape[1]:
            raise ContractError(
                f"档案卡 n_samples={self.info.n_samples} 与实际样本数 {stored.shape[1]} 不一致"
            )

        path = Path(self.path) if self.path is not None else None
        object.__setattr__(self, "path", path)
        sha = self.sha256.strip().lower() if self.sha256 else None
        object.__setattr__(self, "sha256", sha or None)

    @property
    def n_samples(self) -> int:
        """总样本数。"""
        return self.samples.shape[1]

    @property
    def duration_sec(self) -> float:
        """记录总时长（秒），由样本数与采样率推出。"""
        return self.n_samples / self.info.sample_rate_hz

    @property
    def labels(self) -> tuple[str, ...]:
        """出现过的原始标注文字（去重、保序），如 ``("T0", "T1", "T2")``。"""
        seen: dict[str, None] = {}
        for marker in self.markers:
            seen.setdefault(marker.label, None)
        return tuple(seen)


def load_edf(
    path: str | Path,
    *,
    source_id: str = SOURCE_ID,
    with_sha256: bool = True,
) -> Recording:
    """读一条 EDF+ 记录，返回 :class:`Recording`；失败时抛 :class:`ContractError`。

    ``mne`` 只在真正读文件时才导入：``sources`` 的其余部分（分块、节奏、测试）不依赖它。
    文件头里没写参考方式，``SourceInfo.reference`` 就留 ``None``（显示「未提供」），不猜。
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ContractError(f"找不到 EDF 文件：{file_path}")

    try:
        import mne
    except ImportError as exc:
        raise ContractError("读取 EDF 需要 mne：请先安装 requirements.txt 中的依赖") from exc
    except Exception as exc:  # noqa: BLE001 - 配置目录不可写等初始化失败
        raise ContractError(f"无法初始化 mne（依赖缺失或配置目录不可写）：{exc}") from exc

    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    except Exception as exc:
        raise ContractError(f"EDF 读取失败：{file_path.name}：{exc}") from exc

    channel_names = standardize_channel_names(raw.ch_names)
    raw.rename_channels(dict(zip(raw.ch_names, channel_names, strict=True)))

    samples = np.asarray(raw.get_data(), dtype=np.float64)
    sample_rate_hz = float(raw.info["sfreq"])

    record_id = recording_id_of(file_path)
    match = _RECORDING_ID.match(record_id)
    info = SourceInfo(
        source_id=source_id,
        recording_id=record_id,
        sample_rate_hz=sample_rate_hz,
        channel_names=channel_names,
        reference=None,
        subject_id=match.group("subject") if match else None,
        run=str(int(match.group("run"))) if match else None,
        n_samples=int(samples.shape[1]),
    )

    return Recording(
        info=info,
        samples=samples,
        markers=_read_markers(raw, record_id=record_id, sample_rate_hz=sample_rate_hz),
        path=file_path,
        sha256=_sha256_of(file_path) if with_sha256 else None,
    )


def recording_path(data_root: str | Path, record_id: str) -> Path:
    """按 ``S001R04`` 找到 ``<data_root>/S001/S001R04.edf``；不存在时抛 :class:`ContractError`。"""
    match = _RECORDING_ID.match(record_id.strip())
    if not match:
        raise ContractError(f"记录编号应形如 S001R04，收到 {record_id!r}")
    path = Path(data_root) / match.group("subject") / f"{record_id.strip()}.edf"
    if not path.is_file():
        raise ContractError(f"找不到记录文件：{path}")
    return path


def load_recording(
    data_root: str | Path,
    record_id: str,
    *,
    source_id: str = SOURCE_ID,
    with_sha256: bool = True,
) -> Recording:
    """``recording_path`` + ``load_edf`` 的合体，演示与集成测试用。"""
    return load_edf(
        recording_path(data_root, record_id), source_id=source_id, with_sha256=with_sha256
    )


def _read_markers(
    raw: object, *, record_id: str, sample_rate_hz: float
) -> tuple[ReferenceMarker, ...]:
    """把 EDF 内嵌标注换成 :class:`ReferenceMarker`，时间一律用样本编号。

    ``label`` 原样保留（``T0``/``T1``/``T2``），不在这里映射成 idle/左右手——那是评估口径。
    秒→样本号统一用四舍五入（``round``），与 docs/01 §3.2 一致；不要混用 floor。
    """
    annotations = getattr(raw, "annotations", None)
    if annotations is None:
        return ()

    markers: list[ReferenceMarker] = []
    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description, strict=True
    ):
        label = str(description).strip()
        if not label:
            continue
        duration_sec = float(duration)
        markers.append(
            ReferenceMarker(
                recording_id=record_id,
                onset_sample=max(0, int(round(float(onset) * sample_rate_hz))),
                label=label,
                duration_samples=(
                    int(round(duration_sec * sample_rate_hz)) if duration_sec > 0 else None
                ),
            )
        )
    return tuple(markers)


def _sha256_of(path: Path) -> str:
    """整文件 SHA-256，写进 manifest 用来核对「回放的是哪一份文件」。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
