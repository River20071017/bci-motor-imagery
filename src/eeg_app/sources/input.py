"""统一数据源层：文件 → 内存里的样本、档案卡、参考标注。

本模块只管「文件 → 数组」，不管节奏；1 倍速按块供数在 :mod:`eeg_app.sources.replay`。
数据源层只做四件事，每件都可以单独替换/记录：

1. **读格式**：按扩展名分派到 ``mne.io.read_raw_*``（EDF/BDF/GDF/EEGLAB/BrainVision/FIF），
   不为每种设备写一个适配器；
2. **改通道名**：一张别名表（``aliases``）+ 通用规范化，不写设备分支；
3. **可选重采样**：只有显式传 ``resample_to`` 才做，并把原采样率/目标采样率写进
   ``SourceInfo.device_metadata``——用于把 250 Hz 的 BCIC IV 对齐到算法目标 160 Hz；
4. **单位**：MNE 读出即伏，本层不再换算。

三条纪律：

* **单位**：读出后立刻是伏（``EEG_UNIT``），下游不再换算；界面显示时自己乘 ``1e6``。
* **标签隔离**：EEG 样本进 :attr:`Recording.samples`，原始标注单独进
  :attr:`Recording.markers`。前者喂在线路径，后者只给参考轨与评估用（PRD §5.3）。
* **封存集**：run 12 是最终测试集，加载层默认拒绝；只有显式 ``allow_sealed=True``
  （命令行 ``--allow-sealed``）才能读，避免调参时不小心看到封存数据。

通道名默认按 MNE ``mne.datasets.eegbci.standardize`` 的同一规则规范成 10-20 写法
（``Fc5.`` → ``FC5``、``Fcz.`` → ``FCz``），保证 ``FC3/FCz/FC4/C3/Cz/C4/CP3/CPz/CP4``
这九个模型通道能被 ``SourceInfo.check_compatible`` 认出来；其它设备通过 ``aliases`` 表补映射。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from eeg_core.contracts import ContractError, ReferenceMarker, SourceInfo

__all__ = [
    "SEALED_RUNS",
    "SOURCE_ID",
    "Recording",
    "apply_channel_aliases",
    "load_edf",
    "load_file",
    "load_recording",
    "read_raw",
    "recording_from_raw",
    "recording_id_of",
    "recording_path",
    "standardize_channel_names",
]

SOURCE_ID = "physionet-eegmmidb"
"""``EEGChunk.source_id`` 的取值，对应 README §7.1 的数据集。"""

SEALED_RUNS = frozenset({"12"})
"""封存测试 run：默认拒绝加载，必须显式 ``allow_sealed=True``。见 data/eegmmidb/README.md §9。"""

_RECORDING_ID = re.compile(r"^(?P<subject>S\d{3})R(?P<run>\d{2})$")
"""PhysioNet 的文件名约定：``S001R04`` → 受试者 ``S001``、run ``4``。"""


def _ensure_not_sealed(record_id: str, allow_sealed: bool) -> None:
    """``record_id`` 属于封存 run 时拒绝；``allow_sealed`` 只给一次性最终评估用。"""
    match = _RECORDING_ID.match(record_id)
    if allow_sealed or match is None or match.group("run") not in SEALED_RUNS:
        return
    raise ContractError(
        f"{record_id} 是封存测试集（run 12）：加载它就等于看到最终测试数据，不能再用来调参。"
        "确认已经冻结、并且只跑这一次时，显式传 allow_sealed=True（命令行 --allow-sealed）。"
    )


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


_READERS: dict[str, str] = {
    ".edf": "read_raw_edf",
    ".bdf": "read_raw_bdf",
    ".gdf": "read_raw_gdf",
    ".set": "read_raw_eeglab",
    ".fdt": "read_raw_eeglab",
    ".vhdr": "read_raw_brainvision",
    ".fif": "read_raw_fif",
    ".cnt": "read_raw_cnt",
}
"""扩展名 → ``mne.io`` 读取函数名；数据格式靠 MNE，不靠自写解析。"""


def read_raw(path: str | Path, *, preload: bool = True) -> object:
    """按扩展名用 MNE 读取任意受支持格式，返回 ``mne.io.BaseRaw``。

    这是数据源层的**唯一读入口**：换数据集只加一行扩展名映射，不写新适配器。
    ``mne`` 只在真正读文件时才导入，``sources`` 的其余部分（分块、节奏、测试）不依赖它。
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        supported = "、".join(sorted(_READERS))
        raise ContractError(f"不支持的文件格式 {suffix or '（无扩展名）'}；支持：{supported}")
    if not file_path.is_file():
        raise ContractError(f"找不到记录文件：{file_path}")

    try:
        import mne
    except ImportError as exc:
        raise ContractError("读取记录需要 mne：请先安装 requirements.txt 中的依赖") from exc
    except Exception as exc:  # noqa: BLE001 - 配置目录不可写等初始化失败
        raise ContractError(f"无法初始化 mne（依赖缺失或配置目录不可写）：{exc}") from exc

    try:
        return getattr(mne.io, reader)(file_path, preload=preload, verbose="ERROR")
    except Exception as exc:
        raise ContractError(f"读取失败：{file_path.name}：{exc}") from exc


def apply_channel_aliases(names: object, aliases: dict[str, str] | None = None) -> tuple[str, ...]:
    """先按别名表改名，再走通用规范化；别名表只做映射，不做单位/重采样。"""
    if isinstance(names, (str, bytes)):
        raise ContractError("通道名必须是序列，不能是单个字符串")
    raw_names = tuple(str(name) for name in names)  # type: ignore[arg-type]
    if aliases:
        raw_names = tuple(str(aliases.get(name, name)) for name in raw_names)
    return standardize_channel_names(raw_names)


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


def recording_from_raw(
    raw: object,
    *,
    source_id: str,
    recording_id: str,
    subject_id: str | None = None,
    run: str | None = None,
    aliases: dict[str, str] | None = None,
    resample_to: float | None = None,
    path: str | Path | None = None,
) -> Recording:
    """把 MNE ``Raw`` 变成契约 :class:`Recording`：别名 → 可选重采样 → 样本/标注。

    重采样是**显式步骤**：只有 ``resample_to`` 与原始采样率不同才执行，并把
    ``original_sample_rate_hz`` / ``resampled_to_hz`` 写进 ``SourceInfo.device_metadata``，
    让 manifest 能追到“这条记录被重采样过”。单位由 MNE 保证为伏。
    """
    original_rate = float(raw.info["sfreq"])  # type: ignore[attr-defined]
    if resample_to is not None:
        target = float(resample_to)
        if not np.isfinite(target) or target <= 0:
            raise ContractError(f"resample_to 必须是有限正数，收到 {resample_to!r}")
        if abs(original_rate - target) > 1e-9:
            raw = raw.copy().resample(target, verbose="ERROR")  # type: ignore[attr-defined]

    channel_names = apply_channel_aliases(raw.ch_names, aliases)
    if tuple(raw.ch_names) != channel_names:  # type: ignore[attr-defined]
        raw.rename_channels(dict(zip(raw.ch_names, channel_names, strict=True)))
    sample_rate_hz = float(raw.info["sfreq"])  # type: ignore[attr-defined]
    samples = np.asarray(raw.get_data(), dtype=np.float64)

    metadata = {"reader": "mne", "original_sample_rate_hz": f"{original_rate:g}"}
    if abs(sample_rate_hz - original_rate) > 1e-9:
        metadata["resampled_to_hz"] = f"{sample_rate_hz:g}"

    info = SourceInfo(
        source_id=source_id,
        recording_id=recording_id,
        sample_rate_hz=sample_rate_hz,
        channel_names=channel_names,
        reference=None,
        subject_id=subject_id,
        run=run,
        n_samples=int(samples.shape[1]),
        device_metadata=metadata,
        path=str(path) if path is not None else None,
    )
    return Recording(
        info=info,
        samples=samples,
        markers=_read_markers(raw, record_id=recording_id, sample_rate_hz=sample_rate_hz),
        path=Path(path) if path is not None else None,
    )


def load_file(
    path: str | Path,
    *,
    source_id: str = SOURCE_ID,
    recording_id: str | None = None,
    subject_id: str | None = None,
    run: str | None = None,
    aliases: dict[str, str] | None = None,
    resample_to: float | None = None,
    with_sha256: bool = True,
    allow_sealed: bool = False,
) -> Recording:
    """读任意 MNE 支持格式（EDF/BDF/GDF/EEGLAB/BrainVision/FIF）。

    格式由扩展名分派；``aliases`` 补设备命名；``resample_to`` 把数据对齐到算法目标采样率。
    封存集判断仍在读文件之前：记录号命中 ``SEALED_RUNS`` 时默认拒绝。
    """
    file_path = Path(path)
    record_id = recording_id or file_path.stem
    _ensure_not_sealed(record_id, allow_sealed)
    raw = read_raw(file_path)
    recording = recording_from_raw(
        raw,
        source_id=source_id,
        recording_id=record_id,
        subject_id=subject_id,
        run=run,
        aliases=aliases,
        resample_to=resample_to,
        path=file_path,
    )
    if with_sha256:
        recording = replace(recording, sha256=_sha256_of(file_path))
    return recording


def load_edf(
    path: str | Path,
    *,
    source_id: str = SOURCE_ID,
    with_sha256: bool = True,
    allow_sealed: bool = False,
    aliases: dict[str, str] | None = None,
    resample_to: float | None = None,
) -> Recording:
    """EDF 便捷入口（等价于 ``load_file``，并沿用 ``S###R##`` 的封存规则）。

    返回 :class:`Recording`；失败时抛 :class:`ContractError`。``mne`` 只在真正读文件时导入。
    文件头里没写参考方式，``SourceInfo.reference`` 就留 ``None``（显示「未提供」），不猜。

    ``run 12`` 属于封存测试集：默认在**读文件之前**就拒绝，只有 ``allow_sealed=True``
    （命令行 ``--allow-sealed``）才放行，用来做一次性最终评估。
    """
    return load_file(
        path,
        source_id=source_id,
        with_sha256=with_sha256,
        allow_sealed=allow_sealed,
        aliases=aliases,
        resample_to=resample_to,
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
    allow_sealed: bool = False,
) -> Recording:
    """``recording_path`` + ``load_edf`` 的合体，演示与集成测试用。

    与 :func:`load_edf` 一样，``run 12`` 默认被拒；``allow_sealed=True`` 才放行。
    """
    _ensure_not_sealed(record_id.strip(), allow_sealed)
    return load_edf(
        recording_path(data_root, record_id),
        source_id=source_id,
        with_sha256=with_sha256,
        allow_sealed=allow_sealed,
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
