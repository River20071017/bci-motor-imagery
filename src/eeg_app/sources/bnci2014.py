"""BCI Competition IV 2a（BNCI 2014-001）数据源。

2a 的一个 ``.mat`` 文件 = 一个 session（6 个 run × 48 trial，4 类）。本项目只用
**左/右两类**的想象段 + **注视期**作 idle：

* ``T0``：两次想象之间的空档（上一段想象结束 → 本次提示出现），含 run 间休息与 2 s 注视；
* ``T1``：class 1（左手）想象段；
* ``T2``：class 2（右手）想象段；
* ``T3`` / ``T4``：class 3/4（双脚/舌）的想象段——**保留标记但不映射**，
  ``label_of_window`` 会返回 ``None``，这些窗自动排除。

想象段从提示后 1 s 开始（跳过提示呈现与视觉加工），到提示后 4 s 结束（原始协议 t=2–6 s）。
``T0`` 的区间是“上一段想象结束 → 本次提示”，连续覆盖休息与注视，保证 2 s 窗有足够长度。

**封存**：session ``E`` 是最终测试集，默认拒绝加载，只有 ``allow_sealed=True`` 才放行；
与 EEGMMIDB 的 run12 同一语义。session ``T`` 内部按 run 做交叉验证（见 ``training/cv.py``）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from eeg_core.contracts import ContractError, ReferenceMarker, SourceInfo

from .input import Recording, standardize_channel_names

__all__ = [
    "BNCI2014_SOURCE_ID",
    "EEG_CHANNELS_2A",
    "SEALED_SESSIONS",
    "load_bnci2014_session",
    "parse_bnci2014_mat",
    "session_recording_id",
]

BNCI2014_SOURCE_ID = "bnci2014-001"
"""``EEGChunk.source_id``；与 PhysioNet 的 ``physionet-eegmmidb`` 区分。"""

SEALED_SESSIONS = frozenset({"E"})
"""封存 session：``E`` 是最终测试集，默认拒绝加载。"""

EEG_CHANNELS_2A = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)  # fmt: skip
"""22 导 EEG；含本项目 9 个模型通道（FCz 也在内）。"""

EOG_CHANNELS_2A = ("EOG1", "EOG2", "EOG3")

_CLASS_LABELS = {1: "T1", 2: "T2", 3: "T3", 4: "T4"}
"""2a 的 class 1–4 → 原始标签；``LABEL_BY_MARKER`` 只映射 T0/T1/T2。"""


def session_recording_id(subject: int, session: str) -> str:
    """``(1, "T")`` → ``A01T``；文件名与日志都用它。"""
    return f"A{subject:02d}{session.upper()}"


def _field(run: object, name: str) -> object:
    """兼容 scipy ``mat_struct``（属性）与普通 dict（键）。"""
    if isinstance(run, dict):
        return run[name]
    return getattr(run, name)


def _as_runs(payload: object) -> list[object]:
    """``loadmat`` 出来的 ``data`` 可能是单个 struct 或 struct 数组。"""
    if isinstance(payload, (list, tuple)):
        return list(payload)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return list(payload.ravel())
    if isinstance(payload, np.ndarray) and payload.size > 1:
        return list(payload.ravel())
    return [payload]


def parse_bnci2014_mat(
    payload: object,
    *,
    subject: int,
    session: str,
    path: str | Path | None = None,
) -> Recording:
    """把一个 2a session（``data`` 结构，6 个 run）转成契约 :class:`Recording`。

    ``X`` 单位 µV，转成伏；``trial`` 是 1-based 提示样本号；``y`` 是 class 1–4。
    """
    session = session.upper()
    if session not in ("T", "E"):
        raise ContractError(f"session 只能是 'T' 或 'E'，收到 {session!r}")

    runs = _as_runs(payload)
    if not runs:
        raise ContractError("2a 文件里没有任何 run")

    chunks: list[np.ndarray] = []
    markers: list[ReferenceMarker] = []
    record_id = session_recording_id(subject, session)
    n_samples_total = 0
    prev_end = 0
    sample_rate: float | None = None
    channel_names: tuple[str, ...] | None = None

    for run in runs:
        data_uv = np.asarray(_field(run, "X"), dtype=np.float64)
        trials = np.asarray(_field(run, "trial"), dtype=np.int64).ravel()
        labels = np.asarray(_field(run, "y"), dtype=np.int64).ravel()
        rate = float(_field(run, "fs"))
        if data_uv.ndim != 2 or data_uv.shape[1] not in (22, 25):
            raise ContractError(f"2a run 的 X 形状应为 (样本, 22/25)，收到 {data_uv.shape}")
        if trials.size != labels.size:
            raise ContractError("2a run 的 trial 与 y 长度不一致")
        if sample_rate is None:
            sample_rate = rate
            names = (
                EEG_CHANNELS_2A if data_uv.shape[1] == 22 else (*EEG_CHANNELS_2A, *EOG_CHANNELS_2A)
            )
            channel_names = standardize_channel_names(names)
        elif abs(sample_rate - rate) > 1e-9:
            raise ContractError(f"2a 各 run 采样率不一致：{sample_rate:g} vs {rate:g}")

        chunks.append(data_uv)
        offset = n_samples_total
        run_samples = data_uv.shape[0]
        for trial_sample, class_id in zip(trials, labels, strict=True):
            cue = offset + max(0, int(trial_sample) - 1)  # 1-based → 0-based
            if cue <= prev_end:
                raise ContractError(
                    f"2a trial 重叠或乱序：cue={cue} 不晚于上一段想象结束 {prev_end}"
                )
            mi_start = min(cue + int(round(1.0 * rate)), offset + run_samples)
            mi_end = min(cue + int(round(4.0 * rate)), offset + run_samples)
            markers.append(
                ReferenceMarker(
                    recording_id=record_id,
                    onset_sample=prev_end,
                    label="T0",
                    duration_samples=cue - prev_end,
                )
            )
            if mi_end <= mi_start:
                raise ContractError(
                    f"2a trial 的想象段为空：cue={cue}，run 末端 {offset + run_samples}"
                )
            markers.append(
                ReferenceMarker(
                    recording_id=record_id,
                    onset_sample=mi_start,
                    label=_CLASS_LABELS.get(int(class_id), "T9"),
                    duration_samples=mi_end - mi_start,
                )
            )
            prev_end = max(prev_end, mi_end)
        n_samples_total += run_samples

    # 末尾的空档不建 marker：训练/评估只需要有标签的窗，尾部休息没有标签需求。

    assert sample_rate is not None and channel_names is not None  # 由首块设置
    samples = np.concatenate(chunks, axis=0).T * 1e-6
    file_path = Path(path) if path is not None else None
    info = SourceInfo(
        source_id=BNCI2014_SOURCE_ID,
        recording_id=record_id,
        sample_rate_hz=sample_rate,
        channel_names=channel_names,
        reference="left mastoid",
        subject_id=f"A{subject:02d}",
        run=session,
        n_samples=n_samples_total,
        device_metadata={
            "dataset": BNCI2014_SOURCE_ID,
            "session": session,
            "unit_source": "uV",
            "runs": str(len(runs)),
        },
        path=str(file_path) if file_path is not None else None,
    )
    return Recording(
        info=info,
        samples=samples,
        markers=tuple(markers),
        path=file_path,
        sha256=_sha256_of(file_path) if file_path is not None else None,
    )


def load_bnci2014_session(
    path: str | Path,
    *,
    subject: int,
    session: str,
    allow_sealed: bool = False,
) -> Recording:
    """读一个 2a session；``E`` 默认拒绝（最终测试集）。"""
    session = session.upper()
    file_path = Path(path)
    record_id = session_recording_id(subject, session)
    if session in SEALED_SESSIONS and not allow_sealed:
        raise ContractError(
            f"{record_id} 是 2a 的封存测试 session（E）：加载它就不能再用来调参。"
            "确认冻结且只跑一次时，显式传 allow_sealed=True。"
        )
    if not file_path.is_file():
        raise ContractError(f"找不到 2a 文件：{file_path}")

    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - scipy 是硬依赖
        raise ContractError("读取 2a 需要 scipy（见 requirements.txt）") from exc

    try:
        mat = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    except Exception as exc:
        raise ContractError(f"2a 文件读取失败：{file_path.name}：{exc}") from exc
    if "data" not in mat:
        raise ContractError(f"2a 文件缺少 data 结构：{file_path.name}")
    return parse_bnci2014_mat(mat["data"], subject=subject, session=session, path=file_path)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
