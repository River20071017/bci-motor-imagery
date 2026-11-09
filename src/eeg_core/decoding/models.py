"""三状态收缩 LDA：训练、打分、``lda_npz_v1`` 权重读写。

模型包 = ``weights.npz`` + ``ModelInputSpec``。本模块只管数组与数学，不读 EDF、
不碰 UI；加载 ``model.json`` 在 :mod:`eeg_core.decoding.inference`。

固定的类别顺序是 ``("idle", "left_hand", "right_hand")``。左右手互换是最典型的
“看着正常但全反了”的失败，所以这里在训练、保存、加载三处都校验顺序。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from eeg_core.contracts import ContractError

__all__ = [
    "CLASSES",
    "WEIGHTS_FORMAT",
    "LdaArrays",
    "fit_lda",
    "load_weights",
    "predict_proba",
    "save_weights",
    "sha256_file",
]

CLASSES = ("idle", "left_hand", "right_hand")
"""模型类别顺序；与 ``ModelInputSpec.classes`` 和 ``DecisionState`` 对齐。"""

WEIGHTS_FORMAT = "lda_npz_v1"
"""首版权重格式：``npz`` 里存 ``coef`` / ``intercept`` / ``priors`` / scaler 数组。"""


def _readonly(array: object, name: str) -> np.ndarray:
    stored = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(stored).all():
        raise ContractError(f"{name} 必须是有限浮点数组")
    stored.flags.writeable = False
    return stored


@dataclass(frozen=True, slots=True)
class LdaArrays:
    """一个三状态 LDA 的可推理数组；形状在构造时校验。"""

    coef: np.ndarray
    intercept: np.ndarray
    priors: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray

    def __post_init__(self) -> None:
        coef = _readonly(self.coef, "coef")
        intercept = _readonly(self.intercept, "intercept")
        priors = _readonly(self.priors, "priors")
        mean = _readonly(self.scaler_mean, "scaler_mean")
        scale = _readonly(self.scaler_scale, "scaler_scale")

        if coef.ndim != 2 or coef.shape[0] != len(CLASSES):
            raise ContractError(
                f"coef 形状应为 ({len(CLASSES)}, 特征数)，收到 {coef.shape}"
            )
        if intercept.shape != (len(CLASSES),):
            raise ContractError(
                f"intercept 形状应为 ({len(CLASSES)},)，收到 {intercept.shape}"
            )
        if priors.shape != (len(CLASSES),):
            raise ContractError(f"priors 形状应为 ({len(CLASSES)},)，收到 {priors.shape}")
        if mean.shape != (coef.shape[1],) or scale.shape != (coef.shape[1],):
            raise ContractError(
                "scaler_mean / scaler_scale 必须与特征数一致，"
                f"收到 {mean.shape} / {scale.shape}，特征数 {coef.shape[1]}"
            )
        if (scale <= 0).any():
            raise ContractError("scaler_scale 必须全为正数")

        object.__setattr__(self, "coef", coef)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "priors", priors)
        object.__setattr__(self, "scaler_mean", mean)
        object.__setattr__(self, "scaler_scale", scale)

    @property
    def n_features(self) -> int:
        return int(self.coef.shape[1])


def fit_lda(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    solver: str = "lsqr",
    shrinkage: str = "auto",
) -> tuple[LdaArrays, dict[str, object]]:
    """按固定类别顺序拟合三状态 LDA。

    ``features`` 形状 ``(n, 18)``；``labels`` 取 ``CLASSES`` 中的字符串。至少每类
    2 个样本，且三类齐全——模型包契约不允许 2 类或换顺序的模型。
    """
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=str)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ContractError(f"features 必须是 (n, 特征数) 的二维数组，收到 {matrix.shape}")
    if target.ndim != 1 or target.shape[0] != matrix.shape[0]:
        raise ContractError(
            f"labels 长度必须等于样本数，收到 {target.shape} vs {matrix.shape[0]}"
        )
    if not np.isfinite(matrix).all():
        raise ContractError("features 含非有限值")

    unknown = sorted(set(target) - set(CLASSES))
    if unknown:
        raise ContractError("labels 含未知类别：" + "、".join(unknown))
    counts = {name: int((target == name).sum()) for name in CLASSES}
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ContractError("训练数据缺少类别：" + "、".join(missing))
    too_few = [name for name, count in counts.items() if count < 2]
    if too_few:
        raise ContractError("每类至少需要 2 个样本，缺少：" + "、".join(too_few))

    scaler = StandardScaler().fit(matrix)
    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    scaled = (matrix - scaler.mean_) / scale

    lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage).fit(scaled, target)
    observed = tuple(str(name) for name in lda.classes_)
    if set(observed) != set(CLASSES):
        raise ContractError(f"LDA 学到的类别不完整：{observed}")
    order = [observed.index(name) for name in CLASSES]
    arrays = LdaArrays(
        coef=lda.coef_[order],
        intercept=lda.intercept_[order],
        priors=lda.priors_[order],
        scaler_mean=scaler.mean_,
        scaler_scale=scale,
    )
    info: dict[str, object] = {
        "n_samples": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "class_counts": counts,
        "train_accuracy": float(lda.score(scaled, target)),
        "solver": solver,
        "shrinkage": shrinkage,
    }
    return arrays, info


def predict_proba(arrays: LdaArrays, features: np.ndarray) -> np.ndarray:
    """18 维特征 → 三类概率；单样本返回 ``(3,)``，批量返回 ``(n, 3)``。"""
    matrix = np.asarray(features, dtype=np.float64)
    single = matrix.ndim == 1
    if single:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != arrays.n_features:
        raise ContractError(
            f"features 形状应为 (n, {arrays.n_features})，收到 {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ContractError("features 含非有限值")

    scaled = (matrix - arrays.scaler_mean) / arrays.scaler_scale
    decision = scaled @ arrays.coef.T + arrays.intercept
    decision -= decision.max(axis=1, keepdims=True)
    exponent = np.exp(decision)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities[0] if single else probabilities


def sha256_file(path: str | Path) -> str:
    """整文件 SHA-256；模型包的 ``weights_sha256`` 就是它。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_weights(path: str | Path, arrays: LdaArrays) -> str:
    """写 ``lda_npz_v1`` 并返回文件 SHA-256。"""
    target = Path(path)
    if target.suffix != ".npz":
        raise ContractError(f"权重文件必须以 .npz 结尾，收到 {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        coef=arrays.coef,
        intercept=arrays.intercept,
        priors=arrays.priors,
        scaler_mean=arrays.scaler_mean,
        scaler_scale=arrays.scaler_scale,
        classes=np.asarray(CLASSES),
    )
    return sha256_file(target)


def load_weights(path: str | Path) -> LdaArrays:
    """读 ``lda_npz_v1``；缺键、类别顺序不对、形状不对都抛 ``ContractError``。"""
    target = Path(path)
    if not target.is_file():
        raise ContractError(f"找不到权重文件：{target}")
    try:
        with np.load(target, allow_pickle=False) as data:
            required = {"coef", "intercept", "priors", "scaler_mean", "scaler_scale", "classes"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ContractError("权重文件缺少键：" + "、".join(missing))
            classes = tuple(str(name) for name in data["classes"].tolist())
            if classes != CLASSES:
                raise ContractError(
                    f"权重类别顺序必须是 {CLASSES}，收到 {classes}"
                )
            return LdaArrays(
                coef=data["coef"],
                intercept=data["intercept"],
                priors=data["priors"],
                scaler_mean=data["scaler_mean"],
                scaler_scale=data["scaler_scale"],
            )
    except ContractError:
        raise
    except Exception as exc:  # noqa: BLE001 - numpy 的损坏文件异常类型不稳定
        raise ContractError(f"权重文件读取失败：{target}：{exc}") from exc
