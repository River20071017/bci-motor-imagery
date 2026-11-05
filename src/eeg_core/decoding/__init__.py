"""解码层：模型数组、模型包加载、三状态决策。"""

from __future__ import annotations

from .decision import DecisionEngine
from .inference import ModelPackage, load_model_package
from .models import (
    CLASSES,
    WEIGHTS_FORMAT,
    LdaArrays,
    fit_lda,
    load_weights,
    predict_proba,
    save_weights,
    sha256_file,
)

__all__ = [
    "CLASSES",
    "WEIGHTS_FORMAT",
    "DecisionEngine",
    "LdaArrays",
    "ModelPackage",
    "fit_lda",
    "load_model_package",
    "load_weights",
    "predict_proba",
    "save_weights",
    "sha256_file",
]
