"""质量 + 预热 + 三状态分数 → 契约 ``DecisionFrame``。

状态优先级：``invalid`` > ``warming_up`` > ``uncertain`` / 已确认类别。

* 质量不合格：只能发 ``invalid``，不能“猜一个左右手”；
* 连续有效数据不足 3 s：发 ``warming_up``，不带窗口/分数；
* 分数低于门限或领先不够：发 ``uncertain``，带分数和窗口；
* 连续 ``confirm_updates`` 个更新点都是同一类，才发 ``idle`` / ``left_hand`` /
  ``right_hand``。

``valid_for_ms`` 写进每一帧，由接收方（界面/存档）判断结果是否过期；本状态机不
自己“补发”旧动作。
"""

from __future__ import annotations

from eeg_core.contracts import (
    AppConfig,
    ContractError,
    DecisionFrame,
    DecisionState,
    ModelInputSpec,
    QualityReport,
    Scores,
    WindowRange,
    utc_now_iso,
)

__all__ = ["DecisionEngine"]


class DecisionEngine:
    """一个会话内的决策状态机；每条流一个实例。"""

    def __init__(
        self,
        *,
        config: AppConfig,
        spec: ModelInputSpec,
        session_id: str,
        recording_id: str,
    ) -> None:
        if not isinstance(config, AppConfig):
            raise ContractError("config 必须是 AppConfig")
        if not isinstance(spec, ModelInputSpec):
            raise ContractError("spec 必须是 ModelInputSpec")
        self.config = config
        self.spec = spec
        self.session_id = session_id
        self.recording_id = recording_id
        self._seq = 0
        self._candidate: str | None = None
        self._candidate_count = 0

    @property
    def seq(self) -> int:
        """下一个更新点的序号。"""
        return self._seq

    def reset(self) -> None:
        """清空确认历史；会话重新开始/暂停恢复后由调用方决定是否调用。"""
        self._candidate = None
        self._candidate_count = 0

    def _reset_candidate(self) -> None:
        self._candidate = None
        self._candidate_count = 0

    def update(
        self,
        *,
        quality: QualityReport,
        ready: bool,
        scores: Scores | None,
        window: WindowRange | None,
        emitted_monotonic_ms: float,
        source_t_sec: float | None = None,
        generated_at_utc: str | None = None,
    ) -> DecisionFrame:
        """处理一个 250 ms 更新点，返回一帧合法 ``DecisionFrame``。"""
        if not isinstance(quality, QualityReport):
            raise ContractError("quality 必须是 QualityReport")
        if window is not None and not isinstance(window, WindowRange):
            raise ContractError("window 必须是 WindowRange 或 None")
        if not isinstance(ready, bool):
            raise ContractError("ready 必须是 bool")
        if scores is not None and not isinstance(scores, Scores):
            raise ContractError("scores 必须是 Scores 或 None")

        seq = self._seq
        self._seq += 1
        state: DecisionState
        output_scores: Scores | None
        output_window: WindowRange | None
        intent: DecisionState | None

        if not quality.valid:
            state = DecisionState.INVALID
            output_scores = None
            output_window = window
            intent = None
            self._reset_candidate()
        elif not ready:
            state = DecisionState.WARMING_UP
            output_scores = None
            output_window = None
            intent = None
            self._reset_candidate()
        else:
            if scores is None:
                raise ContractError("已预热且质量合格时必须提供 scores")
            if window is None:
                raise ContractError("已预热且质量合格时必须提供证据窗")

            top_name, top_score = scores.top()
            second = sorted(scores.as_dict().values(), reverse=True)[1]
            if (
                top_score < self.config.decision.threshold
                or top_score - second < self.config.decision.margin
            ):
                state = DecisionState.UNCERTAIN
                self._reset_candidate()
            else:
                if self._candidate == top_name:
                    self._candidate_count += 1
                else:
                    self._candidate = top_name
                    self._candidate_count = 1
                if self._candidate_count >= self.config.decision.confirm_updates:
                    state = DecisionState(top_name)
                else:
                    state = DecisionState.UNCERTAIN
            output_scores = scores
            output_window = window
            intent = state if state.is_action else None

        if output_window is not None:
            source_time = output_window.source_t_sec
            sample_rate = output_window.sample_rate_hz
        else:
            source_time = 0.0 if source_t_sec is None else float(source_t_sec)
            sample_rate = self.config.source.sample_rate_hz

        return DecisionFrame(
            session_id=self.session_id,
            recording_id=self.recording_id,
            seq=seq,
            decision_state=state,
            sample_rate_hz=sample_rate,
            source_t_sec=source_time,
            generated_at_utc=generated_at_utc or utc_now_iso(),
            emitted_monotonic_ms=emitted_monotonic_ms,
            model_id=self.spec.model_id,
            pipeline_version=self.spec.pipeline_version,
            config_hash=self.config.config_hash,
            window=output_window,
            intent=intent,
            scores=output_scores,
            quality=quality,
            valid_for_ms=self.config.decision.valid_for_ms,
        )
