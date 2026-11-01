"""界面测试用模拟解码器（非真实算法），实现 DecoderProtocol。

分工方案 4.2 允许 C 在 B 就绪前使用明确标识为"界面测试"的模拟结果对象。
本模块按时间脚本输出三状态 ``DecisionFrame``，覆盖全部界面状态，用于验证
波形 / 时间轴 / 火柴人 / 状态文案的联动。

它严格产出团队契约 ``eeg_core.contracts.DecisionFrame``（含 ``DecisionState`` /
``Scores`` / ``WindowRange`` / ``QualityReport``，构造即校验），因此 B 后续用真实
解码器替换时，回放控制器与界面无需改动。
"""
from __future__ import annotations

from typing import Optional

from eeg_core.contracts import (
    DecisionFrame,
    DecisionState,
    Scores,
    WindowRange,
    QualityReport,
    ResetReason,
    utc_now_iso,
    DEFAULT_CONFIG,
    EEGChunk,
)


class MockDecoder:
    """按脚本输出决策，演示全部状态（界面测试）。

    实现 ``eeg_app.ui.playback.DecoderProtocol``：
    ``set_session`` / ``reset(reason)`` / ``decode(chunk)``。
    """

    def __init__(
        self,
        sample_rate_hz: float = 160.0,
        model_id: str = "MOCK-lda-v1",
        pipeline_version: str = "0.0.0-ui-test",
        config_hash: str = "mock-config-hash",
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.model_id = model_id
        self.pipeline_version = pipeline_version
        self.config_hash = config_hash
        self.session_id = ""
        self._seq = 0
        self._last_emit_sample = -1
        # (持续秒, 状态) 脚本；覆盖全部界面状态
        self._script = [
            (3.0, DecisionState.WARMING_UP),
            (4.0, DecisionState.IDLE),
            (3.0, DecisionState.LEFT_HAND),
            (3.0, DecisionState.IDLE),
            (3.0, DecisionState.RIGHT_HAND),
            (3.0, DecisionState.IDLE),
            (2.0, DecisionState.UNCERTAIN),
            (2.0, DecisionState.IDLE),
            (1.5, DecisionState.INVALID),
            (2.5, DecisionState.IDLE),
            (3.0, DecisionState.LEFT_HAND),
            (3.0, DecisionState.RIGHT_HAND),
            (4.0, DecisionState.IDLE),
        ]

    def set_session(self, session_id: str) -> None:
        self.session_id = session_id

    def reset(self, reason: ResetReason) -> None:
        # PAUSE_RESUME 保留已处理历史，只作废候选确认（这里不重置 _last_emit_sample）；
        # 其余原因（新会话 / 间断 / 质量失败）清空全部状态。
        if reason is ResetReason.PAUSE_RESUME:
            return
        self._seq = 0
        self._last_emit_sample = -1

    def decode(self, chunk: Optional[EEGChunk]) -> Optional[DecisionFrame]:
        if chunk is None:
            return None
        sr = chunk.sample_rate_hz
        start = chunk.start_sample
        elapsed_sec = start / sr if sr else 0.0
        # 按 ~250 ms 步长发布（PRD 6.1）；由解码器决定何时发布，符合 B 的真实行为
        if self._last_emit_sample < 0 or (start - self._last_emit_sample) / sr >= 0.25:
            self._last_emit_sample = start
            state = self._state_at(elapsed_sec)
            scores = self._scores_for(state)
            intent = state if state.is_action else None
            window_samples = int(2.0 * sr)
            if state is DecisionState.WARMING_UP:
                window = None
                quality = QualityReport.ok()
                source_t_sec = elapsed_sec
            else:
                window = WindowRange(
                    sample_rate_hz=sr,
                    start_sample=start,
                    end_sample=start + window_samples,
                )
                if state is DecisionState.INVALID:
                    quality = QualityReport.failed(("amplitude",), window=window)
                else:
                    quality = QualityReport.ok(window=window)
                source_t_sec = window.source_t_sec
            self._seq += 1
            return DecisionFrame(
                session_id=self.session_id,
                recording_id=chunk.recording_id or "S001R12",
                seq=self._seq,
                decision_state=state,
                sample_rate_hz=sr,
                source_t_sec=source_t_sec,
                generated_at_utc=utc_now_iso(),
                emitted_monotonic_ms=chunk.released_monotonic_ms,
                model_id=self.model_id,
                pipeline_version=self.pipeline_version,
                config_hash=self.config_hash,
                window=window,
                intent=intent,
                scores=scores,
                score_type="uncalibrated_probability",
                quality=quality,
                valid_for_ms=DEFAULT_CONFIG.decision.valid_for_ms,
            )
        return None

    def _state_at(self, t: float) -> DecisionState:
        if t < 0:
            return DecisionState.WARMING_UP
        total = 0.0
        for dur, st in self._script:
            total += dur
            if t < total:
                return st
        return DecisionState.IDLE

    def _scores_for(self, state: DecisionState) -> Optional[Scores]:
        if state is DecisionState.IDLE:
            return Scores(idle=0.80, left_hand=0.10, right_hand=0.10)
        if state is DecisionState.LEFT_HAND:
            return Scores(idle=0.10, left_hand=0.80, right_hand=0.10)
        if state is DecisionState.RIGHT_HAND:
            return Scores(idle=0.10, left_hand=0.10, right_hand=0.80)
        if state is DecisionState.UNCERTAIN:
            return Scores(idle=0.34, left_hand=0.33, right_hand=0.33)
        # WARMING_UP / INVALID 不带分数（契约校验要求：仅 IDLE/UNCERTAIN 可带分数）
        return None
