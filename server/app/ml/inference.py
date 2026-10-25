"""체크포인트 독립 추론 코어 (FastAPI 비의존).

게임 세션 없이 **관측 시계열만으로** 행동을 계산한다. API 계층은 이 모듈을
호출하기만 하고, 모델 로드/캐시는 :mod:`app.ml.registry` 가 담당한다.

입력 규약은 학습·게임 서버와 동일하다.

- 각 관측은 ``MazeEnv.observe()`` 가 만든 1차원 벡터 (``obs_dim``)
- 마지막 관측이 현재 스텝
- ``seq_len`` 보다 짧으면 **왼쪽 0 패딩 + attention mask 0**
- ``seq_len`` 보다 길면 **최근 ``seq_len`` 개만** 사용
"""

from __future__ import annotations

import math
import time

import torch

from ..game.actions import ACTION_NAMES
from .policy import BertPolicy


def _device_of(model: BertPolicy) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - 파라미터 없는 모델은 없음
        return torch.device("cpu")


def _validate(model: BertPolicy, observations: list[list[float]]) -> None:
    obs_dim = int(model.cfg.obs_dim)
    if not observations:
        raise ValueError("observations must not be empty")
    for index, row in enumerate(observations):
        if len(row) != obs_dim:
            raise ValueError(
                f"observation[{index}] has {len(row)} values; expected obs_dim={obs_dim}"
            )
        for value in row:
            if not math.isfinite(value):
                raise ValueError(f"observation[{index}] contains a non-finite value")


def build_input(
    model: BertPolicy, observations: list[list[float]]
) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
    """관측 시계열 → ``(1, seq_len, obs_dim)`` 입력과 attention mask."""
    _validate(model, observations)
    cfg = model.cfg
    seq_len = int(cfg.seq_len)
    truncated = len(observations) > seq_len
    window = observations[-seq_len:]
    history = len(window)
    device = _device_of(model)

    seq = torch.zeros(1, seq_len, int(cfg.obs_dim), dtype=torch.float32, device=device)
    mask = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    seq[0, -history:] = torch.tensor(window, dtype=torch.float32, device=device)
    mask[0, -history:] = 1
    return seq, mask, history, truncated


def infer(
    model: BertPolicy,
    observations: list[list[float]],
    return_logits: bool = False,
) -> dict:
    """관측 시계열로 행동·확률을 계산한다.

    반환값은 ``PredictResponse`` 의 필드와 호환되는 dict.
    ``latency_ms`` 는 입력 검증·전처리·순전파를 포함한 서버 처리 시간이다.
    """
    started = time.perf_counter()
    model.eval()
    seq, mask, history, truncated = build_input(model, observations)

    with torch.inference_mode():
        logits = model(seq, mask).squeeze(0)
        probabilities = torch.softmax(logits, dim=-1)

    action = int(torch.argmax(probabilities).item())
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return {
        "action": action,
        "action_name": ACTION_NAMES[action],
        "probabilities": [float(p) for p in probabilities.cpu().tolist()],
        "logits": [float(v) for v in logits.cpu().tolist()] if return_logits else None,
        "obs_dim": int(model.cfg.obs_dim),
        "seq_len": int(model.cfg.seq_len),
        "history_length": history,
        "truncated": truncated,
        "latency_ms": round(elapsed_ms, 3),
    }


def warmup(model: BertPolicy) -> None:
    """첫 요청 지연을 줄이기 위해 ``seq_len`` 길이의 0 입력으로 한 번 순전파한다."""
    cfg = model.cfg
    device = _device_of(model)
    seq = torch.zeros(1, int(cfg.seq_len), int(cfg.obs_dim), device=device)
    mask = torch.ones(1, int(cfg.seq_len), dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        model(seq, mask)
