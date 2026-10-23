"""이동 결정 주체(Brain).

뷰어는 **이동을 결정하지 않는다**. 화면은 Brain 이 돌려준 행동을 그대로 적용할 뿐이다
(PLAN §1 R2). 지금은 두 가지 구현이 있다.

- :class:`HumanBrain`  : 사람이 방향키로 조작
- :class:`PolicyBrain` : LoRA-BERT 체크포인트가 결정

나중에 FastAPI 서버가 붙으면 ``HttpBrain`` 을 추가하면 되고, 뷰어/렌더러는
한 줄도 바뀌지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pygame
import torch

from app.game.actions import Action
from app.game.env import ALLOWED_GLOBAL_SIZES, ALLOWED_WINDOW_SIZES, observation_dim

#: 방향키 / WASD → 행동
KEY_TO_ACTION: dict[int, Action] = {
    pygame.K_UP: Action.UP,
    pygame.K_w: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_s: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_a: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_d: Action.RIGHT,
}


class Brain(Protocol):
    """뷰어가 요구하는 최소 인터페이스."""

    label: str
    window_size: int
    global_size: int
    animation_duration: float

    def reset(self) -> None: ...

    def wants_step(self, now: float) -> bool: ...

    def act(self, env, events: Sequence[pygame.event.Event]) -> int | None: ...

    def note_step(self, now: float) -> None: ...

    def probabilities(self) -> np.ndarray | None: ...


def _resolve_obs_spec(cfg) -> tuple[int, int]:
    """체크포인트의 ``obs_dim`` 과 일치하는 (window, global) 조합을 찾는다.

    ``window_size``/``global_size`` 가 저장되기 전에 만든 구 체크포인트
    (예: 29차원)도 뷰어에서 열 수 있게 하기 위한 호환 처리다.
    """
    window = int(getattr(cfg, "window_size", 3))
    glob = int(getattr(cfg, "global_size", 5))
    if observation_dim(window, glob) == int(cfg.obs_dim):
        return window, glob

    for w in ALLOWED_WINDOW_SIZES:
        for g in ALLOWED_GLOBAL_SIZES:
            if observation_dim(w, g) == int(cfg.obs_dim):
                return w, g

    raise ValueError(
        f"obs_dim={cfg.obs_dim} 에 맞는 (window_size, global_size) 조합이 없습니다"
    )


class HumanBrain:
    """키 입력을 행동으로 바꾼다. 입력이 없으면 ``None`` (이동하지 않음)."""

    label = "HUMAN"
    window_size = 3
    global_size = 5
    animation_duration = 0.10

    def reset(self) -> None:
        pass

    def wants_step(self, now: float) -> bool:
        return True

    def act(self, env, events: Sequence[pygame.event.Event]) -> int | None:
        action: int | None = None
        for event in events:
            if event.type == pygame.KEYDOWN and event.key in KEY_TO_ACTION:
                action = int(KEY_TO_ACTION[event.key])
        return action

    def note_step(self, now: float) -> None:
        pass

    def probabilities(self) -> np.ndarray | None:
        return None


class PolicyBrain:
    """체크포인트를 로드해 매 스텝 행동을 예측한다.

    ``PolicyRunner`` 가 관측 이력(슬라이딩 윈도우)을 관리하므로, 서버에서
    동작할 때와 **완전히 같은 경로**로 추론된다.
    """

    def __init__(
        self,
        checkpoint: str,
        step_interval: float = 0.09,
        temperature: float = 0.0,
        seq_len: int | None = None,
    ) -> None:
        # 순환 import 와 무거운 의존성(torch)을 뷰어 시작 시점까지 미룬다
        from app.ml.policy import BertPolicy
        from app.ml.runner import PolicyRunner

        self.model = BertPolicy.load(checkpoint)
        self.window_size, self.global_size = _resolve_obs_spec(self.model.cfg)
        self.runner = PolicyRunner(self.model, seq_len=seq_len)
        self.interval = max(0.01, float(step_interval))
        self.temperature = float(temperature)
        self.animation_duration = max(0.05, min(0.12, self.interval * 0.8))
        self.label = f"AI · {Path(checkpoint).parent.name}/{Path(checkpoint).name}"
        self._probs: np.ndarray | None = None
        self._next_at = 0.0

    # ------------------------------------------------------------------ 수명주기
    def reset(self) -> None:
        self.runner.reset()
        self._probs = None
        self._next_at = 0.0

    def wants_step(self, now: float) -> bool:
        return now >= self._next_at

    def note_step(self, now: float) -> None:
        self._next_at = now + self.interval

    # ------------------------------------------------------------------ 추론
    def act(self, env, events: Sequence[pygame.event.Event]) -> int | None:
        logits = self.runner.logits(env.observe())
        probs = torch.softmax(logits, dim=-1)
        self._probs = probs.detach().cpu().numpy()

        if self.temperature > 0:
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(torch.argmax(probs).item())

    def probabilities(self) -> np.ndarray | None:
        return self._probs
