"""서버 측 이동 결정 주체(Brain).

**이동 결정은 전부 서버에서 일어난다.** 클라이언트는 관측을 그리거나, 사람 모드에서
의도한 행동을 보낼 뿐이다 (PLAN §1 R2·R3).

세 가지 구현체
--------------
- :class:`PolicyServerBrain` — LoRA-BERT 체크포인트 (``ai``)
- :class:`RuleServerBrain`   — 왼손 법칙 / 지식 지도 탐색 (``rule``)
- :class:`HumanServerBrain`  — 클라이언트가 보낸 행동을 기다린다 (``human``)

규칙 구현체는 서버 안에 있으므로 R1(프론트엔드에 경로 탐색 없음)을 위반하지 않는다.
클라이언트에는 어떤 탐색 코드도 내려가지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from app.game.actions import Action
from app.game.env import MazeEnv
from app.game.expert import frontier_action, wall_follow_action

#: AI 체크포인트 자동 선택 순서
CHECKPOINT_CANDIDATES = ("wall", "v3")

#: 프로세스 전역 모델 캐시 (게임마다 새로 로드하지 않는다)
_POLICY_CACHE: dict[str, object] = {}


def discover_checkpoint(server_root: Path) -> str | None:
    """``MAZE_CHECKPOINT`` → ``checkpoints/wall`` → ``checkpoints/v3`` → 최신 순."""
    import os

    env_path = os.environ.get("MAZE_CHECKPOINT")
    if env_path and Path(env_path).exists():
        return env_path

    for name in CHECKPOINT_CANDIDATES:
        candidate = server_root / "checkpoints" / name / "best.pt"
        if candidate.exists():
            return str(candidate)

    found = sorted(
        (server_root / "checkpoints").glob("*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(found[0]) if found else None


class ServerBrain(Protocol):
    """서버가 요구하는 최소 인터페이스."""

    kind: str
    label: str
    requires_client_action: bool
    #: env 를 만들 때 필요한 관측 설정
    window_size: int
    global_size: int
    map_size: int

    def reset(self) -> None: ...

    def act(self, env: MazeEnv) -> int | None:
        """다음 행동. ``None`` 이면 클라이언트 입력을 기다린다."""

    def probabilities(self) -> list[float] | None: ...


class HumanServerBrain:
    """행동을 스스로 정하지 않는다. 클라이언트가 ``/action`` 으로 보낸다."""

    kind = "human"
    label = "HUMAN · 클라이언트 입력"
    requires_client_action = True
    window_size = 3
    global_size = 5
    map_size = 0

    def reset(self) -> None:
        pass

    def act(self, env: MazeEnv) -> int | None:
        return None

    def probabilities(self) -> list[float] | None:
        return None


class RuleServerBrain:
    """관측만으로 계산 가능한 규칙 전문가.

    - ``wall``     : 왼손 법칙 (완벽 미로에서 도달 보장, 비효율 2~5배)
    - ``frontier`` : 지식 지도 위 BFS 탐색 (평균 10% 정도 더 짧다)

    ``frontier`` 는 지식 지도(``env.known``)가 필요하므로 ``map_size`` 를 켠다.
    """

    kind = "rule"
    requires_client_action = False

    def __init__(self, rule: str = "wall") -> None:
        if rule not in ("wall", "frontier"):
            raise ValueError(f"unknown rule: {rule!r}")
        self.rule = rule
        self.window_size = 3
        self.global_size = 0
        #: frontier 는 누적 관측 지도가 필요하다. 미로 크기는 세션 생성 시 맞춘다.
        self.map_size = 0 if rule == "wall" else -1
        self.label = (
            "RULE · 왼손 법칙" if rule == "wall" else "RULE · frontier 탐색"
        )

    def reset(self) -> None:
        pass

    def act(self, env: MazeEnv) -> int | None:
        if self.rule == "wall":
            return int(wall_follow_action(env.maze, env.position, env.last_action))
        return int(frontier_action(env.known, env.position, env.maze.goal))

    def probabilities(self) -> list[float] | None:
        return None


class PolicyServerBrain:
    """LoRA-BERT 체크포인트로 행동을 고른다 (서버 프로세스에 모델 캐시)."""

    kind = "ai"
    requires_client_action = False

    def __init__(self, checkpoint: str, seq_len: int | None = None) -> None:
        from app.ml.policy import BertPolicy
        from app.ml.runner import PolicyRunner

        model = _POLICY_CACHE.get(checkpoint)
        if model is None:
            model = BertPolicy.load(checkpoint)
            _POLICY_CACHE[checkpoint] = model
        self.model = model
        self.runner = PolicyRunner(model, seq_len=seq_len)
        self.window_size = int(getattr(model.cfg, "window_size", 3))
        self.global_size = int(getattr(model.cfg, "global_size", 5))
        self.map_size = int(getattr(model.cfg, "map_size", 0))
        self.checkpoint = checkpoint
        self.label = f"AI · {Path(checkpoint).parent.name}/{Path(checkpoint).name}"
        self._probs: np.ndarray | None = None

    def reset(self) -> None:
        self.runner.reset()
        self._probs = None

    def act(self, env: MazeEnv) -> int | None:
        import torch

        logits = self.runner.logits(env.observe())
        probs = torch.softmax(logits, dim=-1)
        self._probs = probs.detach().cpu().numpy()
        return int(torch.argmax(probs).item())

    def probabilities(self) -> list[float] | None:
        if self._probs is None:
            return None
        return [round(float(p), 6) for p in self._probs]


def build_brain(
    kind: str,
    rule: str = "wall",
    checkpoint: str | None = None,
) -> ServerBrain:
    """``kind`` → Brain 인스턴스."""
    if kind == "human":
        return HumanServerBrain()
    if kind == "rule":
        return RuleServerBrain(rule)
    if kind == "ai":
        path = checkpoint or discover_checkpoint(Path(__file__).resolve().parents[2])
        if not path:
            raise FileNotFoundError(
                "체크포인트를 찾지 못했습니다. MAZE_CHECKPOINT 환경변수를 지정하세요."
            )
        return PolicyServerBrain(path)
    raise ValueError(f"unknown brain kind: {kind!r}")


__all__ = [
    "HumanServerBrain",
    "PolicyServerBrain",
    "RuleServerBrain",
    "ServerBrain",
    "build_brain",
    "discover_checkpoint",
]
