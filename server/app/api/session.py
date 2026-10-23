"""게임 세션 — 서버가 미로·환경·행동 결정을 모두 소유한다.

클라이언트는 ``StateResponse`` 만 받아서 그린다. 벽 판정·승리 판정·행동 선택은
전부 여기서 일어나므로 R2·R3 가 구조적으로 보장된다.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from app.game.env import MazeEnv, default_max_steps
from app.game.maze import Maze, generate
from app.api.brains import ServerBrain, build_brain
from app.api.schemas import MazeState, StateResponse


def _resolve_map_size(brain: ServerBrain, maze: Maze) -> int:
    """Brain 이 요구하는 지식 지도 크기 (``-1`` = 미로 크기)."""
    if brain.map_size < 0:
        return max(maze.width, maze.height)
    return brain.map_size


def _make_env(maze: Maze, brain: ServerBrain, max_steps: int | None) -> MazeEnv:
    return MazeEnv(
        maze,
        window_size=brain.window_size,
        global_size=brain.global_size,
        map_size=_resolve_map_size(brain, maze),
        max_steps=max_steps or default_max_steps(maze.width, maze.height),
    )


@dataclass
class StepOutcome:
    action: int | None = None
    moved: bool = False
    hit_wall: bool = False


class GameSession:
    """한 판의 상태. 스레드 안전하게 락으로 감싼다."""

    def __init__(
        self,
        size: int = 15,
        algorithm: str = "dfs",
        brain: str = "ai",
        rule: str = "wall",
        seed: int | None = None,
        max_steps: int | None = None,
        randomize_endpoints: bool = True,
        checkpoint: str | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.size = size
        self.algorithm = algorithm
        self.max_steps = max_steps
        self.randomize_endpoints = randomize_endpoints
        self._checkpoint = checkpoint
        self.lock = threading.Lock()

        self.brain = build_brain(brain, rule, checkpoint)
        self.maze = generate(
            size,
            size,
            seed=seed,
            algorithm=algorithm,
            randomize_endpoints=randomize_endpoints,
        )
        self.env = _make_env(self.maze, self.brain, max_steps)
        self.brain.reset()
        self.outcome = StepOutcome()

    # ------------------------------------------------------------------ 상태
    @property
    def brain_kind(self) -> str:
        return self.brain.kind

    @property
    def rule(self) -> str | None:
        return getattr(self.brain, "rule", None)

    def state(self, outcome: StepOutcome | None = None) -> StateResponse:
        env = self.env
        out = outcome or StepOutcome()
        return StateResponse(
            game_id=self.id,
            brain=self.brain.kind,
            brain_label=self.brain.label,
            rule=self.rule,
            window_size=env.window_size,
            global_size=env.global_size,
            map_size=env.map_size,
            maze=MazeState(
                width=self.maze.width,
                height=self.maze.height,
                grid=self.maze.grid.astype(int).tolist(),
                start=list(self.maze.start),
                goal=list(self.maze.goal),
                algorithm=self.maze.algorithm,
                seed=int(self.maze.seed),
            ),
            position=list(env.position),
            visited=[[x, y] for x, y in sorted(env.visited)],
            trajectory=[[x, y] for x, y in env.trajectory],
            step_count=env.step_count,
            max_steps=env.max_steps,
            last_action=int(env.last_action),
            won=env.won,
            done=env.done,
            action=out.action,
            moved=out.moved,
            hit_wall=out.hit_wall,
            probabilities=self.brain.probabilities(),
            awaiting_action=self.brain.requires_client_action and not env.done,
        )

    # ------------------------------------------------------------------ 진행
    def advance(self, steps: int = 1) -> StepOutcome:
        """Brain 이 스스로 ``steps`` 만큼 움직인다 (최대 1회 호출 = 1스텝)."""
        with self.lock:
            outcome = StepOutcome()
            for _ in range(steps):
                if self.env.done:
                    break
                action = self.brain.act(self.env)
                if action is None:
                    break
                result = self.env.step(action)
                outcome = StepOutcome(
                    action=int(action), moved=result.moved, hit_wall=result.hit_wall
                )
            self.outcome = outcome
            return outcome

    def apply(self, action: int) -> StepOutcome:
        """사람(클라이언트)이 보낸 행동을 적용한다. **벽 판정은 서버가 한다.**"""
        with self.lock:
            if self.env.done:
                self.outcome = StepOutcome()
                return self.outcome
            result = self.env.step(action)
            self.outcome = StepOutcome(
                action=int(action), moved=result.moved, hit_wall=result.hit_wall
            )
            return self.outcome

    # ------------------------------------------------------------------ 제어
    def reset(self) -> None:
        with self.lock:
            self.env.reset()
            self.brain.reset()
            self.outcome = StepOutcome()

    def new_maze(self, seed: int | None = None) -> None:
        with self.lock:
            self.maze = generate(
                self.size,
                self.size,
                seed=seed,
                algorithm=self.algorithm,
                randomize_endpoints=self.randomize_endpoints,
            )
            self.env = _make_env(self.maze, self.brain, self.max_steps)
            self.brain.reset()
            self.outcome = StepOutcome()

    def set_brain(self, kind: str, rule: str = "wall") -> None:
        """구현체 교체. 관측 정의가 달라지므로 같은 미로에서 다시 시작한다."""
        with self.lock:
            self.brain = build_brain(kind, rule, self._checkpoint)
            self.env = _make_env(self.maze, self.brain, self.max_steps)
            self.brain.reset()
            self.outcome = StepOutcome()
