"""서버 상태를 뷰어가 쓰는 형태로 감싼다.

렌더러는 ``session.maze`` / ``session.env`` / ``session.brain`` 만 안다.
이 파일은 서버 JSON 을 그 모양으로 바꾸는 **어댑터**일 뿐, 어떤 판단도 하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from viewer.api_client import ApiError, GameApi

#: 방향키 → 서버 행동 번호 (0=UP 1=DOWN 2=LEFT 3=RIGHT)
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3


class Motion:
    """셀 사이 이동을 부드럽게 보간한다 (표시 전용)."""

    def __init__(self, position: tuple[int, int]) -> None:
        self.start = (float(position[0]), float(position[1]))
        self.end = (float(position[0]), float(position[1]))
        self.t0 = 0.0
        self.duration = 0.0

    def reset(self, position: tuple[int, int]) -> None:
        self.start = self.end = (float(position[0]), float(position[1]))
        self.duration = 0.0

    def begin(
        self,
        start: tuple[float, float],
        end: tuple[int, int],
        now: float,
        duration: float,
    ) -> None:
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.t0 = now
        self.duration = max(0.0, duration)

    def position(self, now: float) -> tuple[float, float]:
        if self.duration <= 0:
            return self.end
        t = (now - self.t0) / self.duration
        if t >= 1.0:
            return self.end
        if t <= 0.0:
            return self.start
        ease = 1.0 - (1.0 - t) ** 3  # ease-out cubic
        return (
            self.start[0] + (self.end[0] - self.start[0]) * ease,
            self.start[1] + (self.end[1] - self.start[1]) * ease,
        )


@dataclass
class MazeView:
    """서버가 내려준 미로 (그리기 전용)."""

    width: int
    height: int
    grid: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    algorithm: str
    seed: int

    def is_wall(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return True
        return bool(self.grid[y, x])

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def seed_hex(self) -> str:
        return f"0x{self.seed:08x}"


@dataclass
class EnvView:
    """서버가 내려준 환경 상태 (읽기 전용)."""

    window_size: int
    position: tuple[int, int]
    visited: set[tuple[int, int]]
    trajectory: list[tuple[int, int]]
    step_count: int
    max_steps: int
    last_action: int
    won: bool
    done: bool


class BrainView:
    """서버가 고른 구현체의 라벨과 확률."""

    def __init__(self, kind: str, label: str, animation_duration: float = 0.09) -> None:
        self.kind = kind
        self.label = label
        self.animation_duration = animation_duration
        self._probs: np.ndarray | None = None

    def set_probabilities(self, values: list[float] | None) -> None:
        self._probs = None if values is None else np.asarray(values, dtype=np.float32)

    def probabilities(self) -> np.ndarray | None:
        return self._probs


@dataclass
class RemoteSession:
    """HTTP 로만 진행되는 한 판.

    ``error`` 가 설정되면 뷰어는 **이동하지 않고** 배너를 띄운다 (PLAN §1 R4).
    """

    api: GameApi
    state: dict
    step_interval: float = 0.09
    motion: Motion = field(init=False)
    flash_until: float = 0.0
    error: str | None = None
    _next_at: float = 0.0

    def __post_init__(self) -> None:
        self.maze = MazeView(
            width=self.state["maze"]["width"],
            height=self.state["maze"]["height"],
            grid=np.asarray(self.state["maze"]["grid"], dtype=np.uint8),
            start=tuple(self.state["maze"]["start"]),
            goal=tuple(self.state["maze"]["goal"]),
            algorithm=self.state["maze"]["algorithm"],
            seed=int(self.state["maze"]["seed"]),
        )
        self.env = EnvView(
            window_size=int(self.state["window_size"]),
            position=tuple(self.state["position"]),
            visited={tuple(p) for p in self.state["visited"]},
            trajectory=[tuple(p) for p in self.state["trajectory"]],
            step_count=int(self.state["step_count"]),
            max_steps=int(self.state["max_steps"]),
            last_action=int(self.state["last_action"]),
            won=bool(self.state["won"]),
            done=bool(self.state["done"]),
        )
        self.brain = BrainView(self.state["brain"], self.state["brain_label"])
        self.brain.set_probabilities(self.state.get("probabilities"))
        self.motion = Motion(self.env.position)
        self.floor_total = max(1, int((self.maze.grid == 0).sum()))

    # ------------------------------------------------------------------ 상태
    @property
    def game_id(self) -> str:
        return self.state["game_id"]

    @property
    def brain_kind(self) -> str:
        return self.state["brain"]

    @property
    def awaiting_action(self) -> bool:
        return bool(self.state.get("awaiting_action"))

    def wants_step(self, now: float) -> bool:
        if self.awaiting_action or self.env.done:
            return False
        return now >= self._next_at

    def note_step(self, now: float) -> None:
        self._next_at = now + self.step_interval

    # ------------------------------------------------------------------ 서버 호출
    def _apply(self, new_state: dict, now: float) -> None:
        """서버 응답을 화면 상태로 반영한다."""
        previous = self.env.position
        moved_to = tuple(new_state["position"])

        self.state = new_state
        self.env.position = moved_to
        self.env.visited = {tuple(p) for p in new_state["visited"]}
        self.env.trajectory = [tuple(p) for p in new_state["trajectory"]]
        self.env.step_count = int(new_state["step_count"])
        self.env.max_steps = int(new_state["max_steps"])
        self.env.last_action = int(new_state["last_action"])
        self.env.won = bool(new_state["won"])
        self.env.done = bool(new_state["done"])
        self.brain.kind = str(new_state["brain"])
        self.brain.label = str(new_state["brain_label"])
        self.brain.set_probabilities(new_state.get("probabilities"))

        if moved_to != previous:
            self.motion.begin(
                self.motion.position(now), moved_to, now, self.brain.animation_duration
            )
        if new_state.get("hit_wall"):
            self.flash_until = now + 0.22
        self.error = None

    def _call(self, func, now: float, *args, **kwargs) -> bool:
        try:
            state = func(*args, **kwargs)
        except ApiError as exc:
            self.error = str(exc)
            return False
        self._apply(state, now)
        return True

    def advance(self, now: float) -> bool:
        ok = self._call(self.api.advance, now, self.game_id)
        if ok:
            self.note_step(now)
        return ok

    def send_action(self, action: int, now: float) -> bool:
        return self._call(self.api.action, now, self.game_id, int(action))

    def set_brain(self, kind: str, rule: str, now: float) -> bool:
        ok = self._call(self.api.set_brain, now, self.game_id, kind, rule)
        if ok:
            self.brain.kind = kind
            self.motion.reset(self.env.position)
            self._next_at = 0.0
        return ok

    def restart(self, now: float) -> bool:
        ok = self._call(self.api.reset, now, self.game_id)
        if ok:
            self.motion.reset(self.env.position)
            self._next_at = 0.0
        return ok

    def new_maze(self, now: float, seed: int | None = None) -> bool:
        ok = self._call(self.api.new_maze, now, self.game_id, seed)
        if ok:
            self.maze = MazeView(
                width=self.state["maze"]["width"],
                height=self.state["maze"]["height"],
                grid=np.asarray(self.state["maze"]["grid"], dtype=np.uint8),
                start=tuple(self.state["maze"]["start"]),
                goal=tuple(self.state["maze"]["goal"]),
                algorithm=self.state["maze"]["algorithm"],
                seed=int(self.state["maze"]["seed"]),
            )
            self.floor_total = max(1, int((self.maze.grid == 0).sum()))
            self.motion.reset(self.env.position)
            self._next_at = 0.0
        return ok
