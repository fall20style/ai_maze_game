"""게임 환경 + 관측 인코더.

**학습과 서빙이 이 파일 하나를 공유한다.** 관측 정의가 두 곳에 흩어지면
학습/서빙 불일치(R-3)로 성능이 조용히 망가지므로, 벡터 생성 경로를 단일화한다.

관측 벡터 (``window_size=3`` → 29차원)
--------------------------------------
======  ====  ==============================================
구간     차원  내용
======  ====  ==============================================
pos        2  현재 위치 ``x/(W-1), y/(H-1)``
local     w²  주변 창의 셀 코드 (0=unknown, 1=wall, 2=floor)
visited   w²  주변 창의 방문 여부 (0/1)
goal_rel   2  목적지까지 상대 벡터, 각 축 ``/(W-1)``, ``/(H-1)``
dist       1  맨해튼 거리 ``/(W+H-2)``
last_act   5  직전 행동 one-hot (UP/DOWN/LEFT/RIGHT/NONE)
progress   1  ``step / max_steps``
======  ====  ==============================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .actions import LAST_ACTION_DIM, NO_ACTION_INDEX, Action, apply_action
from .maze import Maze, Point

# --- 셀 코드 (관측용) ---
UNKNOWN = 0
WALL_CELL = 1
FLOOR_CELL = 2

#: 관측 창은 홀수여야 중심 칸이 존재한다
ALLOWED_WINDOW_SIZES = (3, 5, 7)

#: 고정 차원 필드 합계 (pos 2 + goal_rel 2 + dist 1 + last_action 5 + progress 1)
_FIXED_DIM = 2 + 2 + 1 + LAST_ACTION_DIM + 1


def observation_dim(window_size: int = 3) -> int:
    """관측 벡터 차원. ``window_size=3`` 이면 29."""
    if window_size not in ALLOWED_WINDOW_SIZES:
        raise ValueError(f"window_size must be one of {ALLOWED_WINDOW_SIZES}, got {window_size}")
    return _FIXED_DIM + 2 * window_size * window_size


def default_max_steps(width: int, height: int) -> int:
    """에피소드 최대 스텝. 최단 경로 대비 충분히 여유를 둔다."""
    return max(64, 3 * width * height)


@dataclass
class StepResult:
    observation: np.ndarray
    action: Action
    position: Point
    moved: bool
    hit_wall: bool
    win: bool
    done: bool
    step: int


class MazeEnv:
    """미로 위의 단일 에이전트 환경.

    서버에서는 세션당 하나씩 유지되고, 학습에서는 미로마다 새로 만든다.
    """

    def __init__(
        self,
        maze: Maze,
        window_size: int = 3,
        max_steps: int | None = None,
        knows_full_maze: bool = True,
    ) -> None:
        if window_size not in ALLOWED_WINDOW_SIZES:
            raise ValueError(f"window_size must be one of {ALLOWED_WINDOW_SIZES}")
        self.maze = maze
        self.window_size = window_size
        self.max_steps = max_steps or default_max_steps(maze.width, maze.height)
        self.knows_full_maze = knows_full_maze

        self.position: Point = maze.start
        self.visited: set[Point] = set()
        self.last_action: int = NO_ACTION_INDEX
        self.step_count: int = 0
        self.invalid_actions: int = 0
        self.trajectory: list[Point] = []

        self.reset()

    # ------------------------------------------------------------------ 상태
    def reset(self) -> np.ndarray:
        self.position = self.maze.start
        self.visited = {self.maze.start}
        self.last_action = NO_ACTION_INDEX
        self.step_count = 0
        self.invalid_actions = 0
        self.trajectory = [self.maze.start]
        return self.observe()

    @property
    def obs_dim(self) -> int:
        return observation_dim(self.window_size)

    @property
    def won(self) -> bool:
        return self.position == self.maze.goal

    @property
    def done(self) -> bool:
        return self.won or self.step_count >= self.max_steps

    # ------------------------------------------------------------------ 행동
    def step(self, action: Action | int) -> StepResult:
        """행동을 적용한다.

        벽으로의 이동은 **이동하지 않고** 스텝만 소모한다 (``hit_wall=True``).
        """
        if self.done:
            raise RuntimeError("episode is over; call reset() first")

        action = Action(int(action))
        prev = self.position
        target = apply_action(prev, action)

        if self.maze.is_wall(*target):
            self.invalid_actions += 1
            moved = False
            hit_wall = True
        else:
            self.position = target
            self.visited.add(target)
            self.trajectory.append(target)
            moved = True
            hit_wall = False

        self.last_action = int(action)
        self.step_count += 1

        win = self.won
        done = win or self.step_count >= self.max_steps

        return StepResult(
            observation=self.observe(),
            action=action,
            position=self.position,
            moved=moved,
            hit_wall=hit_wall,
            win=win,
            done=done,
            step=self.step_count,
        )

    # ------------------------------------------------------------------ 관측
    def observe(self) -> np.ndarray:
        """학습·서빙 공통 관측 벡터 (float32)."""
        maze = self.maze
        w, h = maze.width, maze.height
        r = self.window_size // 2
        x, y = self.position

        local = np.empty((self.window_size, self.window_size), dtype=np.float32)
        visited = np.zeros((self.window_size, self.window_size), dtype=np.float32)

        for row, dy in enumerate(range(-r, r + 1)):
            for col, dx in enumerate(range(-r, r + 1)):
                cx, cy = x + dx, y + dy
                if not maze.in_bounds(cx, cy):
                    local[row, col] = WALL_CELL
                    continue
                if not self.knows_full_maze and (cx, cy) not in self.visited:
                    local[row, col] = UNKNOWN
                else:
                    local[row, col] = WALL_CELL if maze.is_wall(cx, cy) else FLOOR_CELL
                visited[row, col] = 1.0 if (cx, cy) in self.visited else 0.0

        gx, gy = maze.goal
        sx = w - 1 if w > 1 else 1
        sy = h - 1 if h > 1 else 1
        span = max(1, (w - 1) + (h - 1))
        manhattan = abs(gx - x) + abs(gy - y)

        last_action = np.zeros(LAST_ACTION_DIM, dtype=np.float32)
        last_action[self.last_action] = 1.0

        return np.concatenate(
            [
                np.array([x / sx, y / sy], dtype=np.float32),
                local.ravel(),
                visited.ravel(),
                np.array([(gx - x) / sx, (gy - y) / sy], dtype=np.float32),
                np.array([manhattan / span], dtype=np.float32),
                last_action,
                np.array([min(1.0, self.step_count / self.max_steps)], dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)

    def observe_payload(self) -> dict[str, Any]:
        """API/디버그용 구조화 관측.

        클라이언트는 이 값을 그대로 echo 하고, 서버는 :func:`pack_payload` 로
        동일한 벡터를 복원한다. 두 경로가 일치하는지는 테스트로 보증한다.
        """
        maze = self.maze
        w, h = maze.width, maze.height
        r = self.window_size // 2
        x, y = self.position

        local: list[list[int]] = []
        visited: list[list[int]] = []
        for dy in range(-r, r + 1):
            local_row: list[int] = []
            visited_row: list[int] = []
            for dx in range(-r, r + 1):
                cx, cy = x + dx, y + dy
                if not maze.in_bounds(cx, cy):
                    local_row.append(WALL_CELL)
                elif not self.knows_full_maze and (cx, cy) not in self.visited:
                    local_row.append(UNKNOWN)
                else:
                    local_row.append(WALL_CELL if maze.is_wall(cx, cy) else FLOOR_CELL)
                visited_row.append(1 if (cx, cy) in self.visited else 0)
            local.append(local_row)
            visited.append(visited_row)

        gx, gy = maze.goal
        sx = max(1, w - 1)
        sy = max(1, h - 1)
        span = max(1, (w - 1) + (h - 1))
        manhattan = abs(gx - x) + abs(gy - y)

        return {
            "pos": [x / sx, y / sy],
            "local": local,
            "visited": visited,
            "goal_rel": [(gx - x) / sx, (gy - y) / sy],
            "dist": manhattan / span,
            "last_action": int(self.last_action),
            "progress": min(1.0, self.step_count / self.max_steps),
        }

    # ------------------------------------------------------------------ 기타
    def state_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "visited_count": len(self.visited),
            "step": self.step_count,
            "max_steps": self.max_steps,
            "last_action": int(self.last_action),
            "invalid_actions": self.invalid_actions,
            "won": self.won,
            "done": self.done,
        }


# --------------------------------------------------------------------- packer
def pack_payload(payload: Mapping[str, Any], window_size: int = 3) -> np.ndarray:
    """구조화 관측(클라이언트 전송분)을 벡터로 변환한다.

    ``MazeEnv.observe()`` 와 **비트 단위로 동일한** 결과를 내야 한다.
    """
    expected = observation_dim(window_size)
    pos = np.asarray(payload["pos"], dtype=np.float32)
    local = np.asarray(payload["local"], dtype=np.float32)
    visited = np.asarray(payload["visited"], dtype=np.float32)
    goal_rel = np.asarray(payload["goal_rel"], dtype=np.float32)

    shape = (window_size, window_size)
    if pos.shape != (2,) or goal_rel.shape != (2,):
        raise ValueError("pos and goal_rel must have shape (2,)")
    if local.shape != shape or visited.shape != shape:
        raise ValueError(f"local/visited must have shape {shape}")

    last_action_idx = int(payload["last_action"])
    if not 0 <= last_action_idx < LAST_ACTION_DIM:
        raise ValueError(f"last_action out of range: {last_action_idx}")

    last_action = np.zeros(LAST_ACTION_DIM, dtype=np.float32)
    last_action[last_action_idx] = 1.0

    vector = np.concatenate(
        [
            pos,
            local.ravel(),
            visited.ravel(),
            goal_rel,
            np.array([float(payload["dist"])], dtype=np.float32),
            last_action,
            np.array([float(payload["progress"])], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)

    if vector.shape != (expected,):
        raise ValueError(f"observation dim mismatch: {vector.shape} != {(expected,)}")
    return vector
