"""시드 기반 미로 생성기.

두 가지 알고리즘을 제공한다.

- ``dfs``  : Recursive Backtracker (기본). 길이 길게 이어져 탐색감이 좋다.
- ``prim`` : Randomized Prim. 분기가 많아 상대적으로 쉽다.

규약
----
- 격자 크기는 반드시 **홀수**이고 5 이상이다.
- 외곽 한 줄은 항상 벽이다.
- 좌표계는 ``(x, y)`` 이며 ``grid[y][x]`` 로 접근한다.
- 시작점은 좌상단 내부 ``(1, 1)``, 목적지는 우하단 내부 ``(width-2, height-2)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

WALL = 1
FLOOR = 0

Point = tuple[int, int]
Algorithm = Literal["dfs", "prim"]

#: 2칸 단위 이동 방향 (벽을 하나 건너뛰고 다음 방으로 간다)
_DIRS: tuple[Point, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_SIZE = 5


@dataclass(frozen=True)
class Maze:
    """생성된 미로. 불변 객체로 취급한다."""

    grid: np.ndarray  # (height, width) uint8, WALL=1 / FLOOR=0
    start: Point
    goal: Point
    seed: int
    algorithm: str

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        """경계 밖은 벽으로 취급한다."""
        if not self.in_bounds(x, y):
            return True
        return bool(self.grid[y, x] == WALL)

    def is_floor(self, x: int, y: int) -> bool:
        return not self.is_wall(x, y)

    def seed_hex(self) -> str:
        return f"0x{self.seed:08x}"


def _validate_size(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < MIN_SIZE:
        raise ValueError(f"{name} must be >= {MIN_SIZE}, got {value}")
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd, got {value}")


def generate(
    width: int = 15,
    height: int = 15,
    seed: int | None = None,
    algorithm: Algorithm = "dfs",
) -> Maze:
    """미로를 생성한다.

    ``seed`` 가 주어지면 항상 동일한 미로가 나온다 (재현성).
    """
    _validate_size("width", width)
    _validate_size("height", height)

    if algorithm not in ("dfs", "prim"):
        raise ValueError(f"unknown algorithm: {algorithm!r}")

    if seed is None:
        seed = random.SystemRandom().randrange(2**32)

    rng = random.Random(seed)
    grid = np.full((height, width), WALL, dtype=np.uint8)

    if algorithm == "dfs":
        _carve_dfs(grid, rng)
    else:
        _carve_prim(grid, rng)

    start: Point = (1, 1)
    goal: Point = (width - 2, height - 2)
    grid[start[1], start[0]] = FLOOR
    grid[goal[1], goal[0]] = FLOOR

    return Maze(grid=grid, start=start, goal=goal, seed=seed, algorithm=algorithm)


def _candidate_neighbors(grid: np.ndarray, x: int, y: int) -> list[Point]:
    """2칸 떨어진, 아직 벽인 내부 칸 목록."""
    h, w = grid.shape
    out: list[Point] = []
    for dx, dy in _DIRS:
        nx, ny = x + 2 * dx, y + 2 * dy
        if 1 <= nx < w - 1 and 1 <= ny < h - 1 and grid[ny, nx] == WALL:
            out.append((nx, ny))
    return out


def _carve_dfs(grid: np.ndarray, rng: random.Random) -> None:
    """Recursive Backtracker — 재귀 대신 명시적 스택을 사용한다."""
    grid[1, 1] = FLOOR
    stack: list[Point] = [(1, 1)]

    while stack:
        x, y = stack[-1]
        candidates = _candidate_neighbors(grid, x, y)
        if not candidates:
            stack.pop()
            continue

        nx, ny = rng.choice(candidates)
        # 두 칸 사이의 벽을 뚫는다
        grid[(y + ny) // 2, (x + nx) // 2] = FLOOR
        grid[ny, nx] = FLOOR
        stack.append((nx, ny))


def _carve_prim(grid: np.ndarray, rng: random.Random) -> None:
    """Randomized Prim — frontier 목록에서 무작위로 하나를 골라 확장한다."""
    h, w = grid.shape
    grid[1, 1] = FLOOR

    # (새로 뚫을 칸, 그 칸에서 이어질 기존 칸)
    frontier: list[tuple[Point, Point]] = []

    def push(x: int, y: int) -> None:
        for dx, dy in _DIRS:
            nx, ny = x + 2 * dx, y + 2 * dy
            if 1 <= nx < w - 1 and 1 <= ny < h - 1 and grid[ny, nx] == WALL:
                frontier.append(((nx, ny), (x, y)))

    push(1, 1)

    while frontier:
        idx = rng.randrange(len(frontier))
        (cx, cy), (px, py) = frontier.pop(idx)

        if grid[cy, cx] != WALL:
            continue  # 이미 다른 경로로 뚫린 칸 (중복 엔트리)

        grid[(cy + py) // 2, (cx + px) // 2] = FLOOR
        grid[cy, cx] = FLOOR
        push(cx, cy)
