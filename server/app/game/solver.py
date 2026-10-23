"""미로 탐색 — **학습 라벨 생성 전용** 모듈.

⚠️ 서빙 API에서는 이 모듈을 절대 호출하지 않는다.
   (게임의 길찾기 지능은 전부 신경망 안에 있어야 한다 — PLAN §1 R1)

제공 기능
---------
- :func:`bfs_path`            : 시작→목표 최단 경로 (궤적 롤아웃용)
- :func:`distance_map`        : 목표에서 모든 칸까지의 거리 (라벨 조회 O(1))
- :func:`best_action`         : 임의 칸에서 목표로 향하는 최적 첫 행동

라벨링 시 ``distance_map`` 을 한 번만 계산하면 모든 상태의 정답을 O(1)로 얻을 수 있다.
상태마다 BFS를 다시 돌리면 미로당 O(N²) 이 되어 데이터 생성이 불가능할 정도로 느려진다.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .actions import Action, DELTAS
from .maze import Maze, Point

#: 도달 불가 칸 표시
UNREACHABLE = -1


def neighbors(maze: Maze, pos: Point):
    """벽이 아닌 인접 칸을 (행동, 좌표) 쌍으로 반환한다."""
    x, y = pos
    for action, (dx, dy) in DELTAS.items():
        nx, ny = x + dx, y + dy
        if not maze.is_wall(nx, ny):
            yield action, (nx, ny)


def distance_map(maze: Maze, source: Point) -> np.ndarray:
    """BFS 1회로 ``source`` 로부터의 최단 거리 격자를 만든다.

    반환값은 ``(height, width) int32``, 도달 불가는 ``-1``.
    """
    dist = np.full((maze.height, maze.width), UNREACHABLE, dtype=np.int32)
    sx, sy = source
    dist[sy, sx] = 0
    queue: deque[Point] = deque([source])

    while queue:
        x, y = queue.popleft()
        d = dist[y, x] + 1
        for _, (nx, ny) in neighbors(maze, (x, y)):
            if dist[ny, nx] == UNREACHABLE:
                dist[ny, nx] = d
                queue.append((nx, ny))

    return dist


def bfs_path(maze: Maze, start: Point, goal: Point) -> list[Point] | None:
    """최단 경로를 ``[start, ..., goal]`` 로 반환한다. 도달 불가면 ``None``."""
    if start == goal:
        return [start]

    parent: dict[Point, Point | None] = {start: None}
    queue: deque[Point] = deque([start])

    while queue:
        pos = queue.popleft()
        for _, nxt in neighbors(maze, pos):
            if nxt in parent:
                continue
            parent[nxt] = pos
            if nxt == goal:
                path: list[Point] = [goal]
                cur = pos
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            queue.append(nxt)

    return None


def path_actions(path: list[Point]) -> list[Action]:
    """경로를 연속된 행동 시퀀스로 변환한다."""
    out: list[Action] = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        dx, dy = x1 - x0, y1 - y0
        for action, (adx, ady) in DELTAS.items():
            if (adx, ady) == (dx, dy):
                out.append(action)
                break
        else:
            raise ValueError(f"non-adjacent step: {(x0, y0)} -> {(x1, y1)}")
    return out


def best_action(maze: Maze, pos: Point, dist: np.ndarray) -> Action | None:
    """``dist``(= 목표로부터의 거리 격자)를 이용해 최적 첫 행동을 고른다.

    현재 칸보다 거리가 1 작은 이웃으로 이동하는 것이 곧 최단 경로의 첫 스텝이다.
    현재 칸의 거리가 ``UNREACHABLE`` 이거나 이미 목표면 ``None``.
    """
    x, y = pos
    d = int(dist[y, x])
    if d <= 0:
        return None

    for action, (dx, dy) in DELTAS.items():
        nx, ny = x + dx, y + dy
        if not maze.is_wall(nx, ny) and int(dist[ny, nx]) == d - 1:
            return action
    return None


def is_solvable(maze: Maze) -> bool:
    """시작점에서 목적지로 도달 가능한지 확인한다."""
    return int(distance_map(maze, maze.goal)[maze.start[1], maze.start[0]]) >= 0


def shortest_length(maze: Maze) -> int:
    """최단 경로의 이동 횟수. 도달 불가면 ``-1``."""
    d = int(distance_map(maze, maze.goal)[maze.start[1], maze.start[0]])
    return d
