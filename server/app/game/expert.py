"""관측만으로 계산 가능한 전문가 — 왼손 법칙(wall follower).

왜 필요한가
-----------
기존 라벨은 "전체 미로를 아는 BFS 최단 행동"이었다. 3×3 창 + 압축 방문 요약만
가진 정책에게 이 라벨은 관측으로 결정되지 않는다(unidentifiable). 그래서 모델은
"목표 쪽으로 직진"하는 근시안 정책을 배우고 벽/막다른 길에서 진동했다.

왼손 법칙은 다음 두 가지로만 결정된다.

- 현재 칸의 3×3 벽/바닥 창
- 직전 이동 방향(``last_action``)

따라서 **관측의 함수**이고, 완벽 미로(트리)에서는 목표에 반드시 도달한다.
대가는 비효율이다. 최단 경로의 2~4배를 걷는다.

우선순위
--------
왼쪽 → 직진 → 오른쪽 → 뒤. 항상 열린 칸으로 이동하므로 벽에 부딪히지 않는다.
"""

from __future__ import annotations

from collections import deque

from .actions import DELTAS, NO_ACTION_INDEX, OPPOSITE, Action
from .env import FLOOR_CELL, UNKNOWN
from .maze import Maze, Point

#: 첫 스텝(직전 행동 없음)의 가정 방향
INITIAL_HEADING = Action.UP

#: 화면 좌표계(y 아래로 증가)에서 시계 방향 회전
_CLOCKWISE: dict[Action, Action] = {
    Action.UP: Action.RIGHT,
    Action.RIGHT: Action.DOWN,
    Action.DOWN: Action.LEFT,
    Action.LEFT: Action.UP,
}
_COUNTER: dict[Action, Action] = {v: k for k, v in _CLOCKWISE.items()}


def turn_right(heading: Action) -> Action:
    return _CLOCKWISE[heading]


def turn_left(heading: Action) -> Action:
    return _COUNTER[heading]


def heading_from_last_action(last_action: int | None) -> Action:
    """``last_action`` (0..3, 또는 4=NONE) → 현재 방향."""
    if last_action is None or int(last_action) == NO_ACTION_INDEX:
        return INITIAL_HEADING
    return Action(int(last_action))


def wall_follow_action(
    maze: Maze,
    pos: Point,
    heading: Action | int | None,
) -> Action:
    """왼손 법칙에 따른 다음 행동.

    ``heading`` 은 :class:`Action`, 정수(0..4), 또는 ``None`` 을 받는다.
    ``None``/``NO_ACTION_INDEX`` 는 :data:`INITIAL_HEADING` 으로 해석한다.
    """
    h = heading_from_last_action(heading)
    x, y = pos

    for candidate in (turn_left(h), h, turn_right(h), OPPOSITE[h]):
        dx, dy = DELTAS[candidate]
        if not maze.is_wall(x + dx, y + dy):
            return candidate

    # 완전 연결 미로에서는 도달하지 않는다
    raise RuntimeError(f"isolated cell at {pos}")


# --------------------------------------------------------------------- frontier
#: 미탐색 경계를 고를 때 목표 방향을 얼마나 참작할지
DEFAULT_GOAL_WEIGHT = 0.6


def _bfs(
    known: np.ndarray,
    start: Point,
    goal: Point | None = None,
) -> tuple[dict[Point, int], dict[Point, Point]]:
    """지식 지도 위 BFS. ``known`` 은 0=unknown, 1=wall, 2=floor."""
    h, w = known.shape
    dist: dict[Point, int] = {start: 0}
    parent: dict[Point, Point] = {}
    queue = deque([start])

    while queue:
        cx, cy = queue.popleft()
        if goal is not None and (cx, cy) == goal:
            break
        for dx, dy in DELTAS.values():
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if known[ny, nx] != FLOOR_CELL:
                continue
            if (nx, ny) in dist:
                continue
            dist[(nx, ny)] = dist[(cx, cy)] + 1
            parent[(nx, ny)] = (cx, cy)
            queue.append((nx, ny))

    return dist, parent


def _first_step(parent: dict[Point, Point], start: Point, target: Point) -> Action:
    """``start`` 에서 ``target`` 까지의 BFS 경로에서 첫 행동."""
    node = target
    while True:
        prev = parent.get(node)
        if prev is None:
            raise RuntimeError(f"no path from {start} to {target}")
        if prev == start:
            delta = (node[0] - start[0], node[1] - start[1])
            for action, d in DELTAS.items():
                if d == delta:
                    return Action(action)
            raise RuntimeError(f"non-adjacent step {start} -> {node}")
        node = prev


def frontier_action(
    known: np.ndarray,
    pos: Point,
    goal: Point,
    goal_weight: float = DEFAULT_GOAL_WEIGHT,
) -> Action:
    """**지식 지도만** 보고 고르는 탐색 행동 (관측으로 계산 가능).

    - 목표를 이미 보았다 → 지식 지도 위 최단 경로의 첫 스텝
    - 아직 못 보았다 → 가장 가까운 미탐색 경계(frontier)로 가는 첫 스텝.
      경계 선택은 ``거리 + goal_weight × 목표까지 맨해튼`` 으로 정렬한다.

    ``known`` 은 에이전트가 직접 본 셀만 담은 지도이므로, 이 전문가는
    미로의 정답을 쓰지 않는다. 즉 관측의 함수다.
    """
    h, w = known.shape
    dist, parent = _bfs(known, pos)

    if known[goal[1], goal[0]] == FLOOR_CELL and goal in dist:
        return _first_step(parent, pos, goal)

    best: Point | None = None
    best_score = float("inf")
    for (cx, cy), d in dist.items():
        # 미탐색 이웃이 있는가?
        if not any(
            0 <= cx + dx < w
            and 0 <= cy + dy < h
            and known[cy + dy, cx + dx] == UNKNOWN
            for dx, dy in DELTAS.values()
        ):
            continue
        score = d + goal_weight * (abs(cx - goal[0]) + abs(cy - goal[1]))
        if score < best_score:
            best_score = score
            best = (cx, cy)

    if best is not None:
        return _first_step(parent, pos, best)

    # 이론상 도달 불가 (미로가 연결되어 있으면 항상 경계가 있다)
    raise RuntimeError(f"no frontier reachable from {pos}")
