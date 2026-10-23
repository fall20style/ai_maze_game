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

from .actions import DELTAS, NO_ACTION_INDEX, OPPOSITE, Action
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
