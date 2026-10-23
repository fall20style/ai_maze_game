"""에이전트 행동 정의.

행동 인덱스는 학습 라벨(CrossEntropy target)로 그대로 사용되므로 **순서를 변경하면 안 된다**.
"""

from __future__ import annotations

from enum import IntEnum

from .maze import Point


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


NUM_ACTIONS = len(Action)

#: ``(dx, dy)`` — 화면 좌표계(y가 아래로 증가)를 따른다.
DELTAS: dict[Action, Point] = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

#: 역방향 (잘못된 되돌아가기를 판정할 때 사용)
OPPOSITE: dict[Action, Action] = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}

ACTION_NAMES: tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")

#: 학습·서빙 공통 관측 벡터에서 "직전 행동 없음"을 표현하는 인덱스
NO_ACTION_INDEX = NUM_ACTIONS
LAST_ACTION_DIM = NUM_ACTIONS + 1  # 5


def apply_action(pos: Point, action: Action) -> Point:
    dx, dy = DELTAS[action]
    return (pos[0] + dx, pos[1] + dy)
