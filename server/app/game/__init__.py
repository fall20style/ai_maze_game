"""게임 로직: 미로 생성, 환경, 관측 인코딩.

탐색(solver) 함수는 **지연 로딩**한다. 뷰어나 서빙 코드가 ``app.game`` 을
import 해도 경로 탐색 모듈이 메모리에 올라오지 않게 하기 위해서다
(PLAN §1 R1: 게임 클라이언트에는 경로 탐색 코드가 존재하지 않는다).
"""

from .actions import (
    ACTION_NAMES,
    DELTAS,
    LAST_ACTION_DIM,
    NO_ACTION_INDEX,
    NUM_ACTIONS,
    OPPOSITE,
    Action,
    apply_action,
)
from .env import (
    ALLOWED_GLOBAL_SIZES,
    ALLOWED_WINDOW_SIZES,
    FLOOR_CELL,
    UNKNOWN,
    WALL_CELL,
    MazeEnv,
    StepResult,
    default_max_steps,
    observation_dim,
    pack_payload,
)
from .maze import FLOOR, WALL, Maze, generate

#: ``app.game.solver`` 에서만 제공하는 이름들 (지연 로딩)
_SOLVER_NAMES = frozenset(
    {
        "bfs_path",
        "best_action",
        "distance_map",
        "is_solvable",
        "path_actions",
        "shortest_length",
    }
)


def __getattr__(name: str):
    if name in _SOLVER_NAMES:
        from . import solver

        return getattr(solver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SOLVER_NAMES)


__all__ = [
    "ACTION_NAMES",
    "ALLOWED_GLOBAL_SIZES",
    "ALLOWED_WINDOW_SIZES",
    "DELTAS",
    "FLOOR",
    "FLOOR_CELL",
    "LAST_ACTION_DIM",
    "Maze",
    "MazeEnv",
    "NO_ACTION_INDEX",
    "NUM_ACTIONS",
    "OPPOSITE",
    "Action",
    "StepResult",
    "UNKNOWN",
    "WALL",
    "WALL_CELL",
    "apply_action",
    "best_action",
    "bfs_path",
    "default_max_steps",
    "distance_map",
    "generate",
    "is_solvable",
    "observation_dim",
    "pack_payload",
    "path_actions",
    "shortest_length",
]
