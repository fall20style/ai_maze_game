"""게임 로직: 미로 생성, 탐색 라벨링, 관측 인코딩."""

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
from .solver import bfs_path, best_action, distance_map, is_solvable, path_actions, shortest_length

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
