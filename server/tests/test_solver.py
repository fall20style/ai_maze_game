import numpy as np
import pytest

from app.game.actions import DELTAS
from app.game.maze import generate
from app.game.solver import (
    UNREACHABLE,
    bfs_path,
    best_action,
    distance_map,
    is_solvable,
    path_actions,
    shortest_length,
)

SEEDS = [1, 7, 42, 1234]
ALGOS = ["dfs", "prim"]


@pytest.mark.parametrize("algorithm", ALGOS)
@pytest.mark.parametrize("seed", SEEDS)
def test_every_maze_is_solvable(seed, algorithm):
    maze = generate(15, 15, seed=seed, algorithm=algorithm)
    assert is_solvable(maze)
    assert shortest_length(maze) > 0


@pytest.mark.parametrize("algorithm", ALGOS)
@pytest.mark.parametrize("seed", SEEDS)
def test_path_is_contiguous_and_ends_correctly(seed, algorithm):
    maze = generate(15, 15, seed=seed, algorithm=algorithm)
    path = bfs_path(maze, maze.start, maze.goal)
    assert path is not None
    assert path[0] == maze.start
    assert path[-1] == maze.goal

    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        assert abs(x1 - x0) + abs(y1 - y0) == 1
        assert not maze.is_wall(x1, y1)


@pytest.mark.parametrize("seed", SEEDS)
def test_path_length_matches_distance_map(seed):
    maze = generate(21, 21, seed=seed)
    path = bfs_path(maze, maze.start, maze.goal)
    dist = distance_map(maze, maze.goal)
    assert path is not None
    assert len(path) - 1 == int(dist[maze.start[1], maze.start[0]])


def test_distance_map_is_symmetric():
    maze = generate(15, 15, seed=11)
    a = distance_map(maze, maze.start)
    b = distance_map(maze, maze.goal)
    assert int(a[maze.goal[1], maze.goal[0]]) == int(b[maze.start[1], maze.start[0]])


def test_walls_are_unreachable_in_distance_map():
    maze = generate(15, 15, seed=5)
    dist = distance_map(maze, maze.goal)
    assert (dist[maze.grid == 1] == UNREACHABLE).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_best_action_always_descends(seed):
    """최적 행동은 거리를 정확히 1 줄여야 한다."""
    maze = generate(15, 15, seed=seed)
    dist = distance_map(maze, maze.goal)

    for y in range(maze.height):
        for x in range(maze.width):
            if maze.is_wall(x, y):
                continue
            d = int(dist[y, x])
            if d <= 0:
                continue
            action = best_action(maze, (x, y), dist)
            assert action is not None, f"no action at {(x, y)} (dist={d})"
            dx, dy = DELTAS[action]
            assert int(dist[y + dy, x + dx]) == d - 1


def test_best_action_none_at_goal():
    maze = generate(9, 9, seed=8)
    dist = distance_map(maze, maze.goal)
    assert best_action(maze, maze.goal, dist) is None


def test_path_actions_roundtrip():
    maze = generate(15, 15, seed=17)
    path = bfs_path(maze, maze.start, maze.goal)
    assert path is not None

    actions = path_actions(path)
    assert len(actions) == len(path) - 1

    x, y = maze.start
    for action in actions:
        dx, dy = DELTAS[action]
        x, y = x + dx, y + dy
    assert (x, y) == maze.goal


def test_bfs_path_trivial_when_start_equals_goal():
    maze = generate(9, 9, seed=1)
    assert bfs_path(maze, maze.start, maze.start) == [maze.start]


def test_greedy_following_best_action_reaches_goal():
    """best_action 만으로도 목표에 도달해야 한다 (라벨러 자체 검증)."""
    maze = generate(21, 21, seed=2024)
    dist = distance_map(maze, maze.goal)
    pos = maze.start
    for _ in range(maze.width * maze.height):
        if pos == maze.goal:
            break
        action = best_action(maze, pos, dist)
        assert action is not None
        dx, dy = DELTAS[action]
        pos = (pos[0] + dx, pos[1] + dy)
    assert pos == maze.goal


def test_distance_map_dtype():
    maze = generate(9, 9, seed=2)
    assert distance_map(maze, maze.goal).dtype == np.int32
