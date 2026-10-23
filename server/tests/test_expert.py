"""왼손 법칙 / 지식 지도(frontier) 전문가 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from app.game.actions import DELTAS, Action
from app.game.env import FLOOR_CELL, UNKNOWN, MazeEnv
from app.game.expert import (
    INITIAL_HEADING,
    frontier_action,
    heading_from_last_action,
    turn_left,
    turn_right,
    wall_follow_action,
)
from app.game.maze import generate


def _rollout(maze, max_steps: int = 4096) -> tuple[tuple[int, int], int, list[Action]]:
    pos = maze.start
    heading: Action | None = None
    actions: list[Action] = []
    steps = 0

    while pos != maze.goal and steps < max_steps:
        action = wall_follow_action(maze, pos, heading)
        dx, dy = DELTAS[action]
        pos = (pos[0] + dx, pos[1] + dy)
        assert not maze.is_wall(*pos), f"expert walked into a wall at {pos}"
        heading = action
        actions.append(action)
        steps += 1

    return pos, steps, actions


@pytest.mark.parametrize("size", [9, 11, 15])
def test_wall_follower_solves_random_endpoint_mazes(size):
    for seed in range(8):
        maze = generate(size, size, seed=seed, randomize_endpoints=True)
        pos, steps, _ = _rollout(maze)
        assert pos == maze.goal, f"seed={seed} size={size}: did not reach the goal"
        # 트리 미로의 DFS 순회는 각 간선을 최대 2번 지난다
        assert steps <= 2 * int((maze.grid == 0).sum())


def test_wall_follower_solves_fixed_corner_mazes():
    for seed in range(8):
        maze = generate(15, 15, seed=seed, randomize_endpoints=False)
        pos, steps, _ = _rollout(maze)
        assert pos == maze.goal
        assert steps > 0


def test_wall_follow_action_is_deterministic_and_legal():
    maze = generate(11, 11, seed=5, randomize_endpoints=True)
    for pos, heading in [
        (maze.start, None),
        (maze.start, INITIAL_HEADING),
        (maze.start, Action.DOWN),
        (maze.start, 4),
    ]:
        first = wall_follow_action(maze, pos, heading)
        second = wall_follow_action(maze, pos, heading)
        assert first == second
        dx, dy = DELTAS[first]
        assert not maze.is_wall(pos[0] + dx, pos[1] + dy)


def test_heading_conversion():
    assert heading_from_last_action(None) == INITIAL_HEADING
    assert heading_from_last_action(4) == INITIAL_HEADING
    assert heading_from_last_action(0) == Action.UP
    assert heading_from_last_action(3) == Action.RIGHT


def test_turns_are_inverses():
    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
        assert turn_left(turn_right(action)) == action
        assert turn_right(action) != action


def test_expert_visits_every_cell():
    """목표와 무관하게 끝까지 돌리면 모든 바닥 칸을 지나야 한다."""
    maze = generate(9, 9, seed=2, randomize_endpoints=True)
    floor = int((maze.grid == 0).sum())

    pos = maze.start
    heading: Action | None = None
    visited = {pos}
    for _ in range(4 * floor):
        action = wall_follow_action(maze, pos, heading)
        dx, dy = DELTAS[action]
        pos = (pos[0] + dx, pos[1] + dy)
        heading = action
        visited.add(pos)
        if len(visited) == floor:
            break

    assert len(visited) == floor


# --------------------------------------------------------------- 지식 지도 전문가
def _frontier_rollout(maze, max_steps: int = 4096):
    """지식 지도만 가진 환경에서 frontier 전문가를 돌린다."""
    env = MazeEnv(
        maze,
        window_size=3,
        global_size=0,
        map_size=max(maze.width, maze.height),
        max_steps=max_steps,
    )
    env.reset()
    while not env.done:
        action = frontier_action(env.known, env.position, maze.goal)
        result = env.step(action)
        assert not result.hit_wall, f"frontier expert hit a wall at {env.position}"
    return env.won, env.step_count


@pytest.mark.parametrize("size", [9, 11, 15])
def test_frontier_expert_solves_mazes(size):
    for seed in range(8):
        maze = generate(size, size, seed=seed, randomize_endpoints=True)
        won, steps = _frontier_rollout(maze)
        assert won, f"seed={seed} size={size}: frontier expert did not reach the goal"
        assert steps > 0


def test_frontier_expert_never_moves_into_unknown_cells():
    """경계를 향해 가더라도 지식 지도상 바닥인 칸만 밟는다."""
    maze = generate(11, 11, seed=21, randomize_endpoints=True)
    env = MazeEnv(maze, window_size=3, global_size=0, map_size=11, max_steps=1024)
    env.reset()
    while not env.done:
        action = frontier_action(env.known, env.position, maze.goal)
        dx, dy = DELTAS[action]
        target = (env.position[0] + dx, env.position[1] + dy)
        assert env.known[target[1], target[0]] == FLOOR_CELL
        env.step(action)


def test_frontier_action_depends_only_on_the_belief_map():
    """관측되지 않은 구역을 바꿔도 행동이 변하지 않아야 한다 (관측의 함수)."""
    maze = generate(15, 15, seed=4, randomize_endpoints=True)
    env = MazeEnv(maze, window_size=3, global_size=0, map_size=15, max_steps=512)
    env.reset()
    for _ in range(12):
        env.step(frontier_action(env.known, env.position, maze.goal))

    baseline = frontier_action(env.known, env.position, maze.goal)

    # 지식 지도에서 아직 모르는 칸만 골라 뒤집는다 (에이전트가 볼 수 없는 정보)
    tampered = env.known.copy()
    unknown = np.argwhere(tampered == UNKNOWN)
    assert len(unknown) > 0
    for y, x in unknown[:50]:
        tampered[y, x] = FLOOR_CELL

    assert frontier_action(tampered, env.position, maze.goal) == baseline


def test_frontier_expert_beats_wall_follower_on_average():
    """같은 미로 집합에서 frontier 탐색이 왼손 법칙보다 짧게 도달한다."""
    frontier_steps, wall_steps = [], []
    for seed in range(20):
        maze = generate(15, 15, seed=700 + seed, randomize_endpoints=True)
        _, f = _frontier_rollout(maze)

        env = MazeEnv(maze, window_size=3, global_size=5, max_steps=4096)
        env.reset()
        while not env.done:
            env.step(wall_follow_action(maze, env.position, env.last_action))
        frontier_steps.append(f)
        wall_steps.append(env.step_count)

    assert np.mean(frontier_steps) < np.mean(wall_steps)
