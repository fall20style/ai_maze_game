"""왼손 법칙 전문가 테스트."""

from __future__ import annotations

import pytest

from app.game.actions import DELTAS, Action
from app.game.expert import (
    INITIAL_HEADING,
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
