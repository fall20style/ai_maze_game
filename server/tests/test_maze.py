import numpy as np
import pytest

from app.game.maze import FLOOR, WALL, generate


@pytest.mark.parametrize("size", [5, 9, 15, 21])
@pytest.mark.parametrize("algorithm", ["dfs", "prim"])
def test_generates_requested_size(size, algorithm):
    maze = generate(size, size, seed=42, algorithm=algorithm)
    assert maze.width == size
    assert maze.height == size
    assert maze.grid.shape == (size, size)


@pytest.mark.parametrize("size", [4, 6, 0, -3])
def test_rejects_even_or_tiny_size(size):
    with pytest.raises(ValueError):
        generate(size, 15)


def test_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        generate(15, 15, seed=1, algorithm="kruskal")  # type: ignore[arg-type]


@pytest.mark.parametrize("algorithm", ["dfs", "prim"])
def test_same_seed_same_maze(algorithm):
    a = generate(15, 15, seed=1234, algorithm=algorithm)
    b = generate(15, 15, seed=1234, algorithm=algorithm)
    assert np.array_equal(a.grid, b.grid)


def test_different_seed_different_maze():
    a = generate(15, 15, seed=1)
    b = generate(15, 15, seed=2)
    assert not np.array_equal(a.grid, b.grid)


@pytest.mark.parametrize("algorithm", ["dfs", "prim"])
def test_border_is_solid_wall(algorithm):
    maze = generate(15, 15, seed=7, algorithm=algorithm)
    assert (maze.grid[0, :] == WALL).all()
    assert (maze.grid[-1, :] == WALL).all()
    assert (maze.grid[:, 0] == WALL).all()
    assert (maze.grid[:, -1] == WALL).all()


@pytest.mark.parametrize("algorithm", ["dfs", "prim"])
def test_start_and_goal_are_floor(algorithm):
    maze = generate(15, 15, seed=7, algorithm=algorithm)
    assert maze.grid[maze.start[1], maze.start[0]] == FLOOR
    assert maze.grid[maze.goal[1], maze.goal[0]] == FLOOR


@pytest.mark.parametrize("algorithm", ["dfs", "prim"])
def test_all_floor_cells_connected(algorithm):
    """미로는 완전 연결이어야 한다 (모든 바닥 칸이 하나의 컴포넌트)."""
    maze = generate(21, 21, seed=99, algorithm=algorithm)
    floor_cells = set(map(tuple, np.argwhere(maze.grid == FLOOR)[:, ::-1]))
    assert floor_cells

    stack = [maze.start]
    seen = {maze.start}
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if nxt in floor_cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    assert seen == floor_cells


def test_is_wall_treats_out_of_bounds_as_wall():
    maze = generate(9, 9, seed=3)
    assert maze.is_wall(-1, 0)
    assert maze.is_wall(0, -1)
    assert maze.is_wall(9, 4)
    assert maze.is_wall(4, 9)


def test_seed_is_recorded():
    maze = generate(9, 9, seed=2024)
    assert maze.seed == 2024
    assert maze.seed_hex() == "0x000007e8"


@pytest.mark.parametrize("algorithm", ["dfs", "prim"])
def test_randomized_endpoints_are_valid(algorithm):
    from app.game.solver import is_solvable

    for seed in range(10):
        maze = generate(15, 15, seed=seed, algorithm=algorithm, randomize_endpoints=True)
        assert maze.is_floor(*maze.start)
        assert maze.is_floor(*maze.goal)
        assert maze.start != maze.goal
        assert is_solvable(maze)


def test_randomized_endpoints_are_deterministic():
    a = generate(15, 15, seed=5, randomize_endpoints=True)
    b = generate(15, 15, seed=5, randomize_endpoints=True)
    assert a.start == b.start
    assert a.goal == b.goal


def test_randomized_endpoints_vary_across_seeds():
    starts = {generate(15, 15, seed=s, randomize_endpoints=True).start for s in range(30)}
    goals = {generate(15, 15, seed=s, randomize_endpoints=True).goal for s in range(30)}
    assert len(starts) > 3, "시작점이 거의 변하지 않음"
    assert len(goals) > 3, "목적지가 거의 변하지 않음"


def test_fixed_endpoints_are_default():
    maze = generate(15, 15, seed=1)
    assert maze.start == (1, 1)
    assert maze.goal == (13, 13)
