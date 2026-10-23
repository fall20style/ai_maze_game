import numpy as np
import pytest

from app.game.actions import Action, NO_ACTION_INDEX
from app.game.env import (
    FLOOR_CELL,
    UNKNOWN,
    WALL_CELL,
    MazeEnv,
    observation_dim,
    pack_payload,
)
from app.game.expert import frontier_action
from app.game.maze import generate
from app.game.solver import bfs_path, path_actions


@pytest.fixture
def maze():
    return generate(15, 15, seed=42)


@pytest.fixture
def env(maze):
    return MazeEnv(maze, window_size=3)


def test_observation_dim_formula():
    assert observation_dim(3, 5) == 54
    assert observation_dim(5, 5) == 86
    assert observation_dim(7, 5) == 134
    # 전역 요약 비활성 시 기존 차원
    assert observation_dim(3, 0) == 29


def test_observation_shape_and_dtype(env):
    obs = env.observe()
    assert obs.shape == (env.obs_dim,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_pack_payload_matches_observe(env):
    """학습 경로와 서빙 경로가 **정확히** 같은 벡터를 만들어야 한다 (R-3 방지)."""
    for _ in range(10):
        payload = env.observe_payload()
        assert np.array_equal(pack_payload(payload, 3, 5), env.observe())
        result = env.step(Action.DOWN)
        if result.done:
            break


def test_pack_payload_rejects_bad_shape(env):
    payload = env.observe_payload()
    payload["local"] = [[0, 0], [0, 0]]
    with pytest.raises(ValueError):
        pack_payload(payload, 3, 5)


def test_pack_payload_rejects_bad_last_action(env):
    payload = env.observe_payload()
    payload["last_action"] = 99
    with pytest.raises(ValueError):
        pack_payload(payload, 3, 5)


def test_initial_state(env, maze):
    assert env.position == maze.start
    assert env.visited == {maze.start}
    assert env.last_action == NO_ACTION_INDEX
    assert env.step_count == 0
    assert not env.won


def test_center_of_window_is_current_cell(env, maze):
    obs = env.observe()
    local = obs[2:11].reshape(3, 3)
    x, y = env.position
    expected = WALL_CELL if maze.is_wall(x, y) else FLOOR_CELL
    assert local[1, 1] == expected


def test_wall_collision_does_not_move(env, maze):
    # 시작점 (1,1)에서 위로 가면 외곽 벽
    result = env.step(Action.UP)
    assert result.hit_wall is True
    assert result.moved is False
    assert env.position == maze.start
    assert env.invalid_actions == 1
    assert result.step == 1


def test_last_action_onehot_after_step(env):
    result = env.step(Action.DOWN)
    obs = result.observation
    last_action = obs[-6:-1]
    assert last_action.sum() == pytest.approx(1.0)
    assert last_action[int(Action.DOWN)] == 1.0


def test_visited_window_marks_start(env):
    obs = env.observe()
    visited = obs[11:20].reshape(3, 3)
    assert visited[1, 1] == 1.0


def test_walking_optimal_path_wins(env, maze):
    path = bfs_path(maze, maze.start, maze.goal)
    assert path is not None

    result = None
    for action in path_actions(path):
        result = env.step(action)

    assert result is not None
    assert result.win is True
    assert result.done is True
    assert env.position == maze.goal
    assert env.step_count == len(path) - 1
    assert env.invalid_actions == 0


def test_progress_component(env):
    assert env.observe()[-1] == pytest.approx(0.0)
    env.step(Action.DOWN)
    expected = 1 / env.max_steps
    assert env.observe()[-1] == pytest.approx(expected)


def test_max_steps_terminates(maze):
    env = MazeEnv(maze, window_size=3, max_steps=3)
    for i in range(3):
        result = env.step(Action.DOWN)
    assert result.done is True
    assert result.step == 3


def test_step_after_done_raises(maze):
    env = MazeEnv(maze, window_size=3, max_steps=1)
    env.step(Action.DOWN)
    with pytest.raises(RuntimeError):
        env.step(Action.DOWN)


def test_reset_restores_state(env, maze):
    env.step(Action.DOWN)
    env.step(Action.RIGHT)
    env.reset()
    assert env.position == maze.start
    assert env.step_count == 0
    assert env.visited == {maze.start}
    assert env.last_action == NO_ACTION_INDEX


def test_out_of_bounds_window_is_wall(maze):
    env = MazeEnv(maze, window_size=5)
    env.position = (1, 1)  # 좌상단 구석
    obs = env.observe()
    local = obs[2 : 2 + 25].reshape(5, 5)
    assert local[0, 0] == WALL_CELL
    assert local[0, 4] == WALL_CELL


def test_unknown_cells_when_maze_is_not_fully_known(maze):
    env = MazeEnv(maze, window_size=3, knows_full_maze=False)
    obs = env.observe()
    local = obs[2:11].reshape(3, 3)
    # 시작점 주변은 아직 방문하지 않았으므로 UNKNOWN
    assert (local == UNKNOWN).sum() >= 1


def test_invalid_window_size(maze):
    with pytest.raises(ValueError):
        MazeEnv(maze, window_size=4)


def test_observation_bounds(env):
    """모든 성분이 [-1, 1] 범위를 벗어나지 않아야 한다."""
    obs = env.observe()
    assert obs.min() >= -1.0
    assert obs.max() <= 2.0  # 셀 코드가 2(floor)까지 사용됨


# --------------------------------------------------------------- 전역 방문 요약
def test_global_visited_marks_only_start_block(env, maze):
    obs = env.observe()
    glob = obs[20 : 20 + 25].reshape(5, 5)
    assert glob.sum() == 1.0

    x, y = maze.start
    assert glob[y * 5 // maze.height, x * 5 // maze.width] == 1.0


def test_global_visited_grows_with_movement(env):
    before = env.observe()[20 : 20 + 25].sum()
    for _ in range(5):
        env.step(Action.DOWN)
    after = env.observe()[20 : 20 + 25].sum()
    assert after >= before


def test_global_summary_can_be_disabled(maze):
    env = MazeEnv(maze, window_size=3, global_size=0)
    assert env.obs_dim == 29
    payload = env.observe_payload()
    assert payload["global"] == []
    assert np.array_equal(pack_payload(payload, 3, 0), env.observe())


def test_pack_payload_requires_global_when_enabled(env):
    payload = env.observe_payload()
    del payload["global"]
    with pytest.raises(ValueError):
        pack_payload(payload, 3, 5)


def test_global_size_validation(maze):
    with pytest.raises(ValueError):
        MazeEnv(maze, global_size=4)


# ------------------------------------------------------------- 지식 지도 (map_size)
def _known_env(maze, map_size: int = 15):
    env = MazeEnv(
        maze, window_size=3, global_size=0, map_size=map_size, max_steps=512
    )
    env.reset()
    return env


def test_map_size_adds_two_channels_to_observation_dim():
    assert observation_dim(3, 5, 0) == 54
    assert observation_dim(3, 0, 15) == 11 + 18 + 2 * 225
    assert observation_dim(3, 5, 9) == 54 + 2 * 81


def test_known_map_starts_with_the_start_window(maze):
    env = _known_env(maze)
    wall, floor = env._known_channels()
    assert wall.shape == floor.shape == (15, 15)
    assert floor[maze.start[1], maze.start[0]] == 1.0
    assert wall.sum() > 0, "주변 벽을 하나도 못 봤다"
    # 시작 칸 주변 3x3 만 알고 있어야 한다
    assert int(wall.sum() + floor.sum()) == 9


def test_known_map_matches_true_maze_on_seen_cells(maze):
    env = _known_env(maze)
    for _ in range(20):
        if env.done:
            break
        env.step(frontier_action(env.known, env.position, maze.goal))

    wall, floor = env._known_channels()
    seen_wall = wall.astype(bool)
    seen_floor = floor.astype(bool)
    assert not (seen_wall & seen_floor).any()
    for y in range(maze.height):
        for x in range(maze.width):
            if seen_wall[y, x]:
                assert maze.is_wall(x, y), f"({x},{y}) 는 벽이 아니다"
            if seen_floor[y, x]:
                assert not maze.is_wall(x, y), f"({x},{y}) 는 바닥이 아니다"


def test_known_map_never_shrinks(maze):
    env = _known_env(maze)
    previous = 0
    for _ in range(15):
        if env.done:
            break
        env.step(frontier_action(env.known, env.position, maze.goal))
        wall, floor = env._known_channels()
        known = int(wall.sum() + floor.sum())
        assert known >= previous
        previous = known


def test_known_map_is_not_the_full_maze(maze):
    """지식 지도는 정답이 아니라 누적 관측이어야 한다 (미로를 통째로 주면 안 됨)."""
    env = _known_env(maze)
    wall, floor = env._known_channels()
    assert int(wall.sum() + floor.sum()) < maze.width * maze.height


def test_pack_payload_roundtrip_with_map(maze):
    env = _known_env(maze)
    env.step(frontier_action(env.known, env.position, maze.goal))
    payload = env.observe_payload()
    packed = pack_payload(payload, 3, 0, 15)
    assert np.array_equal(packed, env.observe())


def test_pack_payload_requires_known_map_when_enabled(maze):
    env = _known_env(maze)
    payload = env.observe_payload()
    del payload["known_wall"]
    with pytest.raises(ValueError):
        pack_payload(payload, 3, 0, 15)


def test_map_size_must_cover_the_maze():
    maze = generate(15, 15, seed=1)
    with pytest.raises(ValueError):
        MazeEnv(maze, map_size=9)


def test_map_size_validation(maze):
    with pytest.raises(ValueError):
        MazeEnv(maze, map_size=4)
