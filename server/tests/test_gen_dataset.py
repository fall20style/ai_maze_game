import numpy as np
import pytest

from app.game.actions import DELTAS
from app.game.env import observation_dim
from app.game.maze import generate
from app.game.solver import best_action, distance_map
from training.gen_dataset import (
    DatasetConfig,
    _split_indices,
    build_maze_samples,
    generate_dataset,
)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return DatasetConfig(
        mazes=6,
        sizes=(9, 11),
        base_seed=0,
        random_episodes=1,
        max_episode_steps=128,
        out_dir=str(tmp_path_factory.mktemp("data")),
    )


def test_sample_shapes_are_aligned(cfg):
    batch = build_maze_samples(0, cfg)
    n = len(batch.action)
    assert n > 0
    assert batch.obs.shape == (n, observation_dim(cfg.window_size))
    assert batch.seed.shape == batch.action.shape
    assert batch.kind.shape == batch.action.shape
    assert batch.algo.shape == batch.action.shape
    assert batch.action.dtype == np.int8


def test_every_label_descends_toward_goal(cfg):
    """모든 라벨은 목표 거리를 정확히 1 줄이는 행동이어야 한다."""
    from app.game.actions import Action

    for index in range(cfg.mazes):
        batch = build_maze_samples(index, cfg)
        # 한 배치는 하나의 미로에서만 나온다
        seed, size, algo = int(batch.seed[0]), int(batch.size[0]), int(batch.algo[0])
        maze = generate(
            size,
            size,
            seed=seed,
            algorithm="dfs" if algo == 0 else "prim",
            randomize_endpoints=cfg.randomize_endpoints,
        )
        dist = distance_map(maze, maze.goal)
        sx, sy = maze.width - 1, maze.height - 1

        checked = 0
        for obs, action in zip(batch.obs, batch.action):
            x = int(round(float(obs[0]) * sx))
            y = int(round(float(obs[1]) * sy))
            d = int(dist[y, x])
            if d <= 0:
                continue
            dx, dy = DELTAS[Action(int(action))]
            assert int(dist[y + dy, x + dx]) == d - 1, (
                f"label not optimal at {(x, y)}: action={Action(int(action)).name}"
            )
            checked += 1
            if checked >= 30:
                break
        assert checked > 0


def test_both_trajectory_kinds_present(cfg):
    batch = build_maze_samples(3, cfg)
    assert (batch.kind == 0).any(), "최단 궤적 샘플이 없음"
    assert (batch.kind == 1).any(), "무작위 탐색 샘플이 없음"


def _positions(batch, size: int) -> list[tuple[int, int]]:
    """관측의 정규화된 pos 성분에서 격자 좌표를 복원한다."""
    span = size - 1
    xs = np.rint(batch.obs[:, 0] * span).astype(int)
    ys = np.rint(batch.obs[:, 1] * span).astype(int)
    return list(zip(xs.tolist(), ys.tolist()))


def _taken_action(p0: tuple[int, int], p1: tuple[int, int]):
    from app.game.actions import Action

    delta = (p1[0] - p0[0], p1[1] - p0[1])
    for action, d in DELTAS.items():
        if d == delta:
            return Action(action)
    raise AssertionError(f"non-adjacent transition: {p0} -> {p1}")


def test_optimal_rollout_follows_best_action(cfg):
    """최단 궤적의 모든 전이는 최적 행동이어야 한다."""
    from app.game.actions import Action  # noqa: F401

    for index in range(cfg.mazes):
        batch = build_maze_samples(index, cfg)
        seed, size, algo = int(batch.seed[0]), int(batch.size[0]), int(batch.algo[0])
        maze = generate(
            size,
            size,
            seed=seed,
            algorithm="dfs" if algo == 0 else "prim",
            randomize_endpoints=cfg.randomize_endpoints,
        )
        dist = distance_map(maze, maze.goal)

        idx = np.where(batch.kind == 0)[0]
        pos = _positions(batch, size)
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                continue
            taken = _taken_action(pos[a], pos[b])
            expected = best_action(maze, pos[a], dist)
            assert expected is not None
            assert taken == expected, (
                f"maze {index}: optimal rollout took {taken.name} at {pos[a]}, "
                f"expected {expected.name}"
            )


def test_random_rollout_takes_suboptimal_actions(cfg):
    """무작위 궤적은 최적 행동과 다른 수를 섞어야 한다 (= 분포 확장).

    이 비최적 상태가 "실수 후 복구"를 학습시키는 핵심 샘플이다.
    """
    checked = 0
    diverged = 0
    for index in range(cfg.mazes):
        batch = build_maze_samples(index, cfg)
        seed, size, algo = int(batch.seed[0]), int(batch.size[0]), int(batch.algo[0])
        maze = generate(
            size,
            size,
            seed=seed,
            algorithm="dfs" if algo == 0 else "prim",
            randomize_endpoints=cfg.randomize_endpoints,
        )
        dist = distance_map(maze, maze.goal)

        idx = np.where(batch.kind == 1)[0]
        if len(idx) < 2:
            continue
        pos = _positions(batch, size)

        mismatches = 0
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                continue
            taken = _taken_action(pos[a], pos[b])
            expected = best_action(maze, pos[a], dist)
            if expected is not None and taken != expected:
                mismatches += 1

        if mismatches > 0:
            diverged += 1
        checked += 1

    assert checked > 0, "무작위 샘플이 생성되지 않음"
    assert diverged >= max(1, checked // 2), (
        f"무작위 궤적이 최적 행동에서 벗어난 미로가 너무 적음: {diverged}/{checked}"
    )


def test_split_is_disjoint_and_complete():
    splits = _split_indices(100, seed=1)
    all_idx = np.concatenate([splits["train"], splits["val"], splits["test"]])
    assert len(all_idx) == len(set(all_idx.tolist())) == 100
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_generate_dataset_writes_files(cfg, tmp_path):
    cfg2 = DatasetConfig(
        mazes=10,
        sizes=(9,),
        base_seed=3,
        random_episodes=1,
        max_episode_steps=64,
        out_dir=str(tmp_path / "out"),
    )
    summary = generate_dataset(cfg2, workers=1)

    assert (tmp_path / "out" / "train.npz").exists()
    assert (tmp_path / "out" / "meta.json").exists()
    assert summary["splits"]["train"]["mazes"] == 8

    data = np.load(tmp_path / "out" / "train.npz")
    assert data["obs"].shape[1] == observation_dim(3)
    assert set(np.unique(data["action"])) <= {0, 1, 2, 3}


def test_randomized_endpoints_balance_action_distribution(tmp_path):
    """시작/목적지 무작위화 후 행동 분포가 균형을 이루는지 확인 (A안 검증)."""
    cfg = DatasetConfig(
        mazes=60,
        sizes=(9, 11),
        base_seed=11,
        randomize_endpoints=True,
        out_dir=str(tmp_path),
    )
    generate_dataset(cfg, workers=1)

    data = np.load(tmp_path / "train.npz")
    share = np.bincount(data["action"], minlength=4) / len(data["action"])
    assert share.max() < 0.55, f"여전히 편중됨: {np.round(share, 3).tolist()}"


def test_no_maze_leaks_between_splits(tmp_path):
    """미로 시드는 split 간에 절대 겹치지 않아야 한다."""
    cfg = DatasetConfig(mazes=10, sizes=(9,), base_seed=7, out_dir=str(tmp_path))
    generate_dataset(cfg, workers=1)

    seeds = {
        name: set(np.load(tmp_path / f"{name}.npz")["seed"].tolist())
        for name in ("train", "val", "test")
    }
    assert not (seeds["train"] & seeds["val"])
    assert not (seeds["train"] & seeds["test"])
    assert not (seeds["val"] & seeds["test"])
