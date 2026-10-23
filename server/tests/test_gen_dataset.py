"""데이터셋 생성기 테스트 — 라벨은 왼손 법칙 전문가와 일치해야 한다."""

from __future__ import annotations

import numpy as np
import pytest

from app.game.actions import DELTAS, LAST_ACTION_DIM, Action
from app.game.env import observation_dim
from app.game.expert import wall_follow_action
from app.game.maze import generate
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
        expert_episodes=1,
        noisy_episodes=1,
        noise_prob=0.08,
        max_episode_steps=256,
        out_dir=str(tmp_path_factory.mktemp("data")),
    )


def _maze_of(batch, cfg: DatasetConfig):
    seed, size, algo = int(batch.seed[0]), int(batch.size[0]), int(batch.algo[0])
    return generate(
        size,
        size,
        seed=seed,
        algorithm="dfs" if algo == 0 else "prim",
        randomize_endpoints=cfg.randomize_endpoints,
    )


def _positions(batch, size: int) -> list[tuple[int, int]]:
    """관측의 정규화된 pos 성분에서 격자 좌표를 복원한다."""
    span = size - 1
    xs = np.rint(batch.obs[:, 0] * span).astype(int)
    ys = np.rint(batch.obs[:, 1] * span).astype(int)
    return list(zip(xs.tolist(), ys.tolist()))


def _headings(batch) -> list[int]:
    start = batch.obs.shape[1] - 1 - LAST_ACTION_DIM
    return np.argmax(batch.obs[:, start : start + LAST_ACTION_DIM], axis=1).tolist()


def _taken_action(p0: tuple[int, int], p1: tuple[int, int]) -> Action:
    delta = (p1[0] - p0[0], p1[1] - p0[1])
    for action, d in DELTAS.items():
        if d == delta:
            return Action(action)
    raise AssertionError(f"non-adjacent transition: {p0} -> {p1}")


def test_sample_shapes_are_aligned(cfg):
    batch = build_maze_samples(0, cfg)
    n = len(batch.action)
    assert n > 0
    assert batch.obs.shape == (n, observation_dim(cfg.window_size, cfg.global_size))
    assert batch.seed.shape == batch.action.shape
    assert batch.kind.shape == batch.action.shape
    assert batch.algo.shape == batch.action.shape
    assert batch.action.dtype == np.int8


def test_every_label_matches_wall_expert(cfg):
    """모든 샘플의 라벨이 (위치, 직전 행동)만으로 결정되는 전문가 행동이어야 한다."""
    for index in range(cfg.mazes):
        batch = build_maze_samples(index, cfg)
        maze = _maze_of(batch, cfg)
        size = int(batch.size[0])
        positions = _positions(batch, size)
        headings = _headings(batch)

        for pos, heading, action in zip(positions, headings, batch.action):
            expected = wall_follow_action(maze, pos, heading)
            assert int(action) == int(expected), (
                f"maze {index}: label mismatch at {pos}, heading={heading}: "
                f"got {Action(int(action)).name}, expected {expected.name}"
            )


def test_both_trajectory_kinds_present(cfg):
    batch = build_maze_samples(3, cfg)
    assert (batch.kind == 0).any(), "전문가 궤적 샘플이 없음"
    assert (batch.kind == 1).any(), "noisy 궤적 샘플이 없음"


def test_expert_episode_takes_expert_action(cfg):
    """전문가 궤적의 모든 전이는 라벨과 같은 행동이어야 한다."""
    for index in range(cfg.mazes):
        batch = build_maze_samples(index, cfg)
        maze = _maze_of(batch, cfg)
        positions = _positions(batch, int(batch.size[0]))
        headings = _headings(batch)

        idx = np.where(batch.kind == 0)[0]
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                continue
            taken = _taken_action(positions[a], positions[b])
            expected = wall_follow_action(maze, positions[a], headings[a])
            assert taken == expected


def test_noisy_episode_diverges_from_expert(cfg):
    """noisy 궤적은 전문가 행동에서 벗어난 전이를 포함해야 한다 (= 복구 학습)."""
    checked = 0
    diverged = 0
    for index in range(cfg.mazes):
        batch = build_maze_samples(index, cfg)
        maze = _maze_of(batch, cfg)
        positions = _positions(batch, int(batch.size[0]))
        headings = _headings(batch)

        idx = np.where(batch.kind == 1)[0]
        if len(idx) < 2:
            continue
        mismatches = 0
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                continue
            taken = _taken_action(positions[a], positions[b])
            expected = wall_follow_action(maze, positions[a], headings[a])
            if taken != expected:
                mismatches += 1
        if mismatches > 0:
            diverged += 1
        checked += 1

    assert checked > 0, "noisy 샘플이 생성되지 않음"
    assert diverged >= max(1, checked // 2), (
        f"noisy 궤적이 전문가에서 벗어난 미로가 너무 적음: {diverged}/{checked}"
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
        expert_episodes=1,
        noisy_episodes=1,
        noise_prob=0.05,
        max_episode_steps=128,
        out_dir=str(tmp_path / "out"),
    )
    summary = generate_dataset(cfg2, workers=1)

    assert (tmp_path / "out" / "train.npz").exists()
    assert (tmp_path / "out" / "meta.json").exists()
    assert summary["splits"]["train"]["mazes"] == 8

    data = np.load(tmp_path / "out" / "train.npz")
    assert data["obs"].shape[1] == observation_dim(3, 5)
    assert set(np.unique(data["action"])) <= {0, 1, 2, 3}


def test_action_distribution_is_reasonable(tmp_path):
    """왼손 법칙은 방향 편향이 있을 수 있지만 한 클래스에 과도하게 몰리면 안 된다."""
    cfg = DatasetConfig(
        mazes=60,
        sizes=(9, 11),
        base_seed=11,
        randomize_endpoints=True,
        max_episode_steps=256,
        out_dir=str(tmp_path),
    )
    generate_dataset(cfg, workers=1)

    data = np.load(tmp_path / "train.npz")
    share = np.bincount(data["action"], minlength=4) / len(data["action"])
    assert share.max() < 0.6, f"여전히 편중됨: {np.round(share, 3).tolist()}"


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
