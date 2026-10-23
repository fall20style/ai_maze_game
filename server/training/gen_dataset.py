"""합성 데이터셋 생성기 (W1).

전략
----
1. 미로를 무작위로 생성한다 (DFS / Prim 혼합, 크기 9~21).
2. 목표에서 BFS 거리맵을 **한 번** 계산한다.
3. 두 종류의 궤적을 섞어 샘플을 뽑는다.

   - ``optimal`` : 최단 경로를 그대로 따라간다 (정상 궤적)
   - ``random``  : 방문 칸을 피하는 무작위 탐색 (실수·막다른 길 포함)

4. **모든 상태의 라벨은 "현재 칸에서 목표로 가는 최단 경로의 첫 스텝"** 이다.

왜 최단 경로만 쓰면 안 되는가
-----------------------------
정상 궤적만 학습하면 모델은 "한 번도 실수하지 않은 상태"만 보게 된다.
실제 추론에서는 모델이 스스로 실수하며 분포 밖 상태로 들어가고, 그때 복구 행동을
배운 적이 없으므로 그대로 헤맨다. 무작위 탐색 궤적에 최적 라벨을 붙이는 것은
DAgger(Ross et al., 2011)의 축소판이다.

사용
----
    python -m training.gen_dataset --mazes 200 --sizes 9,11,13 --out data/smoke
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.game.actions import DELTAS, Action
from app.game.env import MazeEnv
from app.game.maze import generate
from app.game.solver import best_action, bfs_path, distance_map

DEFAULT_SIZES: tuple[int, ...] = (9, 11, 13, 15, 17, 19, 21)
ALGORITHMS: tuple[str, ...] = ("dfs", "prim")


@dataclass(frozen=True)
class DatasetConfig:
    mazes: int = 1000
    sizes: tuple[int, ...] = DEFAULT_SIZES
    window_size: int = 3
    base_seed: int = 0
    random_episodes: int = 1
    max_episode_steps: int = 512
    out_dir: str = "data"

    def to_json(self) -> dict:
        d = asdict(self)
        d["sizes"] = list(self.sizes)
        return d


@dataclass
class MazeSamples:
    obs: np.ndarray
    action: np.ndarray
    seed: np.ndarray
    size: np.ndarray
    kind: np.ndarray  # 0=optimal, 1=random
    step: np.ndarray
    algo: np.ndarray  # 0=dfs, 1=prim


def _maze_params(index: int, cfg: DatasetConfig) -> tuple[int, int, int, str]:
    """인덱스 → (width, height, seed, algorithm). 결정적이다."""
    size = cfg.sizes[index % len(cfg.sizes)]
    algorithm = ALGORITHMS[(index // len(cfg.sizes)) % len(ALGORITHMS)]
    seed = (cfg.base_seed * 1_000_003 + index * 7919) % (2**31 - 1)
    return size, size, seed, algorithm


def build_maze_samples(index: int, cfg: DatasetConfig) -> MazeSamples:
    """미로 하나에 대한 학습 샘플을 만든다 (워커 함수 — 반드시 모듈 레벨)."""
    width, height, seed, algorithm = _maze_params(index, cfg)
    maze = generate(width, height, seed=seed, algorithm=algorithm)
    dist = distance_map(maze, maze.goal)

    path = bfs_path(maze, maze.start, maze.goal)
    if path is None:  # 완전 연결 미로에서는 발생하지 않지만 방어
        raise RuntimeError(f"unsolvable maze: seed={seed}")
    shortest = len(path) - 1

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    kinds: list[int] = []
    steps: list[int] = []

    # ---------------------------------------------------------- 1) 최단 궤적
    env = MazeEnv(maze, window_size=cfg.window_size, max_steps=max(1, shortest))
    obs = env.reset()
    t = 0
    while True:
        label = best_action(maze, env.position, dist)
        if label is None:
            break
        obs_list.append(obs)
        action_list.append(int(label))
        kinds.append(0)
        steps.append(t)
        # 최단 경로를 따르므로 라벨과 동일한 행동을 취한다
        result = env.step(label)
        obs = result.observation
        t += 1
        if result.done:
            break

    # ---------------------------------------------------------- 2) 무작위 탐색
    random_cap = min(cfg.max_episode_steps, max(32, 2 * shortest))
    for _ in range(cfg.random_episodes):
        env = MazeEnv(maze, window_size=cfg.window_size, max_steps=random_cap)
        obs = env.reset()
        t = 0
        while True:
            label = best_action(maze, env.position, dist)
            if label is None:
                break
            obs_list.append(obs)
            action_list.append(int(label))
            kinds.append(1)
            steps.append(t)

            action = _explore_action(maze, env)
            result = env.step(action)
            obs = result.observation
            t += 1
            if result.done:
                break

    n = len(action_list)
    return MazeSamples(
        obs=np.stack(obs_list).astype(np.float32) if n else np.empty((0, 0), np.float32),
        action=np.asarray(action_list, dtype=np.int8),
        seed=np.full(n, seed, dtype=np.int64),
        size=np.full(n, width, dtype=np.int16),
        kind=np.asarray(kinds, dtype=np.int8),
        step=np.asarray(steps, dtype=np.int16),
        algo=np.full(n, 0 if algorithm == "dfs" else 1, dtype=np.int8),
    )


def _explore_action(maze, env: MazeEnv) -> Action:
    """방문하지 않은 이웃을 우선하는 무작위 탐색 정책.

    경고: 목표 방향을 전혀 고려하지 않는다. 순수하게 "덜 가본 곳"으로만 간다.
    따라서 라벨(최적 행동)과 실제 행동이 자주 어긋나며, 이것이 의도된 것이다.
    """
    x, y = env.position
    unvisited: list[Action] = []
    all_moves: list[Action] = []

    for action, (dx, dy) in DELTAS.items():
        nx, ny = x + dx, y + dy
        if maze.is_wall(nx, ny):
            continue
        all_moves.append(action)
        if (nx, ny) not in env.visited:
            unvisited.append(action)

    pool = unvisited or all_moves
    if not pool:  # 고립된 칸 (발생 불가)
        return Action.UP
    # 결정적이지만 미로마다 다른 선택을 위해 위치를 해시해 인덱스로 쓴다
    return pool[(x * 31 + y * 17 + env.step_count * 7) % len(pool)]


def _concat(batches: Sequence[MazeSamples]) -> dict[str, np.ndarray]:
    if not batches:
        return {}
    non_empty = [b for b in batches if len(b.action)]
    if not non_empty:
        return {}
    return {
        "obs": np.concatenate([b.obs for b in non_empty]),
        "action": np.concatenate([b.action for b in non_empty]),
        "seed": np.concatenate([b.seed for b in non_empty]),
        "size": np.concatenate([b.size for b in non_empty]),
        "kind": np.concatenate([b.kind for b in non_empty]),
        "step": np.concatenate([b.step for b in non_empty]),
        "algo": np.concatenate([b.algo for b in non_empty]),
    }


def _split_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    """미로 단위로 train/val/test 를 나눈다 (스텝 단위 분할 금지 → 누출)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def generate_dataset(cfg: DatasetConfig, workers: int = 1) -> dict[str, object]:
    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    indices = list(range(cfg.mazes))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            batches = list(pool.map(build_maze_samples, indices, [cfg] * len(indices)))
    else:
        batches = [build_maze_samples(i, cfg) for i in indices]

    splits = _split_indices(cfg.mazes, cfg.base_seed + 1)

    summary: dict[str, object] = {
        "config": cfg.to_json(),
        "elapsed_sec": 0.0,
        "splits": {},
    }

    for name, idx in splits.items():
        part = _concat([batches[i] for i in idx])
        if not part:
            continue
        path = out / f"{name}.npz"
        np.savez_compressed(path, **part)
        counts = np.bincount(part["action"], minlength=4).tolist()
        summary["splits"][name] = {
            "file": str(path),
            "samples": int(len(part["action"])),
            "mazes": int(len(idx)),
            "action_counts": dict(zip(("UP", "DOWN", "LEFT", "RIGHT"), counts)),
            "kind_ratio": {
                "optimal": round(float((part["kind"] == 0).mean()), 3),
                "random": round(float((part["kind"] == 1).mean()), 3),
            },
        }

    summary["elapsed_sec"] = round(time.time() - t0, 2)
    (out / "meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="미로 학습용 합성 데이터셋 생성")
    p.add_argument("--mazes", type=int, default=1000, help="생성할 미로 수")
    p.add_argument("--sizes", type=str, default=",".join(map(str, DEFAULT_SIZES)))
    p.add_argument("--window-size", type=int, default=3, choices=(3, 5, 7))
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--random-episodes", type=int, default=1)
    p.add_argument("--max-episode-steps", type=int, default=512)
    p.add_argument("--out", type=str, default="data")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())
    cfg = DatasetConfig(
        mazes=args.mazes,
        sizes=sizes,
        window_size=args.window_size,
        base_seed=args.base_seed,
        random_episodes=args.random_episodes,
        max_episode_steps=args.max_episode_steps,
        out_dir=args.out,
    )
    summary = generate_dataset(cfg, workers=args.workers)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
