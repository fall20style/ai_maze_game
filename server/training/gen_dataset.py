"""합성 데이터셋 생성기.

전략
----
1. 미로를 무작위로 생성한다 (DFS / Prim 혼합, 크기 9~21).
2. **왼손 법칙(wall follower)** 전문가로 궤적을 만든다.
3. 모든 상태의 라벨은 "그 상태에서 전문가가 고를 행동"이다.

왜 BFS 최단 행동이 아니라 왼손 법칙인가
--------------------------------------
BFS 최단 행동은 **전체 미로를 아는** 전문가의 라벨이다. 3×3 창 + 압축 방문 요약만
가진 정책에게는 관측으로 결정되지 않는(unidentifiable) 목표이므로, 모델은
"목표 쪽으로 직진"하는 근시안 정책을 배우고 벽/막다른 길에서 진동한다
(실측: 단일 스텝 정확도 76%, 롤아웃 성공률 27%).

왼손 법칙은 (현재 칸의 3×3 벽/바닥, 직전 이동 방향)만으로 결정된다.
즉 **관측의 함수**이고, 완벽 미로에서는 목표에 반드시 도달한다.
대가는 비효율(최단 경로의 2~4배)이다. 성공률을 우선한다.

두 종류의 궤적
--------------
- ``expert`` (kind=0): 노이즈 없이 전문가를 그대로 따라간다.
- ``noisy``  (kind=1): ``noise_prob`` 확률로 다른 합법 수를 골라 분포를 넓힌다.
  라벨은 여전히 **그 상태에서의 전문가 행동**이므로, "실수 후 복구"를 정확히 가르친다.

사용
----
    python -m training.gen_dataset --mazes 200 --sizes 9,11,13,15 --out data/wall
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.game.actions import DELTAS, Action
from app.game.env import MazeEnv
from app.game.expert import wall_follow_action
from app.game.maze import generate

DEFAULT_SIZES: tuple[int, ...] = (9, 11, 13, 15, 17, 19, 21)
ALGORITHMS: tuple[str, ...] = ("dfs", "prim")


@dataclass(frozen=True)
class DatasetConfig:
    mazes: int = 1000
    sizes: tuple[int, ...] = DEFAULT_SIZES
    window_size: int = 3
    global_size: int = 5
    base_seed: int = 0
    #: 노이즈 없는 전문가 궤적 수
    expert_episodes: int = 1
    #: 무작위 이탈을 섞은 궤적 수 (실수 복구 학습)
    noisy_episodes: int = 1
    #: noisy 궤적에서 전문가 대신 다른 수를 고를 확률
    noise_prob: float = 0.05
    max_episode_steps: int = 1024
    randomize_endpoints: bool = True
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
    kind: np.ndarray  # 0=expert, 1=noisy
    step: np.ndarray
    algo: np.ndarray  # 0=dfs, 1=prim


def _maze_params(index: int, cfg: DatasetConfig) -> tuple[int, int, int, str]:
    """인덱스 → (width, height, seed, algorithm). 결정적이다."""
    size = cfg.sizes[index % len(cfg.sizes)]
    algorithm = ALGORITHMS[(index // len(cfg.sizes)) % len(ALGORITHMS)]
    seed = (cfg.base_seed * 1_000_003 + index * 7919) % (2**31 - 1)
    return size, size, seed, algorithm


def _random_legal_action(
    maze,
    pos: tuple[int, int],
    rng: random.Random,
    avoid: Action | None = None,
) -> Action:
    """벽이 아닌 이웃 중 하나를 고른다. ``avoid`` 는 가능하면 제외한다."""
    x, y = pos
    options = [
        action
        for action, (dx, dy) in DELTAS.items()
        if not maze.is_wall(x + dx, y + dy)
    ]
    if avoid is not None and len(options) > 1:
        options = [action for action in options if action != avoid]
    return options[rng.randrange(len(options))]


def _rollout(
    maze,
    cfg: DatasetConfig,
    rng: random.Random,
    noise_prob: float,
    kind: int,
    obs_list: list[np.ndarray],
    action_list: list[int],
    kinds: list[int],
    steps: list[int],
) -> None:
    """전문가 롤아웃 하나를 돌며 (관측, 전문가 행동) 쌍을 모은다."""
    env = MazeEnv(
        maze,
        window_size=cfg.window_size,
        global_size=cfg.global_size,
        max_steps=cfg.max_episode_steps,
    )
    obs = env.reset()
    t = 0

    while not env.done:
        label = wall_follow_action(maze, env.position, env.last_action)
        obs_list.append(obs)
        action_list.append(int(label))
        kinds.append(kind)
        steps.append(t)

        if noise_prob > 0 and rng.random() < noise_prob:
            action = _random_legal_action(maze, env.position, rng, avoid=label)
        else:
            action = label

        result = env.step(action)
        obs = result.observation
        t += 1
        if result.done:
            break


def build_maze_samples(index: int, cfg: DatasetConfig) -> MazeSamples:
    """미로 하나에 대한 학습 샘플을 만든다 (워커 함수 — 반드시 모듈 레벨)."""
    width, height, seed, algorithm = _maze_params(index, cfg)
    maze = generate(
        width,
        height,
        seed=seed,
        algorithm=algorithm,
        randomize_endpoints=cfg.randomize_endpoints,
    )

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    kinds: list[int] = []
    steps: list[int] = []
    rng = random.Random(seed ^ 0x9E3779B9)

    for _ in range(cfg.expert_episodes):
        _rollout(
            maze, cfg, rng, 0.0, 0, obs_list, action_list, kinds, steps
        )
    for _ in range(cfg.noisy_episodes):
        _rollout(
            maze, cfg, rng, cfg.noise_prob, 1, obs_list, action_list, kinds, steps
        )

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
                "expert": round(float((part["kind"] == 0).mean()), 3),
                "noisy": round(float((part["kind"] == 1).mean()), 3),
            },
        }

    summary["elapsed_sec"] = round(time.time() - t0, 2)
    (out / "meta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="미로 학습용 합성 데이터셋 생성")
    p.add_argument("--mazes", type=int, default=1000, help="생성할 미로 수")
    p.add_argument("--sizes", type=str, default=",".join(map(str, DEFAULT_SIZES)))
    p.add_argument("--window-size", type=int, default=3, choices=(3, 5, 7))
    p.add_argument(
        "--global-size",
        type=int,
        default=5,
        choices=(0, 3, 5, 7, 9),
        help="전역 방문 요약 격자 크기 (0 = 비활성)",
    )
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--expert-episodes", type=int, default=1)
    p.add_argument("--noisy-episodes", type=int, default=1)
    p.add_argument(
        "--noise-prob",
        type=float,
        default=0.05,
        help="noisy 궤적에서 전문가 대신 다른 수를 고를 확률",
    )
    p.add_argument("--max-episode-steps", type=int, default=1024)
    p.add_argument(
        "--no-randomize-endpoints",
        dest="randomize_endpoints",
        action="store_false",
        help="시작/목적지를 고정 코너에 배치 (기본: 무작위)",
    )
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
        global_size=args.global_size,
        base_seed=args.base_seed,
        expert_episodes=args.expert_episodes,
        noisy_episodes=args.noisy_episodes,
        noise_prob=args.noise_prob,
        max_episode_steps=args.max_episode_steps,
        randomize_endpoints=args.randomize_endpoints,
        out_dir=args.out,
    )
    summary = generate_dataset(cfg, workers=args.workers)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
