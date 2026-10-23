"""DAgger — 모델이 실제로 방문한 상태에 정답 라벨을 붙여 재학습한다.

문제
----
순수 행동 복제(BC)는 **모델이 한 번도 실수하지 않은 상태**만 학습한다.
추론에서는 모델이 스스로 실수하며 학습 분포 밖으로 나가고, 그 상태에서의 복구
행동을 배운 적이 없으므로 그대로 헤맨다.

실측: 5 epoch 학습 후 단일 스텝 정확도 76% 인데 롤아웃 성공률은 27%,
루프율 75%. 전형적인 compounding error.

해법
----
1. 현재 정책으로 미로를 주행한다 (모델이 실제로 가는 상태를 모은다).
2. 각 상태에 **BFS 최적 행동**을 라벨로 붙인다 (전문가 라벨).
3. 원본 데이터에 합쳐 재학습한다.

이렇게 하면 모델이 빠지는 진동 상태가 학습 데이터에 직접 등장한다.

사용
----
    python -m training.dagger --checkpoint checkpoints/v3/best.pt --data data/v3 \
        --out data/v3_dagger --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from app.game.actions import Action
from app.game.env import MazeEnv, default_max_steps, observation_dim
from app.game.maze import generate
from app.game.solver import best_action, distance_map
from app.ml.policy import BertPolicy
from app.ml.runner import PolicyRunner
from training.evaluate import load_meta, unique_mazes


def collect(
    checkpoint: str,
    data_dir: str,
    split: str = "train",
    limit: int | None = None,
    seq_len: int | None = None,
    deterministic: bool = True,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """정책 롤아웃을 돌며 (관측, 최적 행동) 쌍을 모은다."""
    meta = load_meta(Path(data_dir))
    cfg = meta.get("config", {})
    window_size = int(cfg.get("window_size", 3))
    global_size = int(cfg.get("global_size", 5))
    randomize = bool(cfg.get("randomize_endpoints", True))

    model = BertPolicy.load(checkpoint)
    seq_len = seq_len or model.cfg.seq_len

    mazes = unique_mazes(Path(data_dir) / f"{split}.npz")
    if limit:
        mazes = mazes[:limit]

    obs_all: list[np.ndarray] = []
    act_all: list[int] = []
    seed_all: list[int] = []
    size_all: list[int] = []
    step_all: list[int] = []
    algo_all: list[int] = []
    stats = {"mazes": len(mazes), "samples": 0, "wins": 0, "steps": 0}

    for seed, size, algo in mazes:
        maze = generate(
            size,
            size,
            seed=seed,
            algorithm="dfs" if algo == 0 else "prim",
            randomize_endpoints=randomize,
        )
        dist = distance_map(maze, maze.goal)
        env = MazeEnv(
            maze,
            window_size=window_size,
            global_size=global_size,
            max_steps=default_max_steps(maze.width, maze.height),
        )
        runner = PolicyRunner(model, seq_len=seq_len)
        obs = env.observe()

        t = 0
        while not env.done:
            label = best_action(maze, env.position, dist)
            if label is None:
                break
            obs_all.append(obs)
            act_all.append(int(label))
            seed_all.append(seed)
            size_all.append(size)
            step_all.append(t)
            algo_all.append(algo)

            action = runner.act(obs, deterministic=deterministic, temperature=temperature)
            result = env.step(action)
            obs = result.observation
            t += 1
            if result.win:
                break

        stats["wins"] += int(env.won)
        stats["steps"] += env.step_count

    stats["samples"] = len(act_all)
    stats["success_rate"] = round(stats["wins"] / max(1, stats["mazes"]), 4)

    return {
        "obs": np.asarray(obs_all, dtype=np.float32),
        "action": np.asarray(act_all, dtype=np.int8),
        "seed": np.asarray(seed_all, dtype=np.int64),
        "size": np.asarray(size_all, dtype=np.int16),
        "kind": np.full(len(act_all), 2, dtype=np.int8),  # 2 = dagger
        "step": np.asarray(step_all, dtype=np.int16),
        "algo": np.asarray(algo_all, dtype=np.int8),
        "stats": stats,
    }


def merge(base_dir: str, dagger: dict[str, np.ndarray], out_dir: str) -> dict:
    """원본 train 에 DAgger 샘플을 이어붙여 새 데이터 디렉터리를 만든다."""
    base = Path(base_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_train = np.load(base / "train.npz")
    keys = ("obs", "action", "seed", "size", "kind", "step", "algo")

    merged = {
        k: np.concatenate([base_train[k], dagger[k]]) for k in keys
    }
    np.savez_compressed(out / "train.npz", **merged)

    for split in ("val", "test"):
        src = base / f"{split}.npz"
        if src.exists():
            data = np.load(src)
            np.savez_compressed(out / f"{split}.npz", **{k: data[k] for k in keys})

    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    meta.setdefault("dagger", {})
    meta["dagger"] = {
        "samples_added": int(len(dagger["action"])),
        "stats": dagger["stats"],
        "total_train": int(len(merged["action"])),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return meta["dagger"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAgger 데이터 수집 + 병합")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--limit", type=int, default=200, help="수집할 미로 수")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--stochastic", action="store_true", help="샘플링 정책으로 수집")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--raw-out", type=str, default=None, help="DAgger 샘플만 별도 저장")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dagger = collect(
        checkpoint=args.checkpoint,
        data_dir=args.data,
        split=args.split,
        limit=args.limit,
        seq_len=args.seq_len,
        deterministic=not args.stochastic,
        temperature=args.temperature,
    )

    if args.raw_out:
        np.savez_compressed(
            args.raw_out,
            **{k: v for k, v in dagger.items() if k != "stats"},
        )

    info = merge(args.data, dagger, args.out)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
