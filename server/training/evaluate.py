"""롤아웃 평가 — **실제 성공률**을 측정한다.

정확도(accuracy)는 참고용일 뿐이다. 모델이 한 스텝씩 스스로 판단하며 미로를
탈출하는지를 봐야 진짜 성능을 알 수 있다.

    python -m training.evaluate --checkpoint checkpoints/smoke/best.pt --data data/smoke2

측정 지표
---------
- success rate : 목표 도달 비율
- optimality   : 실제 스텝 / 최단 경로 스텝 (1.0 에 가까울수록 효율적)
- loop rate    : 이미 방문한 칸으로 되돌아간 비율 (진동 탐지)
- invalid rate : 벽으로 이동을 시도한 비율
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from app.game.actions import Action, apply_action
from app.game.env import MazeEnv, default_max_steps
from app.game.maze import generate
from app.game.solver import distance_map
from app.ml.policy import BertPolicy
from app.ml.runner import PolicyRunner


def load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {"config": {"window_size": 3, "randomize_endpoints": True}}


def unique_mazes(npz_path: Path) -> list[tuple[int, int, int]]:
    """(seed, size, algo) 고유 조합 목록."""
    data = np.load(npz_path)
    seeds = data["seed"]
    sizes = data["size"]
    algos = data["algo"]
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for s, z, a in zip(seeds.tolist(), sizes.tolist(), algos.tolist()):
        key = (int(s), int(z), int(a))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def rollout(
    model: BertPolicy,
    maze,
    window_size: int,
    seq_len: int,
    max_steps: int | None = None,
    global_size: int = 5,
) -> dict:
    env = MazeEnv(
        maze,
        window_size=window_size,
        global_size=global_size,
        max_steps=max_steps or default_max_steps(maze.width, maze.height),
    )
    runner = PolicyRunner(model, seq_len=seq_len)
    obs = env.observe()

    revisits = 0
    invalid = 0
    actions: list[int] = []

    while not env.done:
        action = runner.act(obs)
        actions.append(action)

        target = apply_action(env.position, Action(action))
        if target in env.visited:
            revisits += 1

        result = env.step(action)
        if result.hit_wall:
            invalid += 1
        obs = result.observation
        if result.win:
            break

    shortest = int(distance_map(maze, maze.goal)[maze.start[1], maze.start[0]])
    steps = env.step_count
    won = env.won

    return {
        "seed": maze.seed,
        "size": maze.width,
        "won": won,
        "steps": steps,
        "shortest": shortest,
        "optimality": round(steps / shortest, 3) if shortest > 0 else 0.0,
        "loop_rate": round(revisits / max(1, steps), 3),
        "invalid_rate": round(invalid / max(1, steps), 3),
    }


def evaluate(
    checkpoint: str,
    data_dir: str = "data",
    split: str = "test",
    limit: int | None = None,
    seq_len: int | None = None,
    out: str | None = None,
) -> dict:
    t0 = time.time()
    data_path = Path(data_dir)
    meta = load_meta(data_path)
    window_size = int(meta.get("config", {}).get("window_size", 3))
    global_size = int(meta.get("config", {}).get("global_size", 5))
    randomize = bool(meta.get("config", {}).get("randomize_endpoints", True))

    model = BertPolicy.load(checkpoint)
    seq_len = seq_len or model.cfg.seq_len

    mazes = unique_mazes(data_path / f"{split}.npz")
    if limit:
        mazes = mazes[:limit]

    results = []
    for seed, size, algo in mazes:
        maze = generate(
            size,
            size,
            seed=seed,
            algorithm="dfs" if algo == 0 else "prim",
            randomize_endpoints=randomize,
        )
        results.append(rollout(model, maze, window_size, seq_len, global_size=global_size))

    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {}
        wins = [r for r in rows if r["won"]]
        return {
            "mazes": n,
            "success_rate": round(len(wins) / n, 4),
            "avg_steps": round(float(np.mean([r["steps"] for r in rows])), 2),
            "avg_optimality_won": round(
                float(np.mean([r["optimality"] for r in wins])), 3
            )
            if wins
            else None,
            "avg_loop_rate": round(float(np.mean([r["loop_rate"] for r in rows])), 3),
            "avg_invalid_rate": round(float(np.mean([r["invalid_rate"] for r in rows])), 3),
        }

    summary = {
        "checkpoint": checkpoint,
        "split": split,
        "seq_len": seq_len,
        "overall": summarize(results),
        "by_size": {str(k): summarize(v) for k, v in sorted(by_size.items())},
        "elapsed_sec": round(time.time() - t0, 1),
        "failures": [r for r in results if not r["won"]][:10],
    }

    if out:
        Path(out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="정책 롤아웃 평가")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data,
        split=args.split,
        limit=args.limit,
        seq_len=args.seq_len,
        out=args.out,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
