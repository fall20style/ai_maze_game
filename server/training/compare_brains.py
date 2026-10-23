"""AI 구현체 vs 규칙 구현체 벤치마크.

같은 미로 집합에서 세 정책을 돌려 성공률·스텝·optimality·루프율을 비교한다.

- ``ai``             : LoRA-BERT 체크포인트 (기본 ``checkpoints/wall``)
- ``rule/wall``      : 왼손 법칙 (관측: 3×3 창 + 직전 행동)
- ``rule/frontier``  : 지식 지도(belief map) 위 탐색 (관측: 누적 벽/바닥)

세 정책 모두 **관측만으로** 행동을 고른다. 미로 전체를 아는 것은 평가용
``distance_map`` 뿐이며, 이는 optimality 계산에만 쓰인다.

사용::

    python -m training.compare_brains --mazes 24 --size 15
    python -m training.compare_brains --mazes 24 --size 15 --out checkpoints/compare.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from app.game.actions import DELTAS
from app.game.env import MazeEnv, default_max_steps
from app.game.expert import frontier_action, wall_follow_action
from app.game.maze import generate
from app.game.solver import distance_map
from app.ml.policy import BertPolicy
from app.ml.runner import PolicyRunner

SERVER_ROOT = Path(__file__).resolve().parents[1]


def _rollout(kind: str, maze, model: BertPolicy | None) -> dict:
    """한 판을 끝까지 돌리고 지표를 돌려준다."""
    if kind == "ai":
        assert model is not None
        cfg = model.cfg
        window = int(getattr(cfg, "window_size", 3))
        glob = int(getattr(cfg, "global_size", 5))
        mapping = int(getattr(cfg, "map_size", 0))
        if mapping and mapping < max(maze.width, maze.height):
            mapping = max(maze.width, maze.height)
    elif kind == "frontier":
        window, glob, mapping = 3, 0, max(maze.width, maze.height)
    else:
        window, glob, mapping = 3, 5, 0

    env = MazeEnv(
        maze,
        window_size=window,
        global_size=glob,
        map_size=mapping,
        max_steps=default_max_steps(maze.width, maze.height),
    )
    env.reset()
    runner = PolicyRunner(model, seq_len=model.cfg.seq_len) if model is not None else None

    revisits = 0
    invalid = 0

    while not env.done:
        if kind == "ai":
            assert runner is not None
            action = runner.act(env.observe())
        elif kind == "frontier":
            action = int(frontier_action(env.known, env.position, maze.goal))
        else:
            action = int(wall_follow_action(maze, env.position, env.last_action))

        dx, dy = DELTAS[action]
        if (env.position[0] + dx, env.position[1] + dy) in env.visited:
            revisits += 1

        result = env.step(action)
        if result.hit_wall:
            invalid += 1

    shortest = int(distance_map(maze, maze.goal)[maze.start[1], maze.start[0]])
    steps = env.step_count
    return {
        "won": bool(env.won),
        "steps": steps,
        "shortest": shortest,
        "optimality": steps / shortest if env.won and shortest else None,
        "loop_rate": revisits / steps if steps else 0.0,
        "invalid_rate": invalid / steps if steps else 0.0,
    }


def _summary(rows: list[dict]) -> dict:
    won = [r for r in rows if r["won"]]
    return {
        "mazes": len(rows),
        "wins": len(won),
        "success_rate": len(won) / len(rows) if rows else 0.0,
        "steps_mean": statistics.mean(r["steps"] for r in rows) if rows else 0.0,
        "optimality_mean": statistics.mean(r["optimality"] for r in won) if won else None,
        "loop_rate": statistics.mean(r["loop_rate"] for r in rows) if rows else 0.0,
        "invalid_rate": statistics.mean(r["invalid_rate"] for r in rows) if rows else 0.0,
    }


def _fmt(name: str, s: dict) -> str:
    opt = "-" if s["optimality_mean"] is None else f"{s['optimality_mean']:.3f}"
    return (
        f"{name:<20} {s['wins']:>3}/{s['mazes']:<3} "
        f"{s['success_rate'] * 100:>6.1f}% "
        f"{s['steps_mean']:>8.1f} {opt:>10} "
        f"{s['loop_rate']:>7.2f} {s['invalid_rate']:>7.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI vs 규칙 구현체 벤치마크")
    parser.add_argument("--mazes", type=int, default=24)
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--seed-base", type=int, default=90000)
    parser.add_argument(
        "--ai-checkpoint",
        default=str(SERVER_ROOT / "checkpoints" / "wall" / "best.pt"),
    )
    parser.add_argument("--out", default=None, help="결과 JSON 경로")
    args = parser.parse_args(argv)

    checkpoint = Path(args.ai_checkpoint)
    if not checkpoint.exists():
        parser.error(f"AI 체크포인트가 없습니다: {checkpoint}")

    model = BertPolicy.load(str(checkpoint))
    policies = (("ai", "AI"), ("wall", "RULE wall"), ("frontier", "RULE frontier"))
    results: dict[str, list[dict]] = {key: [] for key, _ in policies}

    for index in range(args.mazes):
        maze = generate(
            args.size,
            args.size,
            seed=args.seed_base + index,
            randomize_endpoints=True,
        )
        for key, _ in policies:
            results[key].append(_rollout(key, maze, model if key == "ai" else None))

    header = (
        f"{'policy':<20} {'won':>7} {'success':>7} "
        f"{'steps':>8} {'optimality':>10} {'loop':>7} {'invalid':>7}"
    )
    print(f"mazes={args.mazes} size={args.size}x{args.size} ai={checkpoint.parent.name}")
    print(header)
    print("-" * len(header))

    payload = {"config": vars(args), "policies": {}}
    for key, name in policies:
        summary = _summary(results[key])
        payload["policies"][key] = summary
        print(_fmt(name, summary))

    # 같은 미로에서 짝지어 비교한다 (평균만 보면 분산에 가려진다)
    def paired(a: str, b: str, label: str) -> None:
        deltas = [rb["steps"] - ra["steps"] for ra, rb in zip(results[a], results[b])]
        better = sum(d < 0 for d in deltas)
        same = sum(d == 0 for d in deltas)
        worse = sum(d > 0 for d in deltas)
        print(
            f"\n{label}: 더 빠름 {better} · 동일 {same} · 더 느림 {worse} "
            f"· 평균 Δ {statistics.mean(deltas):+.1f} 스텝"
        )
        payload["pairs"] = payload.get("pairs", {})
        payload["pairs"][f"{a}_vs_{b}"] = {
            "better": better,
            "same": same,
            "worse": worse,
            "mean_step_delta": statistics.mean(deltas),
        }

    paired("wall", "frontier", "RULE frontier vs RULE wall")
    paired("wall", "ai", "AI vs RULE wall")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n저장: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
