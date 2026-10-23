"""SDL(pygame) 게임 뷰어.

    # 사람 플레이
    python -m viewer.app --size 15

    # LoRA-BERT 체크포인트 자동 플레이
    python -m viewer.app --checkpoint checkpoints/v3/best.pt --size 15

뷰어는 **이동을 결정하지 않는다**. :class:`viewer.brains.Brain` 이 돌려준 행동을
환경에 적용하고 그리기만 한다 (PLAN §1 R2). 나중에 FastAPI 서버가 준비되면
``HttpBrain`` 을 끼우면 되고 이 파일은 거의 바뀌지 않는다.
"""

from __future__ import annotations

import argparse
import time
from typing import Sequence

import pygame

from app.game.env import MazeEnv
from app.game.maze import FLOOR, generate
from viewer.brains import HumanBrain, PolicyBrain
from viewer.render import Renderer


class Motion:
    """셀 사이 이동을 부드럽게 보간한다 (표시 전용)."""

    def __init__(self, position: tuple[int, int]) -> None:
        self.start = (float(position[0]), float(position[1]))
        self.end = (float(position[0]), float(position[1]))
        self.t0 = 0.0
        self.duration = 0.0

    def reset(self, position: tuple[int, int]) -> None:
        self.start = self.end = (float(position[0]), float(position[1]))
        self.duration = 0.0

    def begin(
        self,
        start: tuple[float, float],
        end: tuple[int, int],
        now: float,
        duration: float,
    ) -> None:
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.t0 = now
        self.duration = max(0.0, duration)

    def position(self, now: float) -> tuple[float, float]:
        if self.duration <= 0:
            return self.end
        t = (now - self.t0) / self.duration
        if t >= 1.0:
            return self.end
        if t <= 0.0:
            return self.start
        ease = 1.0 - (1.0 - t) ** 3  # ease-out cubic
        return (
            self.start[0] + (self.end[0] - self.start[0]) * ease,
            self.start[1] + (self.end[1] - self.start[1]) * ease,
        )


class Session:
    """미로 + 환경 + Brain 을 묶은 한 판."""

    def __init__(
        self,
        brain,
        size: int,
        algorithm: str,
        randomize_endpoints: bool,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.brain = brain
        self.size = size
        self.algorithm = algorithm
        self.randomize_endpoints = randomize_endpoints
        self.max_steps = max_steps
        self.motion = Motion((0, 0))
        self.flash_until = 0.0
        self.last_result = None
        self.floor_total = 1
        self.new_maze(seed)

    # ------------------------------------------------------------------ 생성
    def new_maze(self, seed: int | None = None) -> None:
        self.maze = generate(
            self.size,
            self.size,
            seed=seed,
            algorithm=self.algorithm,
            randomize_endpoints=self.randomize_endpoints,
        )
        self.env = MazeEnv(
            self.maze,
            window_size=self.brain.window_size,
            global_size=self.brain.global_size,
            max_steps=self.max_steps,
        )
        self.floor_total = max(1, int((self.maze.grid == FLOOR).sum()))
        self.motion.reset(self.maze.start)
        self.flash_until = 0.0
        self.last_result = None
        self.brain.reset()

    def restart(self) -> None:
        self.env.reset()
        self.motion.reset(self.maze.start)
        self.flash_until = 0.0
        self.last_result = None
        self.brain.reset()

    # ------------------------------------------------------------------ 진행
    def apply(self, action: int, now: float) -> None:
        start = self.motion.position(now)
        result = self.env.step(action)
        self.last_result = result
        if result.hit_wall:
            self.flash_until = now + 0.22
        self.motion.begin(
            start, self.env.position, now, self.brain.animation_duration
        )
        self.brain.note_step(now)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDL 미로 게임 뷰어")
    parser.add_argument("--size", type=int, default=15, help="미로 크기 (홀수, >=5)")
    parser.add_argument("--seed", type=int, default=None, help="미로 시드 (재현용)")
    parser.add_argument("--algorithm", choices=("dfs", "prim"), default="dfs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="LoRA-BERT 체크포인트. 주어지면 모델이 자동으로 플레이한다.",
    )
    parser.add_argument(
        "--human", action="store_true", help="체크포인트가 있어도 사람이 조작"
    )
    parser.add_argument(
        "--fixed-endpoints",
        action="store_true",
        help="시작/목적지를 고정한다 (기본: AI 모드는 무작위)",
    )
    parser.add_argument(
        "--randomize-endpoints",
        action="store_true",
        help="사람 모드에서도 시작/목적지를 무작위로",
    )
    parser.add_argument("--step-interval", type=float, default=0.09)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1180)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="에피소드 최대 스텝 (0 = 미로 크기 기반 기본값)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="N 프레임 후 종료 (스모크 테스트용)"
    )
    parser.add_argument("--snapshot", type=str, default=None, help="PNG 로 저장할 경로")
    parser.add_argument("--snapshot-frame", type=int, default=60)
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> None:
    if args.size < 5 or args.size % 2 == 0:
        raise SystemExit(f"--size must be odd and >= 5, got {args.size}")
    if args.checkpoint and args.human:
        raise SystemExit("--checkpoint 와 --human 은 함께 쓸 수 없습니다")
    if args.width < 480 or args.height < 400:
        raise SystemExit("window is too small; use --width >= 480, --height >= 400")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate(args)

    if args.checkpoint:
        try:
            brain = PolicyBrain(
                args.checkpoint,
                step_interval=args.step_interval,
                temperature=args.temperature,
            )
        except Exception as exc:  # 체크포인트 경로/버전 문제를 읽기 쉽게 보여준다
            raise SystemExit(f"[viewer] failed to load checkpoint: {exc}") from exc
    else:
        brain = HumanBrain()

    randomize = bool(
        args.randomize_endpoints
        or (args.checkpoint and not args.fixed_endpoints)
    )

    pygame.init()
    pygame.display.set_caption("Maze Runner · LoRA-BERT")
    screen = pygame.display.set_mode((args.width, args.height), pygame.RESIZABLE)
    pygame.key.set_repeat(220, 80)
    clock = pygame.time.Clock()
    renderer = Renderer()

    session = Session(
        brain,
        size=args.size,
        algorithm=args.algorithm,
        randomize_endpoints=randomize,
        seed=args.seed,
        max_steps=args.max_steps or None,
    )

    print(
        f"[viewer] maze {session.maze.width}x{session.maze.height} "
        f"{session.maze.algorithm} seed={session.maze.seed_hex()} "
        f"start={session.maze.start} goal={session.maze.goal}"
    )
    print(
        f"[viewer] brain={brain.label} window={brain.window_size} "
        f"global={brain.global_size} obs_dim={session.env.obs_dim}"
    )
    print(
        "[viewer] keys: ESC quit · R restart · N new maze · SPACE pause"
        + ("" if args.checkpoint else " · arrows/WASD move")
    )

    paused = False
    running = True
    frames = 0

    while running:
        now = time.perf_counter()
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(
                    (max(480, event.w), max(400, event.h)), pygame.RESIZABLE
                )
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    session.restart()
                elif event.key == pygame.K_n:
                    session.new_maze(None)

        if not paused and not session.env.done and session.brain.wants_step(now):
            action = session.brain.act(session.env, events)
            if action is not None:
                session.apply(action, now)

        renderer.draw(screen, session, time.perf_counter(), paused)
        frames += 1

        if args.snapshot and frames == args.snapshot_frame:
            pygame.image.save(screen, args.snapshot)
            print(f"[viewer] snapshot saved: {args.snapshot}")
            running = False

        if args.max_frames and frames >= args.max_frames:
            running = False

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
