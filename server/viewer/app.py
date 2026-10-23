"""SDL(pygame) 게임 클라이언트.

    # 서버 (별도 터미널)
    uvicorn app.api.main:app --port 8000

    # 클라이언트
    python -m viewer.app --size 15

**이 프로세스는 게임을 진행시키지 않는다.** 미로 생성, 이동 결정, 벽 판정, 승리 판정은
전부 서버(``app.api``)에서 일어나고, 여기서는 HTTP 로 받은 상태를 그리고 입력을
전달할 뿐이다 (PLAN §1 R1·R2·R3). 서버 응답이 없으면 이동하지 않는다 (R4).

키
--
1 / 2 / 3   AI / 규칙 / 사람 구현체 선택 (또는 상단 칩 클릭)
방향키·WASD  사람 모드에서 이동
R           같은 미로 다시 시작
N           새 미로
SPACE       일시정지
ESC / Q     종료
"""

from __future__ import annotations

import argparse
import time
from typing import Sequence

import pygame

from viewer.api_client import DEFAULT_SERVER, ApiError, GameApi
from viewer.remote import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    RemoteSession,
)
from viewer.render import Renderer, brain_selector_rects

KEY_TO_ACTION: dict[int, int] = {
    pygame.K_UP: ACTION_UP,
    pygame.K_w: ACTION_UP,
    pygame.K_DOWN: ACTION_DOWN,
    pygame.K_s: ACTION_DOWN,
    pygame.K_LEFT: ACTION_LEFT,
    pygame.K_a: ACTION_LEFT,
    pygame.K_RIGHT: ACTION_RIGHT,
    pygame.K_d: ACTION_RIGHT,
}

KEY_TO_BRAIN: dict[int, str] = {
    pygame.K_1: "ai",
    pygame.K_2: "rule",
    pygame.K_3: "human",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDL 미로 게임 클라이언트")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER, help="API 서버 주소")
    parser.add_argument("--size", type=int, default=15, help="미로 크기 (홀수, >=5)")
    parser.add_argument("--seed", type=int, default=None, help="미로 시드 (재현용)")
    parser.add_argument("--algorithm", choices=("dfs", "prim"), default="dfs")
    parser.add_argument(
        "--brain",
        choices=("ai", "rule", "human"),
        default="ai",
        help="이동 결정 구현체 (기본: ai)",
    )
    parser.add_argument(
        "--rule",
        choices=("wall", "frontier"),
        default="wall",
        help="규칙 구현체 종류 (기본: wall=왼손 법칙)",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="--brain human 과 같다 (하위 호환)",
    )
    parser.add_argument(
        "--fixed-endpoints",
        action="store_true",
        help="시작/목적지를 고정한다 (기본: 무작위)",
    )
    parser.add_argument("--step-interval", type=float, default=0.09, help="AI/규칙 스텝 간격(초)")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1180)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="에피소드 최대 스텝 (0 = 서버 기본값)",
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
    if args.width < 480 or args.height < 400:
        raise SystemExit("window is too small; use --width >= 480, --height >= 400")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate(args)
    brain = "human" if args.human else args.brain

    api = GameApi(args.server)
    try:
        health = api.health()
    except ApiError as exc:
        raise SystemExit(
            f"[viewer] 서버에 연결할 수 없습니다: {exc}\n"
            f"        먼저 실행하세요: uvicorn app.api.main:app --port 8000"
        ) from exc

    try:
        state = api.create(
            size=args.size,
            algorithm=args.algorithm,
            brain=brain,
            rule=args.rule,
            seed=args.seed,
            max_steps=args.max_steps or None,
            randomize_endpoints=not args.fixed_endpoints,
        )
    except ApiError as exc:
        raise SystemExit(f"[viewer] 게임을 만들 수 없습니다: {exc}") from exc

    session = RemoteSession(api, state, step_interval=max(0.01, args.step_interval))

    pygame.init()
    pygame.display.set_caption("Maze Runner · server-authoritative")
    screen = pygame.display.set_mode((args.width, args.height), pygame.RESIZABLE)
    pygame.key.set_repeat(220, 80)
    clock = pygame.time.Clock()
    renderer = Renderer()

    print(f"[viewer] server={args.server} checkpoint={health.get('checkpoint')}")
    print(
        f"[viewer] maze {session.maze.width}x{session.maze.height} "
        f"{session.maze.algorithm} seed={session.maze.seed_hex()} "
        f"start={session.maze.start} goal={session.maze.goal}"
    )
    print(f"[viewer] brain={session.brain.label} (kind={session.brain_kind})")
    print("[viewer] keys: 1 AI · 2 RULE · 3 HUMAN · R restart · N new maze · SPACE pause · ESC quit")

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
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for kind, rect in brain_selector_rects(screen.get_width()):
                    if rect.collidepoint(event.pos):
                        session.set_brain(kind, args.rule, now)
                        print(f"[viewer] brain -> {session.brain.label}")
                        break
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    session.restart(now)
                elif event.key == pygame.K_n:
                    session.new_maze(now)
                elif event.key in KEY_TO_BRAIN:
                    session.set_brain(KEY_TO_BRAIN[event.key], args.rule, now)
                    print(f"[viewer] brain -> {session.brain.label}")

        # --- 진행 (모든 판단은 서버가 한다) --------------------------------
        if not paused and not session.env.done:
            if session.awaiting_action:
                for event in events:
                    if event.type == pygame.KEYDOWN and event.key in KEY_TO_ACTION:
                        session.send_action(KEY_TO_ACTION[event.key], now)
                        break
            elif session.wants_step(now):
                session.advance(now)

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

    try:
        api.delete(session.game_id)
    except ApiError:
        pass
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
