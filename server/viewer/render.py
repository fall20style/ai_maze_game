"""SDL(pygame) 렌더러 — **그리기만** 한다.

이동 결정은 전부 :mod:`viewer.brains` 의 Brain 이 담당한다.
여기에는 경로 탐색/휴리스틱이 존재하지 않는다 (PLAN §1 R1).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import pygame

from . import theme
from .layout import BoardLayout, fit_board

if TYPE_CHECKING:  # 순환 import 회피 (런타임에는 필요 없음)
    from .app import Session

HUD_TOP = 88
HUD_BOTTOM = 112

_ACTION_LABELS = ("UP", "DOWN", "LEFT", "RIGHT")
_ACTION_VECTORS = ((0, -1), (0, 1), (-1, 0), (1, 0))


class FontBook:
    """폰트 캐시. Windows/macOS/Linux 순서로 쓸 만한 글꼴을 찾는다."""

    _FAMILIES = ("segoeui", "malgungothic", "arial", "dejavusans", "liberationsans")

    def __init__(self) -> None:
        self.family = next(
            (name for name in self._FAMILIES if pygame.font.match_font(name)), None
        )
        self._cache: dict[tuple[int, bool], pygame.font.Font] = {}

    def get(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (int(size), bool(bold))
        font = self._cache.get(key)
        if font is None:
            font = pygame.font.SysFont(self.family, key[0], bold=key[1])
            self._cache[key] = font
        return font


def _rect(box: tuple[float, float, float, float]) -> pygame.Rect:
    x, y, w, h = box
    return pygame.Rect(
        int(round(x)),
        int(round(y)),
        max(1, int(round(w))),
        max(1, int(round(h))),
    )


class Renderer:
    def __init__(self) -> None:
        self.fonts = FontBook()
        self._scratch: pygame.Surface | None = None

    # ------------------------------------------------------------------ 유틸
    def _layer(self, size: tuple[int, int]) -> pygame.Surface:
        """투명 레이어를 재사용한다 (매 프레임 Surface 생성 방지)."""
        if self._scratch is None or self._scratch.get_size() != size:
            self._scratch = pygame.Surface(size, pygame.SRCALPHA)
        else:
            self._scratch.fill((0, 0, 0, 0))
        return self._scratch

    def _text(
        self,
        screen: pygame.Surface,
        text: str,
        size: int,
        color: tuple[int, int, int],
        pos: tuple[float, float],
        anchor: str = "topleft",
        bold: bool = False,
    ) -> pygame.Rect:
        surf = self.fonts.get(size, bold).render(text, True, color)
        rect = surf.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        screen.blit(surf, rect)
        return rect

    def _chip(
        self,
        screen: pygame.Surface,
        text: str,
        size: int,
        x: float,
        y: float,
        fg: tuple[int, int, int] = theme.TEXT,
        bg: tuple[int, int, int] = theme.PANEL_EDGE,
    ) -> float:
        font = self.fonts.get(size)
        pad_x, pad_y = 11, 5
        surf = font.render(text, True, fg)
        width = surf.get_width() + 2 * pad_x
        height = surf.get_height() + 2 * pad_y
        rect = pygame.Rect(int(x), int(y), width, height)
        pygame.draw.rect(screen, bg, rect, border_radius=height // 2)
        screen.blit(surf, (rect.x + pad_x, rect.centery - surf.get_height() // 2))
        return float(width)

    def _chip_width(self, text: str, size: int) -> float:
        return float(self.fonts.get(size).size(text)[0] + 22)

    # ------------------------------------------------------------------ 메인
    def draw(
        self,
        screen: pygame.Surface,
        session: "Session",
        now: float,
        paused: bool,
    ) -> None:
        width, height = screen.get_size()
        screen.fill(theme.BG)

        area_h = max(1, height - HUD_TOP - HUD_BOTTOM)
        base = fit_board(
            session.maze.width,
            session.maze.height,
            width,
            area_h,
            margin=16,
        )
        layout = replace(base, origin_y=base.origin_y + HUD_TOP)

        board_panel = pygame.Rect(
            int(layout.origin_x) - 8,
            int(layout.origin_y) - 8,
            int(layout.width) + 16,
            int(layout.height) + 16,
        )

        self._draw_board(screen, session, layout, board_panel, now)
        self._draw_hud(screen, session, width, height, paused)
        if paused and not session.env.done:
            self._draw_paused(screen, board_panel)
        if session.env.done:
            self._draw_result(screen, session, board_panel)

    # ------------------------------------------------------------------ 보드
    def _draw_board(
        self,
        screen: pygame.Surface,
        session: "Session",
        layout: BoardLayout,
        panel: pygame.Rect,
        now: float,
    ) -> None:
        pygame.draw.rect(screen, theme.BOARD_BG, panel, border_radius=14)
        pygame.draw.rect(screen, theme.PANEL_EDGE, panel, width=1, border_radius=14)

        maze = session.maze
        env = session.env
        cell = layout.cell

        # 보드 밖으로 삐져나가는 그리기(관측 창 등)를 잘라낸다
        screen.set_clip(panel)

        for y in range(layout.rows):
            for x in range(layout.cols):
                rect = _rect(layout.cell_rect(x, y))
                if maze.is_wall(x, y):
                    pygame.draw.rect(screen, theme.WALL, rect)
                    continue

                inset = max(1, int(cell * 0.06))
                inner = rect.inflate(-2 * inset, -2 * inset)
                color = theme.FLOOR_VISITED if (x, y) in env.visited else theme.FLOOR
                pygame.draw.rect(
                    screen, color, inner, border_radius=max(1, int(cell * 0.10))
                )

        # 이동 궤적
        if len(env.trajectory) > 1:
            layer = self._layer(screen.get_size())
            points = [layout.center(x, y) for x, y in env.trajectory]
            pygame.draw.lines(
                layer,
                (*theme.TRAIL, 92),
                False,
                points,
                width=max(2, int(cell * 0.09)),
            )
            screen.blit(layer, (0, 0))

        self._draw_goal(screen, layout, session, now)
        self._draw_observation_window(screen, layout, session)
        self._draw_agent(screen, layout, session, now)

        screen.set_clip(None)

    def _draw_goal(
        self,
        screen: pygame.Surface,
        layout: BoardLayout,
        session: "Session",
        now: float,
    ) -> None:
        gx, gy = layout.center(*session.maze.goal)
        pulse = 0.5 + 0.5 * math.sin(now * 2.6)
        radius = layout.cell * 0.26

        pygame.draw.circle(
            screen,
            theme.lerp(theme.FLOOR, theme.GOAL, 0.16),
            (gx, gy),
            radius * (1.95 + 0.15 * pulse),
        )
        pygame.draw.circle(
            screen,
            theme.lerp(theme.FLOOR, theme.GOAL, 0.34),
            (gx, gy),
            radius * (1.35 + 0.08 * pulse),
        )
        pygame.draw.circle(screen, theme.GOAL, (gx, gy), radius)
        pygame.draw.circle(screen, theme.GOAL_CORE, (gx, gy), radius * 0.42)

    def _draw_observation_window(
        self,
        screen: pygame.Surface,
        layout: BoardLayout,
        session: "Session",
    ) -> None:
        half = session.env.window_size // 2
        px, py = session.env.position
        x0, y0, cw, ch = layout.cell_rect(px - half, py - half)
        span = cw * (2 * half + 1)
        box = pygame.Rect(int(round(x0)), int(round(y0)), int(round(span)), int(round(span)))
        pygame.draw.rect(
            screen,
            theme.lerp(theme.BOARD_BG, theme.ACCENT, 0.55),
            box,
            width=1,
            border_radius=5,
        )

    def _draw_agent(
        self,
        screen: pygame.Surface,
        layout: BoardLayout,
        session: "Session",
        now: float,
    ) -> None:
        ax, ay = layout.center(*session.motion.position(now))
        radius = layout.cell * 0.30
        cell = layout.cell

        if now < session.flash_until:
            pygame.draw.circle(
                screen,
                theme.BAD,
                (ax, ay),
                radius * 1.7,
                width=max(1, int(cell * 0.07)),
            )

        pygame.draw.circle(
            screen,
            theme.lerp(theme.FLOOR, theme.AGENT, 0.20),
            (ax, ay),
            radius * 1.8,
        )
        pygame.draw.circle(screen, theme.AGENT, (ax, ay), radius)
        pygame.draw.circle(
            screen,
            theme.AGENT_EDGE,
            (ax, ay),
            radius,
            width=max(1, int(cell * 0.05)),
        )

        last = int(session.env.last_action)
        if 0 <= last < 4:
            dx, dy = _ACTION_VECTORS[last]
            px, py = -dy, dx  # 수직 벡터
            tip = (ax + dx * radius * 1.15, ay + dy * radius * 1.15)
            spread = radius * 0.42
            base_x = ax + dx * radius * 0.15
            base_y = ay + dy * radius * 0.15
            pygame.draw.polygon(
                screen,
                theme.AGENT_EDGE,
                [
                    tip,
                    (base_x + px * spread, base_y + py * spread),
                    (base_x - px * spread, base_y - py * spread),
                ],
            )

    # ------------------------------------------------------------------ HUD
    def _draw_hud(
        self,
        screen: pygame.Surface,
        session: "Session",
        width: int,
        height: int,
        paused: bool,
    ) -> None:
        env = session.env

        # --- 상단
        pygame.draw.rect(screen, theme.PANEL, (0, 0, width, HUD_TOP))
        pygame.draw.line(screen, theme.PANEL_EDGE, (0, HUD_TOP - 1), (width, HUD_TOP - 1))

        self._text(screen, "MAZE RUNNER", 23, theme.TEXT, (26, 18), bold=True)
        self._text(
            screen,
            f"SDL viewer  ·  brain: {session.brain.label}",
            12,
            theme.MUTED,
            (27, 52),
        )

        chips = [
            f"{session.maze.width} x {session.maze.height}",
            session.maze.algorithm.upper(),
            f"seed {session.maze.seed_hex()}",
        ]
        gap = 8.0
        total = sum(self._chip_width(t, 12) for t in chips) + gap * (len(chips) - 1)
        x = width - 26 - total
        for text in chips:
            x += self._chip(screen, text, 12, x, 28) + gap

        # --- 하단
        top = height - HUD_BOTTOM
        pygame.draw.rect(screen, theme.PANEL, (0, top, width, HUD_BOTTOM))
        pygame.draw.line(screen, theme.PANEL_EDGE, (0, top), (width, top))

        if paused:
            status, color = "PAUSED", theme.MUTED
        elif env.won:
            status, color = "GOAL", theme.OK
        elif env.done:
            status, color = "TIMEOUT", theme.BAD
        else:
            status, color = "RUNNING", theme.ACCENT

        self._stat(screen, 26, top + 26, "STEPS", f"{env.step_count} / {env.max_steps}")
        self._stat(
            screen,
            176,
            top + 26,
            "VISITED",
            f"{len(env.visited)} / {session.floor_total}",
        )
        self._stat(screen, 326, top + 26, "STATUS", status, color=color, dot=True)

        probs = session.brain.probabilities()
        if probs is not None:
            self._draw_probabilities(screen, probs, width, top)
        else:
            self._text(
                screen,
                "ARROWS / WASD   move",
                13,
                theme.TEXT,
                (width - 26, top + 26),
                anchor="topright",
            )
            self._text(
                screen,
                "R restart  ·  N new maze  ·  SPACE pause  ·  ESC quit",
                11,
                theme.MUTED,
                (width - 26, top + 52),
                anchor="topright",
            )

    def _stat(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        label: str,
        value: str,
        color: tuple[int, int, int] = theme.TEXT,
        dot: bool = False,
    ) -> None:
        self._text(screen, label, 10, theme.MUTED, (x, y))
        if dot:
            pygame.draw.circle(screen, color, (x + 5, y + 34), 5)
            self._text(screen, value, 19, color, (x + 16, y + 24), bold=True)
        else:
            self._text(screen, value, 19, color, (x, y + 24), bold=True)

    def _draw_probabilities(
        self,
        screen: pygame.Surface,
        probs,
        width: int,
        top: int,
    ) -> None:
        panel_w = 292
        right = width - 26
        left = right - panel_w
        row_h = 18
        y0 = top + 12
        best = int(max(range(len(probs)), key=lambda i: probs[i]))

        for i, name in enumerate(_ACTION_LABELS):
            y = y0 + i * row_h
            color = theme.ACCENT if i == best else theme.MUTED
            self._text(screen, name, 10, color, (left, y + 4))
            bar_x = left + 46
            bar_w = panel_w - 46 - 44
            pygame.draw.rect(
                screen,
                theme.PANEL_EDGE,
                pygame.Rect(bar_x, y + 6, bar_w, 7),
                border_radius=4,
            )
            fill = int(bar_w * float(probs[i]))
            if fill > 0:
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(bar_x, y + 6, max(2, fill), 7),
                    border_radius=4,
                )
            self._text(
                screen,
                f"{float(probs[i]) * 100:3.0f}%",
                10,
                color,
                (right, y + 4),
                anchor="topright",
            )

        self._text(
            screen,
            "SPACE pause  ·  R restart  ·  N new maze  ·  ESC quit",
            10,
            theme.MUTED,
            (right, y0 + 4 * row_h + 2),
            anchor="topright",
        )

    # ------------------------------------------------------------------ 오버레이
    def _draw_paused(self, screen: pygame.Surface, panel: pygame.Rect) -> None:
        text = "PAUSED"
        font = self.fonts.get(12)
        w = font.size(text)[0] + 22
        h = font.get_height() + 10
        rect = pygame.Rect(panel.right - w - 12, panel.top + 12, w, h)
        pygame.draw.rect(screen, theme.PANEL, rect, border_radius=h // 2)
        pygame.draw.rect(screen, theme.MUTED, rect, width=1, border_radius=h // 2)
        self._text(
            screen,
            text,
            12,
            theme.MUTED,
            (rect.centerx, rect.centery),
            anchor="center",
        )

    def _draw_result(
        self,
        screen: pygame.Surface,
        session: "Session",
        panel: pygame.Rect,
    ) -> None:
        width, height = screen.get_size()
        layer = self._layer((width, height))
        pygame.draw.rect(layer, (4, 7, 12, 168), panel, border_radius=14)
        screen.blit(layer, (0, 0))

        won = session.env.won
        color = theme.OK if won else theme.BAD
        title = "GOAL REACHED" if won else "OUT OF STEPS"

        box_w, box_h = 460, 190
        box = pygame.Rect(0, 0, box_w, box_h)
        box.center = panel.center
        pygame.draw.rect(screen, theme.PANEL, box, border_radius=18)
        pygame.draw.rect(screen, theme.PANEL_EDGE, box, width=1, border_radius=18)

        self._text(
            screen, title, 28, color, (box.centerx, box.top + 44), anchor="midtop", bold=True
        )
        self._text(
            screen,
            f"steps {session.env.step_count}   ·   visited {len(session.env.visited)}",
            13,
            theme.MUTED,
            (box.centerx, box.top + 92),
            anchor="midtop",
        )
        self._text(
            screen,
            "R  restart        N  new maze",
            12,
            theme.TEXT,
            (box.centerx, box.top + 132),
            anchor="midtop",
        )
