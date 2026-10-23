"""뷰어 색상 팔레트."""

from __future__ import annotations

Color = tuple[int, int, int]

# --- 배경 / 패널 ---
BG: Color = (10, 14, 24)
PANEL: Color = (17, 23, 37)
PANEL_EDGE: Color = (33, 43, 64)
BOARD_BG: Color = (14, 19, 31)

# --- 미로 ---
WALL: Color = (12, 16, 26)
WALL_TOP: Color = (29, 38, 58)
FLOOR: Color = (33, 42, 62)
FLOOR_VISITED: Color = (24, 66, 72)

# --- 개체 ---
TRAIL: Color = (72, 208, 178)
AGENT: Color = (126, 232, 203)
AGENT_EDGE: Color = (228, 255, 248)
GOAL: Color = (246, 191, 82)
GOAL_CORE: Color = (255, 233, 190)

# --- 텍스트 / 상태 ---
TEXT: Color = (228, 236, 246)
MUTED: Color = (126, 143, 168)
OK: Color = (106, 222, 160)
BAD: Color = (242, 112, 122)
ACCENT: Color = (112, 172, 255)


def lerp(a: Color, b: Color, t: float) -> Color:
    """두 색을 ``t`` (0..1) 비율로 섞는다."""
    t = max(0.0, min(1.0, t))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )
