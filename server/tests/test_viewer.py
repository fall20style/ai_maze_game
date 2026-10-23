"""뷰어 회귀 테스트 — 디스플레이 없이 실행된다."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from viewer.layout import fit_board

VIEWER_DIR = Path(__file__).resolve().parents[1] / "viewer"
SERVER_DIR = VIEWER_DIR.parent

#: 뷰어에 있으면 안 되는 경로 탐색 관련 토큰 (PLAN §1 R1)
FORBIDDEN_TOKENS = (
    "solver",
    "bfs",
    "astar",
    "a_star",
    "dijkstra",
    "distance_map",
    "pathfind",
    "flood",
)


# --------------------------------------------------------------- 레이아웃 계산
def test_board_is_centered_and_fits():
    layout = fit_board(15, 15, 800, 600, margin=10)

    assert layout.cell == pytest.approx(min(780 / 15, 580 / 15))
    assert layout.width <= 800 - 2 * 10 + 1e-6
    assert layout.height <= 600 - 2 * 10 + 1e-6
    assert layout.origin_x + layout.width / 2 == pytest.approx(400)
    assert layout.origin_y + layout.height / 2 == pytest.approx(300)


def test_cell_rect_and_center():
    layout = fit_board(9, 9, 900, 900, margin=0)

    assert layout.cell == pytest.approx(100)
    assert layout.cell_rect(0, 0) == pytest.approx((0, 0, 100, 100))
    assert layout.center(0, 0) == pytest.approx((50, 50))
    assert layout.center(1.5, 2.0) == pytest.approx((200, 250))


def test_cells_stay_square_in_wide_area():
    layout = fit_board(15, 15, 1600, 500, margin=0)

    assert layout.cell == pytest.approx(500 / 15)
    assert layout.origin_x == pytest.approx((1600 - 500) / 2)
    assert layout.origin_y == pytest.approx(0)


def test_layout_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_board(0, 10, 800, 600)
    with pytest.raises(ValueError):
        fit_board(10, 10, 0, 600)


# --------------------------------------------------------------- 설계 불변 규칙
def test_viewer_has_no_pathfinding_code():
    hits: list[str] = []
    for path in sorted(VIEWER_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_TOKENS:
            if token in text:
                hits.append(f"{path.name}: {token}")
    assert not hits, "뷰어에 경로 탐색 코드가 있으면 안 된다: " + ", ".join(hits)


def test_game_package_does_not_eagerly_import_solver():
    """``app.game`` import 만으로 탐색 모듈이 로드되면 안 된다."""
    code = "import app.game, sys; print('app.game.solver' in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SERVER_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"
