"""GUI 클라이언트 테스트 — 클라이언트가 **판단하지 않는지** 검증한다.

핵심
----
- ``viewer`` 는 모델(torch)도, 미로 규칙도, 경로 탐색도 import 하지 않는다 (R1).
- 서버 응답이 없으면 클라이언트는 **이동하지 않는다** (R4).
- 서버 상태는 그대로 화면 상태로 반영된다 (클라이언트가 재계산하지 않는다).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from viewer.api_client import ApiError, GameApi
from viewer.remote import RemoteSession

SERVER_DIR = Path(__file__).resolve().parents[1]


def _state(**overrides) -> dict:
    """서버가 내려주는 최소 상태 (9x9)."""
    grid = [[1] * 9 for _ in range(9)]
    for y in range(1, 8):
        for x in range(1, 8):
            grid[y][x] = 0
    state = {
        "game_id": "abc123",
        "brain": "rule",
        "brain_label": "RULE · 왼손 법칙",
        "rule": "wall",
        "window_size": 3,
        "global_size": 0,
        "map_size": 0,
        "maze": {
            "width": 9,
            "height": 9,
            "grid": grid,
            "start": [1, 1],
            "goal": [7, 7],
            "algorithm": "dfs",
            "seed": 42,
        },
        "position": [1, 1],
        "visited": [[1, 1]],
        "trajectory": [[1, 1]],
        "step_count": 0,
        "max_steps": 128,
        "last_action": 4,
        "won": False,
        "done": False,
        "action": None,
        "moved": False,
        "hit_wall": False,
        "probabilities": None,
        "awaiting_action": False,
    }
    state.update(overrides)
    return state


class StubApi:
    """``GameApi`` 대역. 서버 응답을 그대로 돌려준다."""

    def __init__(self, state: dict) -> None:
        self.state = state
        self.calls: list[tuple] = []
        self.fail = False

    def _maybe_fail(self) -> None:
        if self.fail:
            raise ApiError("서버에 연결할 수 없습니다 (stub)")

    def advance(self, game_id: str, steps: int = 1) -> dict:
        self.calls.append(("advance", game_id, steps))
        self._maybe_fail()
        return self.state

    def action(self, game_id: str, action: int) -> dict:
        self.calls.append(("action", game_id, action))
        self._maybe_fail()
        return self.state

    def set_brain(self, game_id: str, brain: str, rule: str = "wall") -> dict:
        self.calls.append(("brain", game_id, brain, rule))
        self._maybe_fail()
        return self.state

    def reset(self, game_id: str) -> dict:
        self.calls.append(("reset", game_id))
        self._maybe_fail()
        return self.state

    def new_maze(self, game_id: str, seed: int | None = None) -> dict:
        self.calls.append(("new_maze", game_id, seed))
        self._maybe_fail()
        return self.state


# ------------------------------------------------------------------ R1: 순수성
def test_client_modules_do_not_import_torch_or_the_game_rules():
    """클라이언트는 모델도 미로 규칙도 모른다."""
    code = (
        "import sys, viewer.api_client, viewer.remote, viewer.render\n"
        "bad = [m for m in ('torch', 'transformers', 'app.ml.policy', "
        "'app.game.env', 'app.game.maze', 'app.game.solver') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SERVER_DIR,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "SDL_VIDEODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"클라이언트가 서버 코드를 import 했다: {proc.stdout}"


def test_viewer_has_no_pathfinding_code():
    """이름 규칙으로 한 번 더 확인한다 (PLAN §1 R1)."""
    forbidden = ("solver", "bfs", "astar", "a_star", "dijkstra", "distance_map", "pathfind", "flood")
    hits = []
    for path in sorted((SERVER_DIR / "viewer").glob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        hits += [f"{path.name}: {token}" for token in forbidden if token in text]
    assert not hits, "뷰어에 경로 탐색 코드가 있으면 안 된다: " + ", ".join(hits)


# ------------------------------------------------------------------ R4: 오프라인
def test_no_movement_when_the_server_does_not_answer():
    api = StubApi(_state())
    session = RemoteSession(api, _state())
    api.fail = True

    assert session.advance(now=1.0) is False
    assert session.error is not None
    assert session.env.position == (1, 1)
    assert session.env.step_count == 0
    assert session.env.trajectory == [(1, 1)]
    # 실패 직후에는 재요청하지 않는다 (요청 폭주 방지)
    assert session.wants_step(now=1.01) is False
    assert session.wants_step(now=2.0) is True


def test_error_clears_after_the_server_recovers():
    api = StubApi(_state())
    session = RemoteSession(api, _state())
    api.fail = True
    session.advance(now=1.0)
    assert session.error

    api.fail = False
    moved = _state(
        position=[2, 1],
        visited=[[1, 1], [2, 1]],
        trajectory=[[1, 1], [2, 1]],
        step_count=1,
        action=3,
        moved=True,
    )
    api.state = moved
    assert session.advance(now=2.0) is True
    assert session.error is None
    assert session.env.position == (2, 1)
    assert session.env.step_count == 1


def test_game_api_raises_on_unreachable_server():
    api = GameApi("http://127.0.0.1:9", timeout=0.25)  # discard 포트: 연결 거부
    with pytest.raises(ApiError):
        api.health()


# ------------------------------------------------------- 서버 상태 → 화면 상태
def test_server_state_is_mirrored_not_recomputed():
    state = _state()
    api = StubApi(state)
    session = RemoteSession(api, state)

    assert session.maze.width == 9
    assert session.maze.start == (1, 1)
    assert session.maze.goal == (7, 7)
    assert session.floor_total == 49
    assert session.brain.label == "RULE · 왼손 법칙"

    new_state = _state(
        position=[2, 2],
        visited=[[1, 1], [2, 1], [2, 2]],
        trajectory=[[1, 1], [2, 1], [2, 2]],
        step_count=2,
        last_action=1,
    )
    api.state = new_state
    assert session.advance(now=1.0) is True
    assert session.env.position == (2, 2)
    assert session.env.visited == {(1, 1), (2, 1), (2, 2)}
    assert session.env.last_action == 1


def test_human_mode_does_not_poll_the_server_for_steps():
    state = _state(brain="human", brain_label="HUMAN", awaiting_action=True)
    api = StubApi(state)
    session = RemoteSession(api, state)

    assert session.awaiting_action is True
    assert session.wants_step(now=100.0) is False

    session.send_action(0, now=1.0)
    assert api.calls[-1] == ("action", "abc123", 0)


def test_brain_switch_is_forwarded_to_the_server():
    state = _state()
    api = StubApi(state)
    session = RemoteSession(api, state)

    api.state = _state(brain="human", brain_label="HUMAN", awaiting_action=True)
    assert session.set_brain("human", "wall", now=1.0) is True
    assert api.calls[-1] == ("brain", "abc123", "human", "wall")
    assert session.brain_kind == "human"
    assert session.awaiting_action is True


# ----------------------------------------------------------- 구현체 선택 칩
def test_brain_selector_rects_are_ordered_and_clickable():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.font.init()
    from viewer.render import BRAIN_ORDER, brain_selector_rects

    rects = brain_selector_rects(1180)
    assert [kind for kind, _ in rects] == list(BRAIN_ORDER)
    assert all(rect.width > 0 and rect.height > 0 for _, rect in rects)
    for left, right in zip(rects, rects[1:]):
        assert left[1].right < right[1].left, "칩이 겹친다"
    assert rects[-1][1].right < 1180, "칩이 화면 밖으로 나간다"

    # 클릭 판정: 칩 중앙이 그 칩에만 걸린다
    for kind, rect in rects:
        hits = [k for k, r in brain_selector_rects(1180) if r.collidepoint(rect.center)]
        assert hits == [kind]
