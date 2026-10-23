"""서버 API 테스트 — **서버가 유일한 권위**인지 검증한다 (PLAN §1 R2·R3).

여기서 확인하는 것
------------------
1. 미로 생성·이동·승리 판정이 서버에서만 일어난다 (클라이언트는 상태만 받는다).
2. 사람 모드에서도 벽 판정은 서버가 한다 — 클라이언트가 보낸 행동을 그대로 믿지 않는다.
3. 구현체(ai/rule/human)를 실행 중에 바꿀 수 있고, 관측 정의도 함께 바뀐다.
4. 잘못된 game_id / 잘못된 모드의 호출은 4xx 로 거절한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import GAMES, app

SERVER_DIR = Path(__file__).resolve().parents[1]
AI_CHECKPOINT = SERVER_DIR / "checkpoints" / "wall" / "best.pt"

needs_ai = pytest.mark.skipif(
    not AI_CHECKPOINT.exists(), reason="AI 체크포인트가 없다 (gitignore 대상)"
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    GAMES.clear()


def _create(client, **kwargs) -> dict:
    payload = {"size": 9, "seed": 3, "brain": "rule", "rule": "wall"}
    payload.update(kwargs)
    response = client.post("/games", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


# ------------------------------------------------------------------ 기본 계약
def test_health_and_brains(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"

    brains = {b["kind"]: b for b in client.get("/brains").json()}
    assert set(brains) == {"ai", "rule", "human"}
    assert brains["human"]["requires_client_action"] is True
    assert brains["rule"]["requires_client_action"] is False


def test_create_returns_a_full_state(client):
    state = _create(client)
    assert state["game_id"]
    assert state["brain"] == "rule"
    assert state["rule"] == "wall"
    assert state["maze"]["width"] == state["maze"]["height"] == 9
    assert state["position"] == state["maze"]["start"]
    assert state["step_count"] == 0
    assert state["done"] is False
    assert state["awaiting_action"] is False
    assert state["trajectory"] == [state["maze"]["start"]]


def test_rule_brain_solves_the_maze_on_the_server(client):
    state = _create(client)
    game_id = state["game_id"]

    for _ in range(state["max_steps"] + 5):
        if state["done"]:
            break
        response = client.post(f"/games/{game_id}/advance", json={"steps": 1})
        assert response.status_code == 200
        state = response.json()

    assert state["done"] is True
    assert state["won"] is True
    # 클라이언트는 아무 판단도 하지 않았고, 서버가 상태를 완성했다
    assert state["position"] == state["maze"]["goal"]
    assert state["step_count"] > 0


def test_advance_after_done_is_a_no_op(client):
    state = _create(client)
    game_id = state["game_id"]
    while not state["done"]:
        state = client.post(f"/games/{game_id}/advance", json={"steps": 8}).json()

    frozen = client.post(f"/games/{game_id}/advance", json={"steps": 8}).json()
    assert frozen["step_count"] == state["step_count"]
    assert frozen["position"] == state["position"]


# ------------------------------------------------------------- 사람 모드 (R3)
def test_human_mode_rejects_advance_and_accepts_actions(client):
    state = _create(client, brain="human")
    game_id = state["game_id"]
    assert state["awaiting_action"] is True

    conflict = client.post(f"/games/{game_id}/advance", json={"steps": 1})
    assert conflict.status_code == 409

    moved = client.post(f"/games/{game_id}/action", json={"action": 0}).json()
    assert moved["action"] == 0
    assert moved["step_count"] == 1


def test_server_rejects_client_action_that_hits_a_wall(client):
    """클라이언트가 벽으로 이동하라고 해도 서버가 거절한다 (벽 판정 권위)."""
    state = _create(client, brain="human")
    game_id = state["game_id"]
    start = tuple(state["position"])

    # 시작 칸에서 상하좌우 중 벽인 방향을 찾는다 (완벽 미로의 시작 칸은 최소 하나가 벽)
    grid = state["maze"]["grid"]
    blocked = [
        action
        for action, (dx, dy) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0)))
        if grid[start[1] + dy][start[0] + dx] == 1
    ]
    assert blocked, "시작 칸이 사방이 뚫려 있으면 이 테스트를 쓸 수 없다"

    result = client.post(f"/games/{game_id}/action", json={"action": blocked[0]}).json()
    assert result["hit_wall"] is True
    assert result["moved"] is False
    assert tuple(result["position"]) == start
    assert result["step_count"] == 1


def test_auto_brain_rejects_client_actions(client):
    state = _create(client)
    response = client.post(f"/games/{state['game_id']}/action", json={"action": 0})
    assert response.status_code == 409


# ------------------------------------------------------------- 구현체 교체
def test_switching_brain_restarts_the_episode_on_the_same_maze(client):
    state = _create(client)
    game_id = state["game_id"]
    client.post(f"/games/{game_id}/advance", json={"steps": 4})

    switched = client.post(
        f"/games/{game_id}/brain", json={"brain": "rule", "rule": "frontier"}
    ).json()

    assert switched["brain"] == "rule"
    assert switched["rule"] == "frontier"
    assert switched["step_count"] == 0
    assert switched["position"] == switched["maze"]["start"]
    # 같은 미로를 유지한다
    assert switched["maze"]["seed"] == state["maze"]["seed"]
    # frontier 는 지식 지도가 필요하다
    assert switched["map_size"] == switched["maze"]["width"]


def test_switching_to_human_awaits_client_input(client):
    state = _create(client)
    switched = client.post(
        f"/games/{state['game_id']}/brain", json={"brain": "human"}
    ).json()
    assert switched["brain"] == "human"
    assert switched["awaiting_action"] is True


# ------------------------------------------------------------- 미로 / 수명주기
def test_new_maze_and_reset(client):
    state = _create(client)
    game_id = state["game_id"]
    client.post(f"/games/{game_id}/advance", json={"steps": 3})

    reset = client.post(f"/games/{game_id}/reset").json()
    assert reset["step_count"] == 0
    assert reset["maze"]["seed"] == state["maze"]["seed"]

    fresh = client.post(f"/games/{game_id}/new-maze", json={"seed": 12345}).json()
    assert fresh["maze"]["seed"] == 12345
    assert fresh["step_count"] == 0


def test_delete_removes_the_game(client):
    state = _create(client)
    game_id = state["game_id"]
    assert client.delete(f"/games/{game_id}").status_code == 200
    assert client.get(f"/games/{game_id}").status_code == 404
    assert client.delete(f"/games/{game_id}").status_code == 404


def test_unknown_game_is_404(client):
    assert client.get("/games/nope").status_code == 404
    assert client.post("/games/nope/advance", json={"steps": 1}).status_code == 404
    assert client.post("/games/nope/action", json={"action": 0}).status_code == 404


def test_even_size_is_rejected(client):
    assert client.post("/games", json={"size": 10}).status_code == 422


# ------------------------------------------------------------------- AI 구현체
@needs_ai
def test_ai_brain_plays_and_reports_probabilities(client):
    state = _create(client, brain="ai")
    assert state["brain"] == "ai"

    state = client.post(
        f"/games/{state['game_id']}/advance", json={"steps": 1}
    ).json()
    assert state["action"] is not None
    probs = state["probabilities"]
    assert probs is not None and len(probs) == 4
    assert abs(sum(probs) - 1.0) < 1e-4
