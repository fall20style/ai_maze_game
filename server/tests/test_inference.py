"""독립 추론 API 계약 테스트.

검증하는 것
-----------
1. ``/models`` 가 관측 차원·시퀀스 길이 등 모델 계약을 알려준다.
2. ``/predict`` 가 게임 세션 없이 행동·확률을 계산한다.
3. 짧은 입력은 왼쪽 패딩, 긴 입력은 최근 ``seq_len`` 개만 쓴다.
4. 잘못된 관측 차원/체크포인트/경로는 4xx 로 거절한다.
5. 같은 입력은 같은 결과를 낸다 (결정론적 추론).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import GAMES, app

SERVER_DIR = Path(__file__).resolve().parents[1]
AI_CHECKPOINT = SERVER_DIR / "checkpoints" / "wall" / "best.pt"
PRUNED_CHECKPOINT = SERVER_DIR / "checkpoints" / "wall_pruned" / "best.pt"

needs_ai = pytest.mark.skipif(
    not AI_CHECKPOINT.exists(), reason="AI 체크포인트가 없다 (gitignore 대상)"
)
needs_pruned = pytest.mark.skipif(
    not PRUNED_CHECKPOINT.exists(), reason="프루닝 체크포인트가 없다"
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    GAMES.clear()


@pytest.fixture
def contract(client) -> dict:
    response = client.get("/models")
    assert response.status_code == 200, response.text
    return response.json()


def _obs(dim: int, value: float = 0.0) -> list[float]:
    return [value] * dim


@needs_ai
def test_health_reports_the_loaded_model(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model_loaded"] is True
    assert health["checkpoint"]


@needs_ai
def test_models_reports_the_observation_contract(client, contract):
    assert contract["obs_dim"] > 0
    assert contract["seq_len"] > 0
    assert contract["num_actions"] == 4
    assert contract["loaded"] is True
    assert contract["checkpoint"].endswith("best.pt")


@needs_ai
def test_models_can_select_a_checkpoint_by_name(client):
    response = client.get("/models", params={"checkpoint": "wall"})
    assert response.status_code == 200, response.text
    assert response.json()["checkpoint"].replace("\\", "/").endswith("wall/best.pt")


@needs_ai
def test_predict_returns_action_and_probabilities(client, contract):
    dim = contract["obs_dim"]
    response = client.post(
        "/predict",
        json={"observations": [_obs(dim, 0.0), _obs(dim, 1.0), _obs(dim, 0.5)]},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["action"] in range(contract["num_actions"])
    assert body["action_name"] in ("UP", "DOWN", "LEFT", "RIGHT")
    assert len(body["probabilities"]) == contract["num_actions"]
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-5
    assert body["history_length"] == 3
    assert body["truncated"] is False
    assert body["logits"] is None
    assert body["latency_ms"] >= 0.0
    assert body["checkpoint"] == contract["checkpoint"]


@needs_ai
def test_predict_is_deterministic_and_returns_logits(client, contract):
    dim = contract["obs_dim"]
    payload = {
        "observations": [_obs(dim, float(i) / 10.0) for i in range(8)],
        "return_logits": True,
    }
    first = client.post("/predict", json=payload)
    second = client.post("/predict", json=payload)

    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    assert a["action"] == b["action"]
    assert a["probabilities"] == b["probabilities"]
    assert a["logits"] == b["logits"]
    assert len(a["logits"]) == contract["num_actions"]


@needs_ai
def test_predict_left_pads_a_short_sequence(client, contract):
    dim = contract["obs_dim"]
    body = client.post("/predict", json={"observations": [_obs(dim)]}).json()
    assert body["history_length"] == 1
    assert body["truncated"] is False
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-5


@needs_ai
def test_predict_truncates_a_long_sequence(client, contract):
    dim, seq_len = contract["obs_dim"], contract["seq_len"]
    observations = [_obs(dim, float(i)) for i in range(seq_len + 5)]
    body = client.post("/predict", json={"observations": observations}).json()
    assert body["history_length"] == seq_len
    assert body["truncated"] is True


@needs_ai
def test_predict_rejects_wrong_observation_dimension(client, contract):
    dim = contract["obs_dim"]
    response = client.post("/predict", json={"observations": [_obs(dim + 1)]})
    assert response.status_code == 422
    assert "expected obs_dim" in response.json()["detail"]


def test_predict_rejects_empty_observations(client):
    response = client.post("/predict", json={"observations": []})
    assert response.status_code == 422


@needs_ai
def test_predict_rejects_an_unknown_checkpoint(client, contract):
    response = client.post(
        "/predict",
        json={
            "observations": [_obs(contract["obs_dim"])],
            "checkpoint": "does-not-exist",
        },
    )
    assert response.status_code == 404


@needs_ai
def test_predict_rejects_a_checkpoint_outside_the_checkpoint_root(
    client, contract, tmp_path
):
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a checkpoint")
    response = client.post(
        "/predict",
        json={
            "observations": [_obs(contract["obs_dim"])],
            "checkpoint": str(outside),
        },
    )
    assert response.status_code == 400


@needs_pruned
def test_predict_works_with_the_pruned_checkpoint(client, contract):
    response = client.post(
        "/predict",
        json={
            "observations": [_obs(contract["obs_dim"], 0.25)],
            "checkpoint": "wall_pruned",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["checkpoint"].replace("\\", "/").endswith(
        "wall_pruned/best.pt"
    )
