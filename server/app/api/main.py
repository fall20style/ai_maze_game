"""FastAPI 서버 — 게임의 유일한 권위(authority).

실행::

    cd server
    uvicorn app.api.main:app --port 8000

클라이언트(SDL 뷰어)는 ``/games`` 로 판을 만들고 ``/advance`` 또는 ``/action`` 으로
상태를 진행시킨다. **이동 결정·벽 판정·승리 판정은 모두 이 프로세스 안에서만**
일어난다 (PLAN §1 R2·R3).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.api.brains import discover_checkpoint
from app.api.schemas import (
    AdvanceRequest,
    ActionRequest,
    BrainInfo,
    HealthResponse,
    NewGameRequest,
    NewMazeRequest,
    SetBrainRequest,
    StateResponse,
)
from app.api.session import GameSession

SERVER_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="Maze Runner API",
    version="1.0.0",
    description="미로 게임 서버. 이동 결정과 벽 판정의 권위는 서버에 있다.",
)

#: 게임 세션 저장소 (데모용 인메모리)
GAMES: dict[str, GameSession] = {}


def _get(game_id: str) -> GameSession:
    session = GAMES.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown game_id: {game_id}")
    return session


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        checkpoint=discover_checkpoint(SERVER_ROOT),
        games=len(GAMES),
    )


@app.get("/brains", response_model=list[BrainInfo])
def brains() -> list[BrainInfo]:
    checkpoint = discover_checkpoint(SERVER_ROOT)
    ai_label = f"AI · {Path(checkpoint).parent.name}/{Path(checkpoint).name}" if checkpoint else "AI · (체크포인트 없음)"
    return [
        BrainInfo(kind="ai", label=ai_label, requires_client_action=False),
        BrainInfo(kind="rule", label="RULE · 왼손 법칙", requires_client_action=False),
        BrainInfo(kind="human", label="HUMAN · 클라이언트 입력", requires_client_action=True),
    ]


@app.post("/games", response_model=StateResponse, status_code=201)
def create_game(req: NewGameRequest) -> StateResponse:
    if req.size % 2 == 0:
        raise HTTPException(status_code=422, detail="size must be odd")
    try:
        session = GameSession(
            size=req.size,
            algorithm=req.algorithm,
            brain=req.brain,
            rule=req.rule,
            seed=req.seed,
            max_steps=req.max_steps,
            randomize_endpoints=req.randomize_endpoints,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    GAMES[session.id] = session
    return session.state()


@app.get("/games/{game_id}", response_model=StateResponse)
def get_game(game_id: str) -> StateResponse:
    return _get(game_id).state()


@app.post("/games/{game_id}/advance", response_model=StateResponse)
def advance(game_id: str, req: AdvanceRequest) -> StateResponse:
    """Brain 이 스스로 이동한다 (ai/rule)."""
    session = _get(game_id)
    if session.brain.requires_client_action:
        raise HTTPException(
            status_code=409,
            detail="this brain requires /action (human mode)",
        )
    outcome = session.advance(req.steps)
    return session.state(outcome)


@app.post("/games/{game_id}/action", response_model=StateResponse)
def apply_action(game_id: str, req: ActionRequest) -> StateResponse:
    """사람 모드: 클라이언트가 의도한 행동을 보낸다. 서버가 벽을 판정한다."""
    session = _get(game_id)
    if not session.brain.requires_client_action:
        raise HTTPException(
            status_code=409,
            detail="this brain decides on its own; use /advance",
        )
    outcome = session.apply(req.action)
    return session.state(outcome)


@app.post("/games/{game_id}/brain", response_model=StateResponse)
def set_brain(game_id: str, req: SetBrainRequest) -> StateResponse:
    """구현체 교체 (같은 미로에서 다시 시작)."""
    session = _get(game_id)
    try:
        session.set_brain(req.brain, req.rule)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return session.state()


@app.post("/games/{game_id}/reset", response_model=StateResponse)
def reset(game_id: str) -> StateResponse:
    session = _get(game_id)
    session.reset()
    return session.state()


@app.post("/games/{game_id}/new-maze", response_model=StateResponse)
def new_maze(game_id: str, req: NewMazeRequest) -> StateResponse:
    session = _get(game_id)
    session.new_maze(req.seed)
    return session.state()


@app.delete("/games/{game_id}")
def delete_game(game_id: str) -> dict[str, bool]:
    if GAMES.pop(game_id, None) is None:
        raise HTTPException(status_code=404, detail=f"unknown game_id: {game_id}")
    return {"closed": True}
