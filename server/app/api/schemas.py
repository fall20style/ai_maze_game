"""API 스키마 (pydantic v2).

클라이언트는 이 스키마만 알고, 미로 규칙/경로 탐색/모델은 전혀 모른다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BrainKind = Literal["ai", "rule", "human"]
RuleKind = Literal["wall", "frontier"]


class NewGameRequest(BaseModel):
    size: int = Field(default=15, ge=5, le=31)
    algorithm: Literal["dfs", "prim"] = "dfs"
    brain: BrainKind = "ai"
    rule: RuleKind = "wall"
    seed: int | None = None
    max_steps: int | None = Field(default=None, ge=16, le=100_000)
    randomize_endpoints: bool = True


class SetBrainRequest(BaseModel):
    brain: BrainKind
    rule: RuleKind = "wall"


class AdvanceRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=64)


class ActionRequest(BaseModel):
    action: int = Field(ge=0, le=3, description="0=UP 1=DOWN 2=LEFT 3=RIGHT")


class NewMazeRequest(BaseModel):
    seed: int | None = None


class MazeState(BaseModel):
    width: int
    height: int
    grid: list[list[int]]
    start: list[int]
    goal: list[int]
    algorithm: str
    seed: int


class StateResponse(BaseModel):
    game_id: str
    brain: str
    brain_label: str
    rule: str | None
    window_size: int
    global_size: int
    map_size: int

    maze: MazeState
    position: list[int]
    visited: list[list[int]]
    trajectory: list[list[int]]
    step_count: int
    max_steps: int
    last_action: int
    won: bool
    done: bool

    #: 이번 요청으로 적용된 행동 (없으면 null)
    action: int | None = None
    moved: bool = False
    hit_wall: bool = False
    #: AI 브레인의 행동 확률 (rule/human 은 null)
    probabilities: list[float] | None = None
    #: 클라이언트가 행동을 보내야 하는가
    awaiting_action: bool = False


class BrainInfo(BaseModel):
    kind: str
    label: str
    requires_client_action: bool


class HealthResponse(BaseModel):
    status: str
    checkpoint: str | None = None
    games: int = 0
