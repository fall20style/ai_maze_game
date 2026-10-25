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


class PredictRequest(BaseModel):
    """독립 추론 요청.

    ``observations`` 는 ``MazeEnv.observe()`` 가 만드는 1차원 관측 벡터의 시계열이다.
    마지막 원소가 현재 스텝이며, 길이가 ``seq_len`` 보다 짧으면 왼쪽을 0으로 패딩하고
    길면 최근 ``seq_len`` 개만 사용한다.
    """

    observations: list[list[float]] = Field(..., min_length=1, max_length=4096)
    #: 체크포인트 디렉터리 이름(``wall``) 또는 ``server/checkpoints`` 하위 경로.
    #: 생략하면 게임 서버와 같은 자동 탐색 순서를 쓴다.
    checkpoint: str | None = None
    return_logits: bool = False


class PredictResponse(BaseModel):
    checkpoint: str
    action: int
    action_name: str
    probabilities: list[float]
    logits: list[float] | None = None
    obs_dim: int
    seq_len: int
    #: 실제로 사용한 관측 개수 (``seq_len`` 이하)
    history_length: int
    #: 입력이 ``seq_len`` 보다 길어 앞부분을 버렸는가
    truncated: bool
    latency_ms: float


class ModelInfo(BaseModel):
    checkpoint: str
    obs_dim: int
    seq_len: int
    num_actions: int
    window_size: int
    global_size: int
    map_size: int
    loaded: bool


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
    model_loaded: bool = False
    games: int = 0
