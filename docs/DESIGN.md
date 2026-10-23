# 소프트웨어 설계 문서 — Maze Runner

> 버전 1.0 · 작성 2026-10-24 · 기준 커밋 `149afb0`
>
> 관련 문서: [빌드/실행 안내](./BUILD.md) · [의사결정 기록](./DECISIONS.md) · [계획/실험 로그](../PLAN.md) · [README](../README.md)

---

## 1. 목적과 범위

이 문서는 **현재 구현된 시스템**의 구조와 계약을 설명한다. 목표는 다음 한 문장으로 요약된다.

> 미로를 푸는 지능을 **서버 안에** 가두고, GUI 클라이언트는 서버가 보내준 상태를 **그리기만** 한다.

범위에 포함되는 것:

- 미로 생성, 관측 인코딩, 이동 결정, 벽 판정, 승리 판정
- 세 가지 이동 결정 구현체(`ai` / `rule` / `human`)와 실행 중 교체
- HTTP API 계약과 SDL 클라이언트
- 학습 파이프라인(합성 데이터 → LoRA-BERT → 롤아웃 평가 → DAgger → 벤치마크)

범위에서 제외되는 것(현재 데모 구현의 한계):

- 사용자 인증, 영속 저장(DB), 다중 사용자 세션 관리
- 서버 수평 확장(다중 워커/다중 인스턴스)
- 클라이언트 배포 패키징

---

## 2. 설계 원칙 — 불변 규칙 R1~R4

PLAN §1 에서 정의한 네 가지 규칙이 전체 구조를 결정한다. **위반하면 설계 실패**로 간주하며, 테스트로 고정한다.

| # | 규칙 | 구현 방식 | 검증 |
|---|---|---|---|
| R1 | 클라이언트에 경로 탐색 코드가 없다 | `viewer/` 는 `urllib` 로 JSON 만 주고받고, 미로는 받은 격자를 그린다 | `tests/test_client.py::test_viewer_has_no_pathfinding_code`, `tests/test_viewer.py::test_viewer_has_no_pathfinding_code` (금지 토큰 스캔) |
| R2 | 클라이언트는 스스로 이동을 결정하지 않는다 | `/advance`(자동) 또는 `/action`(사람)만 호출. 상태는 응답을 **복사** | `test_server_state_is_mirrored_not_recomputed` |
| R3 | 벽 충돌 판정의 권위는 서버 | `MazeEnv.step` 이 `maze.is_wall` 로 판정. 클라이언트는 판정하지 않음 | `test_server_rejects_client_action_that_hits_a_wall` |
| R4 | 서버 응답이 없으면 이동하지 않는다 | 실패 시 `RemoteSession.error` 설정, 스텝 중단 + 배너 | `test_no_movement_when_the_server_does_not_answer` |

추가로 `test_client_modules_do_not_import_torch_or_the_game_rules` 가 **별도 프로세스**에서
`torch`·`transformers`·`app.game.env`·`app.game.maze`·`app.game.solver` 가 로드되지 않음을 확인한다. 즉 R1은 "코드에 토큰이 없다"를
넘어 "클라이언트 런타임에 게임/모델 코드가 없다"까지 검사한다.

> 미로 격자 자체는 렌더링을 위해 클라이언트로 내려온다. 하지만 클라이언트는 그 격자로
> **이동 가능 여부를 판단하지 않는다**. `MazeView.is_wall` 은 이미 받은 셀을 그리기 위한 조회다.
> 신뢰 경계는 "클라이언트가 정직하다"가 아니라 "서버가 최종 판정한다"에 있다.

---

## 3. 시스템 아키텍처

```
┌──────────────────────────────────────────────┐
│ viewer/  (pygame SDL, 별도 프로세스)          │
│                                              │
│  app.py        이벤트 루프 · 키/마우스 입력    │
│  api_client.py stdlib urllib HTTP            │
│  remote.py     JSON → 화면 상태 어댑터         │
│  render.py     그리기 전용                     │
│  layout.py     순수 레이아웃 계산              │
│                                              │
│  경로 탐색 0줄 · 모델 0줄 · 벽 판정 0줄        │
└──────────────────────┬───────────────────────┘
                       │ HTTP/JSON (127.0.0.1:8000)
                       ▼
┌──────────────────────────────────────────────┐
│ app/api/  (FastAPI + uvicorn, 단일 프로세스)   │
│                                              │
│  main.py      엔드포인트 · GAMES 저장소        │
│  session.py   GameSession (미로+env+brain)     │
│  brains.py    ai | rule | human               │
│  schemas.py   요청/응답 계약 (pydantic v2)     │
│                                              │
│  ┌─────────────── 게임 권위 ───────────────┐  │
│  │ 미로 생성 · 관측 인코딩 · 이동 결정       │  │
│  │ 벽 판정 · 승리 판정 · 최대 스텝 종료      │  │
│  └──────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

### 프로세스 분리

| 프로세스 | 소유하는 것 | 소유하지 않는 것 |
|---|---|---|
| 서버 (`app.api`) | 미로, 환경 상태, 이동 결정, 벽/승리 판정, 체크포인트 | 렌더링, 입력 수집 |
| 클라이언트 (`viewer`) | 렌더링, 입력, 애니메이션 보간, 오류 배너 | 이동 결정, 벽 판정, 경로 탐색 |

두 프로세스는 **HTTP JSON 계약**(`app/api/schemas.py`)만 공유한다. 클라이언트는 서버 패키지를
import 하지 않으므로, 서버 내부를 리팩터링해도 클라이언트가 깨지지 않는다.

---

## 4. 구성 요소

### 4.1 서버 API 계층 (`server/app/api/`)

| 파일 | 책임 |
|---|---|
| `main.py` | FastAPI 앱, 엔드포인트, 인메모리 `GAMES: dict[str, GameSession]` |
| `session.py` | `GameSession` — 한 판의 수명주기와 상태 스냅샷 |
| `brains.py` | `ServerBrain` 프로토콜과 세 구현체, 체크포인트 탐색 |
| `schemas.py` | 요청/응답 pydantic 모델 — 클라이언트와의 유일한 계약 |

`GameSession` 은 다음을 소유한다.

- `maze: Maze` — 시드/알고리즘으로 생성된 불변 격자
- `env: MazeEnv` — 위치, 방문 집합, 지식 지도, 스텝 수
- `brain: ServerBrain` — 이동 결정 주체
- `lock: threading.Lock` — 상태 변경 임계 구역

`state()` 는 `StateResponse` 를 만들어 클라이언트에 그대로 보낸다. 여기에는 **이번 요청으로
적용된 행동**(`action`, `moved`, `hit_wall`)과 AI 확률(`probabilities`)도 포함된다.

### 4.2 게임 코어 (`server/app/game/`)

| 파일 | 책임 | 서버 사용 | 학습 사용 |
|---|---|---|---|
| `maze.py` | 시드 기반 DFS/Prim 생성, 시작/목표 배치, `is_wall` | ✅ | ✅ |
| `actions.py` | `Action` IntEnum(0~3), 델타, 역방향, `LAST_ACTION_DIM=5` | ✅ | ✅ |
| `env.py` | `MazeEnv`, 관측 인코더, `pack_payload`, `default_max_steps` | ✅ | ✅ |
| `expert.py` | 왼손 법칙(`wall_follow_action`), frontier(`frontier_action`) | ✅ (`rule`) | ✅ (라벨) |
| `solver.py` | BFS 최단 경로/거리 지도 | ❌ (API 경로에서 호출 금지) | ✅ (평가 지표) |

`solver.py` 는 **평가 전용**이다. `app/game/__init__.py` 는 solver 를 지연 import 하므로
`import app.game` 만으로 경로 탐색 모듈이 메모리에 올라오지 않는다
(`tests/test_viewer.py::test_game_package_does_not_eagerly_import_solver`).

### 4.3 ML 계층 (`server/app/ml/`)

| 파일 | 책임 |
|---|---|
| `lora.py` | BERT 선형층에 LoRA A/B 주입, 학습 가능 파라미터 표시, 파라미터 그룹 분리 |
| `policy.py` | `PolicyConfig`, `BertPolicy`(Linear → BERT → action head), 저장/로드 |
| `runner.py` | `PolicyRunner` — 슬라이딩 윈도우로 학습과 동일한 `(1, T, D)` 입력 생성 |

`BertPolicy` 구조:

```
obs (B, T, obs_dim)
  → Linear(obs_dim → 128) + LayerNorm + GELU + Dropout
  → BERT-tiny (2 layers, 2 heads, hidden 128, LoRA)
  → 마지막 유효 토큰 선택 (left padding + attention mask)
  → Linear(128 → 4)  → logits
```

`PolicyConfig` 는 `window_size` / `global_size` / `map_size` 를 **체크포인트에 함께 저장**한다.
서버는 이 값으로 `MazeEnv` 를 만들어 학습/서빙 관측 정의를 일치시킨다. 이것이 학습-서빙
불일치를 막는 핵심 장치다.

`PolicyRunner` 는 최근 `seq_len`(기본 32) 개 관측을 유지하고, 부족한 앞부분은 0 + mask 로
왼쪽 패딩한다. 서버와 평가 스크립트가 같은 클래스를 쓰므로 추론 경로가 하나다.

### 4.4 클라이언트 (`server/viewer/`)

| 파일 | 책임 |
|---|---|
| `app.py` | pygame 이벤트 루프, 키/마우스 → API 호출, 스냅샷 |
| `api_client.py` | `GameApi` — stdlib `urllib` 만 사용하는 얇은 REST 래퍼 |
| `remote.py` | `RemoteSession` — 서버 JSON을 `MazeView`/`EnvView`/`BrainView` 로 변환 |
| `render.py` | 미로/궤적/HUD/확률 바/구현체 칩/오프라인 배너 그리기 |
| `layout.py` | pygame 비의존 순수 레이아웃 계산 (테스트 가능) |

`RemoteSession` 은 **판단하지 않는다.** `wants_step()` 은 "지금 `/advance` 를 호출해도 되는가"만
계산하고, 실제 이동 결과는 서버 응답을 복사해 반영한다. `error` 가 설정되면 `advance()` 는
호출되지 않고, 실패 후 최소 0.5초 백오프를 둔다.

### 4.5 학습 파이프라인 (`server/training/`)

| 파일 | 역할 |
|---|---|
| `gen_dataset.py` | 합성 궤적 생성. 전문가 라벨 + 노이즈 궤적(실수 복구 학습) |
| `dataset.py` | `torch.utils.data.Dataset` — 시퀀스 샘플 |
| `train.py` | LoRA-BERT 지도학습, 검증 정확도 기준 `best.pt` 저장 |
| `evaluate.py` | 롤아웃 성공률/스텝/optimality/루프율/벽 충돌 |
| `dagger.py` | 온폴리시 수집 후 기존 데이터셋에 병합 |
| `compare_brains.py` | 같은 미로 집합에서 `ai` vs `rule`(`wall` / `frontier`) 짝지은 비교 |

데이터 분할은 **미로 단위 8:1:1** 이다. 같은 미로가 train/val/test 에 걸치지 않는다
(`test_no_maze_leaks_between_splits`).

### 4.6 런처 (`test_launcher.py`)

서버와 클라이언트를 한 번에 띄우는 개발용 오케스트레이터다.

1. `server/` 를 cwd 로 uvicorn 을 subprocess 실행
2. `/health` 가 200 이 될 때까지 polling (프로세스 조기 종료도 감지)
3. `viewer.app` 을 subprocess 실행 (알 수 없는 인자는 그대로 전달)
4. 클라이언트 종료 시 서버 terminate → kill 순으로 정리
5. 포트가 사용 중이면 OS 가 준 빈 포트로 자동 이동

---

## 5. 데이터 모델과 관측 인코딩

### 5.1 미로

- 격자는 `(height, width)` `uint8`, `1=WALL`, `0=FLOOR`
- 크기는 **홀수, 5 이상** (API 는 5~31 허용)
- 외곽 한 줄은 항상 벽
- 좌표계는 `(x, y)`, 접근은 `grid[y][x]`
- 기본 알고리즘 `dfs`(Recursive Backtracker), 대안 `prim`(Randomized Prim)
- 기본 시작/목표는 **무작위**: 시작점을 무작위로 뽑고, 목표는 시작점에서 BFS 최장 거리 칸
  (`randomize_endpoints=True`). 고정 코너가 필요하면 `False`.

### 5.2 관측 벡터

`window_size=3`, `global_size=5`, `map_size=0` → **54차원**.

| 구간 | 차원 | 내용 |
|---|---|---|
| `pos` | 2 | 현재 위치 `x/(W-1), y/(H-1)` |
| `local` | w² | 주변 창의 셀 코드 (0=unknown, 1=wall, 2=floor) |
| `visited` | w² | 주변 창의 방문 여부 (0/1) |
| `global` | g² | 전역 방문 요약 — 미로를 g×g 블록으로 나눈 방문 여부 |
| `known_wall` | m² | 지식 지도: 직접 본 벽 (0/1) |
| `known_floor` | m² | 지식 지도: 직접 본 바닥 (0/1) |
| `goal_rel` | 2 | 목표까지 상대 벡터 (축별 정규화) |
| `dist` | 1 | 맨해튼 거리 `/(W+H-2)` |
| `last_action` | 5 | 직전 행동 one-hot (UP/DOWN/LEFT/RIGHT/NONE) |
| `progress` | 1 | `step / max_steps` |

차원 공식:

```
observation_dim(w, g, m) = 11 + 2w² + g² + 2m²
```

| 설정 | 차원 | 사용처 |
|---|---|---|
| `(3, 5, 0)` | **54** | `checkpoints/wall`, `checkpoints/v3`, `human` |
| `(3, 0, 15)` | **479** | `checkpoints/frontier` (지식 지도 15×15) |

설계 근거:

- **`global` 이 필요한 이유** — 3×3 창만으로는 "이 구역을 이미 지났다"를 알 수 없어 같은 구역을
  맴돈다(실측 루프율 87%). 전역 요약은 에이전트 자신의 이동 기록일 뿐 정답을 알려주지 않는다.
- **`known_wall`/`known_floor` 가 필요한 이유** — 더 강한 전문가(frontier)의 라벨을 학습하려면
  정책이 **자기 기억(믿음 상태)** 을 관측으로 받아야 한다. 이 채널도 직접 본 것만 누적하며
  정답 미로가 아니다. `map_size` 는 미로 크기 이상이어야 한다(작으면 예외).

`knows_full_maze=False` 이면 방문하지 않은 셀은 `UNKNOWN` 으로 보인다. 기본값은 `True` 이며,
현재 학습/서빙은 모두 `True`(3×3 창 안의 실제 셀을 봄)를 쓴다.

### 5.3 관측 경로의 단일성

같은 관측을 만드는 경로가 두 개다.

1. 서버 내부: `MazeEnv.observe()` → `np.ndarray`
2. API/디버그: `MazeEnv.observe_payload()` → JSON, `pack_payload()` → `np.ndarray`

두 경로가 **비트 단위로 동일**해야 한다. `test_pack_payload_matches_observe` 와
`test_pack_payload_roundtrip_with_map` 이 이를 고정한다. 관측 정의가 두 곳으로 갈라지면
성능이 조용히 망가지기 때문에, 벡터 생성은 `env.py` 한 곳에만 둔다.

---

## 6. 행동과 에피소드 규칙

### 6.1 행동 공간

| 인덱스 | 이름 | 델타 `(dx, dy)` |
|---|---|---|
| 0 | `UP` | `(0, -1)` |
| 1 | `DOWN` | `(0, 1)` |
| 2 | `LEFT` | `(-1, 0)` |
| 3 | `RIGHT` | `(1, 0)` |

인덱스는 CrossEntropy 타깃으로 그대로 쓰이므로 **순서를 바꾸면 안 된다**.
`last_action` 은 0~3 외에 4(`NO_ACTION_INDEX`)를 가질 수 있어 5차원이다.

### 6.2 스텝 규칙 (`MazeEnv.step`)

- 벽으로 이동: **위치 불변**, `moved=False`, `hit_wall=True`, **스텝 1 소모**
- 바닥으로 이동: 위치 갱신, `visited`/`trajectory` 추가, `moved=True`
- 승리: `position == goal` → `won=True`
- 종료: `won` 또는 `step_count >= max_steps`
- 최대 스텝 기본값: `max(64, 3 × W × H)`
- 종료 후 `step()` 호출은 `RuntimeError` (API 는 `done` 이면 no-op 응답)

벽 충돌이 스텝을 소모하는 것은 의도된 설계다. "벽에 박는 것도 비용"이라는 신호를 주고,
무한 대기 없이 에피소드가 끝나도록 한다.

---

## 7. Brain 추상화

### 7.1 인터페이스

```python
class ServerBrain(Protocol):
    kind: str                     # "ai" | "rule" | "human"
    label: str                    # UI 표시 문자열
    requires_client_action: bool  # True → /action 대기
    window_size: int
    global_size: int
    map_size: int                 # -1 = 미로 크기에 맞춤

    def reset(self) -> None: ...
    def act(self, env: MazeEnv) -> int | None: ...   # None = 클라이언트 입력 대기
    def probabilities(self) -> list[float] | None: ...
```

`GameSession` 은 Brain 이 요구하는 관측 설정으로 `MazeEnv` 를 만든다. 따라서 구현체를 바꾸면
**관측 정의가 달라지고, 같은 미로에서 스텝 0으로 재시작**한다. 미로 seed 는 유지된다.

### 7.2 구현체

| kind | 클래스 | 행동 결정 | 관측 설정 | 확률 노출 |
|---|---|---|---|---|
| `ai` | `PolicyServerBrain` | 체크포인트 logits argmax | 체크포인트 config | ✅ |
| `rule`/`wall` | `RuleServerBrain` | 왼손 법칙 | `(3, 0, 0)` | ❌ |
| `rule`/`frontier` | `RuleServerBrain` | 지식 지도 BFS | `(3, 0, -1)` | ❌ |
| `human` | `HumanServerBrain` | 클라이언트 `/action` | `(3, 5, 0)` | ❌ |

**왼손 법칙** — 우선순위 `왼쪽 → 직진 → 오른쪽 → 뒤`. 현재 3×3 창과 직전 방향만으로 결정되므로
관측의 함수다. 완벽 미로(트리)에서 목표 도달을 보장하지만, 최단 경로의 2~4배를 걷는다.
고리 복도(`ring+spur`)처럼 트리가 아닌 미로에서는 보장이 깨진다.

**frontier** — 지식 지도 위 BFS.

1. 목표를 이미 봤으면 → 지식 지도 위 최단 경로의 첫 스텝
2. 아직 못 봤으면 → 미탐색 경계 중 `거리 + 0.6 × 목표까지 맨해튼` 이 최소인 칸으로 가는 첫 스텝

지식 지도는 직접 본 셀만 담으므로 정답을 쓰지 않는다. 즉 이것도 관측의 함수다.
`map_size=-1` 은 세션 생성 시 `max(W, H)` 로 해석된다.

**AI** — `PolicyServerBrain` 은 프로세스 전역 `_POLICY_CACHE` 에 모델을 캐시한다. 같은
체크포인트로 여러 판을 만들 때 torch 모델을 다시 로드하지 않는다. 판마다 `PolicyRunner` 만
새로 만들어 히스토리를 초기화한다.

### 7.3 체크포인트 탐색

`discover_checkpoint(server_root)` 우선순위:

1. `MAZE_CHECKPOINT` 환경변수 (경로가 존재할 때)
2. `checkpoints/wall/best.pt` — 왼손 법칙 증류, 성공률 100%, **기본값**
3. `checkpoints/v3/best.pt` — 54차원 BFS 라벨 실험
4. `checkpoints/*/best.pt` 중 mtime 최신

경로는 **서버만** 안다. 클라이언트에는 `brain_label` 만 내려간다.

---

## 8. API 계약

### 8.1 엔드포인트

| 메서드 | 경로 | 역할 | 오류 |
|---|---|---|---|
| GET | `/health` | 상태 + 선택된 체크포인트 + 게임 수 | — |
| GET | `/brains` | 선택 가능한 구현체 목록 | — |
| POST | `/games` | 판 생성 | `422` 짝수 크기, `503` 체크포인트 없음 |
| GET | `/games/{id}` | 현재 상태 (읽기 전용) | `404` |
| POST | `/games/{id}/advance` | Brain 이 스스로 이동 | `404`, `409` (human 모드) |
| POST | `/games/{id}/action` | 사람 행동 적용 | `404`, `409` (자동 모드) |
| POST | `/games/{id}/brain` | 구현체 교체 (같은 미로에서 재시작) | `404`, `503` |
| POST | `/games/{id}/reset` | 같은 미로 처음부터 | `404` |
| POST | `/games/{id}/new-maze` | 새 미로 | `404` |
| DELETE | `/games/{id}` | 세션 종료 | `404` |

`409` 는 "모드가 맞지 않는 호출"을 조용히 무시하지 않고 명시적으로 거절하기 위한 것이다.
예를 들어 `human` 게임에 `/advance` 를 보내면 서버가 대신 움직여 주지 않는다.

### 8.2 요청

```jsonc
// POST /games
{
  "size": 15,                    // 홀수 5~31
  "algorithm": "dfs",            // dfs | prim
  "brain": "ai",                 // ai | rule | human
  "rule": "wall",                // wall | frontier (brain=rule 일 때)
  "seed": null,                  // null = 무작위
  "max_steps": null,             // null = max(64, 3*W*H)
  "randomize_endpoints": true
}

// POST /games/{id}/advance
{ "steps": 1 }                   // 1~64

// POST /games/{id}/action
{ "action": 0 }                  // 0=UP 1=DOWN 2=LEFT 3=RIGHT

// POST /games/{id}/brain
{ "brain": "rule", "rule": "frontier" }
```

### 8.3 응답 (`StateResponse`)

```jsonc
{
  "game_id": "…",
  "brain": "ai",
  "brain_label": "AI · wall/best.pt",
  "rule": null,
  "window_size": 3,
  "global_size": 5,
  "map_size": 0,

  "maze": { "width": 15, "height": 15, "grid": [[1,1,…]], "start": [x,y], "goal": [x,y],
            "algorithm": "dfs", "seed": 12345 },
  "position": [x, y],
  "visited": [[x,y], …],
  "trajectory": [[x,y], …],
  "step_count": 10,
  "max_steps": 675,
  "last_action": 3,
  "won": false,
  "done": false,

  "action": 3,             // 이번 요청으로 적용된 행동 (없으면 null)
  "moved": true,
  "hit_wall": false,
  "probabilities": [0.1, 0.2, 0.3, 0.4],  // ai 만, 나머지는 null
  "awaiting_action": false // human 이고 미종료일 때 true
}
```

`window_size`/`global_size`/`map_size` 를 응답에 넣는 이유는 클라이언트가 HUD 에 관측 창을
그릴 때 서버와 같은 크기를 쓰게 하기 위해서다.

### 8.4 호출 시퀀스

**자동 모드 (`ai` / `rule`)**

```
Client                          Server
  │  POST /games {brain:"ai"}     │
  │──────────────────────────────▶│  미로 생성 + Brain 로드 + env 생성
  │◀──────────────────────────────│  201 StateResponse
  │                               │
  │  (step_interval 마다)          │
  │  POST /games/{id}/advance     │
  │──────────────────────────────▶│  brain.act(env) → env.step()
  │◀──────────────────────────────│  StateResponse (action/moved/hit_wall)
  │  렌더링 + 이동 보간            │
```

**사람 모드 (`human`)**

```
Client                          Server
  │  POST /games {brain:"human"}  │
  │──────────────────────────────▶│
  │◀──────────────────────────────│  awaiting_action=true
  │                               │
  │  방향키 입력                    │
  │  POST /games/{id}/action {0}  │
  │──────────────────────────────▶│  서버가 벽 판정 → env.step()
  │◀──────────────────────────────│  moved=true/false, hit_wall
```

클라이언트는 사람 모드에서 `/advance` 를 polling 하지 않는다
(`test_human_mode_does_not_poll_the_server_for_steps`).

---

## 9. 동시성과 세션 수명

- `GameSession` 의 상태 변경(`advance`, `apply`, `reset`, `new_maze`, `set_brain`)은
  `threading.Lock` 으로 직렬화한다.
- `state()` 는 락을 잡지 않고 현재 값의 스냅샷을 만든다. FastAPI 의 sync 엔드포인트는
  스레드풀에서 실행되므로, 같은 게임에 대한 **동시 상태 변경**은 락으로 막지만
  `state()` 와 변경의 원자적 결합은 보장하지 않는다. 데모 범위에서는 요청이 사실상 순차적이며,
  다중 사용자 동시성이 필요해지면 세션 단위 락 또는 단일 워커 큐로 확장해야 한다.
- `GAMES` 는 인메모리 dict 다. 서버를 재시작하면 모든 판이 사라진다.
- 세션 TTL/eviction 은 없다. 클라이언트는 종료 시 `DELETE /games/{id}` 를 호출해 정리한다.
- uvicorn 은 **단일 워커**로 실행한다. 다중 워커를 쓰면 `GAMES` 가 워커마다 갈라진다.

---

## 10. 실패 처리와 복구

### 클라이언트

| 상황 | 동작 |
|---|---|
| 연결 실패/타임아웃 (기본 3초) | `ApiError` → `RemoteSession.error` 설정, 스텝 중단 |
| `/advance` 실패 | 0.5초 이상 백오프 후 재시도 (요청 폭주 방지) |
| 서버 복구 | 다음 성공 응답에서 `error=None`, 배너 사라짐 |
| 체크포인트 로드가 걸리는 요청 | `/games`, `/brain` 은 60초 타임아웃 |
| 오프라인 상태 | 상단에 `NO SERVER RESPONSE — NOT MOVING` 배너 |

### 서버

| 상황 | 응답 |
|---|---|
| 없는 `game_id` | `404` |
| 모드 불일치 (`human`+`/advance`, 자동+`/action`) | `409` |
| 짝수 크기 | `422` |
| 체크포인트 없음 | `503` + 안내 메시지 |
| 종료된 게임에 `/advance` | 200 + 상태 그대로 (no-op) |

---

## 11. 테스트 전략

총 **152개** 테스트 (`cd server && python -m pytest`). 계층별로 다음을 고정한다.

| 계층 | 파일 | 검증 내용 |
|---|---|---|
| 미로 | `test_maze.py` | 크기/시드 재현성, 외곽 벽, 완전 연결, 시작·목표 유효성 |
| 솔버(평가용) | `test_solver.py` | 모든 미로 해결 가능, 경로 연속성, 거리 지도 대칭 |
| 환경/관측 | `test_env.py` | 차원 공식, `observe` ≡ `pack_payload`, 벽 충돌 무이동, 승리, 최대 스텝, 지식 지도 불변식 |
| 전문가 | `test_expert.py` | 왼손 법칙 100% 도달, frontier 가 미탐색 칸으로 이동하지 않음, 관측만의 함수 |
| 데이터 | `test_gen_dataset.py` | 라벨 정합성, split 누수 없음, frontier 라벨 재현 가능, 지식 지도에 정답 미로 없음 |
| API | `test_api.py` | 세 구현체, 409/404/422, 벽 판정 권위, 구현체 교체 시 재시작, AI 확률 |
| 클라이언트 | `test_client.py` | R1/R2/R4, 금지 import, 오프라인 정지, 브레인 교체 전달, 칩 클릭 영역 |
| 뷰어 | `test_viewer.py` | 레이아웃, 금지 토큰 스캔, solver 지연 import |
| 런처 | `test_launcher.py` | 인자 전달, 포트 탐색, health 타임아웃, 잘못된 조합 거절 |

특히 R1 은 **문자열 스캔 + subprocess import 검사**의 이중으로 막는다. 문자열만 피한 우회
(동적 import 등)도 런타임 검사에서 걸린다.

헤드리스 렌더링 검증은 `SDL_VIDEODRIVER=dummy` + `--snapshot` 으로 수행한다. 이 경로는
디스플레이 없이 실제 렌더러를 실행하므로, CI 에서 스크린샷 회귀를 만들 수 있다.

---

## 12. 알려진 한계와 향후 과제

1. **모델 표현력** — `Linear(479→128)` + BERT-tiny 로는 지식 지도 위 BFS 를 계산할 수 없다.
   frontier 교사 학습은 BC 62.5% / DAgger 37.5% 에 그쳤다. 공간 연산에는 가중치 공유
   (convolution)라는 inductive bias 가 필요하다.
2. **효율의 정보 한계** — 미로를 모르는 정책은 최단 경로를 걸을 수 없다. 실측 optimality
   하한은 약 1.79~1.90 이며, `≤1.3` 은 원리적으로 불가능하다.
3. **왼손 법칙의 보장 범위** — 완벽 미로(트리)에서만 도달을 보장한다. 고리 복도 + 곁길
   미로에서는 규칙과 모델이 모두 타임아웃한다.
4. **세션 영속성 없음** — 인메모리이며 단일 워커 전제다.
5. **`state()` 읽기 원자성** — 상태 변경과 스냅샷 사이의 원자성은 보장하지 않는다(§9).
6. **보안 경계 아님** — 로컬 데모용이다. 인증/rate limit/CORS 정책은 없다. 클라이언트가
   전체 미로 격자를 받으므로 로컬에서 치팅하는 것은 가능하다. 서버 권위는 "이동 판정"에
   대한 것이지 "클라이언트의 정직성"에 대한 것이 아니다.
7. **체크포인트 탐색이 파일 시스템 의존** — 배포 환경에서는 `MAZE_CHECKPOINT` 를 명시하는
   것이 안전하다.

---

## 부록 A. 좌표계와 규약

- 좌표는 `(x, y)`, `x` 는 열, `y` 는 행. 화면 좌표계와 같다(y 아래로 증가).
- `UP` 은 `y-1`, `DOWN` 은 `y+1`.
- `grid[y][x]` 접근. `maze.is_wall(x, y)` 는 범위 밖을 벽으로 취급한다.
- 미로 크기는 홀수만 허용한다(2칸 단위 carving + 외곽 벽).
- 시드가 같으면 같은 미로·같은 시작/목표가 재현된다.

## 부록 B. 관측 차원 계산 예

```
observation_dim(3, 5, 0)  = 11 + 2·9  + 25 + 0   = 54
observation_dim(3, 0, 15) = 11 + 2·9  + 0  + 450 = 479
observation_dim(5, 5, 0)  = 11 + 2·25 + 25 + 0   = 86
```
