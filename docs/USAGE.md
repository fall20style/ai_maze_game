# 사용 가이드 — GUI 클라이언트와 API 호출

이 문서는 **누가 어떤 API 를 호출하는지**와, GUI·게임 API·독립 추론 API 를 실제로
어떻게 실행/호출하는지 정리한다. API 계약의 상세는 [INFERENCE.md](./INFERENCE.md) 를 본다.

## 0. 한눈에

| 주체 | 호출하는 API | 역할 |
|---|---|---|
| pygame GUI (`server/viewer/`) | `/health`, `/games`, `/games/{id}/advance`, `/action`, `/brain`, `/reset`, `/new-maze` | 게임 플레이·렌더링 |
| 외부 스크립트/도구 | `/models`, `/predict` | 모델 단독 추론 |
| GUI → `/predict` 직접 호출 | **하지 않음** | 호출하면 클라이언트가 관측·이동을 결정하게 되어 R2 위반 |

- **GUI 는 게임 서버의 HTTP 클라이언트**다.
- **`/predict` 는 GUI 가 아니라 모델 단독 호출자를 위한 무상태 API**다.
- GUI 는 AI 모드에서 관측을 보내지도, 행동을 결정하지도 않는다. 서버가
  `MazeEnv.observe()` → `BertPolicy` → `env.step()` 을 모두 수행하고, GUI 는
  `StateResponse` 를 받아 그리기만 한다.

## 1. GUI 호출 흐름

```
pygame GUI (viewer/app.py)
   │  RemoteSession.advance(now)
   ▼
viewer/api_client.py  GameApi
   │  POST /games/{id}/advance   {"steps": 1}
   ▼
FastAPI (app/api/main.py)
   │  GameSession.advance()
   ▼
PolicyServerBrain.act(env)
   │  PolicyRunner.logits(env.observe())
   ▼
BertPolicy (LoRA-BERT)
   │
   ▼
StateResponse { position, action, probabilities, won, ... }
   │
   ▼
GUI 는 응답을 복사해서 그리기만 함 (viewer/render.py)
```

`RemoteSession.wants_step()` 은 `--step-interval`(기본 0.09초)마다 `/advance` 를
호출할지 결정한다. 사람 모드(`awaiting_action=true`)에서는 polling 하지 않고
키 입력마다 `/action` 을 호출한다.

### GUI 동작 → 엔드포인트

| GUI 동작 | HTTP 호출 | 코드 |
|---|---|---|
| 시작 시 서버 확인 | `GET /health` | `viewer/app.py:119` |
| 판 생성 | `POST /games` | `viewer/app.py:127`, `api_client.py:72` |
| AI/규칙 1스텝 | `POST /games/{id}/advance` | `remote.py:212`, `api_client.py:98` |
| 사람 이동 | `POST /games/{id}/action` | `remote.py:220`, `api_client.py:101` |
| `1`/`2`/`3` 구현체 교체 | `POST /games/{id}/brain` | `remote.py:223` |
| `R` 같은 미로 재시작 | `POST /games/{id}/reset` | `remote.py:231` |
| `N` 새 미로 | `POST /games/{id}/new-maze` | `remote.py:238` |
| 창 종료 | `DELETE /games/{id}` | `viewer/app.py:216` |

## 2. 실행

### 한 번에 실행 (권장)

```powershell
# 저장소 루트에서
python test_launcher.py                 # 서버 + AI GUI
python test_launcher.py --brain rule    # 규칙(왼손 법칙)
python test_launcher.py --brain human   # 사람
```

런처는 서버를 먼저 띄우고 `/health` 가 응답하면 GUI 를 실행한다. 창을 닫으면
서버도 함께 정리된다.

### 터미널 두 개로 실행

```powershell
# 터미널 1 — 서버
cd server
uvicorn app.api.main:app --port 8000

# 터미널 2 — GUI
cd server
python -m viewer.app --size 15
python -m viewer.app --brain rule --rule frontier
python -m viewer.app --brain human
```

- API 문서(Swagger): <http://127.0.0.1:8000/docs>
- 상태 확인: <http://127.0.0.1:8000/health>
- 모델 계약: <http://127.0.0.1:8000/models>

### GUI 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--server URL` | `http://127.0.0.1:8000` | 접속할 서버 |
| `--size N` | 15 | 미로 크기 (홀수, 5~31) |
| `--seed N` | 무작위 | 같은 미로 재현 |
| `--algorithm dfs\|prim` | dfs | 미로 생성 알고리즘 |
| `--brain ai\|rule\|human` | ai | 이동 결정 구현체 |
| `--rule wall\|frontier` | wall | 규칙 구현체 종류 |
| `--step-interval 0.09` | 0.09 | AI/규칙 스텝 간격(초) |
| `--fixed-endpoints` | 꺼짐 | 시작/목적지 고정 |

실행 중 키: `1` AI · `2` 규칙 · `3` 사람 · 방향키/WASD(사람) · `R` 재시작 ·
`N` 새 미로 · `SPACE` 일시정지 · `ESC/Q` 종료.

### AI 체크포인트 바꾸기

GUI 에는 체크포인트 선택 UI 가 없다. 서버가 아래 순서로 자동 선택한다.

```
MAZE_CHECKPOINT → checkpoints/wall → checkpoints/v3 → 가장 최근 */best.pt
```

프루닝 모델로 플레이하려면:

```powershell
# 저장소 루트
$env:MAZE_CHECKPOINT = "server/checkpoints/wall_pruned/best.pt"
python test_launcher.py
```

## 3. 게임 API 직접 호출

GUI 없이 게임 서버를 호출할 수도 있다.

```powershell
$game = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/games `
  -ContentType 'application/json' `
  -Body '{"size": 9, "seed": 3, "brain": "rule", "rule": "wall"}'

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/games/$($game.game_id)/advance" `
  -ContentType 'application/json' -Body '{"steps": 1}'
```

- `brain="ai"` 로 만들면 서버가 체크포인트를 로드한 뒤 추론한다.
- 응답의 `position`, `action`, `probabilities`, `won`, `done` 을 그대로 사용한다.
- 사람 모드는 `POST /games/{id}/action` 에 `{"action": 0}` (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)을 보낸다.

## 4. 독립 추론 API 호출

게임 세션 없이 모델만 호출한다. 상세 계약은 [INFERENCE.md](./INFERENCE.md).

### 4.1 모델 계약 확인

```powershell
Invoke-RestMethod http://127.0.0.1:8000/models
```

```jsonc
{
  "checkpoint": "C:\\...\\checkpoints\\wall\\best.pt",
  "obs_dim": 54,
  "seq_len": 32,
  "num_actions": 4,
  "window_size": 3,
  "global_size": 5,
  "map_size": 0,
  "loaded": true
}
```

### 4.2 최소 호출 (0 벡터)

```powershell
# obs_dim=54 인 경우
$body = '{"observations": [[' + (('0,' * 53) + '0') + ']]}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' -Body $body
```

### 4.3 실제 관측으로 호출

```python
import httpx

from app.game.env import MazeEnv
from app.game.maze import generate

BASE = "http://127.0.0.1:8000"
contract = httpx.get(f"{BASE}/models").json()

maze = generate(9, 9, seed=3)
env = MazeEnv(
    maze,
    window_size=contract["window_size"],
    global_size=contract["global_size"],
    map_size=contract["map_size"],
)
env.reset()

observations = [env.observe().tolist()]
response = httpx.post(
    f"{BASE}/predict",
    json={"observations": observations, "return_logits": True},
)
response.raise_for_status()
print(response.json())
```

- 마지막 관측이 현재 스텝이다.
- `seq_len`(32)보다 짧으면 왼쪽 0 패딩, 길면 최근 32개만 사용한다.
- `checkpoint` 로 `"wall"`, `"wall_pruned"` 또는 `server/checkpoints/` 하위 경로를 고를 수 있다.
- 이 API 는 상태를 저장하지 않는다. 관측 이력은 호출자가 매번 보낸다.

## 5. 자주 묻는 질문

**Q. 게임 GUI 가 클라이언트인가?**  
A. 네. pygame GUI 가 FastAPI 게임 서버의 HTTP 클라이언트다.

**Q. GUI 가 `/predict` 를 호출하나?**  
A. 아니다. GUI 는 `/games/{id}/advance` 를 호출하고, 서버 내부에서 같은 모델 추론이
일어난다. `/predict` 는 게임과 분리된 모델 단독 호출용이다.

**Q. 클라이언트가 경로 탐색을 하나?**  
A. 아니다. 클라이언트에는 BFS/A* 등 경로 탐색 코드가 없고, 서버가 보낸 상태를
복사해서 그린다 (R1). `tests/test_client.py` 가 이를 강제한다.

**Q. 서버가 죽으면 GUI 는 어떻게 되나?**  
A. 이동하지 않고 상단에 `NO SERVER RESPONSE — NOT MOVING` 배너를 표시한다 (R4).
실패 후 최소 0.5초 백오프를 둔다.

**Q. AI 모드에서 왜 관측을 클라이언트가 보내지 않나?**  
A. 관측 생성·이동 결정·벽 판정의 권위를 서버에 두기 위해서다 (R2·R3). 학습과 서빙이
같은 `MazeEnv.observe()` 를 쓰므로 학습/서빙 불일치도 막는다.

## 6. 관련 파일

| 파일 | 역할 |
|---|---|
| `server/viewer/app.py` | pygame 이벤트 루프, 키 입력, 스냅샷 |
| `server/viewer/api_client.py` | `GameApi` — stdlib `urllib` REST 래퍼 |
| `server/viewer/remote.py` | `RemoteSession` — 서버 JSON → 화면 상태 |
| `server/app/api/main.py` | FastAPI 엔드포인트 |
| `server/app/api/brains.py` | `ai` / `rule` / `human` 구현체 |
| `server/app/ml/inference.py` | 독립 추론 코어 |
| `server/app/ml/registry.py` | 체크포인트 로드/캐시 (게임·추론 공용) |
