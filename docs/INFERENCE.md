# 독립 추론 API (FastAPI)

학습된 LoRA-BERT 체크포인트를 **게임 세션과 분리해서** 호출하는 HTTP API 다.
관측 시계열을 보내면 서버가 실시간으로 다음 행동과 행동 확률을 돌려준다.

- 구현: [`server/app/api/main.py`](../server/app/api/main.py), [`server/app/api/predictor.py`](../server/app/api/predictor.py), [`server/app/ml/inference.py`](../server/app/ml/inference.py)
- 테스트: [`server/tests/test_inference.py`](../server/tests/test_inference.py)

## 1. 실행

```powershell
cd server
uvicorn app.api.main:app --port 8000
```

서버는 startup(lifespan)에서 기본 체크포인트를 **한 번 로드하고 워밍업**한다.
이후 요청은 프로세스 메모리 캐시(`app/ml/registry.py`)를 재사용한다.
체크포인트는 `MAZE_CHECKPOINT` → `checkpoints/wall` → `checkpoints/v3` → 최신 순으로
자동 선택된다.

## 2. 엔드포인트

| 메서드 | 경로 | 역할 | 오류 |
|---|---|---|---|
| GET | `/models` | 모델의 관측 차원·시퀀스 길이·행동 수를 반환 | `404` 체크포인트 없음, `400` 경로 위반 |
| POST | `/predict` | 관측 시계열 → 행동·확률·(선택) logits | `404`, `400`, `422` |
| GET | `/health` | `model_loaded`, 선택된 체크포인트, 게임 수 | — |

### `GET /models`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/models
```

```jsonc
{
  "checkpoint": "C:\\...\\server\\checkpoints\\wall\\best.pt",
  "obs_dim": 54,
  "seq_len": 32,
  "num_actions": 4,
  "window_size": 3,
  "global_size": 5,
  "map_size": 0,
  "loaded": true
}
```

`?checkpoint=wall` 처럼 체크포인트를 고를 수 있다.
`obs_dim` 은 모델이 기대하는 **관측 벡터 길이**다. 클라이언트는 이 값에 맞춰
`MazeEnv` 를 같은 설정(`window_size`/`global_size`/`map_size`)으로 만들어야 한다.

### `POST /predict`

요청:

```jsonc
{
  "observations": [[0.0, 0.0, /* … obs_dim 개 */]],  // 1개 이상
  "checkpoint": null,        // null = 자동 선택, "wall" / "wall_pruned" / 경로
  "return_logits": false     // true 면 logits 도 반환
}
```

응답:

```jsonc
{
  "checkpoint": "C:\\...\\server\\checkpoints\\wall\\best.pt",
  "action": 0,
  "action_name": "UP",
  "probabilities": [0.8965, 0.0515, 0.0519, 0.0001],
  "logits": null,
  "obs_dim": 54,
  "seq_len": 32,
  "history_length": 1,
  "truncated": false,
  "latency_ms": 14.251
}
```

- `action`: `0=UP 1=DOWN 2=LEFT 3=RIGHT`
- `history_length`: 실제로 사용한 관측 개수 (≤ `seq_len`)
- `truncated`: 입력이 `seq_len` 보다 길어 앞부분을 버렸는지
- `latency_ms`: 입력 검증·전처리·순전파를 포함한 서버 처리 시간

### 시퀀스 규칙

학습·게임 서버와 **완전히 같은 규칙**을 쓴다.

- 마지막 관측이 현재 스텝
- `seq_len`(32) 보다 짧으면 **왼쪽을 0으로 패딩**하고 attention mask 를 0으로 둔다
- `seq_len` 보다 길면 **최근 32개만** 사용한다
- 각 관측의 길이는 반드시 `obs_dim` 과 같아야 한다 (아니면 `422`)

## 3. 호출 예시

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

observations = []
for _ in range(5):
    observations.append(env.observe().tolist())
    # 실제 게임에서는 서버가 고른 행동으로 env.step(action) 을 진행한다
    env.step(0)

response = httpx.post(
    f"{BASE}/predict",
    json={"observations": observations, "return_logits": True},
)
response.raise_for_status()
print(response.json())
```

> 이 API 는 **상태를 저장하지 않는다.** 관측 이력은 호출자가 보낸다.
> 게임 진행(이동·벽 판정·승리 판정)은 기존 `/games/{id}/advance` 가 담당한다.

## 4. 게임 API 와의 관계

| | 게임 API | 독립 추론 API |
|---|---|---|
| 상태 | 서버가 세션 소유 | 무상태 (요청마다 관측 전달) |
| 관측 생성 | 서버가 `MazeEnv.observe()` | 클라이언트가 전달 |
| 이동/승리 판정 | 서버 | 없음 (행동만 반환) |
| 모델 | 세션 생성 시 자동 선택 | `checkpoint` 로 선택 가능 |
| 용도 | 실제 플레이 | 모델 단독 평가·통합·디버깅 |

두 경로 모두 같은 `BertPolicy` 와 같은 체크포인트 캐시를 쓰므로, 동일한 관측
시퀀스에 대해 **같은 행동**을 낸다.

## 5. 오류와 제한

| 상황 | 상태 코드 | 설명 |
|---|---|---|
| 체크포인트 없음 | `404` | `checkpoint` 이름/경로가 존재하지 않음 |
| `server/checkpoints` 밖의 경로 | `400` | 임의 파일 로드 방지 |
| 관측 차원 불일치 | `422` | `expected obs_dim=...` |
| 빈 `observations` | `422` | Pydantic `min_length=1` |
| NaN/Inf 포함 | `422` | JSON 숫자 검증 |

- 명시적 `checkpoint` 는 **`server/checkpoints/` 하위**만 허용한다.
  자동 선택(`null`)은 운영자가 지정한 `MAZE_CHECKPOINT` 를 그대로 존중한다.
- 모델 로드는 프로세스당 1회다. 체크포인트를 교체하면 서버를 재시작하거나
  새 체크포인트를 `checkpoint` 로 지정한다.
- 이 서버는 데모용 단일 워커를 전제로 한다. 다중 워커를 쓰면 워커마다 모델을
  따로 로드한다.
