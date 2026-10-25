# 빌드 및 실행 안내 — Maze Runner

> 버전 1.0 · 작성 2026-10-24 · 기준 커밋 `149afb0`
>
> 관련 문서: [설계 문서](./DESIGN.md) · [사용 가이드](./USAGE.md) · [의사결정 기록](./DECISIONS.md) · [README](../README.md)

이 문서는 **처음 받은 저장소를 실행 가능한 상태로 만드는 방법**과, 학습·평가·문제 해결 절차를
정리한다. 검증 환경은 Windows 11 + Python 3.11(CPU) 이지만, Linux/macOS 에서도 같은 명령이
동작한다(경로 구분자만 다름).

---

## 1. 요구사항

| 항목 | 요구 | 검증된 버전 |
|---|---|---|
| Python | 3.10+ (타입 힌트 `X \| None` 사용) | **3.11.0** |
| OS | Windows / Linux / macOS | Windows 11 |
| GPU | 불필요 (CPU 추론·학습) | CPU |
| 디스크 | torch 등 의존성 포함 약 2~3 GB | — |
| 네트워크 | 최초 1회 Hugging Face 모델 다운로드 | `prajjwal1/bert-tiny` |

주요 패키지(검증 버전):

- `fastapi 0.161.1`, `uvicorn 0.54.0`, `pydantic 2.14.1`
- `torch` (CPU), `transformers 5.20.1`, `numpy`
- `pygame 2.6.1` (SDL 2.28.4)
- `pytest 8.x`, `httpx 0.28.1`

의존성 정의는 [`server/requirements.txt`](../server/requirements.txt) 하나다.

---

## 2. 설치

### 2.1 가상환경 (권장)

PowerShell 기준:

```powershell
cd C:\Users\fall2\work\maze_game

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r server\requirements.txt
```

> PowerShell 실행 정책 때문에 `Activate.ps1` 이 막히면 다음 중 하나를 쓴다.
>
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> # 또는 활성화 없이
> .\.venv\Scripts\python.exe -m pip install -r server\requirements.txt
> ```

Linux/macOS:

```bash
cd maze_game
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r server/requirements.txt
```

### 2.2 최초 실행 시 모델 다운로드

`ai` 구현체는 `prajjwal1/bert-tiny` 를 기반으로 만든 체크포인트를 쓴다. `transformers` 가
최초 로드 시 모델 가중치를 Hugging Face 캐시에 내려받는다(수십 MB). 네트워크가 없으면:

- `rule` / `human` 은 모델이 필요 없으므로 바로 실행할 수 있다.
- 또는 미리 캐시를 채운 뒤 `HF_HUB_OFFLINE=1` 로 실행한다.

```powershell
$env:HF_HUB_OFFLINE = "1"   # 캐시가 이미 있을 때만
```

---

## 3. 체크포인트 주의 — 새 클론에는 AI가 없다

`checkpoints/` 와 `data/` 는 `.gitignore` 대상이다. 따라서 **저장소를 새로 받으면
`checkpoints/wall/best.pt` 가 없고, `ai` 구현체는 `503` 을 돌려준다.**

바로 실행하려면 모델이 필요 없는 구현체를 쓴다.

```powershell
python test_launcher.py --brain rule
python test_launcher.py --brain human
```

AI 를 쓰려면 §7 의 학습 파이프라인으로 `checkpoints/wall` 을 만들거나, 기존 체크포인트를
`server/checkpoints/wall/best.pt` 에 둔다. 서버는 다음 순서로 찾는다.

1. `MAZE_CHECKPOINT` 환경변수
2. `checkpoints/wall/best.pt`
3. `checkpoints/v3/best.pt`
4. `checkpoints/*/best.pt` 중 최신

```powershell
# 예: 다른 위치의 체크포인트를 지정
$env:MAZE_CHECKPOINT = "D:\models\maze-wall\best.pt"
```

---

## 4. 실행 방법

### 4.1 한 번에 실행 (권장) — `test_launcher.py`

저장소 루트에서 실행한다. 서버를 먼저 띄우고 `/health` 가 응답하면 클라이언트를 실행하며,
클라이언트 창을 닫으면 서버도 정리한다.

```powershell
python test_launcher.py                       # AI (체크포인트 필요)
python test_launcher.py --brain rule          # 규칙: 왼손 법칙
python test_launcher.py --brain rule --rule frontier
python test_launcher.py --brain human         # 사람
python test_launcher.py --size 21 --seed 7    # 나머지 인자는 viewer.app 으로 전달
```

동작:

- 기본 포트 `8000`. 사용 중이면 OS 가 준 빈 포트로 자동 이동한다.
- `--port 0` 은 항상 빈 포트 자동 선택.
- `--no-server --url http://127.0.0.1:8000` — 이미 떠 있는 서버에 연결.
- `--no-client` — 서버만 실행 (Ctrl+C 로 종료).
- `--health-timeout 60` — 서버 준비 대기 시간(초).

### 4.2 터미널 두 개로 실행

**터미널 1 — 서버**

```powershell
cd server
python -m uvicorn app.api.main:app --port 8000
```

- API 문서: <http://127.0.0.1:8000/docs>
- 상태 확인: <http://127.0.0.1:8000/health>
- 모델 계약: <http://127.0.0.1:8000/models> · 독립 추론 예시: [`INFERENCE.md`](./INFERENCE.md)
- `--reload` 는 개발 중에만 사용한다(세션이 사라짐).

**터미널 2 — 클라이언트**

```powershell
cd server

python -m viewer.app --size 15                     # AI (기본)
python -m viewer.app --brain rule                  # 규칙: 왼손 법칙
python -m viewer.app --brain rule --rule frontier  # 규칙: 지식 지도 탐색
python -m viewer.app --brain human                 # 사람
```

> 서버와 클라이언트는 **각각 `server/` 를 cwd** 로 실행해야 `app`, `viewer` 패키지가 잡힌다.
> 런처는 이 cwd 와 `PYTHONPATH` 를 자동으로 맞춘다.

### 4.3 클라이언트 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--server URL` | `http://127.0.0.1:8000` | API 주소 |
| `--size N` | `15` | 미로 크기 (홀수, 5~31) |
| `--seed N` | 없음 | 미로 시드 (재현용) |
| `--algorithm dfs\|prim` | `dfs` | 생성 알고리즘 |
| `--brain ai\|rule\|human` | `ai` | 이동 결정 구현체 |
| `--rule wall\|frontier` | `wall` | 규칙 구현체 종류 |
| `--human` | 꺼짐 | `--brain human` 과 동일 (하위 호환) |
| `--fixed-endpoints` | 꺼짐 | 시작/목표를 고정 코너에 배치 |
| `--step-interval 0.09` | `0.09` | AI/규칙 스텝 간격(초) |
| `--fps 60` | `60` | 렌더링 FPS |
| `--width` / `--height` | `1180` / `860` | 창 크기 (최소 480×400) |
| `--max-steps N` | `0` | 에피소드 최대 스텝 (0 = 서버 기본값) |
| `--max-frames N` | `0` | N 프레임 후 종료 (스모크 테스트) |
| `--snapshot PATH` | 없음 | 해당 프레임을 PNG 로 저장하고 종료 |
| `--snapshot-frame N` | `60` | 스냅샷을 찍을 프레임 |

### 4.4 조작 키

| 키 | 동작 |
|---|---|
| `1` / `2` / `3` | AI / 규칙 / 사람 구현체 선택 (상단 칩 클릭도 동일) |
| 방향키 / `WASD` | 사람 모드에서 이동 |
| `R` | 같은 미로 처음부터 |
| `N` | 새 미로 |
| `SPACE` | 일시정지 |
| `ESC` / `Q` | 종료 |

구현체를 바꾸면 **같은 미로에서 스텝 0으로** 돌아간다. 관측 정의가 구현체마다 다르기 때문이며,
미로 seed 는 유지되므로 "같은 판에서 AI와 규칙을 번갈아 비교"할 수 있다.

---

## 5. 헤드리스 스냅샷 (CI/문서용)

디스플레이 없이 실제 렌더러를 실행하고 PNG 로 저장한다. **서버가 먼저 떠 있어야 한다.**

```powershell
cd server
$env:SDL_VIDEODRIVER = "dummy"
python -m viewer.app --brain rule --snapshot shot.png --snapshot-frame 120
```

- `--snapshot` 은 해당 프레임을 저장한 뒤 프로세스를 종료한다.
- 경로는 **클라이언트 cwd 기준**이다. 런처로 실행하면 cwd 가 `server/` 이므로,
  루트에 저장하려면 절대 경로를 쓴다.

```powershell
$env:SDL_VIDEODRIVER = "dummy"
python test_launcher.py --brain rule --snapshot C:\Users\fall2\work\maze_game\docs\shot.png --snapshot-frame 120
```

- `--max-frames N` 은 스냅샷 없이 렌더 루프만 스모크 테스트할 때 쓴다.
- 한글 폰트가 없는 환경에서는 뷰어가 자동으로 시스템 폰트를 찾고, 없으면 기본 폰트로
  대체한다(한글이 □ 로 보일 수 있음 — 렌더링 실패는 아님).

---

## 6. 테스트

```powershell
cd server
python -m pytest
```

- 설정: [`server/pytest.ini`](../server/pytest.ini) — `pythonpath=.`, `testpaths=tests`, `-q`
- 현재 기준: **152 passed**

계층별 실행:

```powershell
python -m pytest tests\test_api.py tests\test_client.py   # 서버 권위 + R1~R4
python -m pytest tests\test_env.py tests\test_expert.py    # 관측/전문가
python -m pytest tests\test_maze.py tests\test_solver.py   # 미로/솔버
python -m pytest tests\test_gen_dataset.py                 # 데이터 라벨/split
python -m pytest tests\test_launcher.py tests\test_viewer.py
```

주의:

- `test_client.py` 는 **subprocess** 로 클라이언트 모듈을 import 해 `torch`/`app.game` 이
  로드되지 않는지 확인한다. 첫 실행은 프로세스 기동 때문에 조금 느릴 수 있다.
- Hugging Face / pygame 의 경고가 stderr 에 나오는 것은 정상이다.
- Windows 콘솔에서 한글 테스트 이름이 깨져 보여도 파일/결과는 UTF-8 이다.

---

## 7. 학습 파이프라인 (선택)

모든 명령은 `server/` 에서 실행한다. 산출물(`data/`, `checkpoints/`)은 git 에 커밋하지 않는다.

### 7.1 왼손 법칙 데이터셋 + LoRA-BERT (`checkpoints/wall`)

```powershell
cd server

# 1) 합성 데이터 — 미로 단위 8:1:1, 전문가 + 노이즈 궤적
python -m training.gen_dataset --mazes 240 --sizes 9,11,13,15 --out data/wall --workers 4

# 2) 학습 (CPU 약 9분, 4 epochs)
python -m training.train --data data/wall --epochs 4 --out checkpoints/wall --lora-r 32 --lora-lr 3e-4

# 3) 롤아웃 평가 (정확도가 아니라 실제 탈출 성공률)
python -m training.evaluate --checkpoint checkpoints/wall/best.pt --data data/wall --limit 40
```

기대 결과(참고): val acc 1.0000, 롤아웃 100%, 평균 optimality 2.3~2.9 (15×15).

### 7.2 frontier 데이터셋 (지식 지도, 479차원)

```powershell
python -m training.gen_dataset `
  --mazes 240 --sizes 9,11,13,15 `
  --expert frontier --map-size 15 --global-size 0 `
  --out data/frontier --workers 4

python -m training.train --data data/frontier --epochs 3 --out checkpoints/frontier --lora-r 32
python -m training.evaluate --checkpoint checkpoints/frontier/best.pt --data data/frontier --limit 40
```

`--map-size` 는 미로 최대 크기 이상이어야 한다(여기서는 15). `--global-size 0` 으로 두면
479차원이 된다. 참고 실측: BC val acc 92.6%, 롤아웃 62.5%.

### 7.3 DAgger

```powershell
python -m training.dagger `
  --checkpoint checkpoints/frontier/best.pt `
  --data data/frontier `
  --out data/frontier_dagger `
  --limit 200

python -m training.train `
  --data data/frontier_dagger `
  --epochs 3 --out checkpoints/frontier_d --lora-r 32 `
  --init checkpoints/frontier/best.pt
```

`dagger.py` 는 기존 데이터에 온폴리시 샘플을 **병합한 새 디렉터리**(`--out`)를 만든다.
원본은 덮어쓰지 않는다.

### 7.4 AI vs 규칙 벤치마크

```powershell
python -m training.compare_brains --mazes 60 --size 15
# 결과 JSON 저장
python -m training.compare_brains --mazes 60 --size 15 --out checkpoints/compare.json
```

같은 미로 집합에서 `ai`(기본 `checkpoints/wall`)와 `rule(wall)`, `rule(frontier)` 를
짝지어 비교한다. 참고 실측(60 미로, 15×15, seed 90000~90059):

| 구현체 | 성공 | 평균 스텝 | optimality |
|---|---|---|---|
| `ai` (`checkpoints/wall`) | 60/60 | 96.9 | 1.865 |
| `rule` wall | 60/60 | 96.9 | 1.865 |
| `rule` frontier | 60/60 | 94.7 | 1.851 |

---

## 8. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MAZE_CHECKPOINT` | 없음 | AI 체크포인트 경로. 있으면 탐색 순서보다 우선 |
| `HF_HUB_OFFLINE` | 없음 | `1` 이면 HF 네트워크 접근 차단(캐시 사용) |
| `SDL_VIDEODRIVER` | 없음 | `dummy` 로 헤드리스 렌더링 |
| `PYGAME_HIDE_SUPPORT_PROMPT` | 런처가 `1` 설정 | pygame 배너 숨김 |
| `PYTHONPATH` | 런처가 `server/` 추가 | 수동 실행 시 `server/` 에서 실행하면 불필요 |

---

## 9. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'app'` | `server/` 가 아닌 곳에서 실행. `cd server` 또는 런처 사용 |
| `503` + "체크포인트를 찾지 못했습니다" | 새 클론에는 `checkpoints/` 가 없음. §3 참고. `--brain rule` 로 우회 가능 |
| `409 this brain requires /action` | `human` 게임에 `/advance` 호출. 클라이언트가 모드를 맞추는지 확인 |
| `422 size must be odd` | 짝수 크기. 홀수 5~31 사용 |
| 클라이언트가 "서버에 연결할 수 없습니다" | 서버 미실행/포트 불일치. `/health` 확인 후 `--server` 지정 |
| AI 첫 스텝이 오래 걸림 | 체크포인트/모델 최초 로드. `/games`, `/brain` 은 60초 타임아웃 |
| 서버를 죽였는데 캐릭터가 계속 움직임 | 있으면 버그(R4 위반). 배너가 뜨고 정지해야 정상 |
| 창이 안 뜸 / `No available video device` | 헤드리스 환경. `SDL_VIDEODRIVER=dummy` + `--snapshot` |
| 한글이 □ 로 보임 | 시스템에 한글 폰트 없음. 렌더링은 계속 동작. 폰트 설치 후 재실행 |
| PowerShell 에서 한글이 깨져 보임 | 콘솔 코드페이지 문제. `chcp 65001` 또는 Windows Terminal 사용 |
| `Activate.ps1` 실행 불가 | `Set-ExecutionPolicy -Scope Process RemoteSigned` |
| 포트 8000 사용 중 | 런처는 자동 회피. 수동 실행은 `--port 8010` |
| `torch` 설치가 너무 느림 | CPU 전용 wheel 사용. GPU 가 없으면 기본 PyPI wheel 로 충분 |
| 테스트가 느림 | 첫 실행은 모델/프로세스 기동 비용. 이후 캐시됨 |

### UTF-8 파일 편집 주의 (Windows PowerShell 5.1)

소스와 문서는 **BOM 없는 UTF-8** 이다. PowerShell 5.1 의 `Set-Content -Encoding UTF8` 은
BOM 을 추가하고, `Get-Content -Raw` 는 BOM 없는 UTF-8 을 잘못 해석할 수 있다. 저장소의
텍스트 파일을 통째로 다시 쓰지 말고, 편집 도구(에디터)로 수정한다.

---

## 10. 설치 후 검증 체크리스트

```powershell
# 1) 테스트
cd server
python -m pytest
# → 152 passed

# 2) 서버 단독 기동
python -m uvicorn app.api.main:app --port 8000
# 다른 터미널:
Invoke-RestMethod http://127.0.0.1:8000/health
# → status=ok, checkpoint=... (없으면 null)

# 3) 규칙 구현체로 GUI
cd ..
python test_launcher.py --brain rule

# 4) 사람 모드
python test_launcher.py --brain human

# 5) R4 확인 — GUI 가 도는 중에 서버 프로세스만 종료
# → 상단에 "NO SERVER RESPONSE — NOT MOVING" 배너, 캐릭터 정지
```

---

## 11. 저장소 구조

```
maze_game/
├── PLAN.md                  # 계획 + 실험 로그
├── README.md                # 개요/빠른 시작
├── test_launcher.py         # 서버 + 클라이언트 동시 실행
├── docs/
│   ├── DESIGN.md            # 이 문서의 자매편 — SW 설계
│   ├── BUILD.md             # 이 문서
│   ├── DECISIONS.md         # 의사결정 기록
│   └── screenshot-rule.png
└── server/
    ├── pytest.ini
    ├── requirements.txt
    ├── app/
    │   ├── api/             # FastAPI — 게임의 유일한 권위
    │   ├── game/            # 미로/환경/전문가/(평가 전용) 솔버
    │   └── ml/              # LoRA/BERT 정책/실행기
    ├── training/            # 데이터/학습/평가/DAgger/비교
    ├── viewer/              # SDL 클라이언트 (경로 탐색 없음)
    └── tests/               # 171 tests
```
