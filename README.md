# Maze Runner — LoRA-BERT 미로 게임

미로를 푸는 지능을 **BERT + LoRA 신경망 안에** 가두고, SDL 뷰어는 그 결과를
그리기만 하는 프로젝트입니다.

```
관측(3×3 창 + 지식 지도) ──▶ LoRA-BERT ──▶ UP/DOWN/LEFT/RIGHT
        ▲                                        │
        └──────────── MazeEnv ◀──────────────────┘
```

- **뷰어에는 경로 탐색 코드가 없습니다.** 이동은 `Brain` 이 돌려준 행동을 그대로 적용할 뿐입니다.
- 학습 라벨(왼손 법칙 전문가)은 `training/` 에서만 쓰이고, 서빙 경로에서는 로드되지 않습니다.
- BFS 는 **평가 지표(최단 경로·optimality) 전용**이며, 학습 라벨로는 쓰지 않습니다.

## 빠른 시작

```powershell
cd server
python -m pip install -r requirements.txt
```

### 1. SDL 게임 화면 — AI 자동 플레이 (기본)

```powershell
python -m viewer.app --size 15
```

체크포인트를 생략하면 `checkpoints/wall/best.pt` → `checkpoints/v3/best.pt` → 가장 최근
`*/best.pt` 순으로 자동 선택해 **AI가 스스로 플레이**합니다. 특정 모델을 쓰려면:

```powershell
python -m viewer.app --checkpoint checkpoints/wall/best.pt --size 15
```

- `--step-interval 0.05` 더 빠르게, `--step-interval 0.3` 천천히
- `--temperature 0.7` 확률적 플레이
- `--max-steps 200` 에피소드 길이 제한
- `--fixed-endpoints` 시작/목적지 고정 (기본: 모델 학습 분포와 같은 무작위 배치)

### 2. 사람 플레이

```powershell
python -m viewer.app --human --size 15
```

| 키 | 동작 |
|---|---|
| 방향키 / WASD | 이동 |
| R | 같은 미로 다시 시작 |
| N | 새 미로 |
| SPACE | 일시정지 |
| ESC / Q | 종료 |

### 3. 헤드리스 스냅샷 (CI/문서용)

```powershell
$env:SDL_VIDEODRIVER = "dummy"
python -m viewer.app --snapshot shot.png --snapshot-frame 120
```

## 학습 파이프라인

```powershell
cd server

# 1) 합성 데이터셋 — 왼손 법칙 전문가 라벨 (미로 단위 8:1:1 분할)
python -m training.gen_dataset --mazes 240 --sizes 9,11,13,15 --out data/wall --workers 4

# 2) LoRA-BERT 학습
python -m training.train --data data/wall --epochs 4 --out checkpoints/wall --lora-r 32 --lora-lr 3e-4

# 3) 롤아웃 평가 (정확도가 아니라 실제 탈출 성공률)
python -m training.evaluate --checkpoint checkpoints/wall/best.pt --data data/wall --limit 40
```

## 구조

```
server/
├── app/
│   ├── game/          # 미로 생성, 환경, 관측 인코더, 왼손 법칙 전문가, (평가 전용) BFS
│   └── ml/            # LoRA, BERT 정책, 슬라이딩 윈도우 실행기
├── training/          # 데이터 생성, 학습, 평가, DAgger
├── viewer/            # SDL(pygame) 뷰어 — 렌더링 + Brain 인터페이스
└── tests/             # 107 tests
```

### 뷰어 설계

```
viewer/app.py      게임 루프 · 입력 · 애니메이션
viewer/brains.py   HumanBrain | PolicyBrain   ← 이동 결정은 여기서만
viewer/render.py   그리기 전용
viewer/layout.py   순수 레이아웃 계산 (pygame 비의존)
```

FastAPI 서버가 준비되면 `HttpBrain` 을 추가하고 `viewer/app.py` 는 그대로 두면 됩니다.
`tests/test_viewer.py` 가 뷰어 소스에 경로 탐색 토큰이 들어오는 것을 막습니다.

## 현재 상태

- 데이터·학습·평가·뷰어 파이프라인이 모두 동작합니다.
- `checkpoints/wall` (왼손 법칙 전문가 라벨, LoRA r=32) 은 **롤아웃 성공률 100%** 입니다.

  | 평가셋 | 미로 | 성공률 | 평균 스텝 | optimality |
  |---|---|---|---|---|
  | 9~15 test | 24 | 100% | 65.8 | 2.28 |
  | 15×15 test | 20 | 100% | 93.2 | 2.92 |
  | 17·19 test (미학습 크기) | 6 | 100% | 185.0 | 4.84 |

- **"규칙을 그냥 쓰면 되지 않나"를 실측으로 검증했습니다** ([`PLAN.md`](./PLAN.md) §14 W2.7).
  - 규칙(왼손 법칙)과 학습 모델은 성공률·스텝·실패 모드가 **동일**합니다. 즉 `wall` 은 규칙의
    충실한 증류이고, 성능 이득은 없습니다.
  - 관측에 지식 지도(belief map, 479차원)를 추가하고 더 나은 전문가(frontier 탐색)를
    학습시켰지만, **BC 62.5% / DAgger 37.5%** 로 실패했습니다. 라벨은 관측으로 완전히
    재현되므로 원인은 라벨이 아니라 **표현력**입니다(선형 투영 + BERT-tiny 로는 BFS 불가).
  - 효율 상한 자체도 작습니다: frontier 1.868 vs 왼손 법칙 2.145 (13% 개선).
- 결론: **이 게임에서는 규칙을 서버에 넣는 것이 정답**입니다. LoRA의 가치는 성능이 아니라
  관측→행동 계약과 "규칙으로 쓸 수 없는 것"을 배우는 데 있습니다. 자세한 분석은 PLAN §14 W2.7.
- FastAPI 서빙(W3)은 아직 없습니다. 뷰어는 현재 로컬 체크포인트를 사용합니다.
