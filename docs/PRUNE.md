# 워드 임베딩 프루닝 — 용량 비교와 실제 플레이 검증

`checkpoints/wall/best.pt` 에서 **추론에 쓰이지 않는 BERT `word_embeddings`** 를 제거해
체크포인트를 **17.38 MiB → 2.48 MiB (−85.8%)** 로 줄였다. 같은 미로에서 실제 롤아웃을
돌려 성능이 **완전히 동일**함을 확인했다.

- 스크립트: [`server/training/prune_embeddings.py`](../server/training/prune_embeddings.py)
- 결과물: [`server/checkpoints/wall_pruned/`](../server/checkpoints/wall_pruned)
  (`best.pt`, `prune_report.json`, `prune_report.md`)
- 테스트: [`server/tests/test_prune.py`](../server/tests/test_prune.py)

## 1. 왜 안전한가

`BertPolicy.forward` 는 항상 `inputs_embeds` 를 넘긴다.

```python
hidden = self.obs_proj(obs_seq)          # (B, T, 128)
encoded = self.encoder(inputs_embeds=hidden, attention_mask=...)
```

HF `BertEmbeddings.forward` 는 `inputs_embeds is not None` 이면
`word_embeddings(input_ids)` 를 **호출하지 않는다**. 따라서 30,522 × 128 어휘표는
순전파에 등장하지 않고 `state_dict` 용량만 차지한다.

프루닝은 어휘표를 **1행짜리 0 임베딩**으로 교체한다. `position_embeddings`(512×128),
`token_type_embeddings`(2×128), LayerNorm 은 실제로 쓰이므로 유지한다.

## 2. Before / After

`python -m training.prune_embeddings --checkpoint checkpoints/wall/best.pt --out checkpoints/wall_pruned`

| 항목 | Before | After | 절감 |
| --- | ---: | ---: | ---: |
| `word_embeddings` 행 수 | 30,522 | 1 | 30,521 |
| `word_embeddings` 파라미터 | 3,906,816 | 128 | 3,906,688 |
| 전체 파라미터 | 4,549,380 | 642,692 | 3,906,688 (85.9%) |
| `state_dict` 텐서 | 17.35 MiB | 2.45 MiB | 14.90 MiB (85.9%) |
| 체크포인트 파일 | 18,221,863 B (17.38 MiB) | 2,595,367 B (2.48 MiB) | 15,626,496 B (85.8%) |

- 남은 642,692 파라미터 = `obs_proj` + BERT-tiny trunk + LoRA + `head`.
- 추론 연산량은 변하지 않는다(제거한 모듈이 호출되지 않으므로). 줄어드는 것은
  **디스크·메모리 적재량과 모델 로드 시간**이다.
- 로짓 동등성: 8샘플 × 3가지 시퀀스 형태(전체/완전 왼쪽 패딩/절반 패딩)에서
  **최대 절대차 0.000e+00, `torch.equal == True`**.

## 3. 실제 플레이 비교

### 3.1 같은 미로 60판 (`compare_brains`)

`training/compare_brains.py` 는 미로를 생성하고 **정책이 스스로 한 스텝씩 행동을 골라**
목표에 도달할 때까지 실제 에피소드를 끝까지 돌린다. 두 체크포인트에 **같은 시드
90000~90059, 같은 15×15 미로**를 사용했다.

```powershell
cd server
python -m training.compare_brains --mazes 60 --size 15 `
  --ai-checkpoint checkpoints/wall/best.pt --out checkpoints/compare_original.json
python -m training.compare_brains --mazes 60 --size 15 `
  --ai-checkpoint checkpoints/wall_pruned/best.pt --out checkpoints/compare_pruned.json
```

| AI 체크포인트 | 성공 | 평균 스텝 | optimality | 루프율 | 벽 충돌 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wall/best.pt` (원본) | 60/60 (100%) | 96.9 | 1.865 | 0.19 | 0.00 |
| `wall_pruned/best.pt` | 60/60 (100%) | 96.9 | 1.865 | 0.19 | 0.00 |

같은 미로에서 짝지어 비교하면 **더 빠름 0 · 동일 60 · 더 느림 0, 평균 Δ 0.0 스텝**.
원본과 프루닝본의 행동 시퀀스가 판마다 동일하다.

### 3.2 데이터셋 test split 24판 (`evaluate`)

`training/evaluate.py` 도 데이터셋의 미로를 실제로 플레이한다 (`data/wall` test, 9~15×15).

| 체크포인트 | 성공 | 평균 스텝 | optimality | 루프율 |
| --- | ---: | ---: | ---: | ---: |
| `wall/best.pt` (원본) | 24/24 (100%) | 65.83 | 2.283 | 0.226 |
| `wall_pruned/best.pt` | 24/24 (100%) | 65.83 | 2.283 | 0.226 |

원본/프루닝본의 `eval_original.json` · `eval_pruned.json` 은 모든 필드가 일치한다.

### 3.3 "GUI로 직접 플레이" 자동 벤치마크는 없다

- **없다.** 저장소에는 사람이 GUI에서 플레이한 결과를 자동 수집·비교하는 벤치마크가 없다.
- 대신 위 두 스크립트가 **게임과 동일한 서버 코드 경로**(`MazeEnv` + `PolicyRunner`)로
  실제 에피소드를 수행한다. GUI 클라이언트는 서버가 계산한 행동을 그리기만 하므로,
  서버 측 롤아웃 결과가 GUI 플레이와 같은 정책 동작을 보장한다.
- GUI에서 눈으로 확인하려면 서버 체크포인트를 바꿔 실행한다:

```powershell
# 저장소 루트
$env:MAZE_CHECKPOINT = "server/checkpoints/wall_pruned/best.pt"
python test_launcher.py
```

## 4. 재현과 주의사항

```powershell
cd server
python -m training.prune_embeddings --checkpoint checkpoints/wall/best.pt --out checkpoints/wall_pruned
python -m pytest tests/test_prune.py
```

- 프루닝본은 `PolicyConfig.prune_word_embeddings=True` 를 체크포인트 config 에 저장한다.
  `BertPolicy.load` 는 이 값을 보고 **가중치를 읽기 전에** 1행 임베딩으로 모델을 구성한다.
  따라서 별도 플래그 없이 바로 로드된다.
- 일반 체크포인트(플래그 없음)는 기존과 똑같이 동작한다. 프루닝은 **opt-in** 이다.
- `prune_word_embeddings()` 는 멱등이다. 이미 1행이면 아무것도 하지 않는다.
- `position_embeddings` 는 `inputs_embeds` 경로에서도 더해지므로 **제거하면 안 된다**.
  (`token_type_embeddings` 도 동일.)
- int8 양자화는 이 환경에서 이득이 없었다: torch 2.14.1+cpu 는 `onednn` 엔진만 지원해
  `quantized::linear` 추론이 실패했고, dynamic int8 은 오히려 느려졌다(2.9ms → 4.9ms).
  이번 작업은 **구조적 프루닝**만 다룬다.
