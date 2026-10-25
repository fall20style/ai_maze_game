"""독립 추론 API 의 체크포인트 해석과 모델 메타데이터.

게임 Brain 과 달리 추론 API 는 사용자가 체크포인트를 고를 수 있다. 경로 탐색을
막기 위해 **명시적으로 지정한 경로는 ``server/checkpoints`` 하위로 제한**한다.
(자동 탐색은 ``MAZE_CHECKPOINT`` 환경변수를 그대로 존중한다.)
"""

from __future__ import annotations

from pathlib import Path

from app.api.brains import discover_checkpoint
from app.api.schemas import ModelInfo
from app.ml.policy import BertPolicy
from app.ml.registry import is_loaded


def resolve_checkpoint(requested: str | None, server_root: Path) -> str:
    """요청된 체크포인트를 실제 파일 경로로 해석한다.

    - ``None``/빈 문자열: 게임 서버와 같은 자동 탐색 순서
    - ``"wall"``: ``checkpoints/wall/best.pt``
    - ``"wall/best.pt"`` 또는 절대 경로: 그 파일 (단, ``checkpoints`` 하위)
    """
    root = (server_root / "checkpoints").resolve()

    if requested is None or not requested.strip():
        found = discover_checkpoint(server_root)
        if not found:
            raise FileNotFoundError(
                "체크포인트를 찾지 못했습니다. MAZE_CHECKPOINT 를 지정하세요."
            )
        return str(Path(found).resolve())

    raw = requested.strip()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    elif candidate.suffix == ".pt":
        resolved = (root / candidate).resolve()
    else:
        by_dir = root / candidate / "best.pt"
        resolved = (by_dir if by_dir.exists() else root / candidate).resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다: {requested}")
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PermissionError(
            "checkpoint 경로는 server/checkpoints/ 하위여야 합니다"
        ) from None
    return str(resolved)


def describe_model(checkpoint: str, model: BertPolicy) -> ModelInfo:
    cfg = model.cfg
    return ModelInfo(
        checkpoint=checkpoint,
        obs_dim=int(cfg.obs_dim),
        seq_len=int(cfg.seq_len),
        num_actions=int(cfg.num_actions),
        window_size=int(cfg.window_size),
        global_size=int(cfg.global_size),
        map_size=int(cfg.map_size),
        loaded=is_loaded(checkpoint),
    )
