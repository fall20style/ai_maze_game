"""정책 체크포인트 로더/캐시.

게임 Brain 과 독립 추론 API 가 **같은 모델 인스턴스**를 공유하도록 한 곳에서
체크포인트를 로드한다. 체크포인트는 프로세스당 한 번만 읽고, 이후 요청은
메모리 캐시를 재사용한다.

캐시 키는 **정규화된 절대 경로**다. 같은 파일을 상대/절대 경로로 번갈아 넘겨도
중복 로드하지 않는다.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .policy import BertPolicy

_LOCK = threading.Lock()
_POLICIES: dict[str, BertPolicy] = {}


def normalize_checkpoint(checkpoint: str | Path) -> str:
    return str(Path(checkpoint).expanduser().resolve())


def load_policy(checkpoint: str | Path) -> BertPolicy:
    """체크포인트를 로드하고 프로세스 캐시에 보관한다 (스레드 안전)."""
    key = normalize_checkpoint(checkpoint)
    model = _POLICIES.get(key)
    if model is not None:
        return model

    with _LOCK:
        model = _POLICIES.get(key)
        if model is None:
            model = BertPolicy.load(key)
            _POLICIES[key] = model
    return model


def is_loaded(checkpoint: str | Path) -> bool:
    return normalize_checkpoint(checkpoint) in _POLICIES


def cached_checkpoints() -> list[str]:
    return sorted(_POLICIES)


def clear_cache() -> None:
    """테스트/핫리로드용. 로드된 모델을 모두 버린다."""
    with _LOCK:
        _POLICIES.clear()
