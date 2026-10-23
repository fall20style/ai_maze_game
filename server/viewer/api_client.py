"""HTTP 클라이언트 — 뷰어가 서버와 통신하는 유일한 통로.

이 모듈에는 **미로 규칙도, 모델도, 경로 탐색도 없다** (PLAN §1 R1·R2).
표준 라이브러리(``urllib``)만 쓰므로 서버 코드를 import 하지 않는다.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_SERVER = "http://127.0.0.1:8000"


class ApiError(RuntimeError):
    """서버가 응답하지 않거나 오류를 돌려줬을 때."""


class GameApi:
    """얇은 REST 래퍼. 모든 이동 결정은 서버에서 일어난다."""

    #: 체크포인트 로드가 걸리는 요청(게임 생성/구현체 교체)의 타임아웃
    LOAD_TIMEOUT = 60.0

    def __init__(self, base_url: str = DEFAULT_SERVER, timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.load_timeout = max(self.timeout, self.LOAD_TIMEOUT)

    # ------------------------------------------------------------------ 내부
    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout if timeout is None else timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except json.JSONDecodeError:
                pass
            raise ApiError(f"{exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiError(f"서버에 연결할 수 없습니다 ({self.base_url})") from exc

        return json.loads(body) if body else None

    # ------------------------------------------------------------------ API
    def health(self) -> dict:
        return self._request("GET", "/health")

    def brains(self) -> list[dict]:
        return self._request("GET", "/brains")

    def create(
        self,
        size: int = 15,
        algorithm: str = "dfs",
        brain: str = "ai",
        rule: str = "wall",
        seed: int | None = None,
        max_steps: int | None = None,
        randomize_endpoints: bool = True,
    ) -> dict:
        # AI 구현체는 첫 요청에서 체크포인트를 로드한다 — 몇 초 걸릴 수 있다.
        return self._request(
            "POST",
            "/games",
            {
                "size": size,
                "algorithm": algorithm,
                "brain": brain,
                "rule": rule,
                "seed": seed,
                "max_steps": max_steps,
                "randomize_endpoints": randomize_endpoints,
            },
            timeout=self.load_timeout,
        )

    def advance(self, game_id: str, steps: int = 1) -> dict:
        return self._request("POST", f"/games/{game_id}/advance", {"steps": steps})

    def action(self, game_id: str, action: int) -> dict:
        return self._request("POST", f"/games/{game_id}/action", {"action": int(action)})

    def set_brain(self, game_id: str, brain: str, rule: str = "wall") -> dict:
        return self._request(
            "POST",
            f"/games/{game_id}/brain",
            {"brain": brain, "rule": rule},
            timeout=self.load_timeout,
        )

    def reset(self, game_id: str) -> dict:
        return self._request("POST", f"/games/{game_id}/reset")

    def new_maze(self, game_id: str, seed: int | None = None) -> dict:
        return self._request("POST", f"/games/{game_id}/new-maze", {"seed": seed})

    def delete(self, game_id: str) -> dict:
        return self._request("DELETE", f"/games/{game_id}")
