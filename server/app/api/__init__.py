"""FastAPI 서버 — 게임 권위 계층.

- :mod:`app.api.main`    HTTP 엔드포인트
- :mod:`app.api.session` 한 판의 상태 (미로 + 환경 + Brain)
- :mod:`app.api.brains`  서버 측 이동 결정 (ai / rule / human)
- :mod:`app.api.schemas` 클라이언트와 공유하는 계약

SDL 뷰어는 이 패키지를 import 하지 않고 HTTP 로만 통신한다.
"""

__all__: list[str] = []
