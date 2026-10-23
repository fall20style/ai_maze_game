"""SDL(pygame) 게임 뷰어 — 렌더링 전용.

이 패키지에는 경로 탐색 코드가 없다. 이동은 :mod:`viewer.brains` 의 Brain 이
결정하고, Brain 은 모델(또는 사람 입력)을 감쌀 뿐이다.

실행::

    python -m viewer.app --size 15
    python -m viewer.app --checkpoint checkpoints/v3/best.pt

패키지 초기화에서 하위 모듈을 import 하지 않는다. 레이아웃 계산만 쓰는 코드
(테스트 등)가 pygame/torch 를 끌고 오지 않게 하기 위해서다.
"""
