"""보드 레이아웃 계산 — pygame 에 의존하지 않는 순수 함수.

창 크기가 바뀌어도 보드는 항상 가운데 정렬되고, 셀은 정사각형을 유지한다.
디스플레이 없이 단위 테스트할 수 있도록 여기에는 그리기 코드를 두지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoardLayout:
    """미로 격자를 화면에 배치한 결과 (픽셀 단위, float)."""

    cell: float
    origin_x: float
    origin_y: float
    cols: int
    rows: int

    @property
    def width(self) -> float:
        return self.cell * self.cols

    @property
    def height(self) -> float:
        return self.cell * self.rows

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.origin_x, self.origin_y, self.width, self.height)

    def cell_rect(self, x: int, y: int) -> tuple[float, float, float, float]:
        return (
            self.origin_x + x * self.cell,
            self.origin_y + y * self.cell,
            self.cell,
            self.cell,
        )

    def center(self, x: float, y: float) -> tuple[float, float]:
        """셀 좌표(정수 또는 보간된 float)의 화면 중심."""
        return (
            self.origin_x + (x + 0.5) * self.cell,
            self.origin_y + (y + 0.5) * self.cell,
        )


def fit_board(
    cols: int,
    rows: int,
    area_w: float,
    area_h: float,
    margin: float = 10.0,
    min_cell: float = 4.0,
) -> BoardLayout:
    """``area_w × area_h`` 영역 안에 ``cols × rows`` 격자를 최대로 맞춘다."""
    if cols <= 0 or rows <= 0:
        raise ValueError(f"cols/rows must be positive, got {cols}x{rows}")
    if area_w <= 0 or area_h <= 0:
        raise ValueError(f"area must be positive, got {area_w}x{area_h}")

    avail_w = max(min_cell * cols, area_w - 2 * margin)
    avail_h = max(min_cell * rows, area_h - 2 * margin)

    cell = max(min_cell, min(avail_w / cols, avail_h / rows))
    origin_x = (area_w - cell * cols) / 2.0
    origin_y = (area_h - cell * rows) / 2.0

    return BoardLayout(
        cell=cell,
        origin_x=origin_x,
        origin_y=origin_y,
        cols=cols,
        rows=rows,
    )
