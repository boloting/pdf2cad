"""Uniform grid for neighbourhood queries over bounding boxes."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Generic, Hashable, TypeVar

T = TypeVar("T", bound=Hashable)


class Grid(Generic[T]):
    def __init__(self, cell: float = 5.0) -> None:
        self.cell = cell
        self.cells: dict[tuple[int, int], list[T]] = defaultdict(list)

    def _range(self, bbox, pad: float = 0.0):
        c = self.cell
        x0, y0 = math.floor((bbox[0] - pad) / c), math.floor((bbox[1] - pad) / c)
        x1, y1 = math.floor((bbox[2] + pad) / c), math.floor((bbox[3] + pad) / c)
        # Very large boxes would touch an enormous number of cells; clamp them.
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 40000:
            x1 = min(x1, x0 + 199)
            y1 = min(y1, y0 + 199)
        for i in range(x0, x1 + 1):
            for j in range(y0, y1 + 1):
                yield (i, j)

    def insert(self, bbox, item: T) -> None:
        for k in self._range(bbox):
            self.cells[k].append(item)

    def insert_point(self, p, item: T) -> None:
        self.insert((p[0], p[1], p[0], p[1]), item)

    def query(self, bbox, pad: float = 0.0) -> set[T]:
        out: set[T] = set()
        for k in self._range(bbox, pad):
            lst = self.cells.get(k)
            if lst:
                out.update(lst)
        return out

    def query_point(self, p, pad: float = 0.0) -> set[T]:
        return self.query((p[0], p[1], p[0], p[1]), pad)
