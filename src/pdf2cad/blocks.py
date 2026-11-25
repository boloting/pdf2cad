"""Find repeated groups of geometry and turn them into BLOCK definitions with INSERTs."""

from __future__ import annotations

import math
from collections import defaultdict

from .geom import Vec, bbox, rotate, segment_segment_distance, sub
from .model import Arc, BlockDef, Bezier, Drawing, Insert, Line, Shape
from .spatial import Grid

TOUCH = 0.05


def _flat_segments(s: Shape) -> list[tuple[Vec, Vec]]:
    pts = s.points()
    return list(zip(pts, pts[1:]))


def components(shapes: list[Shape]) -> list[list[Shape]]:
    """Group shapes that touch or cross each other."""
    grid: Grid[tuple[int, int]] = Grid(4.0)
    segs: dict[int, list[tuple[Vec, Vec]]] = {}
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for idx, s in enumerate(shapes):
        segs[idx] = _flat_segments(s)
        boxes[idx] = []
        for k, (a, b) in enumerate(segs[idx]):
            box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
            boxes[idx].append(box)
            grid.insert(box, (idx, k))

    parent = list(range(len(shapes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for idx in range(len(shapes)):
        for (a, b), box in zip(segs[idx], boxes[idx]):
            x0, y0, x1, y1 = box[0] - TOUCH, box[1] - TOUCH, box[2] + TOUCH, box[3] + TOUCH
            for j, k in grid.query(box, TOUCH):
                if j <= idx:
                    continue
                ob = boxes[j][k]
                if ob[0] > x1 or ob[2] < x0 or ob[1] > y1 or ob[3] < y0:
                    continue
                if find(j) == find(idx):
                    continue
                c, d = segs[j][k]
                if segment_segment_distance(a, b, c, d) <= TOUCH:
                    parent[find(j)] = find(idx)

    groups: dict[int, list[Shape]] = defaultdict(list)
    for idx, s in enumerate(shapes):
        groups[find(idx)].append(s)
    return list(groups.values())


def _r(v: float, q: float = 0.02) -> int:
    return int(round(v / q))


def _pt(p: Vec) -> tuple[int, int]:
    return (_r(p[0]), _r(p[1]))


def _seg_key(g) -> tuple:
    if isinstance(g, Line):
        return ("L", _pt(g.a), _pt(g.b))
    if isinstance(g, Arc):
        if g.is_full:
            return ("C", _pt(g.center), _r(g.radius))
        return ("A", _pt(g.center), _r(g.radius), _pt(g.start()), _pt(g.end()), g.sweep > 0)
    return ("B", _pt(g.p0), _pt(g.p1), _pt(g.p2), _pt(g.p3))


def _shape_key(s: Shape) -> tuple:
    fwd = [_seg_key(g) for g in s.segs]
    rev = [_seg_key(g.reversed()) for g in reversed(s.segs)]
    options = [tuple(fwd), tuple(rev)]
    if s.closed and len(s.segs) > 1:
        n = len(fwd)
        options = [tuple(seq[i:] + seq[:i]) for seq in (fwd, rev) for i in range(n)]
    st = s.style
    return (st.layer, st.color, round(st.width, 2), st.dashes, s.fill, s.closed, min(options))


def _canonical(group: list[Shape], rot: float) -> tuple[tuple, Vec]:
    pts = [rotate(p, rot) for s in group for p in s.points()]
    b = bbox(pts)
    c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def f(p: Vec) -> Vec:
        return sub(rotate(p, rot), c)

    keys = tuple(sorted(_shape_key(s.transformed(f, rot)) for s in group))
    return keys, c


def recognize(d: Drawing, consumed: set[int], min_shapes: int = 2, min_count: int = 2,
              max_size: float = 0.15) -> None:
    diag = math.hypot(d.width, d.height)
    limit = max_size * diag

    def small(s: Shape) -> bool:
        b = s.bbox()
        return math.hypot(b[2] - b[0], b[3] - b[1]) <= limit

    # Long shapes (walls, frames, axes) cannot be part of a block; leaving them out
    # keeps a door that touches a wall from merging into the wall's component.
    shapes = [s for s in d.shapes if s.id not in consumed and small(s)]
    sigs: dict[tuple, list[tuple[list[Shape], float, Vec]]] = defaultdict(list)
    for group in components(shapes):
        if len(group) < min_shapes:
            continue
        pts = [p for s in group for p in s.points()]
        b = bbox(pts)
        if math.hypot(b[2] - b[0], b[3] - b[1]) > limit:
            continue
        best = None
        for rot in (0.0, 90.0, 180.0, 270.0):
            key, c = _canonical(group, rot)
            if best is None or key < best[0]:
                best = (key, rot, c)
        assert best is not None
        sigs[best[0]].append((group, best[1], best[2]))

    n = 0
    for key, instances in sorted(sigs.items(), key=lambda kv: -len(kv[1])):
        if len(instances) < min_count:
            continue
        n += 1
        name = f"PDF_BLOCK_{n:02d}"
        group0, rot0, c0 = instances[0]

        def to_local(p: Vec, rot=rot0, c=c0) -> Vec:
            return sub(rotate(p, rot), c)

        d.blocks.append(BlockDef(name, [s.transformed(to_local, rot0) for s in group0]))
        for group, rot, c in instances:
            styles: dict = defaultdict(int)
            for s in group:
                styles[s.style] += 1
            style = max(styles, key=lambda k: styles[k])
            d.inserts.append(Insert(name, rotate(c, -rot), (-rot) % 360, style.layer, style))
            consumed.update(s.id for s in group)


__all__ = ["recognize", "components", "Bezier"]
