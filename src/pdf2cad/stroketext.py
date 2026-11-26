"""Read dimension numbers that were exported as outlines or SHX strokes instead of text.

Every character of a vector font is drawn the same way each time it appears, so the
glyphs of one drawing fall into a handful of shape classes.  A dimension's measured
length times the drawing scale is the number printed on it; trying the common scales
and keeping the one under which every glyph class always stands for the same digit
yields both the scale and a glyph -> digit table, with no font knowledge needed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .geom import Vec, dot, sub
from .model import Shape
from .spatial import Grid

MAX_GLYPH = 8.0
MAX_RUN = 8
MATCH_TOL = 0.06


@dataclass
class Run:
    """A row of glyphs read along direction u (with n pointing to the glyph tops)."""

    glyphs: list[Shape]
    origin: Vec
    u: Vec
    n: Vec
    x0: float
    x1: float
    y0: float
    y1: float
    descriptors: list[tuple] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def center(self) -> Vec:
        x, y = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        o, u, n = self.origin, self.u, self.n
        return (o[0] + u[0] * x + n[0] * y, o[1] + u[1] * x + n[1] * y)


def glyph_pool(shapes: list[Shape], consumed: set[int]) -> tuple[Grid[int], dict[int, Shape]]:
    grid: Grid[int] = Grid(5.0)
    pool: dict[int, Shape] = {}
    for s in shapes:
        if s.id in consumed:
            continue
        b = s.bbox()
        if max(b[2] - b[0], b[3] - b[1]) > MAX_GLYPH:
            continue
        pool[s.id] = s
        grid.insert(b, s.id)
    return grid, pool


def _local(s: Shape, o: Vec, u: Vec, n: Vec) -> tuple[list[Vec], tuple[float, float, float, float]]:
    pts = [(dot(sub(p, o), u), dot(sub(p, o), n)) for p in s.points()]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return pts, (min(xs), min(ys), max(xs), max(ys))


def find_run(grid: Grid[int], pool: dict[int, Shape], taken: set[int], a: Vec, b: Vec,
             u: Vec, n: Vec, length: float) -> tuple[float, Run] | None:
    """Best glyph row parallel to the segment a-b, just off either side of it."""
    pad = 6.0
    box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
    near = []
    for sid in grid.query(box, pad):
        if sid in taken:
            continue
        s = pool[sid]
        pts, lb = _local(s, a, u, n)
        cx = (lb[0] + lb[2]) / 2
        if cx < -0.5 * length - 4 or cx > 1.5 * length + 4:
            continue
        if lb[1] >= 0.05 and lb[1] <= 3.0:
            near.append((1, s, pts, lb))
        elif lb[3] <= -0.05 and lb[3] >= -9.0:
            near.append((-1, s, pts, lb))
    best: tuple[float, Run] | None = None
    for side in (1, -1):
        items = [x for x in near if x[0] == side]
        seeds = [x for x in items if x[3][3] - x[3][1] >= 1.0]
        if not seeds:
            continue
        for seed in sorted(seeds, key=lambda x: abs((x[3][0] + x[3][2]) / 2 - length / 2))[:3]:
            run = _grow(seed, items)
            if run is None:
                continue
            run.origin, run.u, run.n = a, u, n
            gap = run.y0 if side == 1 else -run.y1
            if side == -1 and gap > 1.5 * run.height + 1.0:
                continue
            if side == 1 and gap > 0.8 * run.height + 0.3:
                continue
            off = abs((run.x0 + run.x1) / 2 - length / 2) / max(length, 1.0)
            score = gap + off * 4 + (0 if side == 1 else 0.5)
            if best is None or score < best[0]:
                best = (score, run)
    return best


def _same_font(a: Shape, b: Shape) -> bool:
    return a.style == b.style and (a.fill is None) == (b.fill is None) and a.stroke == b.stroke


def _grow(seed, items) -> Run | None:
    h = seed[3][3] - seed[3][1]
    base = seed[3][1]
    row = [seed]
    lo, hi = seed[3][0], seed[3][2]
    rest = sorted((x for x in items if x is not seed and _same_font(x[1], seed[1])), key=lambda x: x[3][0])
    changed = True
    while changed:
        changed = False
        for x in rest:
            if any(x is r for r in row):
                continue
            lb = x[3]
            if abs(lb[1] - base) > 0.2 * h or not 0.4 * h <= lb[3] - lb[1] <= 1.15 * h:
                continue
            if lb[0] - hi <= 0.55 * h and lo - lb[2] <= 0.55 * h:
                row.append(x)
                lo, hi = min(lo, lb[0]), max(hi, lb[2])
                changed = True
                if len(row) > MAX_RUN:
                    return None
    row.sort(key=lambda x: x[3][0])
    y0 = min(x[3][1] for x in row)
    y1 = max(x[3][3] for x in row)
    h = y1 - y0
    descs = []
    for _side, s, pts, lb in row:
        descs.append((s.closed, tuple((round((p[0] - lb[0]) / h, 4), round((p[1] - y0) / h, 4)) for p in pts)))
    return Run([x[1] for x in row], (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), lo, hi, y0, y1, descs)


def read_expected(grid: Grid[int], pool: dict[int, Shape], used: set[int], a: Vec, b: Vec, u: Vec, n: Vec,
                  length: float, h: float, classes: "GlyphClasses", table: dict[int, str],
                  expected: float) -> tuple[str, Run] | None:
    """Find the expected value among digit glyphs beside a dimension line.

    In a chain of short dimensions the numbers run into each other, so no single run of
    glyphs belongs to one dimension.  Once the glyph table is known, the digits along the
    line can be read and the expected number picked out next to the span it labels.
    """
    pad = 4 * h + 2
    box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
    tol = max(1.0, expected * 0.1)
    best = None
    near = []
    for sid in grid.query(box, pad):
        if sid in used:
            continue
        s = pool[sid]
        pts, lb = _local(s, a, u, n)
        gh = lb[3] - lb[1]
        if not 0.85 * h <= gh <= 1.15 * h:
            continue
        cx = (lb[0] + lb[2]) / 2
        if cx < -0.5 * length - 4 * h or cx > 1.5 * length + 4 * h:
            continue
        if 0.05 <= lb[1] <= 0.8 * h + 0.3 or -1.5 * h - 1.0 <= lb[3] <= -0.05:
            desc = (s.closed, tuple((round((p[0] - lb[0]) / gh, 4), round((p[1] - lb[1]) / gh, 4)) for p in pts))
            near.append((lb, table.get(classes.classify(desc)), s))
    near.sort(key=lambda x: x[0][1])
    rows: list[list] = []
    for item in near:
        if rows and abs(item[0][1] - rows[-1][0][0][1]) <= 0.2 * h:
            rows[-1].append(item)
        else:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda x: x[0][0])
        for i in range(len(row)):
            if row[i][1] is None or (row[i][1] == "0"):
                continue
            for j in range(i + 1, min(len(row), i + 6) + 1):
                if row[j - 1][1] is None or (j - 1 > i and row[j - 1][0][0] - row[j - 2][0][2] > 0.55 * h):
                    break
                v = float("".join(x[1] for x in row[i:j]))
                if abs(v - expected) > tol:
                    continue
                x0, x1 = row[i][0][0], max(x[0][2] for x in row[i:j])
                cx = (x0 + x1) / 2
                if cx < -0.5 * length - 2 * h or cx > 1.5 * length + 2 * h:
                    continue
                key = (round(abs(v - expected) / expected, 3), abs(cx - length / 2))
                if best is None or key < best[0]:
                    y0 = min(x[0][1] for x in row[i:j])
                    y1 = max(x[0][3] for x in row[i:j])
                    run = Run([x[2] for x in row[i:j]], a, u, n, x0, x1, y0, y1)
                    best = (key, "".join(x[1] for x in row[i:j]), run)
    return None if best is None else (best[1], best[2])


class GlyphClasses:
    """Clusters glyph descriptors that are the same character."""

    def __init__(self) -> None:
        self.reps: dict[tuple, list[tuple[int, tuple]]] = defaultdict(list)
        self.count = 0

    def classify(self, desc: tuple) -> int:
        closed, pts = desc
        key = (closed, len(pts))
        for cid, rep in self.reps[key]:
            if all(abs(p[0] - q[0]) <= MATCH_TOL and abs(p[1] - q[1]) <= MATCH_TOL for p, q in zip(pts, rep)):
                return cid
        cid = self.count
        self.count += 1
        self.reps[key].append((cid, pts))
        return cid


def _labels(value: float) -> list[str]:
    """Plausible printed numbers for a measured value, favouring round ones."""
    tol = max(0.6, value * 0.0015)
    out = []
    for step in (100, 50, 10, 5):
        q = round(value / step) * step
        if q > 0 and abs(value - q) <= tol and str(int(q)) not in out:
            out.append(str(int(q)))
    r = round(value)
    if not out or abs(value - r) <= 0.6:
        if str(int(r)) not in out:
            out.append(str(int(r)))
    return out


def _score(samples: list[tuple[float, list[int]]], s: float):
    votes: dict[int, Counter] = defaultdict(Counter)
    matched = 0
    for measured, cls in samples:
        labels = [lab for lab in _labels(measured * s) if len(lab) == len(cls)]
        if not labels:
            continue
        matched += 1
        w = 1.0 / len(labels)
        for lab in labels:
            for c, ch in zip(cls, lab):
                votes[c][ch] += w
    if matched < 2:
        return 0.0, 0.0, votes
    agree = sum(v.most_common(1)[0][1] for v in votes.values())
    total = sum(sum(v.values()) for v in votes.values())
    purity = agree / total if total else 0.0
    return agree * purity, purity, votes


def learn(samples: list[tuple[float, list[int]]], scales: list[float]) -> tuple[float, dict[int, str]] | None:
    """Pick the scale under which glyph classes map consistently to digits.

    samples: (measured paper length, glyph class ids in reading order).  Besides the
    standard scales, a fine log search covers sheets printed with "fit to page".
    """
    usable = [x for x in samples if 1 <= len(x[1]) <= 6 and x[0] > 0.5]
    if len(usable) < 2:
        return None
    sub = usable[:: max(1, len(usable) // 150)]
    grid = []
    s = 0.5
    while s < 2000:
        grid.append(s)
        s *= 1.002
    coarse = sorted(((_score(sub, s)[0], s) for s in grid), reverse=True)[:4]
    best_score, best_s = 0.0, None
    for _sc, s0 in coarse:
        for i in range(-12, 13):
            s = s0 * (1 + i * 0.0002)
            sc = _score(usable, s)[0]
            if sc > best_score:
                best_score, best_s = sc, s
    std_score, std_s = max((_score(usable, s)[0], s) for s in scales)
    if std_score > 0 and std_score >= 0.97 * best_score:
        best_score, best_s = std_score, std_s
    if best_s is None:
        return None
    _sc, purity, votes = _score(samples, best_s)
    if purity < 0.6:
        return None
    table = {c: v.most_common(1)[0][0] for c, v in votes.items()
             if v.most_common(1)[0][1] / sum(v.values()) >= 0.6 and v.most_common(1)[0][1] >= 0.99}
    return best_s, table
