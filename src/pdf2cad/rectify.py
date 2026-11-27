"""Snap a fit-to-page PDF to its labeled scale and true dimension lengths.

PDF plots printed with “fit to page” come out at a non-standard scale (e.g. 1:104),
and every vertex carries a little plotting noise (coordinates are quantised, so
individual lengths are off by one to three millimetres). The title block still says
1:100, and the numbers on the dimensions are the design sizes. This module maps the
drawing first, then redraws it to the labels:

1. Moves geometry into real millimetres with the detected fit scale.
2. Clusters the ends of orthogonal dimensions into stations (grid lines) per axis and
   arbitrates the labels: a label far from the drawn length, or one that contradicts
   other labels around a loop, is flagged for review instead of being forced.
3. Solves the station positions from the remaining labels and rounds them to whole
   millimetres, so every accepted label is met exactly.
4. Maps everything piecewise-linearly between stations, then snaps nearby vertices
   exactly onto the stations, and parallel wall faces onto whole-10 mm thicknesses.
5. Writes coordinates back in paper millimetres at the intended scale (1:100), so
   AutoCAD’s DIM reads the same numbers that are printed on the drawing.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from statistics import median

import numpy as np

from .dims import parse_value
from .geom import angle_diff, dist
from .model import (
    COMMON_SCALES,
    Arc,
    AxisBubble,
    Bezier,
    Drawing,
    Hatch,
    HatchFamily,
    Leader,
    Line,
    LinearDim,
    RadialDim,
    Shape,
    Style,
    TextItem,
)

# Tolerances in paper millimetres (multiplied by the drawing scale): plotting noise is
# a property of the sheet, not of the building.
# Dimension ends this close are the same station (grid line).
_CLUSTER_PAPER = 0.015
# A label this far from the drawn length is not a rounding difference: flag it.
_GROSS_PAPER = 0.06
# After redraw, vertices this close to a station are put exactly on it.
_SNAP_PAPER = 0.02
# A dimension is horizontal/vertical if its angle is within this of 0°/90°.
_AXIS_TOL = 4.0
# Fit-to-page scales within this fraction of a standard scale snap to that standard.
_FIT_TOL = 0.08

# Wall thickness snap, in real millimetres; only for building scales (1:20 and smaller).
_WALL_MIN_SCALE = 20
_WALL_MIN_LEN = 500.0
_WALL_GAP = (80.0, 400.0)
_WALL_MODULE = 10.0
_WALL_TOL = 1.5


def intended_scale(fit: float) -> float | None:
    """Nearest standard scale when `fit` looks like a fit-to-page print of it."""
    if fit is None or fit <= 0:
        return None
    best = min(COMMON_SCALES, key=lambda c: abs(fit / c - 1.0))
    if abs(fit / best - 1.0) <= _FIT_TOL:
        return float(best)
    return None


def rectify(d: Drawing, fit_scale: float, target_scale: float) -> dict:
    """Rewrite `d` in place to target_scale paper mm with dimension-true lengths.

    Flagged dimensions get ``review = True``. Returns counts, conflicts and residuals.
    """
    if fit_scale <= 0 or target_scale <= 0:
        return {"horiz": 0, "vert": 0, "conflicts": [], "residuals": {}}
    _scale_drawing(d, fit_scale)
    hx = _solve_axis(d, 0, target_scale)
    hy = _solve_axis(d, 1, target_scale)
    for info in (hx, hy):
        for c in info["flagged"]:
            d.linear_dims[c.dim].review = True
    _map_drawing(d, hx, hy)
    snap = _SNAP_PAPER * target_scale
    _place_dims(d, hx, 0)
    _place_dims(d, hy, 1)
    _transform_points(d, _snapper(hx["targets"], snap), _snapper(hy["targets"], snap))
    walls = {"pairs": 0, "snapped": 0}
    if target_scale >= _WALL_MIN_SCALE:
        for axis, info in ((0, hx), (1, hy)):
            w = _snap_walls(d, axis, info["targets"])
            walls["pairs"] += w["pairs"]
            walls["snapped"] += w["snapped"]
    conflicts = _describe_conflicts(d, hx, 0) + _describe_conflicts(d, hy, 1)
    residuals = _residuals(d, hx, hy)
    residuals["walls"] = walls
    _scale_drawing(d, 1.0 / target_scale)
    for dim in d.linear_dims:
        dim.measured = dist(dim.p1, dim.p2)
    for dim in d.radial_dims:
        dim.radius = dist(dim.center, dim.point) if dim.point else dim.radius
    return {
        "horiz": hx["constraints"],
        "vert": hy["constraints"],
        "stations_x": hx["stations"],
        "stations_y": hy["stations"],
        "conflicts": conflicts,
        "residuals": residuals,
    }


# -- uniform scale -----------------------------------------------------------------

def _scale_drawing(d: Drawing, k: float) -> None:
    if abs(k - 1.0) < 1e-15:
        return
    f = lambda p: (p[0] * k, p[1] * k)
    d.width *= k
    d.height *= k
    d.shapes = [_scale_shape(s, f, k) for s in d.shapes]
    d.texts = [_scale_text(t, f, k) for t in d.texts]
    d.linear_dims = [_scale_linear(dim, f, k) for dim in d.linear_dims]
    d.radial_dims = [_scale_radial(dim, f, k) for dim in d.radial_dims]
    d.leaders = [_scale_leader(ld, f, k) for ld in d.leaders]
    d.hatches = [_scale_hatch(h, f, k) for h in d.hatches]
    d.blocks = [replace(b, shapes=[_scale_shape(s, f, k) for s in b.shapes]) for b in d.blocks]
    d.inserts = [replace(ins, insert=f(ins.insert)) for ins in d.inserts]
    d.axis_bubbles = [_scale_bubble(b, f, k) for b in d.axis_bubbles]


def _scale_style(st: Style, k: float) -> Style:
    dashes = tuple(round(v * k, 6) for v in st.dashes) if st.dashes else ()
    return Style(st.layer, st.color, round(st.width * k, 6), dashes)


def _scale_seg(g, f, k):
    if isinstance(g, Line):
        return Line(f(g.a), f(g.b))
    if isinstance(g, Arc):
        return Arc(f(g.center), g.radius * k, g.start_angle, g.sweep)
    if isinstance(g, Bezier):
        return Bezier(f(g.p0), f(g.p1), f(g.p2), f(g.p3))
    return g


def _scale_shape(s: Shape, f, k: float) -> Shape:
    return Shape(
        [_scale_seg(g, f, k) for g in s.segs], s.closed, _scale_style(s.style, k),
        s.stroke, s.fill, s.path_id, s.id,
    )


def _scale_text(t: TextItem, f, k: float) -> TextItem:
    x0, y0, x1, y1 = t.bbox
    return replace(
        t, insert=f(t.insert), height=t.height * k, width=t.width * k,
        bbox=(x0 * k, y0 * k, x1 * k, y1 * k),
    )


def _scale_linear(dim: LinearDim, f, k: float) -> LinearDim:
    out = replace(
        dim, p1=f(dim.p1), p2=f(dim.p2), base=f(dim.base), text_pos=f(dim.text_pos),
        text_height=dim.text_height * k, arrow_size=dim.arrow_size * k,
        exe=dim.exe * k, measured=dim.measured * k,
    )
    for name in ("dle", "ext1", "ext2"):
        if hasattr(dim, name):
            val = getattr(dim, name)
            setattr(out, name, val * k if name == "dle" else val)
    return out


def _scale_radial(dim: RadialDim, f, k: float) -> RadialDim:
    return replace(
        dim, center=f(dim.center), point=f(dim.point), text_pos=f(dim.text_pos),
        radius=dim.radius * k, text_height=dim.text_height * k, arrow_size=dim.arrow_size * k,
    )


def _scale_leader(ld: Leader, f, k: float) -> Leader:
    return replace(
        ld, vertices=[f(v) for v in ld.vertices], text_pos=f(ld.text_pos),
        text_height=ld.text_height * k, arrow_size=ld.arrow_size * k,
    )


def _scale_hatch(h: Hatch, f, k: float) -> Hatch:
    loops = [[_scale_seg(g, f, k) for g in loop] for loop in h.loops]
    fams = [
        HatchFamily(fam.angle, f(fam.base), fam.spacing * k, fam.shift * k,
                    tuple(v * k for v in fam.dashes))
        for fam in h.families
    ]
    return replace(h, loops=loops, families=fams, pattern_scale=h.pattern_scale * k)


def _scale_bubble(b: AxisBubble, f, k: float) -> AxisBubble:
    # slash / part_pos are offsets from the centre; scale them, do not treat as absolute points.
    slash = tuple((p[0] * k, p[1] * k) for p in b.slash) if b.slash else None
    part = tuple((p[0] * k, p[1] * k) for p in b.part_pos) if b.part_pos else None
    return replace(
        b, center=f(b.center), radius=b.radius * k, text_height=b.text_height * k,
        style=_scale_style(b.style, k), slash=slash, part_pos=part,
    )


# -- dimension constraints ---------------------------------------------------------

@dataclass
class _Con:
    dim: int  # index into d.linear_dims
    lo: float  # real-mm coordinates of the two ends along the axis, lo <= hi
    hi: float
    value: float
    dec: int
    i: int = 0  # stations of lo / hi
    j: int = 0
    reason: str = ""

    @property
    def dev(self) -> float:
        """Label minus drawn length."""
        return self.value - (self.hi - self.lo)


def dim_axis(dim: LinearDim) -> int | None:
    """0 for a dimension measuring X, 1 for Y, None for an aligned/rotated one."""
    ang = dim.angle % 180
    if angle_diff(ang, 0.0) <= _AXIS_TOL:
        return 0
    if angle_diff(ang, 90.0) <= _AXIS_TOL:
        return 1
    return None


def _constraints(d: Drawing, axis: int) -> list[_Con]:
    out = []
    for n, dim in enumerate(d.linear_dims):
        if dim_axis(dim) != axis:
            continue
        pv = parse_value(dim.text)
        if not pv or pv[0] <= 0:
            continue
        a, b = dim.p1[axis], dim.p2[axis]
        out.append(_Con(n, min(a, b), max(a, b), pv[0], pv[1]))
    return out


def _cluster(values: list[float], tol: float) -> tuple[list[float], list[int]]:
    """Single-link clusters of values; representative (median) and cluster of each value."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    assign = [0] * len(values)
    groups: list[list[float]] = []
    prev = None
    for i in order:
        v = values[i]
        if prev is None or v - prev > tol:
            groups.append([])
        groups[-1].append(v)
        assign[i] = len(groups) - 1
        prev = v
    return [median(g) for g in groups], assign


def _least_squares(reps: list[float], cons: list[_Con]) -> tuple[np.ndarray, np.ndarray]:
    """p[j] - p[i] = value for every constraint, with a weak pull to the drawn positions."""
    n = len(reps)
    m = len(cons)
    a = np.zeros((m + n, n))
    rhs = np.zeros(m + n)
    for r, c in enumerate(cons):
        a[r, c.j] = 1.0
        a[r, c.i] = -1.0
        rhs[r] = c.value
    w = 1e-3  # anchors each connected group; the labels decide the spacing
    a[m + np.arange(n), np.arange(n)] = w
    rhs[m:] = w * np.asarray(reps)
    sol, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    res = np.array([sol[c.j] - sol[c.i] - c.value for c in cons])
    return sol, res


def _components(n: int, cons: list[_Con]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in cons:
        parent[find(c.i)] = find(c.j)
    groups: dict[int, list[int]] = defaultdict(list)
    for s in {s for c in cons for s in (c.i, c.j)}:
        groups[find(s)].append(s)
    return list(groups.values())


def _solve_axis(d: Drawing, axis: int, target: float) -> dict:
    """Arbitrate the labels on one axis and solve dimension-true station positions."""
    cons = _constraints(d, axis)
    info: dict = {"tol": _CLUSTER_PAPER * target, "map": None, "constraints": 0, "stations": 0, "flagged": [], "kept": [],
                  "targets": [], "station_of": {}, "new": {}, "shift": []}
    if not cons:
        return info
    reps, assign = _cluster([v for c in cons for v in (c.lo, c.hi)], info["tol"])
    for k, c in enumerate(cons):
        c.i, c.j = assign[2 * k], assign[2 * k + 1]
    gross = _GROSS_PAPER * target
    flagged = []
    keep = []
    for c in cons:
        if abs(c.dev) > gross:
            c.reason = "drawn"
            flagged.append(c)
        elif c.i != c.j:
            keep.append(c)
    max_dec = max((c.dec for c in keep), default=0)
    cycle_tol = 0.5 * 10.0 ** -max_dec
    sol = np.asarray(reps, dtype=float)
    while keep:
        sol, res = _least_squares(reps, keep)
        bad = [k for k in range(len(keep)) if abs(res[k]) > cycle_tol]
        if not bad:
            break
        # The labels around a loop disagree: blame the one that also disagrees most
        # with what is drawn, and leave it for review.
        worst = max(bad, key=lambda k: (abs(keep[k].dev), abs(res[k])))
        c = keep.pop(worst)
        c.reason = "labels"
        flagged.append(c)
    info["flagged"] = flagged
    info["kept"] = keep
    if not keep:
        return info

    new = sol.copy()
    if all(float(c.value).is_integer() for c in keep):
        # Whole-millimetre labels: put stations a whole number of mm apart within each
        # connected group so every accepted label is met exactly, not to 0.3 mm.
        degree: dict[int, int] = defaultdict(int)
        for c in keep:
            degree[c.i] += 1
            degree[c.j] += 1
        for comp in _components(len(reps), keep):
            ref = max(comp, key=lambda s: degree[s])
            for s in comp:
                new[s] = sol[ref] + round(sol[s] - sol[ref])
    used = sorted({s for c in keep for s in (c.i, c.j)}, key=lambda s: reps[s])
    old_pos = [reps[s] for s in used]
    new_pos = [float(new[s]) for s in used]
    for k in range(1, len(new_pos)):
        if new_pos[k] <= new_pos[k - 1]:
            new_pos[k] = new_pos[k - 1] + 1e-6
    info.update(
        map=_piecewise(old_pos, new_pos), constraints=len(keep), stations=len(used),
        targets=new_pos, new={s: p for s, p in zip(used, new_pos)},
        shift=[n - o for o, n in zip(old_pos, new_pos)],
    )
    return info


def _piecewise(old: list[float], new: list[float]):
    def fmap(x: float) -> float:
        if x <= old[0]:
            return new[0] + (x - old[0])
        if x >= old[-1]:
            return new[-1] + (x - old[-1])
        hi = bisect.bisect_right(old, x)
        lo = hi - 1
        t = (x - old[lo]) / (old[hi] - old[lo])
        return new[lo] + t * (new[hi] - new[lo])

    return fmap


# -- apply maps --------------------------------------------------------------------

def _map_drawing(d: Drawing, hx: dict, hy: dict) -> None:
    mx = hx.get("map") or (lambda x: x)
    my = hy.get("map") or (lambda y: y)

    def f(p):
        return (mx(p[0]), my(p[1]))

    def local_k(p) -> float:
        # Geometric mean of local stretch, for radii and similar isotropic sizes.
        dx = abs(mx(p[0] + 1.0) - mx(p[0] - 1.0)) / 2.0
        dy = abs(my(p[1] + 1.0) - my(p[1] - 1.0)) / 2.0
        return math.sqrt(max(dx, 1e-9) * max(dy, 1e-9))

    d.width = mx(d.width) - mx(0.0)
    d.height = my(d.height) - my(0.0)
    d.shapes = [_map_shape(s, f, local_k) for s in d.shapes]
    d.texts = [_map_text(t, f, local_k) for t in d.texts]
    d.linear_dims = [_map_linear(dim, f) for dim in d.linear_dims]
    d.radial_dims = [_map_radial(dim, f, local_k) for dim in d.radial_dims]
    d.leaders = [_map_leader(ld, f) for ld in d.leaders]
    d.hatches = [_map_hatch(h, f, local_k) for h in d.hatches]
    # Block definitions are in local coordinates — only the INSERT points move with the grid.
    d.inserts = [replace(ins, insert=f(ins.insert)) for ins in d.inserts]
    d.axis_bubbles = [_map_bubble(b, f, local_k) for b in d.axis_bubbles]


def _map_seg(g, f, local_k):
    if isinstance(g, Line):
        return Line(f(g.a), f(g.b))
    if isinstance(g, Arc):
        c = f(g.center)
        return Arc(c, g.radius * local_k(g.center), g.start_angle, g.sweep)
    if isinstance(g, Bezier):
        return Bezier(f(g.p0), f(g.p1), f(g.p2), f(g.p3))
    return g


def _map_shape(s: Shape, f, local_k) -> Shape:
    return Shape([_map_seg(g, f, local_k) for g in s.segs], s.closed, s.style, s.stroke, s.fill, s.path_id, s.id)


def _map_text(t: TextItem, f, local_k) -> TextItem:
    k = local_k(t.insert)
    x0, y0, x1, y1 = t.bbox
    corners = [f((x0, y0)), f((x1, y0)), f((x0, y1)), f((x1, y1))]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return replace(
        t, insert=f(t.insert), height=t.height * k, width=t.width * k,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
    )


def _map_linear(dim: LinearDim, f) -> LinearDim:
    p1, p2 = f(dim.p1), f(dim.p2)
    out = replace(dim, p1=p1, p2=p2, base=f(dim.base), text_pos=f(dim.text_pos), measured=dist(p1, p2))
    for name in ("dle", "ext1", "ext2"):
        if hasattr(dim, name):
            setattr(out, name, getattr(dim, name))
    return out


def _map_radial(dim: RadialDim, f, local_k) -> RadialDim:
    c = f(dim.center)
    p = f(dim.point)
    return replace(dim, center=c, point=p, text_pos=f(dim.text_pos), radius=dist(c, p))


def _map_leader(ld: Leader, f) -> Leader:
    return replace(ld, vertices=[f(v) for v in ld.vertices], text_pos=f(ld.text_pos))


def _map_hatch(h: Hatch, f, local_k) -> Hatch:
    loops = [[_map_seg(g, f, local_k) for g in loop] for loop in h.loops]
    fams = []
    for fam in h.families:
        k = local_k(fam.base)
        fams.append(HatchFamily(fam.angle, f(fam.base), fam.spacing * k, fam.shift * k, fam.dashes))
    return replace(h, loops=loops, families=fams)


def _map_bubble(b: AxisBubble, f, local_k) -> AxisBubble:
    k = local_k(b.center)
    slash = tuple((p[0] * k, p[1] * k) for p in b.slash) if b.slash else None
    part = tuple((p[0] * k, p[1] * k) for p in b.part_pos) if b.part_pos else None
    return replace(
        b, center=f(b.center), radius=b.radius * k, text_height=b.text_height * k,
        slash=slash, part_pos=part,
    )


# -- redraw onto the stations ------------------------------------------------------

def _place_dims(d: Drawing, info: dict, axis: int) -> None:
    """Put the ends of every accepted dimension exactly on its solved stations."""
    new = info["new"]
    for c in info["kept"]:
        dim = d.linear_dims[c.dim]
        lo, hi = new[c.i], new[c.j]
        swap = dim.p1[axis] > dim.p2[axis]
        a, b = (hi, lo) if swap else (lo, hi)
        old_a = dim.p1[axis]
        dim.p1 = _with(dim.p1, axis, a)
        dim.p2 = _with(dim.p2, axis, b)
        if abs(dim.base[axis] - old_a) <= info["tol"]:
            dim.base = _with(dim.base, axis, a)


def _with(p, axis: int, v: float):
    return (v, p[1]) if axis == 0 else (p[0], v)


def _snapper(targets: list[float], tol: float):
    """Coordinate -> nearest target within tol, else unchanged."""
    arr = sorted(targets)
    if not arr:
        return None

    def f(v: float) -> float:
        k = bisect.bisect_left(arr, v)
        best, best_d = v, tol
        for kk in (k - 1, k):
            if 0 <= kk < len(arr):
                dd = abs(arr[kk] - v)
                if dd <= best_d:
                    best, best_d = arr[kk], dd
        return best

    return f


def _transform_points(d: Drawing, fx, fy) -> None:
    """Apply per-axis coordinate functions to geometry vertices (not to text)."""
    if fx is None and fy is None:
        return
    fx = fx or (lambda v: v)
    fy = fy or (lambda v: v)

    def p(q):
        return (fx(q[0]), fy(q[1]))

    def seg(g):
        if isinstance(g, Line):
            return Line(p(g.a), p(g.b))
        if isinstance(g, Arc):
            return Arc(p(g.center), g.radius, g.start_angle, g.sweep)
        if isinstance(g, Bezier):
            return Bezier(p(g.p0), p(g.p1), p(g.p2), p(g.p3))
        return g

    for s in d.shapes:
        s.segs = [seg(g) for g in s.segs]
    for h in d.hatches:
        h.loops = [[seg(g) for g in loop] for loop in h.loops]
    for dim in d.linear_dims:
        dim.p1, dim.p2, dim.base = p(dim.p1), p(dim.p2), p(dim.base)
    for ld in d.leaders:
        ld.vertices = [p(v) for v in ld.vertices]
    for ins in d.inserts:
        ins.insert = p(ins.insert)
    for b in d.axis_bubbles:
        b.center = p(b.center)


def _snap_walls(d: Drawing, axis: int, anchors: list[float]) -> dict:
    """Snap parallel faces of the same line style onto whole-10 mm thicknesses.

    axis 0 handles faces of constant X (thickness measured in X). A face on a
    dimensioned station stays put; the other face moves. Faces that would be pulled
    to two different places stay where they are.
    """
    faces: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    exact: dict[tuple, float] = {}
    for s in d.shapes:
        if not s.stroke or s.style.dashes:
            continue
        key = (s.style.layer, s.style.color, round(s.style.width, 4))
        for g in s.segs:
            if not isinstance(g, Line) or abs(g.a[axis] - g.b[axis]) > 0.05:
                continue
            lo, hi = sorted((g.a[1 - axis], g.b[1 - axis]))
            if hi - lo >= _WALL_MIN_LEN:
                c = (g.a[axis] + g.b[axis]) / 2
                faces[(key, round(c, 2))].append((lo, hi))
                exact.setdefault((key, round(c, 2)), c)
    by_key: dict[tuple, list[tuple[float, list]]] = defaultdict(list)
    for fk, spans in faces.items():
        by_key[fk[0]].append((exact[fk], spans))
    anchors = sorted(anchors)

    def is_anchor(c: float) -> bool:
        k = bisect.bisect_left(anchors, c - 0.01)
        return k < len(anchors) and abs(anchors[k] - c) <= 0.01

    proposals: dict[float, list[float]] = defaultdict(list)
    refs: set[float] = set()
    pairs = 0
    for key, lst in by_key.items():
        lst.sort(key=lambda t: t[0])
        for n, (c1, s1) in enumerate(lst):
            for c2, s2 in lst[n + 1:]:
                gap = c2 - c1
                if gap > _WALL_GAP[1]:
                    break
                if gap < _WALL_GAP[0] or _overlap(s1, s2) < _WALL_MIN_LEN:
                    continue
                pairs += 1
                module = round(gap / _WALL_MODULE) * _WALL_MODULE
                err = module - gap
                if abs(err) > _WALL_TOL or abs(err) < 1e-3:
                    break
                a1, a2 = is_anchor(c1), is_anchor(c2)
                if a1 and a2:
                    break
                if a1 or (not a2 and _length(s1) >= _length(s2)):
                    proposals[c2].append(c1 + module)
                    refs.add(c1)
                else:
                    proposals[c1].append(c2 - module)
                    refs.add(c2)
                break  # only the nearest overlapping face forms a wall with c1
    moves = {}
    for c, tgts in proposals.items():
        if c in refs or max(tgts) - min(tgts) > 0.05:
            continue
        moves[c] = sum(tgts) / len(tgts)
    if moves:
        keys = sorted(moves)

        def f(v: float) -> float:
            k = bisect.bisect_left(keys, v - 0.05)
            if k < len(keys) and abs(keys[k] - v) <= 0.05:
                return moves[keys[k]]
            return v

        _transform_points(d, f if axis == 0 else None, f if axis == 1 else None)
    return {"pairs": pairs, "snapped": len(moves)}


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(h1, h2) - max(l1, l2)) for l1, h1 in a for l2, h2 in b)


def _length(spans: list[tuple[float, float]]) -> float:
    return sum(h - l for l, h in spans)


# -- report ------------------------------------------------------------------------

def _axis_name(d: Drawing, axis: int, coord: float, tol: float) -> str | None:
    best = None
    for b in d.axis_bubbles:
        dd = abs(b.center[axis] - coord)
        if dd <= tol and (best is None or dd < best[0]):
            best = (dd, b.label)
    return best[1] if best else None


def _describe_conflicts(d: Drawing, info: dict, axis: int) -> list[dict]:
    out = []
    tol = 2.0 * info["tol"]
    for c in info["flagged"]:
        dim = d.linear_dims[c.dim]
        a, b = sorted((dim.p1[axis], dim.p2[axis]))
        out.append({
            "axis": "xy"[axis],
            "label": c.value,
            "text": dim.text,
            "drawn": round(b - a, 1),
            "deviation": round(c.value - (b - a), 1),
            "reason": c.reason,
            "from": _axis_name(d, axis, a, tol),
            "to": _axis_name(d, axis, b, tol),
            "at": (round(a), round(b)),
        })
    return out


def _residuals(d: Drawing, hx: dict, hy: dict) -> dict:
    devs = []
    compromise = []
    for info, axis in ((hx, 0), (hy, 1)):
        for c in info["kept"]:
            dim = d.linear_dims[c.dim]
            got = abs(dim.p2[axis] - dim.p1[axis])
            devs.append(abs(c.value - got))
            if devs[-1] >= 0.005:
                compromise.append({"axis": "xy"[axis], "label": c.value, "drawn": round(got, 2)})
    shift = [abs(v) for v in hx["shift"] + hy["shift"]]
    return {
        "dims_checked": len(devs),
        "dims_exact": sum(1 for v in devs if v < 0.005),
        "dim_max": round(max(devs), 3) if devs else 0.0,
        # Labels that disagree by under a millimetre around a loop (e.g. 3 × 2167 = 6500).
        "compromise": compromise,
        "station_shift_max": round(max(shift), 2) if shift else 0.0,
        "station_shift_p95": round(float(np.percentile(shift, 95)), 2) if shift else 0.0,
    }
