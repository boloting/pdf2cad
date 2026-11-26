"""Snap a fit-to-page PDF to its labeled scale and true dimension lengths.

PDF plots printed with “fit to page” come out at a non-standard scale (e.g. 1:104)
and individual lengths are off by about a millimetre. The title block still says
1:100 and the numbers on the dimensions are the design sizes. This module:

1. Moves geometry into real millimetres with the detected fit scale.
2. Treats orthogonal linear dimensions as distance constraints and solves for
   corrected X and Y coordinates (least squares on clustered grid lines).
3. Writes coordinates back in paper millimetres at the intended scale (1:100),
   so AutoCAD’s DIM reads the same numbers that are printed on the drawing.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from statistics import median

import numpy as np

from .dims import parse_value
from .geom import angle_diff, dist, line_angle
from .model import (
    COMMON_SCALES,
    Arc,
    AxisBubble,
    Bezier,
    Drawing,
    Hatch,
    HatchFamily,
    Insert,
    Leader,
    Line,
    LinearDim,
    RadialDim,
    Shape,
    Style,
    TextItem,
)

# How far two real-mm coordinates may drift and still be the same grid line.
_CLUSTER_TOL = 0.8
# A dimension is horizontal/vertical if its angle is within this of 0°/90°.
_AXIS_TOL = 4.0
# Fit-to-page scales within this fraction of a standard scale snap to that standard.
_FIT_TOL = 0.08


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

    Returns counts for the conversion warning.
    """
    if fit_scale <= 0 or target_scale <= 0:
        return {"horiz": 0, "vert": 0, "clusters_x": 0, "clusters_y": 0}
    _scale_drawing(d, fit_scale)
    hx, hy = _solve_axes(d)
    _map_drawing(d, hx, hy)
    _scale_drawing(d, 1.0 / target_scale)
    for dim in d.linear_dims:
        dim.measured = dist(dim.p1, dim.p2)
    for dim in d.radial_dims:
        dim.radius = dist(dim.center, dim.point) if dim.point else dim.radius
    return {
        "horiz": hx["constraints"],
        "vert": hy["constraints"],
        "clusters_x": hx["clusters"],
        "clusters_y": hy["clusters"],
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


# -- orthogonal solve --------------------------------------------------------------

def _solve_axes(d: Drawing) -> tuple[dict, dict]:
    horiz, vert = [], []
    for dim in d.linear_dims:
        pv = parse_value(dim.text)
        if not pv or pv[0] <= 0:
            continue
        value = pv[0]
        ang = dim.angle % 180
        if angle_diff(ang, 0.0) <= _AXIS_TOL:
            horiz.append((dim.p1[0], dim.p2[0], value, (dim.p1[1] + dim.p2[1]) / 2))
        elif angle_diff(ang, 90.0) <= _AXIS_TOL:
            vert.append((dim.p1[1], dim.p2[1], value, (dim.p1[0] + dim.p2[0]) / 2))
    return _solve_1d(horiz), _solve_1d(vert)


def _cluster(values: list[float], tol: float) -> tuple[list[float], list[int]]:
    """Cluster sorted values; return representative positions and index of each value."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    reps: list[float] = []
    groups: list[list[float]] = []
    assign = [0] * len(values)
    for i in order:
        v = values[i]
        if not groups or v - groups[-1][-1] > tol:
            groups.append([v])
            reps.append(v)
        else:
            groups[-1].append(v)
            reps[-1] = median(groups[-1])
        assign[i] = len(groups) - 1
    # Recompute reps as medians after assignment is stable.
    buckets: list[list[float]] = [[] for _ in reps]
    for i, v in enumerate(values):
        buckets[assign[i]].append(v)
    reps = [median(b) for b in buckets]
    return reps, assign


def _solve_1d(constraints_raw: list[tuple[float, float, float, float]]) -> dict:
    """constraints_raw: (a, b, value, along) in real mm. Returns map info."""
    if len(constraints_raw) < 1:
        return {"map": None, "constraints": 0, "clusters": 0}
    values = []
    edges = []  # (idx_a, idx_b will be filled after clustering)
    for a, b, value, _along in constraints_raw:
        values.append(a)
        values.append(b)
    reps, assign = _cluster(values, _CLUSTER_TOL)
    n = len(reps)
    if n < 2:
        return {"map": None, "constraints": 0, "clusters": n}
    # Build signed constraints between clusters.
    cons: list[tuple[int, int, float]] = []
    for i, (a, b, value, _along) in enumerate(constraints_raw):
        ia, ib = assign[2 * i], assign[2 * i + 1]
        if ia == ib:
            continue
        if reps[ib] >= reps[ia]:
            cons.append((ia, ib, value))
        else:
            cons.append((ib, ia, value))
    if len(cons) < 1:
        return {"map": None, "constraints": 0, "clusters": n}

    # Least squares: for each (i,j,v), p[j]-p[i] = v, plus weak pull to original reps.
    rows = []
    rhs = []
    for i, j, v in cons:
        row = np.zeros(n)
        row[j] = 1.0
        row[i] = -1.0
        rows.append(row)
        rhs.append(v)
    w = 0.005  # keep the grid from drifting; dimensions dominate.
    for i, r in enumerate(reps):
        row = np.zeros(n)
        row[i] = w
        rows.append(row)
        rhs.append(w * r)
    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=float)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    old = list(reps)
    new = list(sol)
    # Keep order: enforce monotonicity by sorting pairs together if a crossing appears.
    pairs = sorted(zip(old, new), key=lambda t: t[0])
    old = [p[0] for p in pairs]
    new = [p[1] for p in pairs]
    for i in range(1, len(new)):
        if new[i] <= new[i - 1]:
            new[i] = new[i - 1] + 1e-6

    def fmap(x: float) -> float:
        if x <= old[0]:
            return new[0] + (x - old[0])
        if x >= old[-1]:
            return new[-1] + (x - old[-1])
        # Piecewise linear between cluster representatives.
        lo, hi = 0, len(old) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if old[mid] <= x:
                lo = mid
            else:
                hi = mid
        t = 0.0 if old[hi] == old[lo] else (x - old[lo]) / (old[hi] - old[lo])
        return new[lo] + t * (new[hi] - new[lo])

    return {"map": fmap, "constraints": len(cons), "clusters": n, "old": old, "new": new}


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
