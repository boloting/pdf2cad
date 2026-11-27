"""Rebuild dimensions and leaders from arrowheads, lines and nearby text."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from . import stroketext
from .geom import (
    Vec, add, angle_diff, angle_of, cross, dist, dot, line_angle, mid, mul, point_line_distance,
    point_segment_distance, project_param, rotate, sub, unit,
)
from .model import COMMON_SCALES, Arc, Arrow, Drawing, Leader, Line, LinearDim, RadialDim, Shape, TextItem
from .spatial import Grid

DIAMETER_MARKS = ("Ø", "Φ", "φ", "%%c", "%%C", "⌀")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
TIP_TOL = 0.15


MULTIPLIER = re.compile(r"^\s*\d+\s*[×xX*]\s*")


def parse_value(text: str) -> tuple[float, int, int, int] | None:
    """Measured value in a dimension string: (value, decimals, start, end).

    A quantity prefix such as the "4×" in "4×Φ12" is skipped.
    """
    m0 = MULTIPLIER.match(text)
    m = NUMBER.search(text, m0.end() if m0 else 0)
    if not m:
        return None
    s = m.group(0).replace(",", ".")
    return float(s), (len(s.split(".")[1]) if "." in s else 0), m.start(), m.end()


def text_anchor(t: TextItem) -> Vec:
    """Centre of the text's cap-height box, which is where CAD puts a dimension's text midpoint."""
    w = t.width if t.width > 0 else (t.bbox[2] - t.bbox[0])
    return add(t.insert, rotate((w / 2, t.height / 2), t.rotation))


def _part_anchor(t: TextItem, start: int, end: int) -> Vec:
    """Centre of characters start..end of a text, assuming even character widths."""
    n = max(1, len(t.text))
    if start == 0 and end >= n:
        return text_anchor(t)
    w = t.width if t.width > 0 else (t.bbox[2] - t.bbox[0])
    return add(t.insert, rotate((w * (start + end) / (2 * n), t.height / 2), t.rotation))


def _shrink(t: TextItem, used: list[tuple[int, int]]) -> None:
    """Drop the characters that went into dimensions, keeping the rest in place."""
    n = len(t.text)
    keep = [i for i in range(n) if not any(s <= i < e for s, e in used)]
    if not keep:
        t.text = ""
        return
    if not any(t.text[i].isalnum() for i in keep):
        t.text = ""
        return
    first, last = keep[0], keep[-1] + 1
    w = t.width if t.width > 0 else (t.bbox[2] - t.bbox[0])
    t.insert = add(t.insert, rotate((w * first / n, 0.0), t.rotation))
    t.width = w * (last - first) / n
    t.text = t.text[first:last]


def _style_match(a: Shape, b: Shape) -> bool:
    return a.style.layer == b.style.layer and a.style.color == b.style.color


def _vertices(s: Shape) -> list[Vec]:
    pts = [g.start() for g in s.segs]
    out: list[Vec] = []
    for p in pts:
        if not out or dist(out[-1], p) > 1e-3:
            out.append(p)
    if len(out) > 1 and dist(out[0], out[-1]) < 1e-3:
        out.pop()
    # drop collinear vertices
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            if abs(cross(unit(sub(b, a)), unit(sub(c, b)))) < 1e-3:
                out.pop(i)
                changed = True
                break
    return out


def _corner_angle(a: Vec, b: Vec, c: Vec) -> float:
    u, v = unit(sub(a, b)), unit(sub(c, b))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(u, v)))))


def detect_arrows(shapes: list[Shape], consumed: set[int], max_size: float) -> list[Arrow]:
    arrows: list[Arrow] = []
    for s in shapes:
        if s.id in consumed:
            continue
        if s.is_circle and s.fill is not None and s.segs[0].radius <= min(1.2, max_size / 3):  # type: ignore[union-attr]
            arc = s.segs[0]
            arrows.append(Arrow(arc.center, (0.0, 0.0), arc.radius * 2, "dot", s))  # type: ignore[union-attr]
            continue
        if not all(isinstance(g, Line) for g in s.segs):
            continue
        if s.closed and len(s.segs) <= 5:
            pts = _vertices(s)
            if len(pts) == 4:
                # Notched arrowhead: drop the reflex tail vertex.
                area_sign = sum(cross(pts[i], pts[(i + 1) % 4]) for i in range(4))
                for i in range(4):
                    a, b, c = pts[i - 1], pts[i], pts[(i + 1) % 4]
                    if cross(sub(b, a), sub(c, b)) * area_sign < 0:
                        pts = pts[:i] + pts[i + 1:]
                        break
            if len(pts) != 3:
                continue
            angles = [_corner_angle(pts[i - 1], pts[i], pts[(i + 1) % 3]) for i in range(3)]
            k = min(range(3), key=lambda i: angles[i])
            if angles[k] > 60:
                continue
            tip, o1, o2 = pts[k], pts[k - 1], pts[(k + 1) % 3]
            l1, l2 = dist(tip, o1), dist(tip, o2)
            if min(l1, l2) / max(l1, l2) < 0.8:
                continue
            base = mid(o1, o2)
            size = dist(tip, base)
            if not (0.3 <= size <= max_size):
                continue
            kind = "closed" if s.fill is not None else "closed_blank"
            arrows.append(Arrow(tip, unit(sub(tip, base)), size, kind, s))
        elif not s.closed and len(s.segs) == 2:
            a, tip, c = s.segs[0].start(), s.segs[0].end(), s.segs[1].end()
            l1, l2 = dist(a, tip), dist(tip, c)
            if not (0.3 <= max(l1, l2) <= max_size) or min(l1, l2) / max(l1, l2) < 0.8:
                continue
            if _corner_angle(a, tip, c) > 60:
                continue
            base = mid(a, c)
            arrows.append(Arrow(tip, unit(sub(tip, base)), dist(tip, base), "open", s))

    # A filled arrowhead is often drawn twice (fill + outline); keep one.
    merged: list[Arrow] = []
    extra: dict[int, list[Shape]] = {}
    for a in arrows:
        for m in merged:
            if dist(a.tip, m.tip) < 0.05 and dot(a.direction, m.direction) > 0.99:
                extra.setdefault(id(m), []).append(a.shape)
                if m.kind == "closed_blank" and a.kind == "closed":
                    m.kind = "closed"
                break
        else:
            merged.append(a)
    for m in merged:
        m.extra_shapes = extra.get(id(m), [])  # type: ignore[attr-defined]
    return merged


def _stroke_axis(s: Shape) -> tuple[Vec, Vec] | None:
    """A line, or a thin closed polygon (a thick stroke exported as an outline), as a segment."""
    if s.is_single_line:
        return s.segs[0].start(), s.segs[0].end()
    if not s.closed or len(s.segs) > 6 or not all(isinstance(g, Line) for g in s.segs):
        return None
    b = s.bbox()
    if max(b[2] - b[0], b[3] - b[1]) > 6.0:
        return None
    pts = [g.start() for g in s.segs]
    p, q = max(((p, q) for i, p in enumerate(pts) for q in pts[i + 1:]), key=lambda pq: dist(*pq))
    ln = dist(p, q)
    if ln <= 0:
        return None
    area = abs(sum(cross(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))) / 2
    if area / ln > min(0.6, 0.2 * ln):
        return None
    return p, q


@dataclass
class _Cand:
    line: Shape
    a: Vec
    b: Vec
    ends: tuple[Arrow, Arrow]
    dle: float


def _arrow_shapes(a: Arrow) -> list[Shape]:
    return [a.shape, *getattr(a, "extra_shapes", [])]


def _span_key(a: Vec, b: Vec) -> tuple:
    return tuple(sorted(((round(a[0], 1), round(a[1], 1)), (round(b[0], 1), round(b[1], 1)))))


def _value_options(text: str, expected: float | None) -> list[tuple[str, int, int]]:
    """Ways to read a dimension value from text: (value text, start, end) within it.

    Without an expected value any number goes.  With one, the number must agree with it,
    and a run of digits that two crowded dimensions were read into may be split in two.
    """
    pv = parse_value(text)
    if pv is None:
        return []
    if expected is None:
        return [(text, 0, len(text))]
    # Drawings often carry rounded or overridden values, so only reject clear mismatches.
    tol = max(1.0, expected * 0.1)
    if len(re.findall(r"\d+(?:\.\d+)?", text)) == 1 and abs(pv[0] - expected) <= tol:
        return [(text, 0, len(text))]
    out = []
    for m in re.finditer(r"\d+", text):
        if m.start() > 0 or m.end() < len(text):
            if abs(float(m.group(0)) - expected) <= tol:
                out.append((m.group(0), m.start(), m.end()))
    if out:
        return out
    s = text.strip()
    if not re.fullmatch(r"\d{3,}", s):
        return []
    for k in range(2, len(s) - 1):
        for st, en in ((0, k), (k, len(s))):
            part = s[st:en]
            if part[0] != "0" and abs(float(part) - expected) <= tol:
                out.append((part, st, en))
    return out


def infer_scale(d: Drawing, consumed: set[int], max_arrow: float = 8.0) -> float | None:
    """Drawing scale from numeric labels next to tick/arrow dimension lines.

    Used after OCR: stroke-glyph dims can lock onto a wrong scale when every short
    module misreads as the same digits; label lengths on the real overall chains
    are a stronger vote.
    """
    shapes = d.shapes
    arrows = detect_arrows(shapes, consumed, max_arrow)
    arrow_grid: Grid[int] = Grid(2.0)
    for i, a in enumerate(arrows):
        arrow_grid.insert_point(a.tip, i)
    lines = [s for s in shapes if s.id not in consumed and s.is_single_line and s.stroke]
    ticks: list[tuple[Vec, float, float, Shape]] = []
    tick_grid: Grid[int] = Grid(3.0)
    for s in shapes:
        if s.id in consumed:
            continue
        ax = _stroke_axis(s)
        if ax is None:
            continue
        p, q = ax
        ln = dist(p, q)
        if 0.5 <= ln <= 6.0:
            m = mid(p, q)
            tick_grid.insert_point(m, len(ticks))
            ticks.append((m, line_angle(p, q), ln, s))

    def ticks_on(line: Shape, a: Vec, b: Vec) -> list[tuple[float, float]]:
        hits = []
        la = line_angle(a, b)
        for i in tick_grid.query((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])), 0.2):
            m, ang, ln, t = ticks[i]
            if t.id == line.id:
                continue
            if abs(angle_diff(ang, la) - 45) > 12:
                continue
            if point_line_distance(m, a, b) < 0.15:
                p = project_param(m, a, b)
                if -0.01 <= p <= 1.01:
                    hits.append((p, ln))
        hits.sort(key=lambda x: x[0])
        merged: list[tuple[float, float]] = []
        total = dist(a, b)
        for p, ln in hits:
            if merged and (p - merged[-1][0]) * total < 0.3:
                merged[-1] = (merged[-1][0], max(merged[-1][1], ln))
            else:
                merged.append((p, ln))
        return merged

    spans: list[tuple[Vec, Vec]] = []
    for s in lines:
        g = s.segs[0]
        a, b = g.start(), g.end()
        if dist(a, b) < 0.8:
            continue
        u = unit(sub(b, a))
        tip_a = tip_b = None
        for i in arrow_grid.query_point(a, TIP_TOL):
            ar = arrows[i]
            if dist(ar.tip, a) <= TIP_TOL and (ar.kind == "dot" or abs(cross(ar.direction, u)) < 0.1):
                tip_a = a
                break
        for i in arrow_grid.query_point(b, TIP_TOL):
            ar = arrows[i]
            if dist(ar.tip, b) <= TIP_TOL and (ar.kind == "dot" or abs(cross(ar.direction, u)) < 0.1):
                tip_b = b
                break
        if tip_a is not None and tip_b is not None:
            spans.append((a, b))
            continue
        tk = ticks_on(s, a, b)
        if len(tk) < 2:
            continue
        for k in range(len(tk) - 1):
            pa = add(a, mul(sub(b, a), tk[k][0]))
            pb = add(a, mul(sub(b, a), tk[k + 1][0]))
            if dist(pa, pb) >= 0.8:
                spans.append((pa, pb))

    texts = {t.id: t for t in d.texts}
    text_grid: Grid[int] = Grid(5.0)
    for t in d.texts:
        text_grid.insert(t.bbox, t.id)
    ratios: list[float] = []
    for a, b in spans:
        ln = dist(a, b)
        if ln < 0.8:
            continue
        la = line_angle(a, b)
        box = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        for tid in text_grid.query(box, 12.0):
            t = texts.get(tid)
            if t is None:
                continue
            pv = parse_value(t.text)
            if pv is None or pv[0] < 100:
                continue
            if angle_diff(t.rotation % 180, la) > 2.0:
                continue
            ctr = text_anchor(t)
            pd = point_line_distance(ctr, a, b)
            if pd > 1.6 * t.height + 1.0:
                continue
            p = project_param(ctr, a, b)
            if p < -0.15 or p > 1.15:
                continue
            ratios.append(pv[0] / ln)
    if len(ratios) < 3:
        return None
    # Prefer a standard scale that many label/line pairs agree with.
    best_c, best_n = None, 0
    for c in COMMON_SCALES:
        n = sum(1 for r in ratios if abs(r / c - 1) <= 0.08)
        if n > best_n:
            best_c, best_n = c, n
    if best_c is None or best_n < max(3, 0.4 * len(ratios)):
        return None
    return float(best_c)


def prune_dims_disagreeing(d: Drawing, scale: float, tol: float = 0.12) -> int:
    """Drop linear dims whose label/length ratio fights `scale`; clear spans if most are bad."""
    if scale <= 0 or not d.linear_dims:
        return 0
    spans = getattr(d, "dim_spans", None)
    keep: list[LinearDim] = []
    dropped = 0
    for dim in d.linear_dims:
        pv = parse_value(dim.text)
        ok = (
            pv is not None and dim.measured > 0.5
            and abs(pv[0] / dim.measured / scale - 1) <= tol
        )
        if ok:
            keep.append(dim)
        else:
            dropped += 1
    if not dropped:
        return 0
    if spans is not None and dropped >= max(1, 0.5 * (dropped + len(keep))):
        # Wrong-scale glyph dims poisoned the span set; allow a full rebuild.
        spans.clear()
        dropped += len(keep)
        d.linear_dims = []
        if hasattr(d, "glyph_dims"):
            d.glyph_dims = 0  # type: ignore[attr-defined]
        return dropped
    d.linear_dims = keep
    return dropped


def recognize(d: Drawing, consumed: set[int], used_texts: set[int], max_arrow: float = 8.0,
              scale: float | None = None, again: bool = False) -> None:
    """Rebuild dimensions.  With again=True, a later pass over texts found since (OCR): spans
    already dimensioned are skipped and, given the drawing scale, values must fit the geometry."""
    shapes = d.shapes
    if not hasattr(d, "dim_spans"):
        d.dim_spans = set()  # type: ignore[attr-defined]
        d.dim_ticks = set()  # type: ignore[attr-defined]
    spans: set[tuple] = d.dim_spans  # type: ignore[attr-defined]
    tick_ids: set[int] = d.dim_ticks  # type: ignore[attr-defined]

    def gone(sid: int) -> bool:
        return sid in consumed and not (again and sid in tick_ids)

    arrows = detect_arrows(shapes, consumed, max_arrow)
    used_arrows: set[int] = set()

    arrow_grid: Grid[int] = Grid(2.0)
    for i, a in enumerate(arrows):
        arrow_grid.insert_point(a.tip, i)

    lines = [s for s in shapes if s.id not in consumed and s.is_single_line and s.stroke]
    line_grid: Grid[int] = Grid(5.0)
    by_id = {s.id: s for s in shapes}
    for s in lines:
        line_grid.insert(s.bbox(), s.id)

    text_grid: Grid[int] = Grid(5.0)
    for t in d.texts:
        if t.id not in used_texts:
            text_grid.insert(t.bbox, t.id)
    texts = {t.id: t for t in d.texts}

    def arrow_at(p: Vec, u: Vec) -> int | None:
        best, best_d = None, TIP_TOL
        for i in arrow_grid.query_point(p, TIP_TOL):
            a = arrows[i]
            if i in used_arrows:
                continue
            dd = dist(a.tip, p)
            if dd <= best_d and (a.kind == "dot" or abs(cross(a.direction, u)) < 0.1):
                best, best_d = i, dd
        return best

    ticks: list[tuple[Vec, float, float, Shape]] = []
    tick_grid: Grid[int] = Grid(3.0)
    for s in shapes:
        if gone(s.id):
            continue
        ax = _stroke_axis(s)
        if ax is None:
            continue
        p, q = ax
        ln = dist(p, q)
        if 0.5 <= ln <= 6.0:
            m = mid(p, q)
            tick_grid.insert_point(m, len(ticks))
            ticks.append((m, line_angle(p, q), ln, s))

    def ticks_on(line: Shape, a: Vec, b: Vec) -> list[tuple[float, float, list[Shape]]]:
        """Tick marks centred on the line, merged per position: (param, size, shapes)."""
        hits = []
        la = line_angle(a, b)
        for i in tick_grid.query((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])), 0.2):
            m, ang, ln, t = ticks[i]
            if t.id == line.id or gone(t.id):
                continue
            if abs(angle_diff(ang, la) - 45) > 12:
                continue
            if point_line_distance(m, a, b) < 0.15:
                p = project_param(m, a, b)
                if -0.01 <= p <= 1.01:
                    hits.append((p, ln, t))
        hits.sort(key=lambda x: x[0])
        merged: list[tuple[float, float, list[Shape]]] = []
        total = dist(a, b)
        for p, ln, t in hits:
            if merged and (p - merged[-1][0]) * total < 0.3:
                p0, l0, ts = merged[-1]
                merged[-1] = (p0, max(l0, ln), ts + [t])
            else:
                merged.append((p, ln, [t]))
        return merged

    # 1. Candidate dimension lines: a straight line with an arrowhead or tick at each end.
    # A row of chained dimensions is often a single line with a tick at every station.
    cands: list[_Cand] = []
    pairs_on_line: dict[int, int] = {}
    seen_spans: set[tuple] = set()
    for s in lines:
        g = s.segs[0]
        a, b = g.start(), g.end()
        ln = dist(a, b)
        if ln < 0.8:
            continue
        u = unit(sub(b, a))
        ia, ib = arrow_at(a, u), arrow_at(b, u)
        if ia is not None and ib is not None and ia != ib:
            if _span_key(a, b) in spans:
                continue
            cands.append(_Cand(s, a, b, (arrows[ia], arrows[ib]), 0.0))
            pairs_on_line[s.id] = 1
            continue
        tk = ticks_on(s, a, b)
        if len(tk) < 2:
            continue
        n_pairs = 0
        for k in range(len(tk) - 1):
            (p0, l0, t0), (p1, l1, t1) = tk[k], tk[k + 1]
            pa = add(a, mul(sub(b, a), p0))
            pb = add(a, mul(sub(b, a), p1))
            if dist(pa, pb) < 0.8:
                continue
            key = _span_key(pa, pb)
            if key in seen_spans or key in spans:
                continue
            seen_spans.add(key)
            a0 = Arrow(pa, mul(u, -1), l0 / math.sqrt(2), "tick", t0[0])
            a1 = Arrow(pb, u, l1 / math.sqrt(2), "tick", t1[0])
            a0.extra_shapes = t0[1:]  # type: ignore[attr-defined]
            a1.extra_shapes = t1[1:]  # type: ignore[attr-defined]
            dle = 0.0
            if len(tk) == 2:
                dle = min(dist(a, pa), dist(b, pb), 3 * max(a0.size, a1.size))
            cands.append(_Cand(s, pa, pb, (a0, a1), dle))
            n_pairs += 1
        if n_pairs:
            pairs_on_line[s.id] = len(tk) - 1

    # 2. Assign texts to candidate lines, nearest pairs first.
    pairs: list[tuple[float, int, int, tuple[str, int, int]]] = []
    for ci, c in enumerate(cands):
        expected = dist(c.a, c.b) * scale if scale else None
        la = line_angle(c.a, c.b)
        ln = dist(c.a, c.b)
        box = (min(c.a[0], c.b[0]), min(c.a[1], c.b[1]), max(c.a[0], c.b[0]), max(c.a[1], c.b[1]))
        for tid in text_grid.query(box, 12.0):
            t = texts[tid]
            if tid in used_texts:
                continue
            options = _value_options(t.text, expected)
            if not options:
                continue
            parallel = angle_diff(t.rotation % 180, la) < 2.0
            if not parallel and angle_diff(t.rotation, 0, 360) > 2.0:
                continue
            for opt in options:
                ctr = _part_anchor(t, opt[1], opt[2])
                pd = point_line_distance(ctr, c.a, c.b)
                limit = 1.6 * t.height + 1.0 if parallel else 2.5 * t.height + 1.0
                if pd > limit:
                    continue
                p = project_param(ctr, c.a, c.b)
                if p < -0.2 - 3 * t.height / ln or p > 1.2 + 3 * t.height / ln:
                    continue
                outside = max(0.0, -p, p - 1.0) * ln
                pairs.append((pd + outside + abs(p - 0.5) * 0.05, ci, tid, opt))
    pairs.sort(key=lambda x: x[0])
    assigned: dict[int, tuple[int, tuple[str, int, int]]] = {}
    parts: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for _score, ci, tid, opt in pairs:
        if ci in assigned or any(opt[1] < e and s0 < opt[2] for s0, e in parts[tid]):
            continue
        assigned[ci] = (tid, opt)
        parts[tid].append((opt[1], opt[2]))

    # 3. Extension lines and the DIMENSION itself.
    def extension(c: _Cand, p: Vec) -> tuple[Vec, float, Shape] | None:
        la = line_angle(c.a, c.b)
        best = None
        for sid in line_grid.query_point(p, TIP_TOL):
            e = by_id[sid]
            if sid == c.line.id or not _style_match(e, c.line) or e.style.dashes:
                continue  # a dash-dot axis is the reference, not the extension line
            g = e.segs[0]
            ea, eb = g.start(), g.end()
            if abs(angle_diff(line_angle(ea, eb), la) - 90) > 1.0:
                continue
            if point_segment_distance(p, ea, eb) > TIP_TOL:
                # the dim line may stop short of an extension line that ends exactly at it
                continue
            da = point_line_distance(ea, c.a, c.b)
            db = point_line_distance(eb, c.a, c.b)
            far, near_d = (ea, db) if da > db else (eb, da)
            far_d = max(da, db)
            if far_d < 0.5:
                continue
            # Overshoot past the dimension line: positive when the near end crosses it.
            near = eb if far is ea else ea
            side_far = cross(sub(c.b, c.a), sub(far, c.a))
            side_near = cross(sub(c.b, c.a), sub(near, c.a))
            exe = near_d if side_far * side_near < 0 else 0.0
            if exe > 6.0:
                continue
            if best is None or far_d > best[0]:
                best = (far_d, far, exe, e)
        return None if best is None else (best[1], best[2], best[3])

    emitted: dict[int, int] = defaultdict(int)

    def emit(c: _Cand, text: str, text_pos: Vec, text_height: float, text_rotation: float) -> None:
        a, b = c.a, c.b
        ang = angle_of(sub(b, a))
        if ang >= 180 - 1e-6:
            a, b = b, a
            ends = (c.ends[1], c.ends[0])
            ang = (ang - 180) % 360
        else:
            ends = c.ends
        e1, e2 = extension(c, a), extension(c, b)
        p1 = e1[0] if e1 else a
        p2 = e2[0] if e2 else b
        exe = max([e[1] for e in (e1, e2) if e] or [0.0])
        for e in (e1, e2):
            if e:
                consumed.add(e[2].id)
        dim = LinearDim(
            p1=p1, p2=p2, base=a, angle=round(ang, 6), text=text, text_pos=text_pos,
            text_height=text_height, text_rotation=text_rotation,
            arrow_size=(ends[0].size + ends[1].size) / 2,
            arrow_kind=ends[0].kind, exe=exe, measured=abs(dot(sub(p2, p1), unit(sub(b, a)))),
            layer=c.line.style.layer,
            color=c.line.style.color,
        )
        dim.ext1 = e1 is not None  # type: ignore[attr-defined]
        dim.ext2 = e2 is not None  # type: ignore[attr-defined]
        dim.dle = c.dle  # type: ignore[attr-defined]
        d.linear_dims.append(dim)
        spans.add(_span_key(c.a, c.b))
        emitted[c.line.id] += 1
        for arr in ends:
            for sh in _arrow_shapes(arr):
                consumed.add(sh.id)
                tick_ids.add(sh.id)
            for i, x in enumerate(arrows):
                if x is arr:
                    used_arrows.add(i)

    for ci, c in enumerate(cands):
        if ci in assigned:
            tid, (val, st, en) = assigned[ci]
            t = texts[tid]
            emit(c, val, _part_anchor(t, st, en), t.height, t.rotation)
            if again:
                d.ocr_dims = getattr(d, "ocr_dims", 0) + 1  # type: ignore[attr-defined]
    for tid, used in parts.items():
        if any(s == 0 and e >= len(texts[tid].text) for s, e in used):
            used_texts.add(tid)
        elif used:
            _shrink(texts[tid], used)
            if not texts[tid].text:
                used_texts.add(tid)

    if again:
        for line_id, n in emitted.items():
            if n >= pairs_on_line.get(line_id, 1):
                consumed.add(line_id)
        _radial_and_leaders(d, arrows, used_arrows, consumed, used_texts, by_id, text_grid, texts)
        return

    # 4. Dimensions whose numbers are outlines/strokes rather than text objects.
    grid, pool = stroketext.glyph_pool(shapes, consumed)
    taken: set[int] = set()
    found: list[tuple[float, int, stroketext.Run]] = []
    pending = [ci for ci in range(len(cands)) if ci not in assigned]

    def frame(c: _Cand) -> tuple[Vec, Vec, Vec, Vec]:
        a, b = c.a, c.b
        u = unit(sub(b, a))
        if u[0] < -1e-9 or (abs(u[0]) <= 1e-9 and u[1] < 0):
            a, b, u = b, a, mul(u, -1)
        return a, b, u, (-u[1], u[0])

    for _round in range(2):
        scored = []
        for ci in pending:
            a, b, u, n = frame(cands[ci])
            r = stroketext.find_run(grid, pool, taken, a, b, u, n, dist(a, b))
            if r:
                scored.append((r[0], ci, r[1]))
        scored.sort(key=lambda x: x[0])
        left = []
        for score, ci, run in scored:
            ids = {g.id for g in run.glyphs}
            if ids & taken:
                left.append(ci)
                continue
            taken |= ids
            found.append((score, ci, run))
        pending = left

    if found:
        classes = stroketext.GlyphClasses()
        samples = []
        for _s, ci, run in found:
            cls = [classes.classify(desc) for desc in run.descriptors]
            run.classes = cls  # type: ignore[attr-defined]
            samples.append((dist(cands[ci].a, cands[ci].b), cls))
        learned = stroketext.learn(samples, COMMON_SCALES)
        if learned:
            scale, table = learned
            d.glyph_scale = scale  # type: ignore[attr-defined]
            d.glyph_dims = 0  # type: ignore[attr-defined]
            done: set[int] = set()
            for _s, ci, run in found:
                chars = [table.get(cl) for cl in run.classes]  # type: ignore[attr-defined]
                if None in chars:
                    continue
                text = "".join(chars)  # type: ignore[arg-type]
                measured = dist(cands[ci].a, cands[ci].b) * scale
                if abs(float(text) - measured) > max(1.0, measured * 0.02):
                    continue
                emit(cands[ci], text, run.center(), run.height, angle_of(run.u))
                consumed.update(g.id for g in run.glyphs)
                done.add(ci)
                d.glyph_dims += 1  # type: ignore[attr-defined]
            # Chains of short dimensions whose numbers run together: pick each value out by length.
            h_ref = median(run.height for _s, _ci, run in found)
            rest = [ci for ci in range(len(cands)) if ci not in assigned and ci not in done]
            rest.sort(key=lambda ci: -dist(cands[ci].a, cands[ci].b))
            for ci in rest:
                a, b, u, n = frame(cands[ci])
                ln = dist(a, b)
                hit = stroketext.read_expected(grid, pool, consumed, a, b, u, n, ln, h_ref,
                                               classes, table, ln * scale)
                if hit is None:
                    continue
                text, run = hit
                emit(cands[ci], text, run.center(), run.height, angle_of(run.u))
                consumed.update(g.id for g in run.glyphs)
                d.glyph_dims += 1  # type: ignore[attr-defined]

    for line_id, n in emitted.items():
        if n >= pairs_on_line.get(line_id, 1):
            consumed.add(line_id)

    _radial_and_leaders(d, arrows, used_arrows, consumed, used_texts, by_id, text_grid, texts)


def _radial_and_leaders(d, arrows, used_arrows, consumed, used_texts, by_id, text_grid, texts) -> None:
    open_shapes = [
        s for s in d.shapes
        if s.id not in consumed and s.stroke and not s.closed and all(isinstance(g, Line) for g in s.segs)
    ]
    end_grid: Grid[int] = Grid(2.0)
    for s in open_shapes:
        end_grid.insert_point(s.start(), s.id)
        end_grid.insert_point(s.end(), s.id)
    circles = [s for s in d.shapes if s.id not in consumed and len(s.segs) == 1 and isinstance(s.segs[0], Arc)]
    circle_grid: Grid[int] = Grid(10.0)
    for s in circles:
        circle_grid.insert(s.bbox(), s.id)

    def chain_from(p: Vec, first_dir: Vec | None, used: set[int]) -> list[tuple[Shape, list[Vec]]]:
        """Follow connected open line shapes starting at p."""
        out = []
        cur = p
        for step in range(4):
            best = None
            for sid in end_grid.query_point(cur, TIP_TOL):
                if sid in used or sid in consumed:
                    continue
                s = by_id[sid]
                if dist(s.start(), cur) <= TIP_TOL:
                    pts = [g.start() for g in s.segs] + [s.end()]
                elif dist(s.end(), cur) <= TIP_TOL:
                    pts = [s.end()] + [g.start() for g in reversed(s.segs)]
                else:
                    continue
                if step == 0 and first_dir is not None:
                    if dot(unit(sub(pts[1], pts[0])), first_dir) < 0.995:
                        continue
                best = (s, pts)
                break
            if best is None:
                break
            used.add(best[0].id)
            out.append(best)
            cur = best[1][-1]
        return out

    for i, arr in enumerate(arrows):
        if i in used_arrows or arr.kind in ("dot", "tick"):
            continue
        used: set[int] = set()
        chain = chain_from(arr.tip, mul(arr.direction, -1), used)
        if not chain:
            continue
        verts: list[Vec] = [arr.tip]
        for _s, pts in chain:
            verts.extend(pts[1:])
        # text closest to the last leader segment
        a, b = verts[-2], verts[-1]
        best_t, best_d = None, math.inf
        for tid in text_grid.query((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])), 10.0):
            if tid in used_texts:
                continue
            t = texts[tid]
            dd = min(
                point_segment_distance(text_anchor(t), a, b),
                point_segment_distance(t.insert, a, b),
            )
            if dd <= 1.6 * t.height + 1.0 and dd < best_d:
                best_t, best_d = t, dd
        if best_t is None:
            continue
        t = best_t
        first_dir = unit(sub(verts[1], verts[0]))
        circle = None
        for sid in circle_grid.query_point(arr.tip, 1.0):
            arc = by_id[sid].segs[0]
            if abs(dist(arc.center, arr.tip) - arc.radius) <= max(TIP_TOL, arc.radius * 0.01):
                radial = unit(sub(arr.tip, arc.center))
                if abs(cross(radial, first_dir)) < 0.03:
                    circle = arc
                    break
        is_dia = any(m in t.text for m in DIAMETER_MARKS)
        is_rad = t.text.lstrip().upper().startswith("R")
        consumed_now = [s.id for s, _ in chain] + [sh.id for sh in _arrow_shapes(arr)]
        if circle is not None and (is_dia or is_rad):
            # A second arrowhead at the far end of a diameter line belongs to it too.
            for j, other in enumerate(arrows):
                if j != i and j not in used_arrows and dist(other.tip, verts[-1]) < TIP_TOL and len(chain) == 1:
                    used_arrows.add(j)
                    consumed_now += [sh.id for sh in _arrow_shapes(other)]
            d.radial_dims.append(RadialDim(
                kind="diameter" if is_dia else "radius", center=circle.center, radius=circle.radius,
                point=arr.tip, text=t.text, text_pos=text_anchor(t), text_height=t.height,
                arrow_size=arr.size, layer=chain[0][0].style.layer, color=chain[0][0].style.color,
            ))
        else:
            d.leaders.append(Leader(
                vertices=verts, text=t.text, text_pos=t.insert, text_height=t.height,
                text_rotation=t.rotation, arrow_size=arr.size, layer=chain[0][0].style.layer,
                color=chain[0][0].style.color,
            ))
            d.leaders[-1].text_item = t  # type: ignore[attr-defined]
        used_arrows.add(i)
        consumed.update(consumed_now)
        used_texts.add(t.id)
