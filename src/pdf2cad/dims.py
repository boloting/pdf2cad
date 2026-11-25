"""Rebuild dimensions and leaders from arrowheads, lines and nearby text."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .geom import (
    Vec, add, angle_diff, angle_of, cross, dist, dot, line_angle, mid, mul, point_line_distance,
    point_segment_distance, project_param, rotate, sub, unit,
)
from .model import Arc, Arrow, Drawing, Leader, Line, LinearDim, RadialDim, Shape, TextItem
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


@dataclass
class _Cand:
    line: Shape
    a: Vec
    b: Vec
    ends: tuple[Arrow, Arrow]
    dle: float


def _arrow_shapes(a: Arrow) -> list[Shape]:
    return [a.shape, *getattr(a, "extra_shapes", [])]


def recognize(d: Drawing, consumed: set[int], used_texts: set[int], max_arrow: float = 8.0) -> None:
    shapes = d.shapes
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

    def ticks_on(line: Shape, a: Vec, b: Vec) -> list[tuple[float, Shape]]:
        out = []
        la = line_angle(a, b)
        for sid in line_grid.query((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])), 3.0):
            t = by_id[sid]
            if sid == line.id or sid in consumed:
                continue
            g = t.segs[0]
            ln = g.length  # type: ignore[union-attr]
            if not (0.5 <= ln <= 6.0):
                continue
            if abs(angle_diff(line_angle(g.a, g.b), la) - 45) > 12:  # type: ignore[union-attr]
                continue
            m = mid(g.a, g.b)  # type: ignore[union-attr]
            if point_line_distance(m, a, b) < 0.1:
                p = project_param(m, a, b)
                if -0.01 <= p <= 1.01:
                    out.append((p, t))
        out.sort(key=lambda x: x[0])
        return out

    # 1. Candidate dimension lines: a straight line with an arrowhead or tick at each end.
    cands: list[_Cand] = []
    tick_used: set[int] = set()
    for s in lines:
        g = s.segs[0]
        a, b = g.start(), g.end()
        ln = dist(a, b)
        if ln < 1.0:
            continue
        u = unit(sub(b, a))
        ia, ib = arrow_at(a, u), arrow_at(b, u)
        if ia is not None and ib is not None and ia != ib:
            cands.append(_Cand(s, a, b, (arrows[ia], arrows[ib]), 0.0))
            continue
        tk = [t for t in ticks_on(s, a, b) if t[1].id not in tick_used]
        if len(tk) >= 2:
            (p0, t0), (p1, t1) = tk[0], tk[-1]
            pa = add(a, mul(sub(b, a), p0))
            pb = add(a, mul(sub(b, a), p1))
            if dist(pa, pb) < 1.0:
                continue
            size = t0.segs[0].length / math.sqrt(2)  # type: ignore[union-attr]
            arr = (Arrow(pa, mul(u, -1), size, "tick", t0), Arrow(pb, u, size, "tick", t1))
            cands.append(_Cand(s, pa, pb, arr, dist(a, pa)))
            tick_used.update((t0.id, t1.id))

    # 2. Assign texts to candidate lines, nearest pairs first.
    pairs: list[tuple[float, int, int]] = []
    for ci, c in enumerate(cands):
        la = line_angle(c.a, c.b)
        ln = dist(c.a, c.b)
        box = (min(c.a[0], c.b[0]), min(c.a[1], c.b[1]), max(c.a[0], c.b[0]), max(c.a[1], c.b[1]))
        for tid in text_grid.query(box, 12.0):
            t = texts[tid]
            if tid in used_texts or not parse_value(t.text):
                continue
            parallel = angle_diff(t.rotation % 180, la) < 2.0
            if not parallel and angle_diff(t.rotation, 0, 360) > 2.0:
                continue
            ctr = text_anchor(t)
            pd = point_line_distance(ctr, c.a, c.b)
            limit = 1.6 * t.height + 1.0 if parallel else 2.5 * t.height + 1.0
            if pd > limit:
                continue
            p = project_param(ctr, c.a, c.b)
            if p < -0.2 - 3 * t.height / ln or p > 1.2 + 3 * t.height / ln:
                continue
            outside = max(0.0, -p, p - 1.0) * ln
            pairs.append((pd + outside + abs(p - 0.5) * 0.05, ci, tid))
    pairs.sort()
    assigned: dict[int, int] = {}
    taken: set[int] = set()
    for _score, ci, tid in pairs:
        if ci in assigned or tid in taken:
            continue
        assigned[ci] = tid
        taken.add(tid)

    # 3. Extension lines and the DIMENSION itself.
    def extension(c: _Cand, p: Vec) -> tuple[Vec, float, Shape] | None:
        la = line_angle(c.a, c.b)
        best = None
        for sid in line_grid.query_point(p, TIP_TOL):
            e = by_id[sid]
            if sid == c.line.id or not _style_match(e, c.line):
                continue
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

    for ci, c in enumerate(cands):
        tid = assigned.get(ci)
        if tid is None:
            continue
        t = texts[tid]
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
            p1=p1, p2=p2, base=a, angle=round(ang, 6), text=t.text, text_pos=text_anchor(t),
            text_height=t.height, text_rotation=t.rotation, arrow_size=(ends[0].size + ends[1].size) / 2,
            arrow_kind=ends[0].kind, exe=exe, measured=dist(a, b), layer=c.line.style.layer,
            color=c.line.style.color,
        )
        dim.ext1 = e1 is not None  # type: ignore[attr-defined]
        dim.ext2 = e2 is not None  # type: ignore[attr-defined]
        dim.dle = c.dle  # type: ignore[attr-defined]
        d.linear_dims.append(dim)
        consumed.add(c.line.id)
        for arr in ends:
            for sh in _arrow_shapes(arr):
                consumed.add(sh.id)
            for i, x in enumerate(arrows):
                if x is arr:
                    used_arrows.add(i)
        used_texts.add(tid)

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
