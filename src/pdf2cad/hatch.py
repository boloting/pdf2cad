"""Rebuild HATCH entities from filled paths and families of evenly spaced parallel lines."""

from __future__ import annotations

import math
from collections import defaultdict

from .geom import (
    Vec, angle_diff, bbox, convex_hull, distance_to_polygon, dot, line_angle, point_in_polygon,
    polygon_area, sub,
)
from .model import Arc, Drawing, Hatch, HatchFamily, Line, Seg, Shape

ON_TOL = 0.12


def _is_white(c) -> bool:
    return c is not None and all(v >= 0.98 for v in c)


def _loop_polygon(segs: list[Seg]) -> list[Vec]:
    pts: list[Vec] = []
    for s in segs:
        fl = s.flatten()
        pts.extend(fl if not pts else fl[1:])
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) < 1e-6:
        pts.pop()
    return pts


def recognize_solid(d: Drawing, consumed: set[int]) -> None:
    by_path: dict[int, list[Shape]] = defaultdict(list)
    for s in d.shapes:
        if s.id in consumed or s.fill is None:
            continue
        by_path[s.path_id].append(s)
    for _pid, group in by_path.items():
        fill = group[0].fill
        if _is_white(fill):
            # White fills are masks under text or wipeouts; CAD has no use for them.
            for s in group:
                if s.stroke:
                    s.fill = None
                else:
                    consumed.add(s.id)
            continue
        d.hatches.append(Hatch(loops=[s.segs for s in group], solid=True, color=fill, layer=group[0].style.layer))
        for s in group:
            if s.stroke:
                s.fill = None
            else:
                consumed.add(s.id)


def _families(lines: list[Shape], min_lines: int) -> list[list[Shape]]:
    """Split parallel same-style lines into clusters of evenly spaced lines."""
    if len(lines) < min_lines:
        return []
    ang = line_angle(lines[0].segs[0].start(), lines[0].segs[0].end())
    r = math.radians(ang)
    u = (math.cos(r), math.sin(r))
    n = (-u[1], u[0])

    def off(s: Shape) -> float:
        return dot(s.segs[0].start(), n)

    items = sorted(lines, key=off)
    offs = [off(s) for s in items]
    # Cluster by gaps in offset: a jump much larger than the local spacing splits it.
    uniq: list[float] = []
    for o in offs:
        if not uniq or o - uniq[-1] > 0.02:
            uniq.append(o)
    if len(uniq) < min_lines:
        return []
    diffs = [b - a for a, b in zip(uniq, uniq[1:])]
    out: list[list[Shape]] = []
    start = 0
    i = 0
    while i < len(diffs):
        s = diffs[i]
        j = i
        while j + 1 < len(diffs) and abs(diffs[j + 1] - s) <= max(0.03, 0.05 * s):
            j += 1
        count = j - i + 2
        if count >= min_lines and s >= 0.3:
            lo, hi = uniq[i] - 0.01, uniq[j + 1] + 0.01
            fam = [x for x in items if lo <= off(x) <= hi]
            out.extend(_split_spatial(fam, u, s))
        i = j + 1
        start = i
    del start
    return [f for f in out if len({round(off(x), 2) for x in f}) >= min_lines]


def _split_spatial(fam: list[Shape], u: Vec, spacing: float) -> list[list[Shape]]:
    """Separate a family into groups whose lines overlap along the hatch direction."""
    groups: list[list[Shape]] = []
    spans: list[tuple[float, float]] = []
    for s in fam:
        a, b = dot(s.segs[0].start(), u), dot(s.segs[0].end(), u)
        lo, hi = min(a, b), max(a, b)
        for gi, (glo, ghi) in enumerate(spans):
            if lo <= ghi + spacing * 2 and hi >= glo - spacing * 2:
                groups[gi].append(s)
                spans[gi] = (min(glo, lo), max(ghi, hi))
                break
        else:
            groups.append([s])
            spans.append((lo, hi))
    return groups


def _endpoints(fam: list[Shape]) -> list[Vec]:
    pts: list[Vec] = []
    for s in fam:
        pts.append(s.segs[0].start())
        pts.append(s.segs[0].end())
    return pts


def _boundary_from_loops(pts: list[Vec], loops: list[Shape]) -> list[list[Seg]] | None:
    fb = bbox(pts)
    cands = []
    for s in loops:
        b = s.bbox()
        if b[2] < fb[0] - 1 or b[0] > fb[2] + 1 or b[3] < fb[1] - 1 or b[1] > fb[3] + 1:
            continue
        poly = _loop_polygon(s.segs)
        if len(poly) < 3:
            continue
        hits = [i for i, p in enumerate(pts) if distance_to_polygon(p, poly) <= ON_TOL]
        if hits:
            cands.append((abs(polygon_area(poly)), s, poly, set(hits)))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    for k, (_area, outer, opoly, ohits) in enumerate(cands):
        mids = [((pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2) for i in range(0, len(pts), 2)]
        inside = sum(1 for m in mids if point_in_polygon(m, opoly))
        if inside < 0.8 * len(mids):
            continue
        chosen = [outer]
        covered = set(ohits)
        for _a, s, poly, hits in cands[k + 1:]:
            if all(point_in_polygon(p, opoly) or distance_to_polygon(p, opoly) <= ON_TOL for p in poly[:: max(1, len(poly) // 8)]):
                chosen.append(s)
                covered |= hits
        if len(covered) >= 0.9 * len(pts):
            return [s.segs for s in chosen]
    return None


def _boundary_from_hull(pts: list[Vec]) -> list[list[Seg]] | None:
    hull = convex_hull(pts)
    if len(hull) < 3 or abs(polygon_area(hull)) < 1e-3:
        return None
    if all(distance_to_polygon(p, hull) <= ON_TOL for p in pts):
        return [[Line(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))]]
    return None


def _boundary_from_sweep(fam: list[Shape], u: Vec) -> list[list[Seg]] | None:
    """Walk the start points up one side and the end points back down the other."""
    n = (-u[1], u[0])
    rows: dict[float, list[Shape]] = defaultdict(list)
    for s in fam:
        rows[round(dot(s.segs[0].start(), n), 2)].append(s)
    if any(len(v) > 1 for v in rows.values()):
        return None
    left, right = [], []
    for k in sorted(rows):
        g = rows[k][0].segs[0]
        a, b = g.start(), g.end()
        if dot(a, u) > dot(b, u):
            a, b = b, a
        left.append(a)
        right.append(b)
    poly = left + right[::-1]
    if abs(polygon_area(poly)) < 1e-3:
        return None
    return [[Line(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]]


def recognize_pattern(d: Drawing, consumed: set[int], min_lines: int = 4) -> None:
    lines = [
        s for s in d.shapes
        if s.id not in consumed and s.stroke and s.is_single_line and not s.style.dashes
    ]
    groups: dict[tuple, list[Shape]] = defaultdict(list)
    for s in lines:
        a = line_angle(s.segs[0].start(), s.segs[0].end())
        key = (s.style, round(a * 2) / 2 % 180)
        groups[key].append(s)
    loops = [
        s for s in d.shapes
        if s.id not in consumed and s.closed and (len(s.segs) > 1 or s.is_circle)
    ]

    found: list[tuple[Hatch, list[Shape], list[Seg] | None]] = []
    for (style, ang), group in groups.items():
        for fam in _families(group, min_lines):
            pts = _endpoints(fam)
            r = math.radians(line_angle(fam[0].segs[0].start(), fam[0].segs[0].end()))
            u = (math.cos(r), math.sin(r))
            boundary = _boundary_from_loops(pts, loops) or _boundary_from_hull(pts) or _boundary_from_sweep(fam, u)
            if boundary is None:
                continue
            n = (-u[1], u[0])
            offs = sorted({round(dot(s.segs[0].start(), n), 4) for s in fam})
            diffs = sorted(b - a for a, b in zip(offs, offs[1:]) if b - a > 0.02)
            spacing = diffs[len(diffs) // 2]
            base = fam[0].segs[0].start()
            angle = math.degrees(r) % 180
            h = Hatch(
                loops=boundary, solid=False, color=style.color, layer=style.layer,
                families=[HatchFamily(angle, base, spacing)],
            )
            h.line_width = style.width  # type: ignore[attr-defined]
            found.append((h, fam, boundary[0]))

    # Cross hatching: two families with the same boundary become one pattern.
    merged: list[tuple[Hatch, list[Shape]]] = []
    for h, fam, outer in found:
        for mh, mfam in merged:
            if mh.layer == h.layer and mh.color == h.color and _same_loop(mh.loops[0], h.loops[0]):
                mh.families.extend(h.families)
                mfam.extend(fam)
                break
        else:
            merged.append((h, list(fam)))

    for h, fam in merged:
        _name_pattern(h)
        d.hatches.append(h)
        consumed.update(s.id for s in fam)


def _same_loop(a: list[Seg], b: list[Seg]) -> bool:
    pa, pb = _loop_polygon(a), _loop_polygon(b)
    ba, bb = bbox(pa), bbox(pb)
    return all(abs(x - y) < 0.2 for x, y in zip(ba, bb))


ANSI_SPACING = 3.175


def _name_pattern(h: Hatch) -> None:
    """Name the pattern after the AutoCAD predefined one it reproduces, when it does."""
    fams = h.families
    if len(fams) == 1:
        h.name = "ANSI31" if angle_diff(fams[0].angle, 45) < 0.5 or angle_diff(fams[0].angle, 135) < 0.5 else "USER"
    elif len(fams) == 2 and abs(fams[0].spacing - fams[1].spacing) < 0.02 and abs(angle_diff(fams[0].angle, fams[1].angle) - 90) < 0.5:
        h.name = "ANSI37" if any(angle_diff(f.angle, 45) < 0.5 for f in fams) else "USER"
    else:
        h.name = "USER"
    if h.name.startswith("ANSI"):
        h.pattern_scale = round(fams[0].spacing / ANSI_SPACING, 4)


__all__ = ["recognize_pattern", "recognize_solid", "Arc"]
