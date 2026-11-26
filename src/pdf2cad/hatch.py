"""Rebuild HATCH entities from filled paths and families of evenly spaced parallel lines."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median

from .geom import (
    Vec, angle_diff, bbox, convex_hull, distance_to_polygon, dot, line_angle, point_in_polygon,
    polygon_area, sub,
)
from .model import Arc, Drawing, Hatch, HatchFamily, Line, Seg, Shape, Style

ON_TOL = 0.12
TINY_FILL = 1.0


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
    tiny: dict[tuple, Hatch] = {}
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
        b = bbox([p for s in group for p in s.points()])
        if max(b[2] - b[0], b[3] - b[1]) < TINY_FILL:
            # Dots of a stipple pattern, glyph slivers: one HATCH per area instead of thousands.
            key = (fill, group[0].style.layer, int(b[0] // 50), int(b[1] // 50))
            if key not in tiny:
                tiny[key] = Hatch(loops=[], solid=True, color=fill, layer=group[0].style.layer)
                d.hatches.append(tiny[key])
            tiny[key].loops.extend(s.segs for s in group)
        else:
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
            # Short pieces that merely line up (curved symbols repeated side by side) are
            # not a hatch: real hatch lines run well past their spacing.
            if median(x.segs[0].length for x in fam) >= 1.5 * s:  # type: ignore[union-attr]
                out.extend(_split_spatial(fam, u, s))
        i = j + 1
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


def _rows(segs: list[tuple[float, float, float]], join: float) -> dict[float, list[list[float]]]:
    """(offset, t0, t1) pieces -> per offset, the covered intervals (pieces closer than join merged)."""
    rows: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for o, t0, t1 in segs:
        rows[round(o, 2)].append((min(t0, t1), max(t0, t1)))
    out: dict[float, list[list[float]]] = {}
    for o, ivs in rows.items():
        ivs.sort()
        merged = [list(ivs[0])]
        for lo, hi in ivs[1:]:
            if lo - merged[-1][1] <= join:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out[o] = merged
    return out


def _region_from_rows(rows: dict[float, list[list[float]]], u: Vec, spacing: float, pad: float):
    """The area the hatch lines cover: each row's extent, half a spacing to either side.

    Edges along the rows sit midway between pattern lines; an edge lying on a pattern line
    makes CAD programs fill the wrong side of it.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    n = (-u[1], u[0])
    half = 0.5 * spacing
    pieces = [box(lo - pad, o - half, hi + pad, o + half) for o in rows for lo, hi in rows[o]]
    area = unary_union([p.buffer(0.01 * spacing, join_style=2) for p in pieces]).buffer(-0.01 * spacing, join_style=2)
    area = area.simplify(0.05 * spacing)

    def world(coords):
        return [(t * u[0] + o * n[0], t * u[1] + o * n[1]) for t, o in list(coords)[:-1]]

    loops: list[list[Seg]] = []
    for poly in getattr(area, "geoms", [area]):
        if poly.is_empty or poly.area < spacing * spacing:
            continue
        for ring in [poly.exterior, *poly.interiors]:
            pts = world(ring.coords)
            if len(pts) >= 3:
                loops.append([Line(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))])
    return loops or None


def _boundary_from_strips(fam: list[Shape], u: Vec, spacing: float) -> list[list[Seg]] | None:
    n = (-u[1], u[0])
    segs = [(dot(s.segs[0].start(), n), dot(s.segs[0].start(), u), dot(s.segs[0].end(), u)) for s in fam]
    return _region_from_rows(_rows(segs, 0.0), u, spacing, 0.0)


MAX_DASH = 6.0


def recognize_dashed(d: Drawing, consumed: set[int], min_rows: int = 4, min_dashes: int = 12) -> None:
    """Patterns made of short dashes on a regular lattice ("+" planting, brick, dot-dash fills).

    Each pattern line of a CAD hatch is exported as a row of equal dashes; rows repeat at a
    fixed spacing with a fixed shift along the line.  Arms of different directions that cover
    the same area (the two strokes of a "+") end up in one HATCH.
    """
    short = [
        s for s in d.shapes
        if s.id not in consumed and s.stroke and s.is_single_line and not s.style.dashes
        and s.segs[0].length <= MAX_DASH  # type: ignore[union-attr]
    ]
    groups: dict[tuple, list[Shape]] = defaultdict(list)
    for s in short:
        a = line_angle(s.segs[0].start(), s.segs[0].end())
        groups[(s.style, round(a * 2) / 2 % 180, round(s.segs[0].length, 1))].append(s)  # type: ignore[union-attr]
    found = []
    for (style, ang, _ln), group in groups.items():
        if len(group) < min_dashes:
            continue
        r = math.radians(ang)
        u = (math.cos(r), math.sin(r))
        n = (-u[1], u[0])
        info = {}
        for s in group:
            t0, t1 = dot(s.segs[0].start(), u), dot(s.segs[0].end(), u)
            info[s.id] = (dot(s.segs[0].start(), n), min(t0, t1), max(t0, t1))
        for comp in _dash_components(group, info):
            # Neighbouring areas may be hatched with different origins: peel off one lattice at a time.
            for _ in range(4):
                res = _dash_family(comp, info, ang, u, n, min_rows, min_dashes)
                if not res:
                    break
                fam_def, region, fitted = res
                found.append((style, fam_def, region, fitted))
                ids = {s.id for s in fitted}
                comp = [s for s in comp if s.id not in ids]
    # Arms of one symbol: same style, overlapping areas.
    used = [False] * len(found)
    for i, (style, fam_i, region_i, shapes_i) in enumerate(found):
        if used[i]:
            continue
        fams, region, shapes = [fam_i], region_i, list(shapes_i)
        for j in range(i + 1, len(found)):
            st, fj, rj, sj = found[j]
            if used[j] or st != style:
                continue
            inter = region.intersection(rj).area
            if inter >= 0.6 * min(region.area, rj.area):
                fams.append(fj)
                region = region.union(rj)
                shapes += sj
                used[j] = True
        loops = _loops_from_area(region.simplify(0.05 * fam_i.spacing))
        if not loops:
            continue
        h = Hatch(loops=loops, solid=False, color=style.color, layer=style.layer, families=fams, name="USER")
        h.line_width = style.width  # type: ignore[attr-defined]
        d.hatches.append(h)
        consumed.update(s.id for s in shapes)


def _loops_from_area(area) -> list[list[Seg]]:
    loops: list[list[Seg]] = []
    for poly in getattr(area, "geoms", [area]):
        if poly.is_empty or not hasattr(poly, "exterior"):
            continue
        for ring in [poly.exterior, *poly.interiors]:
            pts = list(ring.coords)[:-1]
            if len(pts) >= 3:
                loops.append([Line(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))])
    return loops


def _dash_components(group: list[Shape], info: dict) -> list[list[Shape]]:
    """Split dashes into areas: neighbours lie within a few dash lengths of each other."""
    ln = median(info[s.id][2] - info[s.id][1] for s in group)
    return _components(group, max(3.0 * ln, 1.0))


def _components(group: list[Shape], reach: float) -> list[list[Shape]]:
    from .spatial import Grid

    grid: Grid[int] = Grid(reach)
    for i, s in enumerate(group):
        grid.insert(s.bbox(), i)
    parent = list(range(len(group)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, s in enumerate(group):
        for j in grid.query(s.bbox(), reach):
            if j > i:
                parent[root(j)] = root(i)
    comps: dict[int, list[Shape]] = defaultdict(list)
    for i, s in enumerate(group):
        comps[root(i)].append(s)
    return list(comps.values())


def _dash_family(fam: list[Shape], info: dict, ang: float, u: Vec, n: Vec, min_rows: int, min_dashes: int):
    if len(fam) < min_dashes:
        return None
    rows: dict[float, list[float]] = defaultdict(list)
    for s in fam:
        o, t0, _t1 = info[s.id]
        rows[round(o, 2)].append(t0)
    if len(rows) < min_rows:
        return None
    offs = sorted(rows)
    steps = [b - a for a, b in zip(offs, offs[1:])]
    spacing = median(steps)
    if spacing < 0.3:
        return None
    gaps = sorted(b - a for o in offs for a, b in zip(sorted(rows[o]), sorted(rows[o])[1:]) if b - a > 0.05)
    if len(gaps) < min_dashes // 2:
        return None
    period = gaps[len(gaps) // 4]
    ln = median(info[s.id][2] - info[s.id][1] for s in fam)
    if period <= ln * 1.05:
        return None
    # Every dash start must fall on the lattice: t = t00 + k * shift + m * period on row k.
    o0, t00 = offs[0], min(rows[offs[0]])
    shifts = []
    for a, b in zip(offs, offs[1:]):
        if abs((b - a) - spacing) < 0.05 * spacing:
            shifts.append((min(rows[b]) - min(rows[a])) % period)
    if not shifts:
        return None
    shift = sorted(shifts)[len(shifts) // 2]
    # Small errors add up over a hundred rows: refine from the far ends, snap to round fractions.
    k_far = round((offs[-1] - o0) / spacing)
    if k_far > 0:
        spacing = (offs[-1] - o0) / k_far
    spans = []
    for o in offs:
        ts = sorted(rows[o])
        m = round((ts[-1] - ts[0]) / period)
        if m >= 3:
            spans.append((ts[-1] - ts[0]) / m)
    if spans:
        period = median(spans)
    for frac in (0, 1 / 2, 1 / 3, 2 / 3, 1 / 4, 3 / 4, 1):
        if abs(shift - frac * period) <= 0.03 * period:
            shift = frac * period % period
            break
    fitted = []
    for s in fam:
        o, t, _t1 = info[s.id]
        k = round((o - o0) / spacing)
        if abs((o - o0) - k * spacing) > 0.05 * spacing:
            continue
        e = (t - t00 - k * shift) % period
        if min(e, period - e) <= max(0.05, 0.02 * period):
            fitted.append(s)
    if len(fitted) < max(min_dashes, 0.3 * len(fam)):
        return None
    if len({round(info[s.id][0], 2) for s in fitted}) < min_rows:
        return None
    if shift > period / 2:
        shift -= period
    base = (t00 * u[0] + o0 * n[0], t00 * u[1] + o0 * n[1])
    segs = [info[s.id] for s in fitted]
    loops = _region_from_rows(_rows(segs, period - ln + 0.05 * period), u, spacing, 0.5 * (period - ln))
    if not loops:
        return None
    region = _rebuild(loops)
    return HatchFamily(ang, base, spacing, shift, (ln, -(period - ln))), region, fitted


def _rebuild(loops: list[list[Seg]]):
    """Loops back to one shapely area (outer rings minus the rings inside them)."""
    from shapely.geometry import Polygon

    polys = [Polygon(pts).buffer(0) for pts in (_loop_polygon(lp) for lp in loops) if len(pts) >= 3]
    polys.sort(key=lambda p: -p.area)
    area = Polygon()
    for p in polys:
        if area.is_empty:
            area = p
        elif area.contains(p.representative_point()) and area.contains(p):
            area = area.symmetric_difference(p)
        else:
            area = area.union(p)
    return area


def recognize_pattern(d: Drawing, consumed: set[int], min_lines: int = 4) -> None:
    lines = [
        s for s in d.shapes
        if s.id not in consumed and s.stroke and s.is_single_line and not s.style.dashes
    ]
    # CAD exporters usually write one hatch as one path; lines from such a path are
    # grouped on their own so stray parallel lines elsewhere cannot join them.
    per_path: dict[int, int] = defaultdict(int)
    for s in lines:
        per_path[s.path_id] += 1
    groups: dict[tuple, list[Shape]] = defaultdict(list)
    for s in lines:
        a = line_angle(s.segs[0].start(), s.segs[0].end())
        pid = s.path_id if per_path[s.path_id] >= min_lines else -1
        groups[(s.style, round(a * 2) / 2 % 180, pid)].append(s)
    loops = [
        s for s in d.shapes
        if s.id not in consumed and s.closed and (len(s.segs) > 1 or s.is_circle)
    ]

    found: list[tuple[Hatch, list[Shape], list[Seg] | None]] = []
    split: list[tuple[Style, list[Shape]]] = []
    for (style, _ang, pid), group in groups.items():
        # Lines from separate paths: keep areas apart so their spacings do not interleave.
        parts = [group] if pid != -1 else _components(group, 5.0)
        split.extend((style, g) for g in parts)
    for style, group in split:
        for fam in _families(group, min_lines):
            pts = _endpoints(fam)
            r = math.radians(line_angle(fam[0].segs[0].start(), fam[0].segs[0].end()))
            u = (math.cos(r), math.sin(r))
            n = (-u[1], u[0])
            offs = sorted({round(dot(s.segs[0].start(), n), 4) for s in fam})
            diffs = sorted(b - a for a, b in zip(offs, offs[1:]) if b - a > 0.02)
            spacing = diffs[len(diffs) // 2]
            strips = _boundary_from_strips(fam, u, spacing)
            boundary = _boundary_from_loops(pts, loops) or _boundary_from_hull(pts) or _boundary_from_sweep(fam, u)
            if boundary is not None and strips is not None:
                # A drawn outline that also takes in unhatched ground is the wrong one.
                extra = _rebuild(boundary).difference(_rebuild(strips).buffer(spacing))
                if extra.area > 0.03 * _rebuild(strips).area + 4 * spacing * spacing:
                    boundary = None
            boundary = boundary or strips
            if boundary is None:
                continue
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


__all__ = ["recognize_dashed", "recognize_pattern", "recognize_solid", "Arc"]
