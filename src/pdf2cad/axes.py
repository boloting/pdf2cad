"""Axis grid: centre lines plotted as loose dashes, and the numbered bubbles at their ends.

CAD plots often explode a dash-dot linetype into one short stroke per dash and per dot,
and draw circles as many-sided polygons.  Collinear strokes that repeat one rhythm are
joined back into a single LINE carrying that rhythm as a linetype; regular polygons become
circles; a circle that a long line runs into radially, with a short label inside, is an
axis bubble.  Its label is read from the glyphs inside the circle alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from statistics import median

from .model import Arc, AxisBubble, Drawing, Line, Shape, Style
from .spatial import Grid

AXIS_LAYER = "PDF_轴线"
BUBBLE_LAYER = "PDF_轴号"
MAX_PIECE = 40.0  # mm; longer strokes are not dashes
MAX_GAP = 3.0
ANGLE_TOL = 0.05  # degrees
OFFSET_TOL = 0.03  # mm
_PART = r"(?:\d{1,3}|0?[A-Z]{1,2}\d{0,2})"
LABEL = re.compile(rf"{_PART}(?:/{_PART})?'?")  # 1, 12, A, AA, A1, 1/A, 1/0A
# Axis letters never use I, O and Z; a recognizer that reads one meant the digit.
LOOKALIKE = str.maketrans({"I": "1", "l": "1", "|": "1", "O": "0", "o": "0", "Z": "2", "z": "2"})


def polygons_to_circles(d: Drawing, consumed: set[int]) -> int:
    """Closed, unfilled polygons of 12+ equal sides whose corners lie on one circle."""
    n = 0
    for s in d.shapes:
        if s.id in consumed or not s.closed or s.fill is not None or not s.stroke or len(s.segs) < 12:
            continue
        if not all(isinstance(g, Line) for g in s.segs):
            continue
        pts = [g.a for g in s.segs]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        rs = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
        r = sum(rs) / len(rs)
        if r < 0.3 or max(abs(v - r) for v in rs) > 0.01 * r + 0.003:
            continue
        sides = [g.length for g in s.segs]
        if max(sides) > 1.3 * median(sides) or max(sides) > 2 * math.pi * r / 10:
            continue
        s.segs = [Arc((cx, cy), r, 0.0, 360.0)]
        n += 1
    return n


def merge_dashes(d: Drawing, consumed: set[int]) -> int:
    """Join collinear dash and dot strokes of one style into single LINEs with a linetype."""
    groups: dict[tuple, list[tuple[float, float, float, Shape]]] = defaultdict(list)
    for s in d.shapes:
        if s.id in consumed or not s.stroke or s.fill is not None or not s.is_single_line or s.style.dashes:
            continue
        g = s.segs[0]
        ln = g.length
        if ln > MAX_PIECE:
            continue
        if ln < 1e-6:
            continue
        ang = math.degrees(math.atan2(g.b[1] - g.a[1], g.b[0] - g.a[0])) % 180.0
        if ang > 180.0 - ANGLE_TOL:
            ang -= 180.0
        groups[(s.style.layer, s.style.color, s.style.width)].append((ang, 0.0, 0.0, s))

    runs = []
    for key, items in groups.items():
        items.sort(key=lambda it: it[0])
        # Cluster by direction, then by offset from the origin across that direction.
        start = 0
        for i in range(1, len(items) + 1):
            if i < len(items) and items[i][0] - items[i - 1][0] <= ANGLE_TOL:
                continue
            bundle = items[start:i]
            start = i
            if len(bundle) < 5:
                continue
            ang = math.radians(median(it[0] for it in bundle))
            u, nrm = (math.cos(ang), math.sin(ang)), (-math.sin(ang), math.cos(ang))
            placed = []
            for _a, _o, _t, s in bundle:
                g = s.segs[0]
                ta, tb = g.a[0] * u[0] + g.a[1] * u[1], g.b[0] * u[0] + g.b[1] * u[1]
                o = (g.a[0] + g.b[0]) / 2 * nrm[0] + (g.a[1] + g.b[1]) / 2 * nrm[1]
                placed.append((o, min(ta, tb), max(ta, tb), s))
            placed.sort(key=lambda it: it[0])
            j0 = 0
            for j in range(1, len(placed) + 1):
                if j < len(placed) and placed[j][0] - placed[j - 1][0] <= OFFSET_TOL:
                    continue
                line = placed[j0:j]
                j0 = j
                if len(line) >= 5:
                    runs += _runs(key, u, nrm, line)

    # Rows of dashes a few dash lengths apart are a hatch, not lines.
    by_dir: dict[tuple, list] = defaultdict(list)
    for r in runs:
        by_dir[(r["key"], round(r["angle"], 1))].append(r)
    # One linetype per rhythm: the same pattern measures a hair differently line to line.
    canon: list[tuple[float, ...]] = []
    for r in sorted(runs, key=lambda r: -(r["t1"] - r["t0"])):
        for c in canon:
            if len(c) == len(r["dashes"]) and all(abs(a - b) <= 0.04 * max(a, 0.25) for a, b in zip(c, r["dashes"])):
                r["dashes"] = c
                break
        else:
            canon.append(r["dashes"])
    n = 0
    for r in runs:
        near = [o for o in by_dir[(r["key"], round(r["angle"], 1))] if o is not r
                and o["dashes"] == r["dashes"] and 1e-3 < abs(o["offset"] - r["offset"]) < r["dashes"][0]
                and o["t0"] < r["t1"] and o["t1"] > r["t0"]]
        if len(near) >= 2:
            continue
        first = r["shapes"][0]
        u, nrm, o = r["u"], r["n"], r["offset"]
        a = (u[0] * r["t0"] + nrm[0] * o, u[1] * r["t0"] + nrm[1] * o)
        b = (u[0] * r["t1"] + nrm[0] * o, u[1] * r["t1"] + nrm[1] * o)
        first.segs = [Line(a, b)]
        first.closed = False
        first.style = Style(first.style.layer, first.style.color, first.style.width, r["dashes"])
        consumed.update(s.id for s in r["shapes"][1:])
        n += 1
    return n


def _runs(key, u, nrm, line) -> list[dict]:
    """Dashed runs along one line: equal dashes at a steady pitch, with the dots between.

    Other strokes on the same line (a wall face, an extension line drawn over the axis)
    are left alone; a centre line may also lose a few dashes under them.
    """
    line.sort(key=lambda it: it[1])
    longs = [it for it in line if it[2] - it[1] > 0.5]
    if len(longs) < 4:
        return []
    top = Counter(round(it[2] - it[1], 1) for it in longs).most_common(1)[0][0]
    dash = median(it[2] - it[1] for it in longs if abs(it[2] - it[1] - top) <= 0.1 * top)
    pat = []
    for it in longs:
        if abs(it[2] - it[1] - dash) <= 0.08 * dash and (not pat or it[1] - pat[-1][1] >= 0.5 * dash):
            pat.append(it)
    if len(pat) < 4:
        return []
    steps = [b[1] - a[1] for a, b in zip(pat, pat[1:])]
    common = Counter(round(v, 1) for v in steps).most_common(1)[0][0]
    pitch = median(v for v in steps if abs(v - common) <= 0.03 * common)
    if pitch <= 1.02 * dash or sum(1 for v in steps if abs(v - pitch) <= 0.08 * pitch) < 3:
        return []
    # A second dashed line laid over this one, out of step: keep the steady dashes only.
    kept = [pat[0]]
    for it in pat[1:]:
        if it[1] - kept[-1][1] >= 0.92 * pitch:
            kept.append(it)
    pat = kept
    dots = [it for it in line if it[2] - it[1] <= 0.3]
    counts = []
    for a, b in zip(pat, pat[1:]):
        if b[1] - a[1] <= 1.1 * pitch:
            counts.append(sum(1 for it in dots if a[2] < it[1] < b[1]))
    k = round(median(counts)) if counts else 0
    if k == 0 and pitch - dash > dash:
        return []  # sparse strokes ("+" rows) are marks, not a dashed line
    space = pitch - dash
    dashes = (round(dash, 3), round(space, 3)) if k == 0 else \
        (round(dash, 3), *([round(space / (k + 1), 3), 0.0] * k), round(space / (k + 1), 3))
    bridge = 3.2 * pitch if k else 1.3 * pitch

    def regular(v: float) -> bool:
        n = round(v / pitch)
        return n >= 1 and abs(v - n * pitch) <= 0.08 * pitch

    out = []
    chains: list[list] = [[pat[0]]]
    for a, b in zip(pat, pat[1:]):
        if b[1] - a[1] > bridge or not regular(b[1] - a[1]):
            chains.append([b])
        else:
            chains[-1].append(b)
    for ch in chains:
        if len(ch) < 4:
            continue
        t0, t1 = ch[0][1], ch[-1][2]
        members = ch + [it for it in dots if t0 - pitch < it[1] < t1 + pitch]
        out.append({
            "key": key, "angle": math.degrees(math.atan2(u[1], u[0])) % 180.0,
            "u": u, "n": nrm, "offset": median(it[0] for it in ch),
            "t0": t0, "t1": t1, "dashes": dashes, "shapes": [it[3] for it in members],
        })
    return out


def _clean_label(text: str) -> str:
    t = re.sub(r"\s+", "", text).strip("()[]{}<>.·,，。")
    t = t.translate(LOOKALIKE).upper()
    return t if LABEL.fullmatch(t) else ""


def recognize(d: Drawing, consumed: set[int], used_texts: set[int], eng=None) -> dict:
    """Axis bubbles: a circle a line runs into radially, with a short label inside."""
    circles = [s for s in d.shapes if s.id not in consumed and s.is_circle and 2.0 <= s.segs[0].radius <= 8.0]
    if not circles:
        return {"bubbles": 0, "lines": 0}
    # End segments of lines and of short bent leaders, by end point.
    ends: Grid[tuple[int, int]] = Grid(10.0)
    lines: dict[int, Shape] = {}
    end_seg: dict[tuple[int, int], Line] = {}
    # Lines taken as extension lines by a dimension still connect a bubble to its axis.
    for s in d.shapes:
        if not s.stroke or s.closed or len(s.segs) > 6 or (s.id in consumed and not s.is_single_line):
            continue
        if not all(isinstance(g, Line) for g in s.segs):
            continue
        if s.is_single_line:
            lines[s.id] = s
        first, last = s.segs[0], s.segs[-1]
        end_seg[(s.id, 0)] = first
        end_seg[(s.id, 1)] = Line(last.b, last.a)
        ends.insert_point(first.a, (s.id, 0))
        ends.insert_point(last.b, (s.id, 1))
    glyph_grid: Grid[int] = Grid(5.0)
    by_id = {}
    for s in d.shapes:
        if s.id in consumed or s.is_circle:
            continue
        b = s.bbox()
        if max(b[2] - b[0], b[3] - b[1]) <= 16.0:
            glyph_grid.insert(b, s.id)
            by_id[s.id] = s
    text_grid: Grid[int] = Grid(10.0)
    for i, t in enumerate(d.texts):
        if t.id not in used_texts:
            text_grid.insert(t.bbox if t.bbox != (0, 0, 0, 0) else (*t.insert, *t.insert), i)
    long_grid: Grid[int] = Grid(20.0)
    for sid, s in lines.items():
        if s.segs[0].length > 6.0:
            long_grid.insert(s.bbox(), sid)

    axis_lines: set[int] = set()
    found = 0
    for c in circles:
        arc = c.segs[0]
        ctr, r = arc.center, arc.radius
        cx, cy = ctr
        hits, dirs, leads = [], [], []
        for key in ends.query_point(ctr, r * 1.3):
            g = end_seg[key]
            p, q = g.a, g.b
            dp = math.hypot(p[0] - cx, p[1] - cy)
            ln = g.length
            if abs(dp - r) > 0.15 * r + 0.2 or ln < 0.3 * r:
                continue
            # The line must point at the centre: it is radial, not a tangent or a chord.
            vx, vy = (p[0] - q[0]) / ln, (p[1] - q[1]) / ln
            wx, wy = (cx - p[0]) / dp, (cy - p[1]) / dp
            if vx * wx + vy * wy < math.cos(math.radians(5)):
                continue
            if key[0] in lines and ln >= 1.2 * r:
                hits.append(key[0])
                dirs.append((vx, vy))
            else:
                leads.append(key[0])
        # An axis drawn on through its bubble.
        for sid in long_grid.query_point(ctr):
            g = lines[sid].segs[0]
            if sid in hits or g.length < 3 * r or _seg_dist(ctr, g.a, g.b) > 0.05 * r:
                continue
            if min(math.dist(g.a, ctr), math.dist(g.b, ctr)) < 1.5 * r:
                continue
            hits.append(sid)
            dirs.append(((g.b[0] - g.a[0]) / g.length, (g.b[1] - g.a[1]) / g.length))
        if not hits and not leads:
            continue
        box = (cx - r, cy - r, cx + r, cy + r)
        inner, slash = [], None
        for gid in glyph_grid.query(box):
            s = by_id[gid]
            if gid in hits or gid in leads:
                continue
            if s.is_single_line and slash is None and _is_slash(s.segs[0], ctr, r, dirs):
                slash = s
                continue
            b = s.bbox()
            corners = [(b[0], b[1]), (b[0], b[3]), (b[2], b[1]), (b[2], b[3])]
            if not all(math.hypot(x - cx, y - cy) <= 0.95 * r for x, y in corners):
                continue
            if s.is_single_line and _on_axis(s.segs[0], ctr, r, dirs):
                continue
            inner.append(s)
        texts = []
        for i in text_grid.query(box):
            t = d.texts[i]
            tc = t.center if t.bbox != (0, 0, 0, 0) else t.insert
            if math.hypot(tc[0] - cx, tc[1] - cy) <= 0.8 * r:
                texts.append(t)
        if slash is None and not hits:
            continue  # a bent leader alone does not make a circle an axis bubble
        bubble = _label(eng, c, slash, inner, texts)
        if bubble is None:
            continue
        consumed.add(c.id)
        consumed.update(s.id for s in inner)
        if slash is not None:
            consumed.add(slash.id)
        used_texts.update(t.id for t in texts)
        d.axis_bubbles.append(bubble)
        axis_lines.update(hits)
        found += 1
    axis_lines = {i for i in _extend(axis_lines, lines, ends) if i not in consumed}
    for sid in axis_lines:
        s = lines[sid]
        s.style = Style(AXIS_LAYER, s.style.color, s.style.width, s.style.dashes)
    runs = _count_runs(axis_lines, lines)
    return {"bubbles": found, "lines": runs}


def _is_slash(g: Line, c, r: float, dirs) -> bool:
    """The diagonal across an additional-axis bubble, from rim to rim through the centre."""
    if not 1.4 * r <= g.length <= 2.05 * r or _seg_dist(c, g.a, g.b) > 0.2 * r:
        return False
    if not all(0.8 * r <= math.dist(p, c) <= 1.08 * r for p in (g.a, g.b)):
        return False
    ux, uy = (g.b[0] - g.a[0]) / g.length, (g.b[1] - g.a[1]) / g.length
    return all(abs(ux * vy - uy * vx) > math.sin(math.radians(20)) for vx, vy in dirs)


def _label(eng, c: Shape, slash: Shape | None, inner: list[Shape], texts) -> AxisBubble | None:
    arc = c.segs[0]
    ctr, r = arc.center, arc.radius
    if slash is None:
        label, height = "", 0.0
        if texts:
            label = _clean_label("".join(t.text for t in sorted(texts, key=lambda t: t.insert[0])))
            height = max(t.height for t in texts)
        if not label and inner and eng is not None:
            label, height = _read(eng, inner)
        if not label or "/" in label:
            return None
        return AxisBubble(ctr, r, label, height or 0.6 * r, c.style)
    # Numerator above the slash, the axis it follows below.
    g = slash.segs[0]
    ux, uy = g.b[0] - g.a[0], g.b[1] - g.a[1]

    def side(p) -> float:
        s = ux * (p[1] - g.a[1]) - uy * (p[0] - g.a[0])
        return s if ux >= 0 else -s

    parts: dict[bool, list] = {True: [], False: []}
    for s in inner:
        b = s.bbox()
        parts[side(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) > 0].append(s)
    tparts: dict[bool, list] = {True: [], False: []}
    for t in texts:
        tparts[side(t.center if t.bbox != (0, 0, 0, 0) else t.insert) > 0].append(t)
    read, pos = [], []
    for up in (True, False):
        txt, h = "", 0.0
        if tparts[up]:
            txt = _clean_label("".join(t.text for t in tparts[up]))
            h = max(t.height for t in tparts[up])
        if not txt and parts[up] and eng is not None:
            txt, h = _read(eng, parts[up])
        if not txt or "/" in txt:
            return None
        src = [s.bbox() for s in parts[up]] or [t.bbox for t in tparts[up]]
        bx = ((min(b[0] for b in src) + max(b[2] for b in src)) / 2, (min(b[1] for b in src) + max(b[3] for b in src)) / 2)
        read.append((txt, h))
        pos.append((bx[0] - ctr[0], bx[1] - ctr[1]))
    rel = ((g.a[0] - ctr[0], g.a[1] - ctr[1]), (g.b[0] - ctr[0], g.b[1] - ctr[1]))
    return AxisBubble(ctr, r, f"{read[0][0]}/{read[1][0]}", max(h for _t, h in read), c.style,
                      slash=rel, part_pos=(pos[0], pos[1]))


def _seg_dist(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _on_axis(g: Line, c, r: float, dirs) -> bool:
    """Dots and dashes of an axis line crossing the bubble, but not a "1" set along it."""
    if 0.15 * r <= g.length <= 1.2 * r:
        return False
    for vx, vy in dirs:
        if all(abs((p[0] - c[0]) * vy - (p[1] - c[1]) * vx) < 0.05 * r for p in (g.a, g.b)):
            return True
    return False


def _collinear(g: Line, h: Line, tol: float = 0.08) -> bool:
    ux, uy = (g.b[0] - g.a[0]) / g.length, (g.b[1] - g.a[1]) / g.length
    return all(abs((p[0] - g.a[0]) * uy - (p[1] - g.a[1]) * ux) < tol for p in (h.a, h.b))


def _extend(seed: set[int], lines: dict[int, Shape], ends: Grid) -> set[int]:
    """Follow an axis from its bubble through the collinear pieces that continue it."""
    out = set(seed)
    todo = list(seed)
    while todo:
        cur = lines[todo.pop()]
        g = cur.segs[0]
        for p in (g.a, g.b):
            for sid, _k in ends.query_point(p, MAX_GAP + MAX_PIECE / 2):
                if sid in out or sid not in lines:
                    continue
                o = lines[sid]
                # A dash pattern rarely stops exactly at the lead: allow up to one pattern repeat.
                gap = MAX_GAP + min(sum(o.style.dashes or ()), MAX_PIECE / 2)
                # Walls and extension lines often run along an axis; only its dashes continue it.
                # The lead may be drawn in another pen than the dashes, so only dash to dash must match.
                if not o.style.dashes:
                    continue
                if cur.style.dashes and (o.style.color, o.style.width) != (cur.style.color, cur.style.width):
                    continue
                h = o.segs[0]
                if h.length < 1.0 or not _collinear(g, h):
                    continue
                if min(math.dist(p, h.a), math.dist(p, h.b)) > gap:
                    continue
                out.add(sid)
                todo.append(sid)
    return out


def _count_runs(ids: set[int], lines: dict[int, Shape]) -> int:
    """Axis lines counted once however many pieces each is drawn in."""
    parent = {i: i for i in ids}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    items = list(ids)
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if root(a) != root(b) and _collinear(lines[a].segs[0], lines[b].segs[0]):
                parent[root(a)] = root(b)
    return len({root(i) for i in ids})


def _read(eng, shapes: list[Shape]) -> tuple[str, float]:
    import numpy as np

    from .ocrtext import _render

    bs = [s.bbox() for s in shapes]
    x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
    x1, y1 = max(b[2] for b in bs), max(b[3] for b in bs)
    h = y1 - y0
    if h <= 0.3:
        return "", 0.0
    pad = 0.6 * h
    img = _render(shapes, (x0 - pad, y0 - pad, x1 + pad, y1 + pad), False)
    res, _ = eng(np.array(img.convert("RGB")), use_det=False, use_cls=False, use_rec=True)
    if not res:
        return "", 0.0
    text, score = res[0][0], float(res[0][1])
    if score < 0.5:
        return "", 0.0
    return _clean_label(text), h
