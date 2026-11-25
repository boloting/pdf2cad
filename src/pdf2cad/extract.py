"""Read vector paths and text from a PDF page into the intermediate model."""

from __future__ import annotations

import math
import re

import pymupdf

from .geom import (
    Vec, angle_of, circumcenter, cross, dist, length, point_line_distance, sub, unit,
)
from .model import Arc, Bezier, Drawing, Line, Seg, Shape, Style, TextItem

PT = 25.4 / 72  # PDF points to millimetres
EPS = 1e-3  # mm; points closer than this are the same point


def _parse_dashes(s: str | None) -> tuple[float, ...]:
    if not s:
        return ()
    m = re.match(r"\s*\[([^\]]*)\]", s)
    if not m:
        return ()
    vals = [float(v) * PT for v in m.group(1).replace(",", " ").split()]
    if not vals or all(v <= 1e-6 for v in vals):
        return ()
    if len(vals) % 2:
        vals = vals * 2
    return tuple(round(v, 4) for v in vals)


def _color(c) -> tuple[float, float, float] | None:
    if c is None:
        return None
    if isinstance(c, (int, float)):
        return (float(c),) * 3
    if len(c) == 1:
        return (float(c[0]),) * 3
    if len(c) == 4:  # CMYK
        k = c[3]
        return tuple(round((1 - min(1.0, v + k)), 4) for v in c[:3])  # type: ignore[return-value]
    return tuple(round(float(v), 4) for v in c[:3])  # type: ignore[return-value]


def bezier_to_arc(b: Bezier, rel_tol: float = 0.002, abs_tol: float = 0.005) -> Seg:
    """Return a Line or Arc when the cubic Bézier is one, else the Bézier itself."""
    chord = dist(b.p0, b.p3)
    span = max(chord, dist(b.p0, b.p1), dist(b.p2, b.p3))
    if span < EPS:
        return Line(b.p0, b.p3)
    if (
        chord > EPS
        and point_line_distance(b.p1, b.p0, b.p3) < abs_tol
        and point_line_distance(b.p2, b.p0, b.p3) < abs_tol
    ):
        return Line(b.p0, b.p3)
    if chord < EPS:
        return b
    pm = b.at(0.5)
    c = circumcenter(b.p0, pm, b.p3)
    if c is None:
        return b
    r = dist(c, b.p0)
    tol = max(abs_tol, r * rel_tol)
    for t in (0.125, 0.25, 0.375, 0.625, 0.75, 0.875):
        if abs(dist(c, b.at(t)) - r) > tol:
            return b
    # The control polygon must leave the endpoints tangentially, otherwise an
    # S-curve through three points on a circle would pass.
    for p, q in ((b.p0, b.p1), (b.p3, b.p2)):
        if dist(p, q) > EPS:
            rad = unit(sub(p, c))
            tan = unit(sub(q, p))
            if abs(rad[0] * tan[0] + rad[1] * tan[1]) > 0.05:
                return b
    a0 = angle_of(sub(b.p0, c))
    am = angle_of(sub(pm, c))
    a3 = angle_of(sub(b.p3, c))
    ccw = (a3 - a0) % 360
    if (am - a0) % 360 < ccw:
        sweep = ccw
    else:
        sweep = -((a0 - a3) % 360)
    return Arc(c, r, a0, sweep)


def _same_circle(a: Arc, b: Arc) -> bool:
    tol = max(0.01, a.radius * 0.003)
    return dist(a.center, b.center) < tol and abs(a.radius - b.radius) < tol and (a.sweep > 0) == (b.sweep > 0)


def simplify(segs: list[Seg], closed: bool) -> list[Seg]:
    """Turn Béziers into arcs where possible and join arcs of one circle and collinear lines."""
    out: list[Seg] = []
    for s in segs:
        if isinstance(s, Bezier):
            s = bezier_to_arc(s)
        if isinstance(s, Line) and s.length < EPS:
            continue
        if out:
            prev = out[-1]
            if isinstance(prev, Arc) and isinstance(s, Arc) and _same_circle(prev, s):
                out[-1] = Arc(prev.center, (prev.radius + s.radius) / 2, prev.start_angle, prev.sweep + s.sweep)
                continue
            if isinstance(prev, Line) and isinstance(s, Line):
                d1, d2 = sub(prev.b, prev.a), sub(s.b, s.a)
                if abs(cross(unit(d1), unit(d2))) < 1e-4 and d1[0] * d2[0] + d1[1] * d2[1] > 0:
                    out[-1] = Line(prev.a, s.b)
                    continue
        out.append(s)
    if closed and len(out) > 1:
        first, last = out[0], out[-1]
        if isinstance(first, Arc) and isinstance(last, Arc) and _same_circle(first, last):
            out[0] = Arc(last.center, last.radius, last.start_angle, last.sweep + first.sweep)
            out.pop()
    for i, s in enumerate(out):
        if isinstance(s, Arc) and abs(s.sweep) > 360:
            out[i] = Arc(s.center, s.radius, s.start_angle, math.copysign(360.0, s.sweep))
    return out


def extract_shapes(page: pymupdf.Page) -> tuple[list[Shape], bool]:
    # Drawing coordinates are in the unrotated page space; rotate them to the
    # page as it is displayed.
    h = page.rect.height
    m = page.rotation_matrix

    ma, mb, mc, md, me, mf = m.a, m.b, m.c, m.d, m.e, m.f

    def cv(p) -> Vec:
        x, y = p[0], p[1]
        return ((x * ma + y * mc + me) * PT, (h - (x * mb + y * md + mf)) * PT)

    shapes: list[Shape] = []
    has_layers = False
    for pid, d in enumerate(page.get_drawings()):
        typ = d.get("type") or ""
        if typ not in ("s", "f", "fs"):
            continue
        stroke = "s" in typ
        fill = _color(d.get("fill")) if "f" in typ else None
        if (d.get("stroke_opacity") or 1) == 0 and fill is None:
            continue
        layer = d.get("layer") or None
        has_layers = has_layers or bool(layer)
        width = (d.get("width") or 0) * PT if stroke else 0.0
        color = _color(d.get("color")) if stroke else fill
        style = Style(layer, color, round(width, 4), _parse_dashes(d.get("dashes")) if stroke else ())

        subpaths: list[tuple[list[Seg], bool]] = []
        cur: list[Seg] = []

        def flush():
            nonlocal cur
            if cur:
                subpaths.append((cur, False))
            cur = []

        for it in d["items"]:
            op = it[0]
            if op == "l":
                seg: Seg = Line(cv(it[1]), cv(it[2]))
            elif op == "c":
                seg = Bezier(cv(it[1]), cv(it[2]), cv(it[3]), cv(it[4]))
            elif op in ("re", "qu"):
                flush()
                if op == "re":
                    r = it[1]
                    pts = [cv(r.tl), cv(r.tr), cv(r.br), cv(r.bl)]
                else:
                    q = it[1]
                    pts = [cv(q.ul), cv(q.ur), cv(q.lr), cv(q.ll)]
                subpaths.append(([Line(pts[i], pts[(i + 1) % 4]) for i in range(4)], True))
                continue
            else:
                continue
            if cur and dist(cur[-1].end(), seg.start()) > EPS:
                flush()
            cur.append(seg)
        flush()
        if d.get("closePath") and subpaths and not subpaths[-1][1]:
            subpaths[-1] = (subpaths[-1][0], True)

        for segs, closed in subpaths:
            closed = closed or fill is not None or (len(segs) > 1 and dist(segs[0].start(), segs[-1].end()) < EPS)
            if closed and dist(segs[0].start(), segs[-1].end()) > EPS:
                segs = segs + [Line(segs[-1].end(), segs[0].start())]
            segs = simplify(segs, closed)
            if not segs:
                continue
            if closed and len(segs) == 1 and not (isinstance(segs[0], Arc) and segs[0].is_full):
                closed = False
            shapes.append(Shape(segs, closed, style, stroke, fill, pid, len(shapes)))
    return shapes, has_layers


_REPLACE = {"\u2300": "Ø", "\uf06e": "Ø"}


def extract_texts(page: pymupdf.Page, height_factor: float = 0.7) -> list[TextItem]:
    """Group text-trace characters into single-line strings that share a baseline."""
    h = page.rect.height
    m = page.rotation_matrix
    runs: list[dict] = []
    cur: dict | None = None

    for span in page.get_texttrace():
        if span.get("type") == 3 or (span.get("opacity") or 1) == 0:
            continue
        d0 = span["dir"]
        dx, dy = d0[0] * m.a + d0[1] * m.c, d0[0] * m.b + d0[1] * m.d
        size = span["size"]
        layer = span.get("layer") or None
        color = _color(span.get("color"))
        for ch in span["chars"]:
            code, _glyph, origin, cb = ch[0], ch[1], ch[2], ch[3]
            c = chr(code) if code > 0 else "?"
            c = _REPLACE.get(c, c)
            op = pymupdf.Point(origin) * m
            o = (op.x, op.y)
            cb = tuple(pymupdf.Rect(cb) * m)
            key = (round(dx, 3), round(dy, 3), round(size, 2), layer, color)
            if cur is not None and cur["key"] == key:
                # PDF y points down: along = (dx, dy), up = (dy, -dx) in PDF space.
                rel = (o[0] - cur["end"][0], o[1] - cur["end"][1])
                along = rel[0] * dx + rel[1] * dy
                across = rel[0] * dy - rel[1] * dx
                if abs(across) < 0.15 * size and -0.35 * size < along < 1.2 * size:
                    cur["text"] += c
                    cur["bboxes"].append(cb)
                    cur["end"] = _char_end(o, cb, dx, dy)
                    continue
            cur = {
                "key": key, "text": c, "origin": o, "bboxes": [cb], "end": _char_end(o, cb, dx, dy),
                "size": size, "dir": (dx, dy), "layer": layer, "color": color, "font": span.get("font", ""),
            }
            runs.append(cur)

    items: list[TextItem] = []
    for r in runs:
        text = r["text"].rstrip()
        stripped = text.lstrip()
        if not stripped:
            continue
        dx, dy = r["dir"]
        x0 = min(b[0] for b in r["bboxes"])
        y0 = min(b[1] for b in r["bboxes"])
        x1 = max(b[2] for b in r["bboxes"])
        y1 = max(b[3] for b in r["bboxes"])
        o = r["origin"]
        lead = len(text) - len(stripped)
        if lead:
            first = r["bboxes"][lead]
            o = (first[0], o[1]) if abs(dx) >= abs(dy) else (o[0], first[3] if dy < 0 else first[1])
        rotation = math.degrees(math.atan2(-dy, dx)) % 360
        insert = (o[0] * PT, (h - o[1]) * PT)
        end = (r["end"][0] * PT, (h - r["end"][1]) * PT)
        items.append(TextItem(
            text=stripped, insert=insert, height=r["size"] * height_factor * PT, rotation=round(rotation, 4),
            layer=r["layer"], color=r["color"], font=r["font"],
            bbox=(x0 * PT, (h - y1) * PT, x1 * PT, (h - y0) * PT),
            width=length(sub(end, insert)), id=len(items),
        ))
    return items


def _char_end(origin, cb, dx, dy) -> tuple[float, float]:
    """Origin of the next character: the far edge of this glyph's box along the baseline."""
    # The box centre is the glyph rectangle's centre at any rotation, and it sits
    # half an advance along the baseline from the origin.
    cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
    adv = max(0.0, 2 * ((cx - origin[0]) * dx + (cy - origin[1]) * dy))
    return (origin[0] + dx * adv, origin[1] + dy * adv)


def extract_page(page: pymupdf.Page, text_height_factor: float = 0.7) -> Drawing:
    shapes, has_layers = extract_shapes(page)
    texts = extract_texts(page, text_height_factor)
    has_layers = has_layers or any(t.layer for t in texts)
    return Drawing(
        width=page.rect.width * PT, height=page.rect.height * PT,
        shapes=shapes, texts=texts, has_pdf_layers=has_layers,
    )
