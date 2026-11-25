"""Turn text that was exported as strokes or filled outlines back into TEXT, using OCR.

Only small shapes are rendered (glyphs are small; walls, axes and frames are not), so
the recognizer sees little besides the lettering.  Each detected line of text is mapped
back to the shapes inside its box, which are then replaced by one TEXT.  A piece of
lettering that appears many times is drawn identically every time, so identical runs
vote on a common reading, which fixes most single misreadings.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Callable

from .dims import _stroke_axis
from .model import Drawing, Shape, TextItem
from .spatial import Grid

PX = 14.0  # render resolution, pixels per paper millimetre
TILE = 2400  # tile size in pixels
OVERLAP = 25.0  # tile overlap in millimetres; longer lines are read from the tile that holds them
MAX_GLYPH = 12.0
MIN_GLYPH = 0.3
MIN_SCORE = 0.6
REVIEW_SCORE = 0.85
CJK = re.compile(r"[\u3400-\u9fff]")
MEANINGFUL = re.compile(r"[0-9A-Za-z\u3400-\u9fff]")
SYMBOLS = set("十一二丨|+-—_=口〇○()[]{}<>/\\.·、，,:：;；'\"`~^*#×xX冏囧。．")
EDGE_JUNK = "[]|丨+十。．冏囧"
# Misreadings that recur with CAD lettering, and the reading meant in a building drawing.
CORRECTIONS = [
    (re.compile(r"至项"), "至顶"),
    (re.compile(r"项板"), "顶板"),
    (re.compile(r"(?<![A-Za-z])(F?M)Z(?=\d)"), r"\1乙"),
    (re.compile(r"(?<![A-Za-z])(F?M)[甲申由](?=\d)"), r"\1甲"),
    (re.compile(r"[再而币西](?=水管|水斗|水口)"), "雨"),
    (re.compile(r"羊见"), "详见"),
    (re.compile(r"留润"), "留洞"),
    (re.compile(r"(?<=[\u4e00-\u9fff])问"), "间"),
    (re.compile(r"(?<=[\d.])[OoQ]+(?=[^A-Za-z]|[Mm](?![a-z])|$)"), lambda m: "0" * len(m.group(0))),
    (re.compile(r"^[Il|](?=CH=)"), ""),
]

_engine = None


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


@dataclass
class Found:
    text: str
    score: float
    shapes: list[Shape]
    vertical: bool
    x0: float  # glyph extent in the reading frame (mm)
    y0: float
    x1: float
    y1: float


def _to_frame(p, vertical: bool):
    """Page mm -> reading frame: x along the text, y towards the glyph tops."""
    return (p[1], -p[0]) if vertical else (p[0], p[1])


def _from_frame(q, vertical: bool):
    return (-q[1], q[0]) if vertical else (q[0], q[1])


def _frame_bbox(s: Shape, vertical: bool):
    pts = [_to_frame(p, vertical) for p in s.points()]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _render(shapes: list[Shape], box, vertical: bool):
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = box
    w, h = max(1, int((x1 - x0) * PX)), max(1, int((y1 - y0) * PX))
    img = Image.new("L", (w, h), 255)
    dr = ImageDraw.Draw(img)

    def tp(p):
        q = _to_frame(p, vertical)
        return ((q[0] - x0) * PX, (y1 - q[1]) * PX)

    for s in shapes:
        pts = [tp(p) for p in s.points()]
        if s.fill is not None and len(pts) > 2:
            dr.polygon(pts, fill=0)
        if s.stroke and len(pts) > 1:
            if s.closed:
                pts = pts + [pts[0]]
            dr.line(pts, fill=0, width=max(1, round(s.style.width * PX)))
    return img


def _candidates(d: Drawing, consumed: set[int]) -> list[Shape]:
    out = []
    for s in d.shapes:
        if s.id in consumed:
            continue
        b = s.bbox()
        size = max(b[2] - b[0], b[3] - b[1])
        if size > MAX_GLYPH:
            continue
        # Tiny fills are stipple dots; tiny strokes are full stops and decimal points.
        if size < MIN_GLYPH and (s.fill is not None or not s.stroke):
            continue
        if s.is_single_line and size > 8.0:
            continue
        out.append(s)
    return out


def _plausible(text: str, score: float) -> bool:
    if score < MIN_SCORE or not MEANINGFUL.search(text):
        return False
    if all(ch in SYMBOLS or ch.isspace() for ch in text):
        return False  # "+" planting marks read as 十, dashes as 一, and so on
    if len(text) == 1 and text.isascii() and text.isalpha() and score < 0.95:
        return False
    return True


class _Frame:
    """Candidate shapes indexed in one reading orientation."""

    def __init__(self, cands: list[Shape], vertical: bool) -> None:
        self.vertical = vertical
        self.boxes = {s.id: _frame_bbox(s, vertical) for s in cands}
        self.by_id = {s.id: s for s in cands}
        self.grid: Grid[int] = Grid(10.0)
        for sid, b in self.boxes.items():
            self.grid.insert(b, sid)

    def inside(self, region, exclude: set[int]) -> list[Shape]:
        x0, y0, x1, y1 = region
        out = []
        for sid in self.grid.query(region):
            b = self.boxes[sid]
            if sid not in exclude and b[0] >= x0 and b[2] <= x1 and b[1] >= y0 and b[3] <= y1:
                out.append(self.by_id[sid])
        return out

    def glyphs(self, box, exclude: set[int]) -> list[Shape]:
        pad = 0.25
        bx0, by0, bx1, by1 = box
        gs = self.inside((bx0 - pad, by0 - pad, bx1 + pad, by1 + pad), exclude)
        if not gs:
            return []
        # Table borders and leader lines that cross the box are not part of the text.
        shaped = [s for s in gs if not s.is_single_line]
        ref = max((self.boxes[s.id][3] - self.boxes[s.id][1] for s in shaped), default=by1 - by0)
        keep = []
        for s in gs:
            if s.is_single_line:
                b = self.boxes[s.id]
                if max(b[2] - b[0], b[3] - b[1]) > 1.15 * ref:
                    continue
            keep.append(s)
        return keep


def _read(eng, frame: _Frame, region, exclude: set[int]):
    """OCR the candidate shapes inside region; yields (text, score, box) in frame mm."""
    import numpy as np

    shapes = frame.inside(region, exclude)
    if not shapes:
        return
    x0, y0, x1, y1 = region
    img = _render(shapes, region, frame.vertical)
    res, _ = eng(np.array(img.convert("RGB")))
    for quad, text, score in res or []:
        qx = [x0 + p[0] / PX for p in quad]
        qy = [y1 - p[1] / PX for p in quad]
        yield text.strip(), float(score), (min(qx), min(qy), max(qx), max(qy))


def clean(text: str) -> str:
    text = text.strip().strip(EDGE_JUNK).strip()
    for pat, rep in CORRECTIONS:
        text = pat.sub(rep, text)
    return text


def _accept(found: list[Found], frame: _Frame, text: str, score: float, box, claimed: set[int]) -> bool:
    text = clean(text)
    if not _plausible(text, score):
        return False
    glyphs = frame.glyphs(box, claimed)
    if not glyphs:
        return False
    # Ticks or symbols caught in the box would inflate the text height: keep glyph-sized shapes.
    core = [s for s in glyphs if not s.is_single_line and _stroke_axis(s) is None]
    if len(core) >= 2:
        cb = [frame.boxes[s.id] for s in core]
        cy0, cy1 = min(b[1] for b in cb), max(b[3] for b in cb)
        slack = 0.15 * (cy1 - cy0)
        ids = {s.id for s in core}
        glyphs = core + [s for s in glyphs if s.id not in ids
                         and frame.boxes[s.id][1] >= cy0 - slack and frame.boxes[s.id][3] <= cy1 + slack]
        bs = cb
    else:
        bs = [frame.boxes[s.id] for s in glyphs]
    gx0, gy0 = min(b[0] for b in bs), min(b[1] for b in bs)
    gx1, gy1 = max(b[2] for b in bs), max(b[3] for b in bs)
    if gy1 - gy0 < 0.8:
        return False
    claimed.update(s.id for s in glyphs)
    found.append(Found(text, score, glyphs, frame.vertical, gx0, gy0, gx1, gy1))
    return True


def _horizontal_pass(eng, frame: _Frame, claimed: set[int], progress, p0: float, p1: float):
    found: list[Found] = []
    upright: list[tuple] = []
    cut: list[tuple] = []
    bs = frame.boxes.values()
    X0, Y0 = min(b[0] for b in bs), min(b[1] for b in bs)
    X1, Y1 = max(b[2] for b in bs), max(b[3] for b in bs)
    span = TILE / PX
    step = span - OVERLAP
    tiles = [(x, y, x + span, y + span)
             for y in _frange(Y0, Y1, step) for x in _frange(X0, X1, step)]
    for ti, (tx0, ty0, tx1, ty1) in enumerate(tiles):
        if progress:
            progress(p0 + (p1 - p0) * ti / max(1, len(tiles)))
        for text, score, (bx0, by0, bx1, by1) in _read(eng, frame, (tx0, ty0, tx1, ty1), claimed):
            m = 1.0
            if ((bx0 < tx0 + m and tx0 > X0) or (by0 < ty0 + m and ty0 > Y0)
                    or (bx1 > tx1 - m and tx1 < X1) or (by1 > ty1 - m and ty1 < Y1)):
                cut.append((bx0, by0, bx1, by1))
                continue
            if (by1 - by0) > 1.5 * (bx1 - bx0):
                upright.append((bx0, by0, bx1, by1))  # text set at 90°: read it turned
                if len(text) > 1:
                    continue
            _accept(found, frame, text, score, (bx0, by0, bx1, by1), claimed)
    # Lines cut by a tile edge: read the joined pieces once more as a whole.
    for bx0, by0, bx1, by1 in _merge_boxes(cut):
        region = (bx0 - 3, by0 - 1.5, bx1 + 3, by1 + 1.5)
        for text, score, box in _read(eng, frame, region, claimed):
            if (box[3] - box[1]) <= 1.5 * (box[2] - box[0]) or len(text) == 1:
                _accept(found, frame, text, score, box, claimed)
    return found, upright


def _vertical_pass(eng, frame: _Frame, boxes: list[tuple], claimed: set[int], progress, p0: float, p1: float):
    found: list[Found] = []
    merged = _merge_boxes(boxes)
    for i, (px0, py0, px1, py1) in enumerate(merged):
        if progress:
            progress(p0 + (p1 - p0) * i / max(1, len(merged)))
        pad = 1.5
        region = (py0 - pad, -px1 - pad, py1 + pad, -px0 + pad)
        for text, score, box in _read(eng, frame, region, claimed):
            if (box[3] - box[1]) > 1.5 * (box[2] - box[0]):
                continue
            _accept(found, frame, text, score, box, claimed)
    return found


def _merge_boxes(boxes: list[tuple]) -> list[tuple]:
    out: list[list[float]] = []
    for b in sorted(boxes):
        for o in out:
            if b[0] <= o[2] + 1 and b[2] >= o[0] - 1 and b[1] <= o[3] + 1 and b[3] >= o[1] - 1:
                o[0], o[1], o[2], o[3] = min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])
                break
        else:
            out.append(list(b))
    return [tuple(o) for o in out]


def _frange(a: float, b: float, step: float):
    v = a
    while v < b:
        yield v
        v += step


def _signature(f: Found) -> tuple:
    keys = []
    for s in f.shapes:
        b = _frame_bbox(s, f.vertical)
        keys.append((len(s.segs), s.fill is not None,
                     round((b[0] - f.x0) / 0.1), round((b[1] - f.y0) / 0.1),
                     round((b[2] - b[0]) / 0.1), round((b[3] - b[1]) / 0.1)))
    return (round((f.y1 - f.y0) / 0.1), tuple(sorted(keys)))


def _vote(found: list[Found]) -> None:
    groups: dict[tuple, list[Found]] = defaultdict(list)
    for f in found:
        groups[_signature(f)].append(f)
    for fs in groups.values():
        if len(fs) < 2:
            continue
        tally: Counter = Counter()
        for f in fs:
            tally[f.text] += f.score
        best, weight = tally.most_common(1)[0]
        share = weight / sum(tally.values())
        for f in fs:
            if f.text != best:
                f.text = best
                f.score = max(f.score, share)
            else:
                f.score = max(f.score, min(1.0, share + 0.1 * (len(fs) - 1)))


def recognize(d: Drawing, consumed: set[int], progress: Callable[[float], None] | None = None) -> dict:
    """Replace outlined/stroked lettering with TEXT items. Returns counts for the report."""
    cands = _candidates(d, consumed)
    if not cands:
        return {"text": 0, "review": 0}
    eng = _get_engine()
    claimed: set[int] = set()
    found, upright = _horizontal_pass(eng, _Frame(cands, False), claimed, progress, 0.0, 0.85)
    found += _vertical_pass(eng, _Frame(cands, True), upright, claimed, progress, 0.85, 1.0)
    _vote(found)
    if found:
        # Door swings and similar loops read as "0"/"O" stand out by their size.
        mh = median(f.y1 - f.y0 for f in found)
        found = [f for f in found
                 if not (re.fullmatch(r"[0Oo]{1,2}", f.text) and f.y1 - f.y0 > 1.5 * mh)]
    next_id = max((t.id for t in d.texts), default=-1) + 1
    review = 0
    for f in found:
        gh = f.y1 - f.y0
        has_cjk = bool(CJK.search(f.text))
        height = gh / 1.15 if has_cjk else gh
        insert = _from_frame((f.x0, f.y0), f.vertical)
        styles = Counter((s.fill if s.fill is not None else s.style.color) for s in f.shapes)
        t = TextItem(
            text=f.text, insert=insert, height=round(height, 3), rotation=90.0 if f.vertical else 0.0,
            layer=None, color=styles.most_common(1)[0][0], font="ocr",
            bbox=(0, 0, 0, 0), width=f.x1 - f.x0, id=next_id,
        )
        t.review = f.score < REVIEW_SCORE  # type: ignore[attr-defined]
        review += t.review  # type: ignore[attr-defined]
        next_id += 1
        d.texts.append(t)
        consumed.update(s.id for s in f.shapes)
    return {"text": len(found), "review": review}


__all__ = ["available", "recognize"]
