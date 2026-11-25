"""Conversion pipeline: PDF page -> recognized drawing -> DXF document."""

from __future__ import annotations

import io
import math
from typing import Callable
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from statistics import median

import pymupdf
from ezdxf.document import Drawing as DXFDocument

from . import blocks, dims, hatch, ocrtext
from .dims import parse_value
from .extract import extract_page
from .geom import dot, line_angle
from .model import COMMON_SCALES, Drawing, Line, Shape, Style
from .writer import Writer


@dataclass
class Options:
    page: int = 0
    scale: float | None = None  # model units per paper millimetre; None = detect from dimensions
    dimensions: bool = True
    hatches: bool = True
    blocks: bool = True
    text_height_factor: float = 0.7
    max_arrow_size: float = 8.0
    min_hatch_lines: int = 4
    # Read outlined/stroked lettering with OCR: True, False, or "auto" (when the page
    # has almost no real text and the OCR extra is installed).
    ocr: bool | str = "auto"
    progress: Callable[[float], None] | None = None


@dataclass
class Report:
    page: int
    page_count: int
    page_size_mm: tuple[float, float]
    scale: float
    scale_source: str
    pdf_layers: bool
    layers: list[str] = field(default_factory=list)
    entities: dict[str, int] = field(default_factory=dict)
    recognized: dict[str, int] = field(default_factory=dict)
    block_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    doc: DXFDocument
    report: Report
    drawing: Drawing

    def dxf_bytes(self) -> bytes:
        buf = io.StringIO()
        self.doc.write(buf)
        return buf.getvalue().encode("utf-8")


def dedupe_lines(d: Drawing, consumed: set[int]) -> int:
    """Remove single lines that lie on top of another line of the same style."""
    groups: dict[tuple, list[Shape]] = defaultdict(list)
    for s in d.shapes:
        if s.id in consumed or not s.is_single_line or not s.stroke:
            continue
        g = s.segs[0]
        a = round(line_angle(g.start(), g.end()), 1) % 180
        r = math.radians(a)
        off = round(dot(g.start(), (-math.sin(r), math.cos(r))), 2)
        groups[(s.style, a, off)].append(s)
    removed = 0
    for (_st, a, _off), lines in groups.items():
        if len(lines) < 2:
            continue
        r = math.radians(a)
        u = (math.cos(r), math.sin(r))
        spans = []
        for s in lines:
            p, q = dot(s.segs[0].start(), u), dot(s.segs[0].end(), u)
            spans.append((min(p, q), max(p, q), s))
        spans.sort(key=lambda x: (x[0], -x[1]))
        keep_hi = -math.inf
        for lo, hi, s in spans:
            if hi <= keep_hi + 1e-3:
                consumed.add(s.id)
                removed += 1
            else:
                keep_hi = max(keep_hi, hi)
    return removed


def strips_to_lines(d: Drawing, consumed: set[int]) -> int:
    """Turn long thin filled rectangles (heavy lines exported as outlines) back into lines."""
    n = 0
    for s in d.shapes:
        if s.id in consumed or s.fill is None or not s.closed:
            continue
        if not 4 <= len(s.segs) <= 5 or not all(isinstance(g, Line) for g in s.segs):
            continue
        pts: list = []
        for g in s.segs:
            p = g.start()
            if not pts or math.dist(pts[-1], p) > 1e-3:
                pts.append(p)
        if len(pts) == 5 and math.dist(pts[0], pts[-1]) <= 1e-3:
            pts.pop()
        if len(pts) != 4:
            continue
        sides = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        if abs(sides[0] - sides[2]) > 0.02 or abs(sides[1] - sides[3]) > 0.02:
            continue
        k = 0 if sides[0] < sides[1] else 1
        w, length = sides[k], sides[k + 1]
        if w <= 0 or w > 0.8 or length < 3.0 or length / w < 12:
            continue
        a = ((pts[k][0] + pts[k + 1][0]) / 2, (pts[k][1] + pts[k + 1][1]) / 2)
        b = ((pts[k + 2][0] + pts[(k + 3) % 4][0]) / 2, (pts[k + 2][1] + pts[(k + 3) % 4][1]) / 2)
        s.segs = [Line(a, b)]
        s.closed = False
        s.style = Style(s.style.layer, s.fill, round(w, 2), s.style.dashes)
        s.fill = None
        s.stroke = True
        n += 1
    return n


def detect_scale(d: Drawing) -> float | None:
    ratios = []
    for dim in d.linear_dims:
        pv = parse_value(dim.text)
        if pv and dim.measured > 0.5 and pv[0] > 0:
            ratios.append(pv[0] / dim.measured)
    for dim in d.radial_dims:
        pv = parse_value(dim.text)
        if pv and pv[0] > 0:
            ratios.append(pv[0] / (dim.radius * (2 if dim.kind == "diameter" else 1)))
    if not ratios:
        return None
    m = median(ratios)
    agree = [r for r in ratios if abs(r / m - 1) < 0.01]
    if len(agree) < max(1, 0.6 * len(ratios)):
        return None
    m = median(agree)
    for c in COMMON_SCALES:
        for v in (c, 1 / c):
            if abs(m / v - 1) < 0.01:
                return float(v)
    return float(f"{m:.6g}")


def convert_page(doc: pymupdf.Document, opts: Options) -> Result:
    if not 0 <= opts.page < doc.page_count:
        raise ValueError(f"页码超出范围: 共 {doc.page_count} 页")
    page = doc[opts.page]
    d = extract_page(page, opts.text_height_factor)
    consumed: set[int] = set()
    used_texts: set[int] = set()
    warnings: list[str] = []

    if not d.shapes and not d.texts:
        warnings.append("这一页没有矢量路径或文字对象，可能是扫描件或图片，无法转换。")
    if d.shapes and not d.texts:
        if opts.ocr is False or not ocrtext.available():
            warnings.append("页面里没有文字对象。如果图上能看到文字，说明导出时文字被转成了轮廓，只能作为图形保留。")
        else:
            warnings.append("页面里没有文字对象，图上的文字是转曲的轮廓或笔画，已尝试用文字识别还原。")
    if page.get_images():
        warnings.append("页面含有位图图片，图片内容不会被转换。")

    if opts.dimensions:
        dims.recognize(d, consumed, used_texts, opts.max_arrow_size)
    ocr_counts = None
    want_ocr = opts.ocr is True or (opts.ocr == "auto" and len(d.texts) < 3 and len(d.shapes) > 0)
    if want_ocr:
        if ocrtext.available():
            ocr_counts = ocrtext.recognize(d, consumed, opts.progress)
            if opts.dimensions and ocr_counts["text"]:
                hint = (opts.scale if opts.scale and opts.scale > 0
                        else getattr(d, "glyph_scale", None) or detect_scale(d))
                dims.recognize(d, consumed, used_texts, opts.max_arrow_size, scale=hint, again=True)
        elif opts.ocr is True:
            warnings.append("没有安装文字识别组件（rapidocr-onnxruntime），转曲文字未识别。")
    strips_to_lines(d, consumed)
    if opts.hatches:
        hatch.recognize_solid(d, consumed)
        hatch.recognize_pattern(d, consumed, opts.min_hatch_lines)
    dup = dedupe_lines(d, consumed)
    if opts.blocks:
        blocks.recognize(d, consumed)

    if opts.scale and opts.scale > 0:
        scale, source = float(opts.scale), "manual"
    else:
        auto = detect_scale(d)
        scale, source = (auto, "dimensions") if auto else (1.0, "default")
        if auto is None and d.linear_dims:
            warnings.append("标注数值和图上长度对不上同一个比例，已按 1:1（图纸毫米）输出，可手动指定比例。")
        if auto and auto >= 1 and not any(abs(auto / c - 1) < 1e-6 for c in COMMON_SCALES):
            warnings.append(
                f"按标注数值反推出的比例是 1:{auto:g}，不是标准比例，PDF 可能是缩放打印的（例如“适合纸张”）。"
                "已按这个比例还原真实尺寸。"
            )
    if ocr_counts and ocr_counts["text"]:
        msg = f"用文字识别把 {ocr_counts['text']} 处转曲文字还原成了可编辑文字。"
        if ocr_counts["review"]:
            msg += f"其中 {ocr_counts['review']} 处把握不大，放在“PDF_文字_待核对”图层，请在 CAD 里核对。"
        if getattr(d, "ocr_dims", 0):
            msg += f"识别出的数字里有 {d.ocr_dims} 个和图上尺寸对得上，已重建为标注。"  # type: ignore[attr-defined]
        warnings.append(msg)
    glyph_dims = getattr(d, "glyph_dims", 0)
    if glyph_dims:
        warnings.append(f"有 {glyph_dims} 个标注的数字是转曲的笔画，已按字形自动识别并重建为标注对象。")

    if not d.has_pdf_layers:
        warnings.append("PDF 没有图层信息，已按颜色和线宽分层，标注、文字、填充各自单独成层。")
    if dup:
        warnings.append(f"去掉了 {dup} 条完全重叠的重复线。")

    w = Writer(d, scale, layer_by_type=not d.has_pdf_layers)
    dxf = w.write(consumed, used_texts)

    entities = Counter(e.dxftype() for e in dxf.modelspace())
    recognized = {
        "线性标注": len(d.linear_dims),
        "半径/直径标注": len(d.radial_dims),
        "引线": len(d.leaders),
        "实心填充": sum(1 for h in d.hatches if h.solid),
        "图案填充": sum(1 for h in d.hatches if not h.solid),
        "图块定义": len(d.blocks),
        "图块引用": len(d.inserts),
        "文字": sum(1 for t in d.texts if t.id not in used_texts) + len(d.leaders),
        "圆": w.counts.get("CIRCLE", 0) + w.block_entity_counts.get("CIRCLE", 0),
        "圆弧": w.counts.get("ARC", 0) + w.block_entity_counts.get("ARC", 0),
        "样条曲线": w.counts.get("SPLINE", 0) + w.block_entity_counts.get("SPLINE", 0),
    }
    report = Report(
        page=opts.page, page_count=doc.page_count,
        page_size_mm=(round(d.width, 1), round(d.height, 1)), scale=scale, scale_source=source,
        pdf_layers=d.has_pdf_layers, layers=sorted(l.dxf.name for l in dxf.layers),
        entities=dict(sorted(entities.items())), recognized=recognized,
        block_names=[b.name for b in d.blocks], warnings=warnings,
    )
    return Result(dxf, report, d)


def convert_bytes(data: bytes, opts: Options | None = None) -> Result:
    opts = opts or Options()
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return convert_page(doc, opts)


def convert_file(path: str, opts: Options | None = None) -> Result:
    opts = opts or Options()
    with pymupdf.open(path) as doc:
        return convert_page(doc, opts)
