"""SVG previews of the source PDF page and of the converted DXF."""

from __future__ import annotations

import re

import pymupdf
from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
from ezdxf.addons.drawing.properties import LayoutProperties
from ezdxf.document import Drawing as DXFDocument
from ezdxf.math import BoundingBox2d

# Colours for the "what was recognized" view, keyed by DXF entity type.
HIGHLIGHT = {
    "DIMENSION": "#2563eb",
    "LEADER": "#2563eb",
    "HATCH": "#ea580c",
    "INSERT": "#16a34a",
    "TEXT": "#9333ea",
    "MTEXT": "#9333ea",
}
PLAIN = "#9ca3af"


def pdf_page_svg(data: bytes, page: int) -> str:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return doc[page].get_svg_image(text_as_path=True)


PREVIEW_FONTS = ("simsun.ttc", "NotoSansCJK-Regular.ttc", "wqy-microhei.ttc", "DroidSansFallbackFull.ttf")


def _preview_font() -> str | None:
    """A CJK font the renderer can find on this machine; CAD itself resolves simsun.ttc."""
    from ezdxf.fonts import fonts

    for name in PREVIEW_FONTS:
        try:
            if fonts.font_manager.has_font(name):
                return name
        except Exception:
            continue
    return None


def dxf_svg(doc: DXFDocument, width_mm: float, height_mm: float, scale: float = 1.0, highlight: bool = False) -> str:
    """Render model space framed to the sheet: (0, 0) to the page size times the drawing scale."""
    style = doc.styles.get("PDF_TEXT")
    original = style.dxf.font if style is not None else None
    font = _preview_font()
    if style is not None and font:
        style.dxf.font = font
    try:
        return _render(doc, width_mm, height_mm, scale, highlight)
    finally:
        if style is not None and original:
            style.dxf.font = original


def _render(doc: DXFDocument, width_mm: float, height_mm: float, scale: float, highlight: bool) -> str:
    msp = doc.modelspace()
    ctx = RenderContext(doc)
    backend = svg.SVGBackend()
    cfg = config.Configuration(
        background_policy=config.BackgroundPolicy.WHITE,
        color_policy=config.ColorPolicy.COLOR,
        lineweight_policy=config.LineweightPolicy.ABSOLUTE,
        lineweight_scaling=1.0,
    )
    frontend = Frontend(ctx, backend, config=cfg)
    if highlight:
        top_level: dict[int, str] = {}

        def recolor(entity, props) -> None:
            kind = top_level.get(id(entity))
            props.color = HIGHLIGHT.get(kind or entity.dxftype(), PLAIN)

        for e in msp:
            kind = e.dxftype()
            if kind in HIGHLIGHT:
                for v in _virtual(e):
                    top_level[id(v)] = kind
        frontend.push_property_override_function(recolor)
        backend.set_background("#ffffff")
        frontend.draw_entities(_expanded(msp, top_level))
        backend.finalize()
    else:
        props = LayoutProperties.from_layout(msp)
        props.set_colors("#ffffff")
        frontend.draw_layout(msp, layout_properties=props)
    page = layout.Page(width_mm, height_mm, layout.Units.mm, margins=layout.Margins.all(0))
    settings = layout.Settings(fit_page=True, min_stroke_width=0.08, max_stroke_width=0.004)
    box = BoundingBox2d([(0, 0), (width_mm * scale, height_mm * scale)])
    return _inline_classes(backend.get_string(page, settings=settings, render_box=box))


_STYLE = re.compile(r"<style>\.(C[0-9A-Za-z]+) \{([^}]*)\}</style>")
_CLASS = re.compile(r'class="(C[0-9A-Za-z]+)"')


def _inline_classes(s: str) -> str:
    """Move CSS class rules onto the elements; some SVG viewers ignore <style>."""
    rules = {m.group(1): m.group(2).strip() for m in _STYLE.finditer(s)}
    s = _STYLE.sub("", s)
    return _CLASS.sub(lambda m: f'style="{rules.get(m.group(1), "")}"', s)


_cache: dict[int, list] = {}


def _virtual(e) -> list:
    key = id(e)
    if key not in _cache:
        try:
            _cache[key] = list(e.virtual_entities())
        except Exception:
            _cache[key] = []
    return _cache[key]


def _expanded(msp, top_level: dict[int, str]):
    """Yield model-space entities, replacing highlighted composites by their parts."""
    for e in msp:
        kind = e.dxftype()
        if kind in ("DIMENSION", "INSERT", "LEADER"):
            parts = _virtual(e)
            if parts:
                for v in parts:
                    if v.dxftype() == "INSERT":
                        for vv in v.virtual_entities():
                            top_level[id(vv)] = kind
                            yield vv
                    else:
                        yield v
                continue
        yield e
    _cache.clear()
