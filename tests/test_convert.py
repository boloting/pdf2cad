import io
import math

import ezdxf
import pytest

from drawings import (
    axis_grid_pdf, centered_dims_pdf, crowded_dims_pdf, floor_fill_under_walls_pdf,
    outlined_dims_pdf, outlined_text_pdf, plan_pdf,
)
from pdf2cad.model import LinearDim
from pdf2cad.writer import infer_dimtad
from pdf2cad.convert import Options, convert_bytes
from pdf2cad.extract import bezier_to_arc
from pdf2cad.dims import parse_value
from pdf2cad.model import Arc, Bezier, Line
from pdf2cad.sample import sample_pdf_bytes


def reload(result):
    """Round-trip the DXF through text so we test what CAD would read."""
    return ezdxf.read(io.StringIO(result.dxf_bytes().decode("utf-8")))


@pytest.fixture(scope="module")
def sample():
    return convert_bytes(sample_pdf_bytes())


@pytest.fixture(scope="module")
def plan():
    return convert_bytes(plan_pdf())


def test_sample_scale_and_layers(sample):
    rep = sample.report
    assert rep.scale == 2.0 and rep.scale_source == "dimensions"
    assert rep.pdf_layers
    assert {"轮廓线", "中心线", "标注", "剖面线", "文字", "图框"} <= set(rep.layers)


def test_sample_dimensions_are_live(sample):
    doc = reload(sample)
    dims = doc.modelspace().query("DIMENSION")
    assert len(dims) == 5
    values = sorted(round(d.get_measurement()) for d in dims)
    assert values == [12, 80, 120, 160, 200]
    # Every PDF value matched the geometry, so the text stays associative.
    assert all("<>" in d.dxf.text for d in dims)
    dia = [d for d in dims if d.dimtype == 3]
    assert len(dia) == 1 and dia[0].dxf.text == "4×<>"


def test_sample_block_hatch_text(sample):
    doc = reload(sample)
    msp = doc.modelspace()
    inserts = msp.query("INSERT")
    assert len(inserts) == 4 and len({i.dxf.name for i in inserts}) == 1
    blk = doc.blocks.get(inserts[0].dxf.name)
    kinds = sorted(e.dxftype() for e in blk)
    assert kinds == ["CIRCLE", "LINE", "LINE"]
    circle = next(e for e in blk if e.dxftype() == "CIRCLE")
    assert circle.dxf.radius == pytest.approx(6.0, abs=1e-3)

    hatches = msp.query("HATCH")
    assert len(hatches) == 1
    h = hatches[0]
    assert h.dxf.pattern_name == "ANSI31" and h.dxf.layer == "剖面线"
    line = h.pattern.lines[0]
    assert line.angle == pytest.approx(45)
    assert math.hypot(*line.offset) == pytest.approx(4.0, abs=1e-3)  # 2 mm on paper at 1:2

    texts = {t.dxf.text for t in msp.query("TEXT")}
    assert {"技术要求:", "安装板  1:2", "图号 AZB-001"} <= texts
    assert all(t.dxf.height == pytest.approx(7.0, abs=0.01) for t in msp.query("TEXT") if t.dxf.text.startswith("1."))


def test_sample_curves_and_linetypes(sample):
    doc = reload(sample)
    msp = doc.modelspace()
    arcs = msp.query("ARC")
    assert len(arcs) == 2 and all(a.dxf.radius == pytest.approx(12.0, abs=1e-3) for a in arcs)
    centre_lines = [e for e in msp.query("LINE") if e.dxf.layer == "中心线"]
    assert centre_lines and all(e.dxf.linetype == "PDF_CENTER" for e in centre_lines)
    assert doc.audit().has_errors is False


def test_plan_chained_tick_dims(plan):
    doc = reload(plan)
    dims = doc.modelspace().query("DIMENSION")
    measures = sorted(round(d.get_measurement()) for d in dims)
    assert measures == [3000, 4000, 4000, 5000]
    assert plan.report.scale == 100
    ticks = [d for d in dims if d.get_dxf_attrib("defpoint")[1] > 13000]
    assert len(ticks) == 3


def test_outlined_digits_chain_and_fit_to_page_scale():
    data, hx, vy = outlined_dims_pdf()
    r = convert_bytes(data)
    texts = sorted(d.text for d in r.drawing.linear_dims)
    assert texts == sorted(hx + vy)
    # Fit-to-page (~1:104) snaps back to the labeled 1:100 with true lengths.
    assert r.report.scale == pytest.approx(100, abs=1e-6)
    assert r.report.recognized["线性标注"] == len(hx) + len(vy)
    assert any("笔画" in w for w in r.report.warnings)
    doc = reload(r)
    msp = doc.modelspace()
    # glyph strokes, ticks and the two row lines are all absorbed into the dimensions
    assert len(msp.query("LWPOLYLINE")) == 0
    assert len(msp.query("HATCH")) == 0
    measures = sorted(round(d.get_measurement()) for d in msp.query("DIMENSION"))
    assert measures == sorted(int(v) for v in hx + vy)
    for d in msp.query("DIMENSION"):
        shown = d.dxf.text if d.dxf.text not in ("", "<>") else str(round(d.get_measurement()))
        assert shown in hx + vy


def test_crowded_outlined_chain_dims():
    data, vals = crowded_dims_pdf()
    r = convert_bytes(data, Options(ocr=False))
    assert sorted(d.text for d in r.drawing.linear_dims) == sorted(vals)


def test_outlined_lettering_becomes_text():
    pytest.importorskip("rapidocr_onnxruntime")
    items = [
        ("雨水管", (40, 160), 4, 0), ("CH=3500", (120, 160), 4, 0), ("卫生间", (40, 120), 5, 0),
        ("FM甲1022", (120, 120), 4, 0), ("玻璃幕墙", (240, 60), 4, 90), ("±0.000", (40, 80), 4, 0),
    ]
    r = convert_bytes(outlined_text_pdf(items))
    got = {t.text: t for t in r.drawing.texts}
    hits = [text for text, *_ in items if text in got]
    assert len(hits) >= len(items) - 1, sorted(got)
    assert got["卫生间"].rotation == 0 and got.get("玻璃幕墙", got["卫生间"]).rotation in (0, 90)
    if "玻璃幕墙" in got:
        assert got["玻璃幕墙"].rotation == 90
    t = got["卫生间"]
    assert math.dist(t.insert, (40, 120)) < 1.5
    assert any("文字识别" in w for w in r.report.warnings)
    doc = reload(r)
    msp = doc.modelspace()
    assert len(msp.query("TEXT")) >= len(items) - 1
    assert len(msp.query("HATCH")) == 0


def test_outlined_table_cells_read_separately():
    pytest.importorskip("rapidocr_onnxruntime")
    items = [("0", (150, 60), 4, 0), ("-70", (155.5, 60), 4, 0), ("CH=3500", (150, 50.5), 4, 0)]
    x0, x1, y0, ym, y1 = 147, 172, 48.5, 57.5, 66
    lines = [((x0, y0), (x1, y0)), ((x0, ym), (x1, ym)), ((x0, y1), (x1, y1)),
             ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1)), ((154, ym), (154, y1))]
    r = convert_bytes(outlined_text_pdf(items, lines))
    got = {t.text.replace("−", "-") for t in r.drawing.texts}
    assert "0" in got and "-70" in got, got
    assert not any("0-70" in t for t in got)


def _plus_field(x0, y0, x1, y1, step=5.0, arm=1.2):
    """Planting marks: staggered rows of "+" drawn as loose strokes."""
    lines = []
    row = 0
    y = y0 + step / 2
    while y < y1:
        x = x0 + step / 2 + (step / 2 if row % 2 else 0)
        while x < x1:
            lines += [((x - arm, y), (x + arm, y)), ((x, y - arm), (x, y + arm))]
            x += step
        y += step / 2
        row += 1
    return lines


def test_plus_marks_become_one_hatch():
    lines = _plus_field(60, 40, 160, 120)
    r = convert_bytes(outlined_text_pdf([], lines), Options(ocr=False))
    patterns = [h for h in r.drawing.hatches if not h.solid]
    assert len(patterns) == 1, len(patterns)
    assert len(r.drawing.inserts) == 0
    doc = reload(r)
    hatches = [h for h in doc.modelspace().query("HATCH") if not h.dxf.solid_fill]
    assert len(hatches) == 1 and len(hatches[0].pattern.lines) >= 2
    assert any(line.dash_length_items for line in hatches[0].pattern.lines)


def test_vertical_text_over_plus_marks():
    pytest.importorskip("rapidocr_onnxruntime")
    lines = _plus_field(60, 40, 160, 120)
    items = [("8400", (102, 60), 4, 90), ("43050", (125, 70), 4, 90)]
    r = convert_bytes(outlined_text_pdf(items, lines))
    got = [(t.text, t.rotation) for t in r.drawing.texts]
    assert ("8400", 90) in got and ("43050", 90) in got, got
    assert not [t for t, rot in got if rot == 0 and t.isdigit()], got


def test_exploded_axes_and_bubbles():
    pytest.importorskip("rapidocr_onnxruntime")
    data, labels = axis_grid_pdf()
    r = convert_bytes(data, Options(scale=100))
    assert sorted(b.label for b in r.drawing.axis_bubbles) == sorted(labels)
    doc = reload(r)
    msp = doc.modelspace()
    axes = [e for e in msp.query("LINE") if e.dxf.layer == "PDF_轴线" and e.dxf.linetype != "BYLAYER"]
    assert len(axes) == 6
    assert all(e.dxf.start.distance(e.dxf.end) > 90 * 100 for e in axes)
    assert not [e for e in msp.query("LINE") if e.dxf.start.distance(e.dxf.end) < 700]
    tags = {}
    for ins in msp.query("INSERT[layer=='PDF_轴号']"):
        tags.update({a.dxf.tag: a.dxf.text for a in ins.attribs})
    assert tags.get("分子") == "1" and tags.get("分母") == "2"


def test_dimension_values_from_crowded_text():
    from pdf2cad.dims import _value_options

    assert _value_options("1300", 1296.0) == [("1300", 0, 4)]
    assert _value_options("200,200", 191.0) == [("200", 0, 3), ("200", 4, 7)]
    assert ("600", 0, 3) in _value_options("600200", 600.4)
    assert ("200", 3, 6) in _value_options("600200", 199.0)
    assert _value_options("FM甲1022", 300.0) == []
    assert _value_options("2700", None) == [("2700", 0, 4)]


def test_ocr_cleanup_rules():
    from pdf2cad.ocrtext import clean

    assert clean("CH=至项") == "CH=至顶"
    assert clean("FMZ1522") == "FM乙1522"
    assert clean("1.2OOM标高") == "1.200M标高"
    assert clean("[ICH=3500") == "CH=3500"
    assert clean("十冏十") == ""


def test_fit_to_page_rectified_to_labeled_scale():
    """A sheet printed at ~96% of 1:100 is restored to 1:100 with exact labeled lengths."""
    from drawings import Page, filled_tick

    pg = Page()
    # Designed 3600×2400 at 1:100 → 36×24 paper mm; print shrink 4% → 34.56×23.04.
    shrink = 0.96
    x0, y0 = 40.0, 40.0
    x1, y1 = x0 + 36 * shrink, y0 + 24 * shrink
    pg.line((x0, y0), (x1, y0), width=0.25)
    pg.line((x1, y0), (x1, y1), width=0.25)
    pg.line((x1, y1), (x0, y1), width=0.25)
    pg.line((x0, y1), (x0, y0), width=0.25)
    yd = y0 - 8
    pg.line((x0, yd), (x1, yd), width=0.15)
    filled_tick(pg, (x0, yd))
    filled_tick(pg, (x1, yd))
    pg.line((x0, y0), (x0, yd - 1), width=0.15)
    pg.line((x1, y0), (x1, yd - 1), width=0.15)
    pg.text(((x0 + x1) / 2 - 3, yd + 1.2), "3600", height=2.5)
    xd = x1 + 8
    pg.line((xd, y0), (xd, y1), width=0.15)
    filled_tick(pg, (xd, y0))
    filled_tick(pg, (xd, y1))
    pg.line((x1, y0), (xd + 1, y0), width=0.15)
    pg.line((x1, y1), (xd + 1, y1), width=0.15)
    pg.text((xd + 1.2, (y0 + y1) / 2 - 3), "2400", height=2.5, rotate=90)
    r = convert_bytes(pg.save(), Options(ocr=False))
    assert abs(r.report.scale - 100) < 1e-6, r.report
    doc = reload(r)
    measures = sorted(round(d.get_measurement()) for d in doc.modelspace().query("DIMENSION"))
    assert measures == [2400, 3600], measures
    assert any("1:100" in w for w in r.report.warnings)


def test_dimstyles_carry_the_sizes():
    data, *_ = outlined_dims_pdf()
    doc = reload(convert_bytes(data, Options(ocr=False)))
    dims = list(doc.modelspace().query("DIMENSION"))
    assert {d.dxf.dimstyle for d in dims} == {"PDF"}
    for d in dims:
        assert not {"dimtxt", "dimasz", "dimblk", "dimtsz"} & set(d.override().dimstyle_attribs)
    ds = doc.dimstyles.get("PDF").dxf
    # Paper millimetres × DIMSCALE (= drawing scale), as AutoCAD expects for new dimensions.
    assert abs(ds.dimtxt - 2.5) < 0.15 and ds.dimblk == "_ARCHTICK" and ds.dimscale > 50
    assert abs(doc.header["$DIMSCALE"] - ds.dimscale) < 1e-6
    # ezdxf sample styles (DIMLFAC=100) must not remain — AutoCAD may use them for DIM.
    assert not [s.dxf.name for s in doc.dimstyles if s.dxf.name.startswith("EZ")]
    std = doc.dimstyles.get("Standard").dxf
    assert std.dimscale == ds.dimscale and std.dimblk == ds.dimblk and std.dimlfac == 1
    # Header dimvars must match PDF — AutoCAD uses $DIMLFAC / $DIMTXSTY for new DIMs.
    assert doc.header["$DIMLFAC"] == 1
    assert doc.header["$DIMDEC"] == 0
    assert doc.header["$DIMTXSTY"] == "PDF_TEXT"
    assert doc.header["$DIMSTYLE"] == "PDF"
    assert doc.header["$DIMBLK"] in ("_ARCHTICK", "ARCHTICK")
    # outlined_dims_pdf puts numbers above the line → DIMTAD=1 for new DIMs.
    assert ds.dimtad == 1 and ds.dimtmove == 0
    assert doc.header["$DIMTAD"] == 1 and doc.header["$DIMTMOVE"] == 0
    st = doc.styles.get("PDF_TEXT").dxf
    assert (st.font, st.bigfont) == ("simplex.shx", "gbcbig.shx")


def test_infer_dimtad_from_text_offset():
    def dim(ty: float, h: float = 2.5) -> LinearDim:
        return LinearDim(
            p1=(0, 0), p2=(30, 0), base=(0, 10), angle=0.0, text="3000",
            text_pos=(15, 10 + ty), text_height=h, text_rotation=0.0,
            arrow_size=1.5, arrow_kind="tick", exe=1.0, measured=30.0,
            layer=None, color=None,
        )
    assert infer_dimtad([dim(2.0), dim(1.8), dim(0.0)]) == 1
    assert infer_dimtad([dim(0.0), dim(0.2), dim(2.0)]) == 0
    assert infer_dimtad([]) == 1


def test_dimstyle_follows_centered_pdf_text():
    data, vals = centered_dims_pdf()
    doc = reload(convert_bytes(data, Options(ocr=False)))
    dims = list(doc.modelspace().query("DIMENSION"))
    assert len(dims) == len(vals)
    ds = doc.dimstyles.get("PDF").dxf
    assert ds.dimtad == 0 and doc.header["$DIMTAD"] == 0
    assert ds.dimtmove == 0 and doc.header["$DIMTMOVE"] == 0


def test_plan_aligned_dim(plan):
    aligned = [d for d in plan.drawing.linear_dims if abs(d.angle - 30) < 0.1]
    assert len(aligned) == 1 and aligned[0].text == "4000"


def test_plan_rotated_blocks(plan):
    ins = plan.drawing.inserts
    assert len(plan.drawing.blocks) == 1
    assert sorted(round(i.rotation) % 360 for i in ins) == [0, 90, 180, 270]


def test_plan_hatch_with_island_and_solid(plan):
    hs = plan.drawing.hatches
    pattern = [h for h in hs if not h.solid]
    solid = [h for h in hs if h.solid]
    assert len(pattern) == 1 and len(pattern[0].loops) == 2
    assert pattern[0].families[0].spacing == pytest.approx(1.5, abs=1e-3)
    assert len(solid) == 1


def test_hatches_drawn_under_geometry_when_pdf_fills_are_background():
    # Floor fill painted before wall strokes → HATCH under LINE + SORTENTS order.
    r = convert_bytes(floor_fill_under_walls_pdf(), Options(dimensions=False, blocks=False, ocr=False))
    assert getattr(r.drawing, "fills_under_content", None) is True
    assert any("填充" in w and "底层" in w for w in r.report.warnings)
    msp = reload(r).modelspace()
    kinds = [e.dxftype() for e in msp]
    assert kinds.index("HATCH") < kinds.index("LINE")
    order = dict(msp.get_redraw_order())
    hatch_h = next(e.dxf.handle for e in msp.query("HATCH"))
    line_h = next(e.dxf.handle for e in msp.query("LINE"))
    assert int(order[hatch_h], 16) < int(order[line_h], 16)


def test_plan_layers_without_ocg(plan):
    rep = plan.report
    assert not rep.pdf_layers
    assert {"PDF_标注", "PDF_文字", "PDF_填充"} <= set(rep.layers)
    assert any("图层" in w for w in rep.warnings)


def test_manual_scale_and_switches():
    r = convert_bytes(sample_pdf_bytes(), Options(scale=10, blocks=False, hatches=False, dimensions=False))
    rep = r.report
    assert rep.scale == 10 and rep.scale_source == "manual"
    assert rep.recognized["图块引用"] == 0
    assert rep.recognized["线性标注"] == 0
    assert rep.recognized["图案填充"] == 0


def test_bezier_to_arc():
    r = 10.0
    k = 4 / 3 * math.tan(math.pi / 8)
    quarter = Bezier((r, 0), (r, k * r), (k * r, r), (0, r))
    arc = bezier_to_arc(quarter)
    assert isinstance(arc, Arc)
    assert arc.radius == pytest.approx(r, rel=1e-3) and arc.sweep == pytest.approx(90, abs=0.1)
    assert isinstance(bezier_to_arc(Bezier((0, 0), (1, 0), (2, 0), (3, 0))), Line)
    s_curve = Bezier((0, 0), (5, 5), (5, -5), (10, 0))
    assert isinstance(bezier_to_arc(s_curve), Bezier)


@pytest.mark.parametrize("text,value", [("200", 200), ("4×Φ12", 12), ("R6", 6), ("12.50", 12.5), ("2x45°", 45)])
def test_parse_value(text, value):
    assert parse_value(text)[0] == value


def test_blank_page_warns():
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    r = convert_bytes(doc.tobytes())
    assert any("扫描件" in w for w in r.report.warnings)
