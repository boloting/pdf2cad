import io
import math

import ezdxf
import pytest

from drawings import outlined_dims_pdf, plan_pdf
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
    assert r.report.scale == pytest.approx(100 / 0.96, rel=2e-4)
    assert r.report.recognized["线性标注"] == len(hx) + len(vy)
    assert any("笔画" in w for w in r.report.warnings)
    doc = reload(r)
    msp = doc.modelspace()
    # glyph strokes, ticks and the two row lines are all absorbed into the dimensions
    assert len(msp.query("LWPOLYLINE")) == 0
    assert len(msp.query("HATCH")) == 0
    for d in msp.query("DIMENSION"):
        shown = d.dxf.text if d.dxf.text not in ("", "<>") else str(round(d.get_measurement()))
        assert shown in hx + vy


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
