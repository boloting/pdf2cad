import io

import ezdxf
import pytest

from drawings import Page, filled_tick
from pdf2cad import measure
from pdf2cad.convert import Options, convert_bytes
from pdf2cad.writer import DIM_REVIEW_LAYER

SHRINK = 0.96  # printed "fit to page" at about 1:104 of a 1:100 sheet


def reload(result):
    return ezdxf.read(io.StringIO(result.dxf_bytes().decode("utf-8")))


def _hdim(pg, xa, xb, y_obj, y, text):
    pg.line((xa, y), (xb, y), width=0.15)
    filled_tick(pg, (xa, y))
    filled_tick(pg, (xb, y))
    pg.line((xa, y_obj), (xa, y - 1), width=0.15)
    pg.line((xb, y_obj), (xb, y - 1), width=0.15)
    pg.text(((xa + xb) / 2 - 3, y + 1.2), text, height=2.5)


def _vdim(pg, ya, yb, x_obj, x, text):
    pg.line((x, ya), (x, yb), width=0.15)
    filled_tick(pg, (x, ya))
    filled_tick(pg, (x, yb))
    pg.line((x_obj, ya), (x + 1, ya), width=0.15)
    pg.line((x_obj, yb), (x + 1, yb), width=0.15)
    pg.text((x + 1.2, (ya + yb) / 2 - 3), text, height=2.5, rotate=90)


def conflict_plan_pdf() -> bytes:
    """6000 × 2400 room at 1:100, shrunk 4 %, with an interior wall drawn 201 mm thick.

    The chain says 3000 + 3000 and the overall dimension says 6050: the overall label
    contradicts the drawing and the chain, so it must be flagged, not forced.
    """
    pg = Page()
    k = SHRINK / 100  # real mm -> paper mm on the printed sheet
    x0, y0 = 40.0, 60.0
    X = lambda v: x0 + v * k  # noqa: E731
    Y = lambda v: y0 + v * k  # noqa: E731
    for a, b in [((0, 0), (6000, 0)), ((6000, 0), (6000, 2400)), ((6000, 2400), (0, 2400)), ((0, 2400), (0, 0))]:
        pg.line((X(a[0]), Y(a[1])), (X(b[0]), Y(b[1])), width=0.35)
    for xw in (4000, 4201):
        pg.line((X(xw), Y(0)), (X(xw), Y(2400)), width=0.35)
    _hdim(pg, X(0), X(3000), Y(0), Y(0) - 8, "3000")
    _hdim(pg, X(3000), X(6000), Y(0), Y(0) - 8, "3000")
    _hdim(pg, X(0), X(6000), Y(0), Y(0) - 16, "6050")
    _vdim(pg, Y(0), Y(2400), X(6000), X(6000) + 8, "2400")
    return pg.save()


@pytest.fixture(scope="module")
def conflict():
    return convert_bytes(conflict_plan_pdf(), Options(ocr=False))


def test_conflicting_label_is_flagged_not_forced(conflict):
    r = conflict
    assert r.report.scale == pytest.approx(100)
    check = r.report.dimension_check
    assert [c["text"] for c in check["conflicts"]] == ["6050"]
    assert check["conflicts"][0]["drawn"] == pytest.approx(6000, abs=0.01)
    assert any("PDF_标注_待核对" in w and "6050" in w for w in r.report.warnings)
    doc = reload(r)
    dims = list(doc.modelspace().query("DIMENSION"))
    flagged = [d for d in dims if d.dxf.layer == DIM_REVIEW_LAYER]
    assert len(flagged) == 1
    # Keeps the drawn length and shows the printed number for a person to check.
    assert flagged[0].dxf.text == "6050"
    assert flagged[0].get_measurement() == pytest.approx(6000, abs=1e-6)
    accepted = sorted(d.get_measurement() for d in dims if d.dxf.layer != DIM_REVIEW_LAYER)
    assert accepted == pytest.approx([2400, 3000, 3000], abs=1e-6)


def test_units_are_millimetres(conflict):
    doc = reload(conflict)
    assert doc.header["$INSUNITS"] == 4
    assert doc.header["$MEASUREMENT"] == 1


def test_wall_thickness_snapped_to_whole_ten_mm(conflict):
    doc = reload(conflict)
    xs = sorted({
        round(e.dxf.start.x, 6) for e in doc.modelspace().query("LINE")
        if abs(e.dxf.start.x - e.dxf.end.x) < 1e-9 and abs(e.dxf.start.y - e.dxf.end.y) > 2000
    })
    left = xs[0]
    # 201 mm drawn, nearest whole-10 thickness is 200; nothing anchors either face.
    walls = [x for x in xs if 3900 < x - left < 4300]
    assert len(walls) == 2, xs
    assert walls[1] - walls[0] == pytest.approx(200, abs=1e-6)


def test_measure_reports_residuals(conflict, tmp_path):
    doc = reload(conflict)
    res = measure.check(doc)
    assert res["units"]["insunits"] == 4
    dm = res["dimensions"]
    assert dm["review"] == 1 and dm["accepted_exact"] == dm["accepted_checked"] == 3
    assert res["walls"]["on_10mm"] >= 1
    path = tmp_path / "plan.dxf"
    path.write_bytes(conflict.dxf_bytes())
    assert measure.main([str(path)]) == 0
