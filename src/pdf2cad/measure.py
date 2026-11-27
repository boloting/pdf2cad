"""Measure a DXF against its own dimension labels: pdf2cad-check drawing.dxf [--json].

Reads only the DXF, the way a person checks it in CAD: what units the drawing is in,
whether each linear dimension's printed number equals the length it measures, whether
labeled axis spacings equal the drawn axis lines, and what wall thicknesses come out.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import median

import ezdxf

from .axes import AXIS_LAYER, BUBBLE_LAYER
from .dims import parse_value
from .writer import DIM_REVIEW_LAYER

_UNITS = {0: "无单位", 1: "英寸", 2: "英尺", 4: "毫米", 5: "厘米", 6: "米"}
_ENDPOINT_TOL = 1.0  # mm: a dimension end this close to an axis line measures that axis
_AXIS_NEAR = 60.0  # mm: a dimension whose ends are this close to two axes is about their spacing
_ON_GEOMETRY = 0.05  # mm: a dimension end this close to a drawn line or vertex sits on it
_SAME_AXIS = 100.0  # mm: bubbles with one label further apart than this belong to different plans
_WALL_MIN_LEN = 500.0
_WALL_GAP = (80.0, 400.0)


def check(doc) -> dict:
    msp = doc.modelspace()
    hdr = doc.header
    out = {
        "units": {
            "insunits": hdr.get("$INSUNITS", 0),
            "insunits_name": _UNITS.get(hdr.get("$INSUNITS", 0), "其它"),
            "measurement": hdr.get("$MEASUREMENT", 0),
            "dimscale": hdr.get("$DIMSCALE", 1.0),
            "dimlfac": hdr.get("$DIMLFAC", 1.0),
        },
    }
    dims = _dims(msp)
    gap = _geometry_gap(msp)
    out["dimensions"] = _dim_stats(dims)
    out["axes"] = _axis_stats(msp, dims, gap)
    out["walls"] = _wall_stats(msp)
    out["dimension_ends"] = _end_stats(dims, gap)
    return out


# -- dimensions --------------------------------------------------------------------

def _dims(msp) -> list[dict]:
    rows = []
    for e in msp.query("DIMENSION"):
        if e.dimtype not in (0, 1):
            continue
        measured = e.get_measurement()
        if not isinstance(measured, float | int):
            continue
        raw = e.dxf.get("text", "")
        live = raw in ("", "<>") or "<>" in raw
        if live:
            # CAD prints the measurement rounded to DIMDEC; that rounded number is the label.
            dec = int(e.override().get("dimdec", 0) or 0)
            label = round(float(measured), dec)
        else:
            pv = parse_value(raw)
            label = pv[0] if pv else None
        p1, p2 = e.dxf.defpoint2, e.dxf.defpoint3
        ang = e.dxf.get("angle", 0.0) % 180
        axis = 0 if min(ang, 180 - ang) < 1 else 1 if abs(ang - 90) < 1 else None
        rows.append({
            "measured": float(measured), "label": label, "text": raw, "live": live,
            "review": e.dxf.layer == DIM_REVIEW_LAYER, "layer": e.dxf.layer, "axis": axis,
            "ends": (p1[axis], p2[axis]) if axis is not None else None,
        })
    return rows


def _dim_stats(dims: list[dict]) -> dict:
    accepted = [d for d in dims if not d["review"]]
    resid = []
    for d in accepted:
        if d["label"] is not None:
            resid.append(abs(d["label"] - d["measured"]))
    review = [
        {"text": d["text"], "measured": round(d["measured"], 1)}
        for d in dims if d["review"]
    ]
    return {
        "total": len(dims),
        "live": sum(1 for d in dims if d["live"]),
        "static_text": sum(1 for d in dims if not d["live"]),
        "review": len(review),
        "accepted_checked": len(resid),
        "accepted_exact": sum(1 for v in resid if v < 0.005),
        "accepted_within_1mm": sum(1 for v in resid if v <= 1.0 + 1e-6),
        "accepted_max_residual": round(max(resid), 3) if resid else 0.0,
        "review_list": review,
    }


def _geometry_gap(msp):
    """(axis, coordinate) -> distance to the nearest drawn vertex coordinate on that axis."""
    coords: tuple[list[float], list[float]] = ([], [])
    for _layer, a, b in _segments(msp, skip=()):
        for q in (a, b):
            coords[0].append(q[0])
            coords[1].append(q[1])
    pts = []
    for e in msp.query("CIRCLE ARC"):
        pts.append(e.dxf.center)
        if e.dxftype() == "ARC":
            pts.extend((e.start_point, e.end_point))
    for ins in msp.query("INSERT"):
        for ve in ins.virtual_entities():
            if ve.dxftype() == "LINE":
                pts.extend((ve.dxf.start, ve.dxf.end))
            elif ve.dxftype() == "LWPOLYLINE":
                pts.extend(ve.vertices_in_wcs())
            elif ve.dxftype() in ("ARC", "CIRCLE"):
                pts.append(ve.dxf.center)
                if ve.dxftype() == "ARC":
                    pts.extend((ve.start_point, ve.end_point))
    for q in pts:
        coords[0].append(q[0])
        coords[1].append(q[1])
    for c in coords:
        c.sort()

    def gap(axis: int, v: float) -> float:
        arr = coords[axis]
        k = bisect.bisect_left(arr, v)
        return min((abs(arr[i] - v) for i in (k - 1, k) if 0 <= i < len(arr)), default=math.inf)

    return gap


def _end_stats(dims: list[dict], gap) -> dict:
    """Do the dimension ends sit on drawn lines? A gap means the label was forced."""
    gaps = [gap(d["axis"], v) for d in dims if d["ends"] and not d["review"] for v in d["ends"]]
    return {
        "checked": len(gaps),
        "on_geometry": sum(1 for g in gaps if g <= _ON_GEOMETRY),
        "off_0_05_to_2mm": sum(1 for g in gaps if _ON_GEOMETRY < g <= 2.0),
        "off_more_than_2mm": sum(1 for g in gaps if g > 2.0),
    }


# -- axes --------------------------------------------------------------------------

def _bubble_label(ins) -> str | None:
    attrs = {a.dxf.tag: a.dxf.text for a in ins.attribs}
    if "轴号" in attrs:
        return attrs["轴号"]
    if "分子" in attrs and "分母" in attrs:
        return f"{attrs['分子']}/{attrs['分母']}"
    return None


def _axes(msp) -> dict[int, dict[str, list[float]]]:
    """Axis label -> drawn positions, per measured axis (0 = vertical lines, positions in X)."""
    lines = []
    for e in msp.query("LINE"):
        if e.dxf.layer != AXIS_LAYER:
            continue
        a, b = e.dxf.start, e.dxf.end
        if abs(a.x - b.x) < 0.01:
            lines.append((0, a.x, (a.x, a.y), (b.x, b.y)))
        elif abs(a.y - b.y) < 0.01:
            lines.append((1, a.y, (a.x, a.y), (b.x, b.y)))
    out: dict[int, dict[str, list[float]]] = {0: defaultdict(list), 1: defaultdict(list)}
    for ins in msp.query("INSERT"):
        if ins.dxf.layer != BUBBLE_LAYER:
            continue
        label = _bubble_label(ins)
        if not label:
            continue
        c = ins.dxf.insert
        best = None
        for axis, pos, a, b in lines:
            gap = min(math.dist((c.x, c.y), a), math.dist((c.x, c.y), b))
            if best is None or gap < best[0]:
                best = (gap, axis, pos)
        if best is not None:
            out[best[1]][label].append(best[2])
    return out


def _axis_stats(msp, dims: list[dict], gap) -> dict:
    # One sheet may hold several plans that reuse the same axis labels: positions of a
    # label further apart than _SAME_AXIS are different axes.
    clusters: dict[int, list[tuple[str, list[float]]]] = {0: [], 1: []}
    for axis, by_label in _axes(msp).items():
        for label, vals in by_label.items():
            vals = sorted(vals)
            group = [vals[0]]
            for v in vals[1:]:
                if v - group[-1] > _SAME_AXIS:
                    clusters[axis].append((label, group))
                    group = []
                group.append(v)
            clusters[axis].append((label, group))
    spacing = []
    for axis in (0, 1):
        pos = sorted(((median(v), k) for k, v in clusters[axis]), key=lambda t: t[0])
        for (p1, n1), (p2, n2) in pairwise(pos):
            near = [
                d for d in dims
                if d["label"] is not None and d["axis"] == axis and d["ends"] and not d["review"]
                and abs(min(d["ends"]) - p1) <= _AXIS_NEAR and abs(max(d["ends"]) - p2) <= _AXIS_NEAR
            ]
            labels = sorted({
                d["label"] for d in near
                if abs(min(d["ends"]) - p1) <= _ENDPOINT_TOL and abs(max(d["ends"]) - p2) <= _ENDPOINT_TOL
            })
            # A dimension meant for this spacing whose end stops short of the axis line.
            loose = [
                round(max(abs(min(d["ends"]) - p1), abs(max(d["ends"]) - p2)), 2)
                for d in near
                if max(abs(min(d["ends"]) - p1), abs(max(d["ends"]) - p2)) > _ENDPOINT_TOL
                and max(gap(axis, v) for v in d["ends"]) > _ON_GEOMETRY
            ]
            flagged = sorted({
                d["text"] for d in dims
                if d["axis"] == axis and d["ends"] and d["review"]
                and abs(min(d["ends"]) - p1) <= _AXIS_NEAR and abs(max(d["ends"]) - p2) <= _AXIS_NEAR
            })
            row = {"axis": "xy"[axis], "from": n1, "to": n2, "drawn": round(p2 - p1, 2), "labels": labels}
            if labels:
                row["residual"] = round(max(abs(v - (p2 - p1)) for v in labels), 3)
            if flagged:
                row["review"] = flagged
            if loose:
                row["dim_end_off_axis"] = max(loose)
            spacing.append(row)
    checked = [r["residual"] for r in spacing if "residual" in r]
    spread = [max(v) - min(v) for c in clusters.values() for _k, v in c if len(v) > 1]
    return {
        "count": sum(len(c) for c in clusters.values()),
        "spacings": spacing,
        "spacings_labeled": len(checked),
        "spacings_exact": sum(1 for v in checked if v < 0.005),
        "spacing_max_residual": round(max(checked), 3) if checked else 0.0,
        # The same axis drawn at both ends of the plan should be one straight line.
        "same_axis_spread_max": round(max(spread), 3) if spread else 0.0,
        "spacings_with_dim_end_off_axis": sum(1 for r in spacing if "dim_end_off_axis" in r),
    }


# -- walls -------------------------------------------------------------------------

def _segments(msp, skip=(AXIS_LAYER, BUBBLE_LAYER)):
    for e in msp:
        if e.dxf.layer in skip:
            continue
        if e.dxftype() == "LINE":
            yield e.dxf.layer, (e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)
        elif e.dxftype() == "LWPOLYLINE":
            pts = [(p[0], p[1], p[4]) for p in e.get_points("xyseb")]
            if e.closed:
                pts.append(pts[0])
            for (x1, y1, bulge), (x2, y2, _b) in pairwise(pts):
                if not bulge:
                    yield e.dxf.layer, (x1, y1), (x2, y2)


def _wall_stats(msp) -> dict:
    faces: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for layer, a, b in _segments(msp):
        for axis in (0, 1):
            if abs(a[axis] - b[axis]) < 0.01:
                lo, hi = sorted((a[1 - axis], b[1 - axis]))
                if hi - lo >= _WALL_MIN_LEN:
                    faces[(layer, axis, round(a[axis], 2))].append((lo, hi))
    grouped: dict[tuple, list] = defaultdict(list)
    for (layer, axis, c), spans in faces.items():
        grouped[(layer, axis)].append((c, spans))
    thick = []
    for lst in grouped.values():
        lst.sort(key=lambda t: t[0])
        for n, (c1, s1) in enumerate(lst):
            for c2, s2 in lst[n + 1:]:
                gap = c2 - c1
                if gap > _WALL_GAP[1]:
                    break
                if gap < _WALL_GAP[0]:
                    continue
                ov = sum(max(0.0, min(h1, h2) - max(l1, l2)) for l1, h1 in s1 for l2, h2 in s2)
                if ov >= _WALL_MIN_LEN:
                    thick.append(gap)
                    break
    off = [abs(t - round(t / 10) * 10) for t in thick]
    common = Counter(round(t / 10) * 10 for t in thick).most_common(6)
    return {
        "pairs": len(thick),
        "on_10mm": sum(1 for v in off if v < 0.01),
        "within_0_5mm": sum(1 for v in off if v <= 0.5),
        "within_1_5mm": sum(1 for v in off if v <= 1.5),
        "off_more_than_1_5mm": sum(1 for v in off if v > 1.5),
        "common_thickness": [{"mm": int(k), "pairs": v} for k, v in common],
    }


# -- CLI ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pdf2cad-check", description="量一张 DXF：单位、标注数值与实际长度、轴距、墙厚。")
    p.add_argument("dxf", type=Path, help="要检查的 DXF 文件")
    p.add_argument("--json", action="store_true", help="以 JSON 输出全部结果")
    args = p.parse_args(argv)
    try:
        doc = ezdxf.readfile(args.dxf)
    except (OSError, ezdxf.DXFError) as e:
        print(f"无法读取 DXF: {e}", file=sys.stderr)
        return 2
    res = check(doc)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        _print(res)
    return 0


def _print(r: dict) -> None:
    u, dm, ax, w = r["units"], r["dimensions"], r["axes"], r["walls"]
    print(f"单位: INSUNITS={u['insunits']}（{u['insunits_name']}），DIMSCALE={u['dimscale']:g}，DIMLFAC={u['dimlfac']:g}")
    print(
        f"线性标注 {dm['total']} 个：采用 {dm['accepted_checked']} 个，其中与实际长度完全一致 {dm['accepted_exact']} 个，"
        f"差 ≤1 mm {dm['accepted_within_1mm']} 个，最大差 {dm['accepted_max_residual']:g} mm；待核对 {dm['review']} 个"
    )
    for d in dm["review_list"]:
        print(f"  待核对: 标“{d['text']}”，实际 {d['measured']:g}")
    print(
        f"轴线 {ax['count']} 条：有标注的相邻轴距 {ax['spacings_labeled']} 个，完全一致 {ax['spacings_exact']} 个，"
        f"最大差 {ax['spacing_max_residual']:g} mm；同一轴线两端偏差最大 {ax['same_axis_spread_max']:g} mm"
    )
    for s in ax["spacings"]:
        if "review" in s:
            print(f"  {s['from']}–{s['to']}：图上 {s['drawn']:g}，待核对标注 {'、'.join(s['review'])}")
        if "dim_end_off_axis" in s:
            print(f"  {s['from']}–{s['to']}：有标注的端点离轴线 {s['dim_end_off_axis']:g} mm（标注被硬拉到数值）")
    e = r["dimension_ends"]
    print(
        f"标注端点 {e['checked']} 个：落在图线上 {e['on_geometry']} 个，偏 0.05–2 mm {e['off_0_05_to_2mm']} 个，"
        f"偏 >2 mm {e['off_more_than_2mm']} 个"
    )
    print(
        f"墙厚 {w['pairs']} 对：整 10 mm {w['on_10mm']} 对，差 ≤0.5 mm {w['within_0_5mm']} 对，"
        f"≤1.5 mm {w['within_1_5mm']} 对，差 >1.5 mm {w['off_more_than_1_5mm']} 对"
    )
    print("  常见墙厚: " + "，".join(f"{c['mm']} mm×{c['pairs']}" for c in w["common_thickness"]))


if __name__ == "__main__":
    raise SystemExit(main())
