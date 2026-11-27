"""Write the recognized drawing model to a DXF document with ezdxf."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from statistics import median

import ezdxf
from ezdxf.document import Drawing as DXFDocument
from ezdxf.lldxf.const import VALID_DXF_LINEWEIGHTS
from ezdxf.math import Bezier4P, Vec3, bezier_to_bspline

from .dims import parse_value
from .model import Arc, Bezier, Color, Drawing, Hatch, Line, Shape, Style

TEXT_STYLE = "PDF_TEXT"
REVIEW_LAYER = "PDF_文字_待核对"
DIM_REVIEW_LAYER = "PDF_标注_待核对"
_CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")
DIM_STYLE = "PDF"

_ACI = {
    (1.0, 0.0, 0.0): 1, (1.0, 1.0, 0.0): 2, (0.0, 1.0, 0.0): 3, (0.0, 1.0, 1.0): 4,
    (0.0, 0.0, 1.0): 5, (1.0, 0.0, 1.0): 6, (0.0, 0.0, 0.0): 7, (1.0, 1.0, 1.0): 7,
}
_ARROW_BLOCKS = {"closed": "", "closed_blank": "CLOSEDBLANK", "open": "OPEN", "tick": "ARCHTICK", "dot": "DOT"}
_ARROW_NAMES = {"closed": "箭头", "closed_blank": "空心箭头", "open": "开口箭头", "tick": "斜线", "dot": "圆点"}
# Most Chinese drawings use these SHX fonts, so CAD finds them without substituting.
TEXT_FONT, TEXT_BIGFONT = "simplex.shx", "gbcbig.shx"
_DIA = re.compile("[ØΦφ⌀]")
_BAD_NAME = re.compile(r'[<>/\\":;?*|=,`]')


def color_key(c: Color | None) -> tuple[float, float, float] | None:
    return None if c is None else tuple(round(v, 2) for v in c)  # type: ignore[return-value]


def color_attribs(c: Color | None) -> dict:
    k = color_key(c)
    if k is None:
        return {}
    if k in _ACI:
        return {"color": _ACI[k]}
    return {"true_color": ezdxf.rgb2int(tuple(int(round(v * 255)) for v in k))}


def lineweight(width_mm: float) -> int:
    v = width_mm * 100
    if v <= 0:
        return 0
    return min(VALID_DXF_LINEWEIGHTS, key=lambda w: abs(w - v))


def _hex(c: Color | None) -> str:
    if c is None:
        return "000000"
    return "".join(f"{int(round(v * 255)):02X}" for v in c)


def _advance(ch: str) -> float:
    if _CJK.match(ch):
        return 1.0
    if ch in "Iil.,:;|!'":
        return 0.45
    return 0.95


def _nice(v: float) -> float:
    """Paper-millimetre size rounded to a round number within 2 %, else two significant digits.

    2.69 -> 2.7, 1.45 -> 1.5, 0.95 -> 1.0, 99.2 -> 100.
    """
    if v <= 0:
        return 0.0
    top = 10 ** math.floor(math.log10(v))
    for p in (top / 2, top / 5, top / 10):
        r = round(v / p) * p
        if abs(r - v) <= 0.02 * v or p == top / 10:
            return round(r, 6)
    return v


def infer_dimtad(linear_dims) -> int:
    """DIMTAD from where PDF numbers sit relative to the dimension line.

    0 = centered on the line (line breaks around the text); 1 = above/beside
    (unbroken line under the text). Majority vote; default 1 when empty.
    """
    centered = offset = 0
    for dim in linear_dims:
        ang = math.radians(dim.angle)
        nx, ny = -math.sin(ang), math.cos(ang)
        off = (dim.text_pos[0] - dim.base[0]) * nx + (dim.text_pos[1] - dim.base[1]) * ny
        h = max(float(dim.text_height), 0.5)
        if abs(off) < 0.35 * h:
            centered += 1
        else:
            offset += 1
    if centered > offset:
        return 0
    return 1


def _snapper(values, tol: float = 0.04):
    """Map each measured size to the rounded median of the values within tol of it."""
    groups: list[list[float]] = []
    for v in sorted(v for v in values if v > 0):
        if groups and v <= groups[-1][0] * (1 + tol):
            groups[-1].append(v)
        else:
            groups.append([v])
    table = [(g[0], g[-1], _nice(median(g))) for g in groups]

    def snap(v: float) -> float:
        for lo, hi, rep in table:
            if lo <= v <= hi:
                return rep
        return _nice(v)

    return snap


def _dxf_symbols(s: str) -> str:
    return _DIA.sub("%%c", s).replace("±", "%%p").replace("°", "%%d")


class Writer:
    def __init__(self, drawing: Drawing, scale: float, layer_by_type: bool) -> None:
        self.d = drawing
        self.s = scale
        self.layer_by_type = layer_by_type
        self.doc: DXFDocument = ezdxf.new("R2018", setup=True, units=ezdxf.units.MM)
        self.msp = self.doc.modelspace()
        self.linetypes: dict[tuple[float, ...], str] = {}
        self.layer_colors: dict[str, tuple | None] = {}
        self._layer_usage: dict[str, Counter] = defaultdict(Counter)
        self.counts: Counter = Counter()

    # -- coordinates and attributes -------------------------------------------------
    def X(self, p) -> tuple[float, float]:
        return (p[0] * self.s, p[1] * self.s)

    def L(self, v: float) -> float:
        """A paper length (text height, arrow size) in model units."""
        return round(v * self.s, 4)

    def layer_name(self, layer: str | None, kind: str, style: Style | None = None) -> str:
        if layer:
            name = layer
        elif kind == "geometry" or not self.layer_by_type:
            st = style
            name = f"PDF_{_hex(st.color if st else None)}_{(st.width if st else 0):.2f}"
        else:
            name = {"dim": "PDF_标注", "text": "PDF_文字", "hatch": "PDF_填充"}.get(kind, "PDF_其他")
        return _BAD_NAME.sub("_", name).strip() or "0"

    def attribs(self, layer: str | None, kind: str, color: Color | None, style: Style | None = None) -> dict:
        name = self.layer_name(layer, kind, style)
        self._layer_usage[name][color_key(color)] += 1
        a: dict = {"layer": name}
        a.update(color_attribs(color))
        if style is not None:
            a["lineweight"] = lineweight(style.width)
            if style.dashes:
                a["linetype"] = self.linetype(style.dashes)
        return a

    def linetype(self, dashes: tuple[float, ...]) -> str:
        if dashes in self.linetypes:
            return self.linetypes[dashes]
        n = len(dashes) // 2
        base = {1: "PDF_DASHED", 2: "PDF_CENTER", 3: "PDF_PHANTOM"}.get(n, "PDF_DASH")
        name = base if base not in self.linetypes.values() else f"{base}{len(self.linetypes) + 1}"
        pattern = []
        for i, v in enumerate(dashes):
            v = v * self.s
            pattern.append(round(v, 6) if i % 2 == 0 else -round(v, 6))
        total = sum(abs(v) for v in pattern)
        desc = " ".join(("__" if v > 0 else ". " if v == 0 else "  ") for v in pattern)
        self.doc.linetypes.add(name, pattern=[total, *pattern], description=desc)
        self.linetypes[dashes] = name
        return name

    # -- geometry -------------------------------------------------------------------
    def shape(self, target, s: Shape) -> None:
        if not s.stroke:
            return
        a = self.attribs(s.style.layer, "geometry", s.style.color, s.style)
        runs: list[list] = []
        for g in s.segs:
            is_b = isinstance(g, Bezier)
            if runs and (isinstance(runs[-1][0], Bezier) == is_b):
                runs[-1].append(g)
            else:
                runs.append([g])
        whole = len(runs) == 1
        for run in runs:
            if isinstance(run[0], Bezier):
                curves = [Bezier4P([Vec3(self.X(p)) for p in (b.p0, b.p1, b.p2, b.p3)]) for b in run]
                target.add_spline(dxfattribs=a).apply_construction_tool(bezier_to_bspline(curves))
                self.counts["SPLINE"] += 1
            elif len(run) == 1 and isinstance(run[0], Arc) and run[0].is_full:
                g = run[0]
                target.add_circle(self.X(g.center), g.radius * self.s, dxfattribs=a)
                self.counts["CIRCLE"] += 1
            elif len(run) == 1 and isinstance(run[0], Line) and not (whole and s.closed):
                target.add_line(self.X(run[0].a), self.X(run[0].b), dxfattribs=a)
                self.counts["LINE"] += 1
            elif len(run) == 1 and isinstance(run[0], Arc) and not (whole and s.closed):
                g = run[0]
                start, end = g.ccw_angles()
                target.add_arc(self.X(g.center), g.radius * self.s, start, end, dxfattribs=a)
                self.counts["ARC"] += 1
            else:
                closed = whole and s.closed
                target.add_lwpolyline(self.bulge_points(run, closed), format="xyb", close=closed, dxfattribs=a)
                self.counts["LWPOLYLINE"] += 1

    def bulge_points(self, segs, closed: bool) -> list[tuple[float, float, float]]:
        pts = []
        for g in segs:
            if isinstance(g, Bezier):
                for p in g.flatten(12)[:-1]:
                    pts.append((*self.X(p), 0.0))
                continue
            if isinstance(g, Arc):
                # A bulge cannot describe a full turn; split big arcs in halves.
                parts = 2 if abs(g.sweep) > 180 else 1
                step = g.sweep / parts
                bulge = math.tan(math.radians(step) / 4)
                for k in range(parts):
                    pts.append((*self.X(g.point_at(g.start_angle + k * step)), bulge))
                continue
            pts.append((*self.X(g.start()), 0.0))
        if not closed:
            pts.append((*self.X(segs[-1].end()), 0.0))
        return pts

    # -- text -----------------------------------------------------------------------
    def text(self, target, t, layer_kind: str = "text"):
        layer = REVIEW_LAYER if getattr(t, "review", False) else t.layer
        a = self.attribs(layer, layer_kind, t.color)
        a.update(style=TEXT_STYLE, insert=self.X(t.insert), rotation=t.rotation)
        if t.font == "ocr" and t.width > 0 and t.text:
            # Width of the lettering in the PDF over its natural width in the CAD font
            # (gbcbig.shx glyphs are square; simplex.shx digits and capitals about 0.95 h wide).
            natural = t.height * sum(_advance(ch) for ch in t.text)
            a["width"] = round(min(2.0, max(0.4, t.width / natural)), 3)
        self.counts["TEXT"] += 1
        return target.add_text(t.text, height=t.height * self.s, dxfattribs=a)

    # -- dimensions -----------------------------------------------------------------
    def setup_dimstyle(self) -> None:
        """One dimension style per look; sizes are paper millimetres, scaled by DIMSCALE.

        AutoCAD multiplies style sizes by DIMSCALE when you create a new dimension. Writing
        paper sizes (2.7, 1.5, …) with DIMSCALE = drawing scale makes new dimensions match
        the ones already on the drawing. Values within 4 % are one setting.
        """
        d = self.d
        # Dimension geometry is measured in paper mm on the PDF.
        self.snap_txt = _snapper([x.text_height for x in (*d.linear_dims, *d.radial_dims, *d.leaders)])
        self.snap_asz = _snapper([x.arrow_size for x in (*d.linear_dims, *d.radial_dims, *d.leaders)])
        self.snap_exe = _snapper([x.exe for x in d.linear_dims])
        self.snap_dle = _snapper([getattr(x, "dle", 0) for x in d.linear_dims])
        acis = Counter(color_attribs(x.color).get("color", 0) for x in (*d.linear_dims, *d.radial_dims, *d.leaders))
        self.dim_aci = acis.most_common(1)[0][0] if acis else 0
        # Match new AutoCAD DIMs to how this PDF places numbers (on the line vs above it).
        self.dimtad = infer_dimtad(d.linear_dims)
        self.dim_styles: dict[tuple, str] = {}
        # Extension-line overshoot and dimension-line extension vary from dimension to dimension
        # (CAD users stretch them); the style takes each look's most common value.
        looks: dict[tuple, list] = defaultdict(list)
        for x in d.linear_dims:
            looks[self.look(x)].append(self.extents(x))
        for look, ext in sorted(looks.items(), key=lambda kv: -len(kv[1])):
            exe, dle = Counter(ext).most_common(1)[0][0]
            self.dim_style(*look, exe, dle)
        if not self.dim_styles:
            self.dim_style(2.5, 2.5, "closed", 1.25, 0.0)

    def look(self, dim) -> tuple:
        return (self.snap_txt(dim.text_height), self.snap_asz(dim.arrow_size), dim.arrow_kind)

    def extents(self, dim) -> tuple[float, float]:
        dle = self.snap_dle(getattr(dim, "dle", 0))
        # A dimension line running a few hundredths past the tick is measurement noise.
        if dim.arrow_kind != "tick" or dle < 0.1 * self.snap_asz(dim.arrow_size):
            dle = 0.0
        return self.snap_exe(dim.exe), dle

    def style_for(self, look: tuple) -> str:
        return next((n for k, n in self.dim_styles.items() if k[:3] == look), None) or self.dim_style(
            *look, self.snap_exe(1.25), 0.0)

    def dim_style(self, txt: float, asz: float, kind: str, exe: float, dle: float) -> str:
        key = (txt, asz, kind, exe, dle)
        if key in self.dim_styles:
            return self.dim_styles[key]
        if not self.dim_styles:
            name = DIM_STYLE
        else:
            base = f"{DIM_STYLE}_{_ARROW_NAMES.get(kind, kind)}{txt:g}_{asz:g}"
            name, k = base, 2
            while name in self.dim_styles.values():
                name, k = f"{base}_{k}", k + 1
        self.dim_styles[key] = name
        ds = self.doc.dimstyles.new(name)
        # Paper millimetres × DIMSCALE (= drawing scale) → model size, as AutoCAD expects.
        ds.dxf.dimscale = self.s
        ds.dxf.dimtxt = txt
        ds.dxf.dimasz = asz
        ds.dxf.dimexe = exe
        # Architectural ticks: a short run of the dimension line past the tick looks like CAD plots.
        ds.dxf.dimdle = dle if kind == "tick" else 0.0
        if kind == "tick" and ds.dxf.dimdle <= 0:
            ds.dxf.dimdle = _nice(0.5 * asz)
        # Gap between the extension-line origin and the object — common on Chinese drawings.
        ds.dxf.dimexo = 0.5
        ds.dxf.dimgap = _nice(0.25 * txt)
        ds.dxf.dimdli = _nice(max(7.0, 3 * txt))
        # From the PDF: 1 = numbers above the line, 0 = numbers on the line.
        ds.dxf.dimtad = getattr(self, "dimtad", 1)
        ds.dxf.dimtih = 0
        ds.dxf.dimtoh = 0
        # 0 = AutoCAD places text from DIMTAD. 2 would pin new DIM text to the click point.
        # Existing dims still override dimtmove=2 so PDF text positions keep.
        ds.dxf.dimtmove = 0
        ds.dxf.dimupt = 0
        ds.dxf.dimdec = 0
        ds.dxf.dimzin = 8
        ds.dxf.dimdsep = ord(".")
        ds.dxf.dimclrd = self.dim_aci
        ds.dxf.dimclre = self.dim_aci
        ds.dxf.dimclrt = self.dim_aci
        ds.dxf.dimtxsty = TEXT_STYLE
        if kind == "tick":
            # Use AutoCAD's _ARCHTICK so a new DIM matches the ticks we bake into existing dims.
            ds.set_arrows(blk=_ARROW_BLOCKS["tick"])
            ds.dxf.dimtsz = 0.0
        else:
            ds.set_arrows(blk=_ARROW_BLOCKS.get(kind, ""))
            ds.dxf.dimtsz = 0.0
        return name

    def finish_dimstyles(self) -> None:
        """Drop ezdxf sample styles, make Standard match PDF, and sync header dimvars.

        setup=True leaves header variables like $DIMLFAC=100, $DIMDEC=2 and
        $DIMTXSTY=OpenSansCondensed-Light. AutoCAD uses those for new DIM commands
        even when $DIMSTYLE is PDF, so measurements come out 100× too large with the
        wrong font and two decimal places.
        """
        doc = self.doc
        keep = {"Standard", *self.dim_styles.values()}
        for ds in list(doc.dimstyles):
            if ds.dxf.name not in keep:
                doc.dimstyles.remove(ds.dxf.name)
        primary = doc.dimstyles.get(DIM_STYLE)
        if primary is None:
            return
        keys = (
            "dimtxt", "dimasz", "dimtsz", "dimexe", "dimdle", "dimexo", "dimgap", "dimdli",
            "dimtad", "dimtih", "dimtoh", "dimtmove", "dimupt", "dimscale", "dimdec", "dimzin", "dimdsep",
            "dimclrd", "dimclre", "dimclrt", "dimtxsty", "dimlfac", "dimblk", "dimblk1", "dimblk2",
            "dimtofl", "dimtdec",
        )
        std = doc.dimstyles.get("Standard")
        if std is not None:
            for key in keys:
                if primary.dxf.hasattr(key):
                    std.dxf.set(key, primary.dxf.get(key))
        # Header $DIMxxx must match the current style — AutoCAD reads these for new dims.
        for key in keys:
            if not primary.dxf.hasattr(key):
                continue
            header = f"${key.upper()}"
            try:
                doc.header[header] = primary.dxf.get(key)
            except Exception:
                pass
        doc.header["$DIMSTYLE"] = DIM_STYLE
        # Tolerance decimals follow primary; keep them from staying at ezdxf's "2".
        doc.header["$DIMTDEC"] = int(primary.dxf.dimdec)
        doc.header["$DIMLFAC"] = float(primary.dxf.dimlfac or 1.0)

    def dim_text(self, text: str, measured_model: float) -> tuple[str, dict]:
        """Keep the measured value live ('<>') when the PDF text agrees with the geometry."""
        text = _dxf_symbols(text)
        pv = parse_value(text)
        override: dict = {}
        if pv is None:
            return text, override
        value, dec, start, end = pv
        if abs(value - measured_model) <= 0.5 * 10 ** -dec + 1e-9:
            override["dimdec"] = dec
            override["dimzin"] = 0 if dec else 8
            return text[:start] + "<>" + text[end:], override
        return text, override

    def dim_colors(self, attribs: dict) -> dict:
        """Explicit part colours: viewers disagree on BYBLOCK under a BYLAYER parent."""
        aci = attribs.get("color", 0)
        return {} if aci == self.dim_aci else {"dimclrd": aci, "dimclre": aci, "dimclrt": aci}

    def linear_dims(self) -> None:
        for dim in self.d.linear_dims:
            layer = DIM_REVIEW_LAYER if dim.review else dim.layer
            a = self.attribs(layer, "dim", dim.color)
            text, ov = self.dim_text(dim.text, dim.measured * self.s)
            ov.update(self.dim_colors(a))
            if not getattr(dim, "ext1", True):
                ov["dimse1"] = 1
            if not getattr(dim, "ext2", True):
                ov["dimse2"] = 1
            # Keep the PDF's own text position; style default is 0 for new AutoCAD DIMs.
            ov["dimtmove"] = 2
            style = self.style_for(self.look(dim))
            exe, dle = self.extents(dim)
            sd = self.doc.dimstyles.get(style).dxf
            if exe != sd.dimexe:
                ov["dimexe"] = exe
            if dle != sd.dimdle:
                ov["dimdle"] = dle
            text_rot = None
            if abs(((dim.text_rotation - dim.angle + 90) % 180) - 90) > 2:
                text_rot = dim.text_rotation
            r = self.msp.add_linear_dim(
                base=self.X(dim.base), p1=self.X(dim.p1), p2=self.X(dim.p2), angle=dim.angle,
                location=self.X(dim.text_pos), text=text, text_rotation=text_rot,
                dimstyle=style, override=ov, dxfattribs=a,
            )
            r.render()
            self.counts["DIMENSION"] += 1

    def radial_dims(self) -> None:
        for dim in self.d.radial_dims:
            a = self.attribs(dim.layer, "dim", dim.color)
            measured = dim.radius * self.s * (2 if dim.kind == "diameter" else 1)
            text, ov = self.dim_text(dim.text, measured)
            # CAD prefixes a live radial measurement with Ø or R itself.
            prefix = "%%c" if dim.kind == "diameter" else "R"
            if f"{prefix}<>" in text or f"{prefix.lower()}<>" in text:
                text = text.replace(f"{prefix}<>", "<>").replace(f"{prefix.lower()}<>", "<>")
            ov.update(dimtix=0, dimtofl=0, dimtih=1, dimtoh=1, **self.dim_colors(a))
            style = self.style_for((self.snap_txt(dim.text_height), self.snap_asz(dim.arrow_size), "closed"))
            fn = self.msp.add_diameter_dim if dim.kind == "diameter" else self.msp.add_radius_dim
            r = fn(
                self.X(dim.center), mpoint=self.X(dim.point), radius=dim.radius * self.s,
                location=self.X(dim.text_pos), text=text,
                dimstyle=style, override=ov, dxfattribs=a,
            )
            r.render()
            self.counts["DIMENSION"] += 1

    def leaders(self) -> None:
        for ld in self.d.leaders:
            a = self.attribs(ld.layer, "dim", ld.color)
            asz = self.snap_asz(ld.arrow_size)
            style = next((n for k, n in self.dim_styles.items() if k[1] == asz), DIM_STYLE)
            ov = self.dim_colors(a)
            if self.doc.dimstyles.get(style).dxf.dimasz != asz:
                ov["dimasz"] = asz
            self.msp.add_leader([self.X(v) for v in ld.vertices], dimstyle=style, override=ov, dxfattribs=a)
            self.counts["LEADER"] += 1
            t = getattr(ld, "text_item", None)
            if t is not None:
                self.text(self.msp, t, layer_kind="dim").dxf.layer = a["layer"]

    # -- hatches --------------------------------------------------------------------
    def hatch(self, h: Hatch) -> None:
        a = self.attribs(h.layer, "hatch", h.color)
        hatch = self.msp.add_hatch(dxfattribs={"layer": a["layer"]})
        for loop in h.loops:
            hatch.paths.add_polyline_path(self.bulge_points(loop, True), is_closed=True)
        ca = color_attribs(h.color)
        aci = ca.get("color", 7)
        if h.solid:
            rgb = None
            if "true_color" in ca:
                rgb = tuple(int(round(v * 255)) for v in color_key(h.color))  # type: ignore[union-attr]
            hatch.set_solid_fill(color=aci, rgb=rgb)
        else:
            lines = []
            for f in h.families:
                r = math.radians(f.angle)
                sp = f.spacing * self.s
                # Offset between successive pattern lines in world coordinates.
                sh = f.shift * self.s
                off = (math.cos(r) * sh - math.sin(r) * sp, math.sin(r) * sh + math.cos(r) * sp)
                # Float noise such as cos(90°) = 6e-17 makes viewers pair line crossings wrongly.
                off = (round(off[0], 6), round(off[1], 6))
                lines.append([round(f.angle, 6), self.X(f.base), off, [round(v * self.s, 6) for v in f.dashes]])
            hatch.set_pattern_fill(h.name, color=aci, angle=0.0, scale=1.0, definition=lines)
            hatch.dxf.pattern_scale = h.pattern_scale * self.s if h.name.startswith("ANSI") else 1.0
            if h.name.startswith("ANSI"):
                hatch.dxf.pattern_angle = (h.families[0].angle - 45) % 360
            if "true_color" in ca:
                hatch.dxf.true_color = ca["true_color"]
        self.counts["HATCH"] += 1

    # -- axis grid ------------------------------------------------------------------
    def axis_bubbles(self) -> None:
        """One block per bubble size: a circle and an editable 轴号 attribute at its centre."""
        from ezdxf.enums import TextEntityAlignment

        from .axes import BUBBLE_LAYER

        made: dict[tuple, str] = {}

        def q(p) -> tuple:
            return (round(p[0] * self.s), round(p[1] * self.s))

        for b in self.d.axis_bubbles:
            r, h = round(b.radius * self.s), round(b.text_height * self.s)
            # Bubbles of one kind measure a hair differently; the first of a kind sets the block.
            snap = max(1, round(0.05 * r))
            key = (round(r / snap), round(h / snap), b.style.width, b.slash is not None)
            sl = tuple(q(p) for p in b.slash) if b.slash else None
            pos = tuple(q(p) for p in b.part_pos) if b.part_pos else None
            name = made.get(key)
            if name is None:
                base = f"轴号_附加_{2 * r}" if sl else f"轴号_{2 * r}"
                name = base if base not in made.values() else f"{base}_{len(made) + 1}"
                blk = self.doc.blocks.new(name)
                ga = {"layer": "0", "color": 0, "lineweight": lineweight(b.style.width)}
                blk.add_circle((0, 0), r, dxfattribs=ga)
                ta = {"height": h, "style": TEXT_STYLE, "layer": "0", "color": 0}
                if sl and pos:
                    blk.add_line(sl[0], sl[1], dxfattribs=ga)
                    for tag, p in (("分子", pos[0]), ("分母", pos[1])):
                        blk.add_attdef(tag, p, dxfattribs={**ta, "prompt": tag}).set_placement(
                            p, align=TextEntityAlignment.MIDDLE_CENTER)
                else:
                    blk.add_attdef("轴号", (0, 0), dxfattribs={**ta, "prompt": "轴号"}).set_placement(
                        (0, 0), align=TextEntityAlignment.MIDDLE_CENTER)
                made[key] = name
            a = self.attribs(BUBBLE_LAYER, "geometry", b.style.color)
            ref = self.msp.add_blockref(name, self.X(b.center), dxfattribs=a)
            if sl:
                num, den = b.label.split("/", 1)
                ref.add_auto_attribs({"分子": num, "分母": den})
            else:
                ref.add_auto_attribs({"轴号": b.label})
            self.counts["INSERT"] += 1
        self.axis_blocks = len(made)

    # -- assemble -------------------------------------------------------------------
    def write(self, consumed: set[int], used_texts: set[int]) -> DXFDocument:
        d = self.d
        doc = self.doc
        doc.styles.new(TEXT_STYLE, dxfattribs={"font": TEXT_FONT, "bigfont": TEXT_BIGFONT, "width": 1.0})
        doc.header["$LWDISPLAY"] = 1
        doc.header["$LTSCALE"] = 1.0
        self.setup_dimstyle()
        self.finish_dimstyles()
        doc.header["$DIMSCALE"] = self.s
        doc.header["$DIMSTYLE"] = DIM_STYLE

        for b in d.blocks:
            blk = doc.blocks.new(b.name)
            for s in b.shapes:
                self.shape(blk, s)
        block_counts = Counter(self.counts)
        self.counts.clear()
        # PDF paint order: if strokes sit on top of fills, emit HATCH first so CAD
        # does not cover walls/text. Otherwise keep fills on top like the PDF.
        self.fills_under = bool(getattr(d, "fills_under_content", True)) and bool(d.hatches)
        if self.fills_under:
            for h in d.hatches:
                self.hatch(h)
        for ins in d.inserts:
            a = {"layer": self.layer_name(ins.layer, "geometry", ins.style), "rotation": ins.rotation}
            self.msp.add_blockref(ins.name, self.X(ins.insert), dxfattribs=a)
            self.counts["INSERT"] += 1

        self.axis_bubbles()
        for s in d.shapes:
            if s.id not in consumed:
                self.shape(self.msp, s)
        if not self.fills_under:
            for h in d.hatches:
                self.hatch(h)
        for t in d.texts:
            if t.id not in used_texts:
                self.text(self.msp, t)
        self.linear_dims()
        self.radial_dims()
        self.leaders()
        self._finish_layers()
        self._finish_draw_order()

        w, h = d.width * self.s, d.height * self.s
        doc.header["$EXTMIN"] = (0, 0, 0)
        doc.header["$EXTMAX"] = (w, h, 0)
        doc.header["$LIMMAX"] = (w, h)
        doc.set_modelspace_vport(height=h * 1.05, center=(w / 2, h / 2))
        self.block_entity_counts = block_counts
        return doc

    def _finish_layers(self) -> None:
        for name, usage in self._layer_usage.items():
            if name == "0":
                continue
            if name not in self.doc.layers:
                self.doc.layers.add(name)
            common = usage.most_common(1)[0][0]
            ca = color_attribs(common)
            layer = self.doc.layers.get(name)
            if "color" in ca:
                layer.color = ca["color"]
            elif "true_color" in ca:
                layer.rgb = tuple(int(round(v * 255)) for v in common)  # type: ignore[union-attr]
        for name in {self.layer_name(i.layer, "geometry", i.style) for i in self.d.inserts}:
            if name not in self.doc.layers:
                self.doc.layers.add(name)
        self._colors_bylayer()

    def _finish_draw_order(self) -> None:
        """Pin solid fills under geometry when the PDF painted them that way."""
        if not getattr(self, "fills_under", False):
            return
        hatches = [e for e in self.msp if e.dxftype() == "HATCH"]
        others = [e for e in self.msp if e.dxftype() != "HATCH"]
        if not hatches or not others:
            return
        # Non-zero sort handles: "0" would force entities to the top in AutoCAD.
        mapping = [(e.dxf.handle, f"{i + 1:X}") for i, e in enumerate(hatches)]
        base = len(hatches) + 1
        mapping.extend((e.dxf.handle, f"{base + i:X}") for i, e in enumerate(others))
        self.msp.set_redraw_order(mapping)
        # Bit 16 = honour SORTENTSTABLE on regen; 127 matches common CAD defaults.
        self.doc.header["$SORTENTS"] = 127

    def _colors_bylayer(self) -> None:
        """Drop explicit colours that equal their layer's colour so they read BYLAYER."""
        layouts = [self.msp, *(b for b in self.doc.blocks if not b.name.startswith("*"))]
        for lay in layouts:
            for e in lay:
                # Dimension and leader parts are BYBLOCK; viewers disagree on resolving
                # BYBLOCK through a BYLAYER parent, so those keep an explicit colour.
                if e.dxftype() in ("HATCH", "DIMENSION", "LEADER") or not e.dxf.hasattr("layer"):
                    continue
                layer = self.doc.layers.get(e.dxf.layer)
                if layer is None:
                    continue
                if e.dxf.hasattr("true_color"):
                    if layer.has_dxf_attrib("true_color") and layer.dxf.true_color == e.dxf.true_color:
                        e.dxf.discard("true_color")
                        e.dxf.color = 256
                elif e.dxf.hasattr("color") and e.dxf.color == layer.color:
                    e.dxf.color = 256
