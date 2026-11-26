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
_CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")
DIM_STYLE = "PDF"

_ACI = {
    (1.0, 0.0, 0.0): 1, (1.0, 1.0, 0.0): 2, (0.0, 1.0, 0.0): 3, (0.0, 1.0, 1.0): 4,
    (0.0, 0.0, 1.0): 5, (1.0, 0.0, 1.0): 6, (0.0, 0.0, 0.0): 7, (1.0, 1.0, 1.0): 7,
}
_ARROW_BLOCKS = {"closed": "", "closed_blank": "CLOSEDBLANK", "open": "OPEN", "tick": "ARCHTICK", "dot": "DOT"}
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
            # Width of the lettering in the PDF over its natural width in the CAD font.
            natural = t.height * sum(1.15 if _CJK.match(ch) else 0.6 for ch in t.text)
            a["width"] = round(min(2.0, max(0.4, t.width / natural)), 3)
        self.counts["TEXT"] += 1
        return target.add_text(t.text, height=t.height * self.s, dxfattribs=a)

    # -- dimensions -----------------------------------------------------------------
    def setup_dimstyle(self) -> None:
        d = self.d
        th = [x.text_height for x in d.linear_dims] + [x.text_height for x in d.radial_dims] + [x.text_height for x in d.leaders]
        az = [x.arrow_size for x in d.linear_dims] + [x.arrow_size for x in d.radial_dims] + [x.arrow_size for x in d.leaders]
        ex = [x.exe for x in d.linear_dims if x.exe > 0]
        kinds = Counter(x.arrow_kind for x in d.linear_dims)
        ds = self.doc.dimstyles.new(DIM_STYLE)
        # Sizes are written in model units with DIMSCALE 1: some CAD programs
        # (LibreCAD among them) ignore DIMSCALE when they regenerate dimensions.
        ds.dxf.dimtxt = self.L(median(th) if th else 2.5)
        ds.dxf.dimasz = self.L(median(az) if az else 2.5)
        ds.dxf.dimexe = self.L(median(ex) if ex else 1.25)
        ds.dxf.dimexo = 0.0
        ds.dxf.dimgap = self.L(0.6)
        ds.dxf.dimtad = 1
        ds.dxf.dimtih = 0
        ds.dxf.dimtoh = 0
        ds.dxf.dimtmove = 2
        ds.dxf.dimscale = 1.0
        ds.dxf.dimdec = 0
        ds.dxf.dimzin = 8
        ds.dxf.dimdsep = ord(".")
        ds.dxf.dimclrd = 0
        ds.dxf.dimclre = 0
        ds.dxf.dimclrt = 0
        ds.dxf.dimtxsty = TEXT_STYLE
        if kinds:
            ds.set_arrows(blk=_ARROW_BLOCKS.get(kinds.most_common(1)[0][0], ""))

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

    @staticmethod
    def dim_colors(attribs: dict) -> dict:
        """Explicit part colours: viewers disagree on BYBLOCK under a BYLAYER parent."""
        aci = attribs.get("color")
        return {"dimclrd": aci, "dimclre": aci, "dimclrt": aci} if aci else {}

    def linear_dims(self) -> None:
        for dim in self.d.linear_dims:
            a = self.attribs(dim.layer, "dim", dim.color)
            text, ov = self.dim_text(dim.text, dim.measured * self.s)
            ov.update(self.dim_colors(a))
            ov.update(
                dimtxt=self.L(dim.text_height), dimasz=self.L(dim.arrow_size),
                dimexe=self.L(dim.exe), dimse1=0 if getattr(dim, "ext1", True) else 1,
                dimse2=0 if getattr(dim, "ext2", True) else 1,
            )
            if getattr(dim, "dle", 0) > 0:
                ov["dimdle"] = self.L(dim.dle)
            blk = _ARROW_BLOCKS.get(dim.arrow_kind, "")
            ov["dimblk"] = blk
            text_rot = None
            if abs(((dim.text_rotation - dim.angle + 90) % 180) - 90) > 2:
                text_rot = dim.text_rotation
            r = self.msp.add_linear_dim(
                base=self.X(dim.base), p1=self.X(dim.p1), p2=self.X(dim.p2), angle=dim.angle,
                location=self.X(dim.text_pos), text=text, text_rotation=text_rot,
                dimstyle=DIM_STYLE, override=ov, dxfattribs=a,
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
            ov.update(
                dimtxt=self.L(dim.text_height), dimasz=self.L(dim.arrow_size), dimtix=0, dimtofl=0,
                dimtih=1, dimtoh=1, **self.dim_colors(a),
            )
            fn = self.msp.add_diameter_dim if dim.kind == "diameter" else self.msp.add_radius_dim
            r = fn(
                self.X(dim.center), mpoint=self.X(dim.point), radius=dim.radius * self.s,
                location=self.X(dim.text_pos), text=text,
                dimstyle=DIM_STYLE, override=ov, dxfattribs=a,
            )
            r.render()
            self.counts["DIMENSION"] += 1

    def leaders(self) -> None:
        for ld in self.d.leaders:
            a = self.attribs(ld.layer, "dim", ld.color)
            self.msp.add_leader(
                [self.X(v) for v in ld.vertices], dimstyle=DIM_STYLE,
                override={"dimasz": self.L(ld.arrow_size), **self.dim_colors(a)}, dxfattribs=a,
            )
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
        doc.styles.new(TEXT_STYLE, dxfattribs={"font": "simsun.ttc", "width": 1.0})
        doc.header["$LWDISPLAY"] = 1
        doc.header["$LTSCALE"] = 1.0
        doc.header["$DIMSCALE"] = 1.0
        self.setup_dimstyle()
        doc.header["$DIMSTYLE"] = DIM_STYLE

        for b in d.blocks:
            blk = doc.blocks.new(b.name)
            for s in b.shapes:
                self.shape(blk, s)
        block_counts = Counter(self.counts)
        self.counts.clear()
        for ins in d.inserts:
            a = {"layer": self.layer_name(ins.layer, "geometry", ins.style), "rotation": ins.rotation}
            self.msp.add_blockref(ins.name, self.X(ins.insert), dxfattribs=a)
            self.counts["INSERT"] += 1

        self.axis_bubbles()
        for s in d.shapes:
            if s.id not in consumed:
                self.shape(self.msp, s)
        for h in d.hatches:
            self.hatch(h)
        for t in d.texts:
            if t.id not in used_texts:
                self.text(self.msp, t)
        self.linear_dims()
        self.radial_dims()
        self.leaders()
        self._finish_layers()

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
