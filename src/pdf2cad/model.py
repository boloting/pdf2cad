"""Intermediate drawing model shared by the extractor, recognizers and writer.

Coordinates are paper millimetres with the y axis pointing up and the origin at
the lower-left corner of the PDF page. The writer applies the drawing scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Union

from .geom import Vec, add, bezier_point, dist, mul, rotate

Color = tuple[float, float, float]
Transform = Callable[[Vec], Vec]


@dataclass(frozen=True)
class Style:
    layer: str | None
    color: Color | None
    width: float
    dashes: tuple[float, ...] = ()


@dataclass
class Line:
    a: Vec
    b: Vec

    def start(self) -> Vec:
        return self.a

    def end(self) -> Vec:
        return self.b

    def reversed(self) -> "Line":
        return Line(self.b, self.a)

    def flatten(self) -> list[Vec]:
        return [self.a, self.b]

    def transformed(self, f: Transform, rot: float = 0.0) -> "Line":
        return Line(f(self.a), f(self.b))

    @property
    def length(self) -> float:
        return dist(self.a, self.b)


@dataclass
class Arc:
    center: Vec
    radius: float
    start_angle: float
    sweep: float  # degrees, positive counter-clockwise

    @property
    def is_full(self) -> bool:
        return abs(self.sweep) >= 359.9

    def point_at(self, deg: float) -> Vec:
        r = math.radians(deg)
        return (self.center[0] + self.radius * math.cos(r), self.center[1] + self.radius * math.sin(r))

    def start(self) -> Vec:
        return self.point_at(self.start_angle)

    def end(self) -> Vec:
        return self.point_at(self.start_angle + self.sweep)

    def reversed(self) -> "Arc":
        return Arc(self.center, self.radius, (self.start_angle + self.sweep) % 360, -self.sweep)

    def flatten(self, step_deg: float = 10.0) -> list[Vec]:
        n = max(2, int(abs(self.sweep) / step_deg) + 1)
        return [self.point_at(self.start_angle + self.sweep * i / (n - 1)) for i in range(n)]

    def ccw_angles(self) -> tuple[float, float]:
        """Start and end angles of the same arc traversed counter-clockwise, as DXF stores it."""
        if self.sweep >= 0:
            return self.start_angle % 360, (self.start_angle + self.sweep) % 360
        return (self.start_angle + self.sweep) % 360, self.start_angle % 360

    def transformed(self, f: Transform, rot: float = 0.0) -> "Arc":
        c = f(self.center)
        return Arc(c, self.radius, (self.start_angle + rot) % 360, self.sweep)


@dataclass
class Bezier:
    p0: Vec
    p1: Vec
    p2: Vec
    p3: Vec

    def start(self) -> Vec:
        return self.p0

    def end(self) -> Vec:
        return self.p3

    def reversed(self) -> "Bezier":
        return Bezier(self.p3, self.p2, self.p1, self.p0)

    def at(self, t: float) -> Vec:
        return bezier_point(self.p0, self.p1, self.p2, self.p3, t)

    def flatten(self, n: int = 8) -> list[Vec]:
        return [self.at(i / n) for i in range(n + 1)]

    def transformed(self, f: Transform, rot: float = 0.0) -> "Bezier":
        return Bezier(f(self.p0), f(self.p1), f(self.p2), f(self.p3))


Seg = Union[Line, Arc, Bezier]


@dataclass
class Shape:
    """One PDF subpath with the style of the path that drew it."""

    segs: list[Seg]
    closed: bool
    style: Style
    stroke: bool = True
    fill: Color | None = None
    path_id: int = -1
    id: int = -1

    def points(self) -> list[Vec]:
        pts: list[Vec] = []
        for s in self.segs:
            fl = s.flatten()
            pts.extend(fl if not pts else fl[1:])
        return pts

    def start(self) -> Vec:
        return self.segs[0].start()

    def end(self) -> Vec:
        return self.segs[-1].end()

    def bbox(self) -> tuple[float, float, float, float]:
        pts = self.points()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def is_single_line(self) -> bool:
        return len(self.segs) == 1 and isinstance(self.segs[0], Line)

    @property
    def is_circle(self) -> bool:
        return len(self.segs) == 1 and isinstance(self.segs[0], Arc) and self.segs[0].is_full

    def transformed(self, f: Transform, rot: float) -> "Shape":
        return Shape(
            [s.transformed(f, rot) for s in self.segs], self.closed, self.style,
            self.stroke, self.fill, self.path_id, self.id,
        )


@dataclass
class TextItem:
    text: str
    insert: Vec
    height: float
    rotation: float
    layer: str | None
    color: Color | None
    font: str = ""
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    width: float = 0.0
    id: int = -1

    @property
    def center(self) -> Vec:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


@dataclass
class Arrow:
    tip: Vec
    direction: Vec  # unit vector from the arrow's base towards its tip
    size: float
    kind: str  # "closed" (filled triangle) or "tick" (oblique stroke)
    shape: Shape


@dataclass
class LinearDim:
    p1: Vec
    p2: Vec
    base: Vec
    angle: float
    text: str
    text_pos: Vec
    text_height: float
    text_rotation: float
    arrow_size: float
    arrow_kind: str
    exe: float
    measured: float
    layer: str | None
    color: Color | None


@dataclass
class RadialDim:
    kind: str  # "radius" or "diameter"
    center: Vec
    radius: float
    point: Vec
    text: str
    text_pos: Vec
    text_height: float
    arrow_size: float
    layer: str | None
    color: Color | None


@dataclass
class Leader:
    vertices: list[Vec]
    text: str
    text_pos: Vec
    text_height: float
    text_rotation: float
    arrow_size: float
    layer: str | None
    color: Color | None


@dataclass
class HatchFamily:
    angle: float
    base: Vec
    spacing: float


@dataclass
class Hatch:
    loops: list[list[Seg]]
    solid: bool
    color: Color | None
    layer: str | None
    families: list[HatchFamily] = field(default_factory=list)
    name: str = "SOLID"
    pattern_scale: float = 1.0


@dataclass
class BlockDef:
    name: str
    shapes: list[Shape]


@dataclass
class Insert:
    name: str
    insert: Vec
    rotation: float
    layer: str | None
    style: Style | None = None


@dataclass
class Drawing:
    width: float
    height: float
    shapes: list[Shape] = field(default_factory=list)
    texts: list[TextItem] = field(default_factory=list)
    linear_dims: list[LinearDim] = field(default_factory=list)
    radial_dims: list[RadialDim] = field(default_factory=list)
    leaders: list[Leader] = field(default_factory=list)
    hatches: list[Hatch] = field(default_factory=list)
    blocks: list[BlockDef] = field(default_factory=list)
    inserts: list[Insert] = field(default_factory=list)
    has_pdf_layers: bool = False


def translate_rotate(offset: Vec, deg: float) -> Transform:
    """p -> rotate(p, deg) + offset."""
    return lambda p: add(rotate(p, deg), offset)


__all__ = [
    "Arc", "Arrow", "BlockDef", "Bezier", "Color", "Drawing", "Hatch", "HatchFamily", "Insert",
    "Leader", "Line", "LinearDim", "RadialDim", "Seg", "Shape", "Style", "TextItem", "mul",
    "translate_rotate",
]
