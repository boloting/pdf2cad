"""Small 2D geometry toolkit. Points are ``(x, y)`` tuples in paper millimetres."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Vec = tuple[float, float]


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k)


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Vec, b: Vec) -> float:
    return a[0] * b[1] - a[1] * b[0]


def length(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def dist(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mid(a: Vec, b: Vec) -> Vec:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def unit(a: Vec) -> Vec:
    n = length(a)
    return (a[0] / n, a[1] / n) if n > 1e-12 else (0.0, 0.0)


def perp(a: Vec) -> Vec:
    return (-a[1], a[0])


def rotate(p: Vec, deg: float, origin: Vec = (0.0, 0.0)) -> Vec:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    x, y = p[0] - origin[0], p[1] - origin[1]
    return (origin[0] + x * c - y * s, origin[1] + x * s + y * c)


def angle_of(v: Vec) -> float:
    """Direction of ``v`` in degrees, in [0, 360)."""
    return math.degrees(math.atan2(v[1], v[0])) % 360.0


def line_angle(a: Vec, b: Vec) -> float:
    """Undirected angle of segment ab in degrees, in [0, 180)."""
    return angle_of(sub(b, a)) % 180.0


def angle_diff(a: float, b: float, period: float = 180.0) -> float:
    d = (a - b) % period
    return min(d, period - d)


def point_line_distance(p: Vec, a: Vec, b: Vec) -> float:
    """Distance from p to the infinite line through a and b."""
    d = sub(b, a)
    n = length(d)
    if n < 1e-12:
        return dist(p, a)
    return abs(cross(d, sub(p, a))) / n


def point_segment_distance(p: Vec, a: Vec, b: Vec) -> float:
    d = sub(b, a)
    n2 = dot(d, d)
    if n2 < 1e-18:
        return dist(p, a)
    t = max(0.0, min(1.0, dot(sub(p, a), d) / n2))
    return dist(p, add(a, mul(d, t)))


def project_param(p: Vec, a: Vec, b: Vec) -> float:
    d = sub(b, a)
    n2 = dot(d, d)
    return dot(sub(p, a), d) / n2 if n2 > 1e-18 else 0.0


def segments_intersect(a: Vec, b: Vec, c: Vec, d: Vec) -> bool:
    d1 = cross(sub(b, a), sub(c, a))
    d2 = cross(sub(b, a), sub(d, a))
    d3 = cross(sub(d, c), sub(a, c))
    d4 = cross(sub(d, c), sub(b, c))
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def segment_segment_distance(a: Vec, b: Vec, c: Vec, d: Vec) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def circumcenter(a: Vec, b: Vec, c: Vec) -> Vec | None:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy)


def bezier_point(p0: Vec, p1: Vec, p2: Vec, p3: Vec, t: float) -> Vec:
    u = 1 - t
    a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    )


def polygon_area(pts: Sequence[Vec]) -> float:
    """Signed area, positive for counter-clockwise polygons."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2


def point_in_polygon(p: Vec, pts: Sequence[Vec]) -> bool:
    x, y = p
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            xc = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < xc:
                inside = not inside
        j = i
    return inside


def distance_to_polygon(p: Vec, pts: Sequence[Vec], closed: bool = True) -> float:
    n = len(pts)
    if n == 1:
        return dist(p, pts[0])
    best = math.inf
    last = n if closed else n - 1
    for i in range(last):
        best = min(best, point_segment_distance(p, pts[i], pts[(i + 1) % n]))
    return best


def convex_hull(points: Iterable[Vec]) -> list[Vec]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def half(seq):
        out: list[Vec] = []
        for p in seq:
            while len(out) >= 2 and cross(sub(out[-1], out[-2]), sub(p, out[-2])) <= 1e-12:
                out.pop()
            out.append(p)
        return out

    lower = half(pts)
    upper = half(reversed(pts))
    return lower[:-1] + upper[:-1]


def bbox(points: Iterable[Vec]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_overlap(a, b, tol: float = 0.0) -> bool:
    return not (a[2] + tol < b[0] or b[2] + tol < a[0] or a[3] + tol < b[1] or b[3] + tol < a[1])
