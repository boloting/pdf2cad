"""Synthetic vector PDFs that mimic CAD plots, for recognizer tests."""

from __future__ import annotations

import math

import pymupdf

MM = 72 / 25.4
W, H = 297, 210


class Page:
    """Paper millimetres, y up, no PDF layers: style is carried by colour and width only."""

    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=W * MM, height=H * MM)

    def P(self, x, y):
        return pymupdf.Point(x * MM, (H - y) * MM)

    def line(self, a, b, width=0.25, color=(0, 0, 0)):
        sh = self.page.new_shape()
        sh.draw_line(self.P(*a), self.P(*b))
        sh.finish(color=color, width=width * MM, closePath=False)
        sh.commit()

    def poly(self, pts, closed=True, width=0.35, color=(0, 0, 0), fill=None):
        sh = self.page.new_shape()
        sh.draw_polyline([self.P(*p) for p in pts])
        sh.finish(color=None if fill is not None and width == 0 else color, fill=fill, width=width * MM, closePath=closed)
        sh.commit()

    def circle(self, c, r, width=0.35, color=(0, 0, 0)):
        sh = self.page.new_shape()
        sh.draw_circle(self.P(*c), r * MM)
        sh.finish(color=color, width=width * MM)
        sh.commit()

    def arc(self, c, r, a0, a1, width=0.25, color=(0, 0, 0)):
        sh = self.page.new_shape()
        n = max(1, math.ceil(abs(a1 - a0) / 90))
        step = (a1 - a0) / n
        k = 4 / 3 * math.tan(math.radians(step) / 4)
        for i in range(n):
            t0, t1 = math.radians(a0 + i * step), math.radians(a0 + (i + 1) * step)
            p0 = (c[0] + r * math.cos(t0), c[1] + r * math.sin(t0))
            p3 = (c[0] + r * math.cos(t1), c[1] + r * math.sin(t1))
            p1 = (p0[0] - k * r * math.sin(t0), p0[1] + k * r * math.cos(t0))
            p2 = (p3[0] + k * r * math.sin(t1), p3[1] - k * r * math.cos(t1))
            sh.draw_bezier(self.P(*p0), self.P(*p1), self.P(*p2), self.P(*p3))
        sh.finish(color=color, width=width * MM, closePath=False)
        sh.commit()

    def text(self, pos, s, height=2.5, rotate=0, color=(0, 0, 0)):
        # insert_text only rotates in 90° steps; use a morph for anything else.
        p = self.P(*pos)
        if rotate % 90 == 0:
            self.page.insert_text(p, s, fontsize=height / 0.7 * MM, fontname="helv", rotate=rotate, color=color)
        else:
            m = pymupdf.Matrix(rotate)
            self.page.insert_text(p, s, fontsize=height / 0.7 * MM, fontname="helv", morph=(p, m), color=color)

    def arrow(self, tip, d, size=2.5, color=(0, 0, 0)):
        n = math.hypot(*d)
        dx, dy = d[0] / n, d[1] / n
        bx, by = tip[0] - dx * size, tip[1] - dy * size
        w = size / 6
        self.poly([tip, (bx - dy * w, by + dx * w), (bx + dy * w, by - dx * w)], width=0.1, color=color, fill=color)

    def tick(self, p, size=2.0, color=(0, 0, 0)):
        h = size / 2
        self.line((p[0] - h, p[1] - h), (p[0] + h, p[1] + h), width=0.35, color=color)

    def save(self) -> bytes:
        return self.doc.tobytes()


BLUE = (0, 0, 1)
GREEN = (0, 0.6, 0)


def aligned_dim(pg: Page, p1, p2, offset, text, color=BLUE):
    """Dimension parallel to p1-p2, offset to its left, with filled arrows."""
    ux, uy = p2[0] - p1[0], p2[1] - p1[1]
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    nx, ny = -uy, ux
    a = (p1[0] + nx * offset, p1[1] + ny * offset)
    b = (p2[0] + nx * offset, p2[1] + ny * offset)
    for p, q in ((p1, a), (p2, b)):
        pg.line((p[0] + nx * 1.0, p[1] + ny * 1.0), (q[0] + nx * 1.5, q[1] + ny * 1.5), width=0.18, color=color)
    pg.line(a, b, width=0.18, color=color)
    pg.arrow(a, (-ux, -uy), color=color)
    pg.arrow(b, (ux, uy), color=color)
    ang = math.degrees(math.atan2(uy, ux))
    w = len(text) * 2.5 * 0.55
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    pos = (mx - ux * w / 2 + nx * 0.8, my - uy * w / 2 + ny * 0.8)
    pg.text(pos, text, 2.5, rotate=ang, color=color)


def tick_dim(pg: Page, x1, x2, y, y_obj, text, dle=1.5, color=BLUE):
    """Horizontal architectural dimension with oblique ticks and extended dim line."""
    for x in (x1, x2):
        pg.line((x, y_obj + 1.0), (x, y + 2.0), width=0.18, color=color)
        pg.tick((x, y), color=color)
    pg.line((x1 - dle, y), (x2 + dle, y), width=0.18, color=color)
    w = len(text) * 2.5 * 0.55
    pg.text(((x1 + x2) / 2 - w / 2, y + 0.8), text, 2.5, color=color)


def door(pg: Page, hinge, rot, r=9.0):
    """Door leaf plus 90° swing arc, rotated by rot degrees around the hinge."""
    def R(p):
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        return (hinge[0] + p[0] * c - p[1] * s, hinge[1] + p[0] * s + p[1] * c)

    pg.line(R((0, 0)), R((0, r)), width=0.25, color=GREEN)
    pg.arc(hinge, r, rot, rot + 90, width=0.18, color=GREEN)
    pg.line(R((0, 0)), R((r, 0)), width=0.18, color=GREEN)


def plan_pdf() -> bytes:
    """A 1:100 floor plan: walls, rotated door blocks, chained tick dims, aligned dim, hatches."""
    pg = Page()
    # Outer walls (one closed polyline) — 120 x 70 paper mm = 12 m x 7 m at 1:100
    x0, y0 = 40, 60
    pg.poly([(x0, y0), (x0 + 120, y0), (x0 + 120, y0 + 70), (x0, y0 + 70)], width=0.5)

    # Four identical doors in four orientations, standing free of the walls
    for i, rot in enumerate((0, 90, 180, 270)):
        door(pg, (x0 + 20 + i * 25, y0 + 30), rot)

    # Chained dimensions along the top: 4000 + 5000 + 3000 = 12000
    xs = [x0, x0 + 40, x0 + 90, x0 + 120]
    for (a, b), t in zip(zip(xs, xs[1:]), ("4000", "5000", "3000")):
        tick_dim(pg, a, b, y0 + 80, y0 + 70, t)

    # Aligned dimension on a 30° chamfer edge drawn separately
    c1 = (190, 60)
    c2 = (190 + 40 * math.cos(math.radians(30)), 60 + 40 * math.sin(math.radians(30)))
    pg.line(c1, c2, width=0.35)
    aligned_dim(pg, c1, c2, 8, "4000")

    # Hatched slab with a round opening (island), 45° lines 1.5 mm apart
    hx, hy, hw, hh = 40, 20, 60, 25
    pg.poly([(hx, hy), (hx + hw, hy), (hx + hw, hy + hh), (hx, hy + hh)], width=0.35)
    cc, cr = (hx + 30, hy + 12.5), 6.0
    pg.circle(cc, cr, width=0.35)
    sh = pg.page.new_shape()
    s = 1.5
    k = -hh
    while k < hw:
        # line x - y = k - hx + hy clipped to the rectangle, then split around the circle
        ax, bx = max(hx, hx + k), min(hx + hw, hx + k + hh)
        if bx - ax > 0.05:
            a = (ax, hy + (ax - hx - k))
            b = (bx, hy + (bx - hx - k))
            # intersection with circle along direction (1,1)/sqrt2
            ux = uy = 1 / math.sqrt(2)
            fx, fy = a[0] - cc[0], a[1] - cc[1]
            bq = 2 * (fx * ux + fy * uy)
            cq = fx * fx + fy * fy - cr * cr
            disc = bq * bq - 4 * cq
            segs = [(a, b)]
            if disc > 0:
                t1, t2 = (-bq - math.sqrt(disc)) / 2, (-bq + math.sqrt(disc)) / 2
                L = math.dist(a, b)
                if 0 < t1 < L and 0 < t2 < L:
                    segs = [(a, (a[0] + ux * t1, a[1] + uy * t1)), ((a[0] + ux * t2, a[1] + uy * t2), b)]
            for p, q in segs:
                if math.dist(p, q) > 0.05:
                    sh.draw_line(pg.P(*p), pg.P(*q))
        k += s * math.sqrt(2)
    sh.finish(color=(0.5, 0.5, 0.5), width=0.13 * MM, closePath=False)
    sh.commit()

    # Solid black column, fill only
    pg.poly([(120, 20), (130, 20), (130, 30), (120, 30)], width=0, fill=(0, 0, 0))

    pg.text((40, 190), "FLOOR PLAN 1:100", 5)
    return pg.save()
