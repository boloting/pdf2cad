"""Generate a sample vector PDF drawing, shaped like a PDF plotted from CAD.

The drawing is a mounting plate at 1:2 on A4 landscape. It contains layered
geometry, centre lines, filled-arrow dimensions, a section hatch, repeated
bolt-hole symbols and Chinese text, so every recognizer has something to find.
"""

from __future__ import annotations

import math

import pymupdf

MM = 72 / 25.4
PAGE_W, PAGE_H = 297 * MM, 210 * MM
SCALE = 2  # 1:2 — one paper mm is two model mm


def P(x: float, y: float) -> pymupdf.Point:
    """Paper millimetres with y up, from the lower-left corner of the sheet."""
    return pymupdf.Point(x * MM, PAGE_H - y * MM)


class Sheet:
    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.layers = {
            name: self.doc.add_ocg(name, on=True)
            for name in ("轮廓线", "中心线", "标注", "剖面线", "文字", "图框")
        }

    def shape(self) -> pymupdf.Shape:
        return self.page.new_shape()

    def lines(self, segs, layer, width=0.35, dashes=None, color=(0, 0, 0)):
        sh = self.shape()
        for a, b in segs:
            sh.draw_line(P(*a), P(*b))
            sh.finish(color=color, width=width * MM, dashes=dashes, closePath=False, oc=self.layers[layer])
        sh.commit()

    def polyline(self, pts, layer, width=0.35, closed=True):
        sh = self.shape()
        sh.draw_polyline([P(*p) for p in pts])
        sh.finish(color=(0, 0, 0), width=width * MM, closePath=closed, oc=self.layers[layer])
        sh.commit()

    def circle(self, c, r, layer, width=0.35):
        sh = self.shape()
        sh.draw_circle(P(*c), r * MM)
        sh.finish(color=(0, 0, 0), width=width * MM, oc=self.layers[layer])
        sh.commit()

    def arc(self, c, r, a0, a1, layer, width=0.35):
        """Counter-clockwise arc in paper coordinates, drawn as cubic Béziers."""
        sh = self.shape()
        n = max(1, math.ceil(abs(a1 - a0) / 90))
        step = (a1 - a0) / n
        k = 4 / 3 * math.tan(math.radians(step) / 4)
        for i in range(n):
            t0 = math.radians(a0 + i * step)
            t1 = math.radians(a0 + (i + 1) * step)
            p0 = (c[0] + r * math.cos(t0), c[1] + r * math.sin(t0))
            p3 = (c[0] + r * math.cos(t1), c[1] + r * math.sin(t1))
            p1 = (p0[0] - k * r * math.sin(t0), p0[1] + k * r * math.cos(t0))
            p2 = (p3[0] + k * r * math.sin(t1), p3[1] - k * r * math.cos(t1))
            sh.draw_bezier(P(*p0), P(*p1), P(*p2), P(*p3))
        sh.finish(color=(0, 0, 0), width=width * MM, closePath=False, oc=self.layers[layer])
        sh.commit()

    def text(self, pos, s, height, layer="文字", rotate=0):
        # CAD text height is the cap height; the PDF font size is the em size.
        self.page.insert_text(
            P(*pos), s, fontsize=height / 0.7 * MM, fontname="china-s",
            rotate=rotate, oc=self.layers[layer], color=(0, 0, 0),
        )

    def arrow(self, tip, direction, size=3.0):
        """Filled arrowhead with its tip at ``tip`` pointing along ``direction``."""
        dx, dy = direction
        n = math.hypot(dx, dy)
        dx, dy = dx / n, dy / n
        bx, by = tip[0] - dx * size, tip[1] - dy * size
        w = size / 6
        sh = self.shape()
        sh.draw_polyline([P(*tip), P(bx - dy * w, by + dx * w), P(bx + dy * w, by - dx * w)])
        sh.finish(color=(0, 0, 0), fill=(0, 0, 0), width=0.18 * MM, closePath=True, oc=self.layers["标注"])
        sh.commit()

    def linear_dim(self, p1, p2, offset, text, horizontal=True):
        """Linear dimension in CAD style: extension lines, dim line, two arrows, text."""
        thin = 0.18
        if horizontal:
            y = offset
            ext = [((p1[0], p1[1] + (1 if y > p1[1] else -1) * 1.5), (p1[0], y + (2 if y > p1[1] else -2))),
                   ((p2[0], p2[1] + (1 if y > p2[1] else -1) * 1.5), (p2[0], y + (2 if y > p2[1] else -2)))]
            a, b = (p1[0], y), (p2[0], y)
            self.lines(ext + [(a, b)], "标注", width=thin)
            self.arrow(a, (-1, 0))
            self.arrow(b, (1, 0))
            w = len(text) * 3.5 * 0.7
            self.text(((a[0] + b[0]) / 2 - w / 2, y + 1), text, 3.5, layer="标注")
        else:
            x = offset
            ext = [((p1[0] + (1 if x > p1[0] else -1) * 1.5, p1[1]), (x + (2 if x > p1[0] else -2), p1[1])),
                   ((p2[0] + (1 if x > p2[0] else -1) * 1.5, p2[1]), (x + (2 if x > p2[0] else -2), p2[1]))]
            a, b = (x, p1[1]), (x, p2[1])
            self.lines(ext + [(a, b)], "标注", width=thin)
            self.arrow(a, (0, -1))
            self.arrow(b, (0, 1))
            w = len(text) * 3.5 * 0.7
            self.text((x - 1, (a[1] + b[1]) / 2 - w / 2), text, 3.5, layer="标注", rotate=90)


def bolt_hole(s: Sheet, c, r=3.0):
    s.circle(c, r, "轮廓线")
    cl = r + 2
    s.lines([((c[0] - cl, c[1]), (c[0] + cl, c[1])), ((c[0], c[1] - cl), (c[0], c[1] + cl))],
            "中心线", width=0.18, dashes="[6 1.5 1 1.5] 0", color=(1, 0, 0))


def build() -> pymupdf.Document:
    s = Sheet()

    s.polyline([(10, 10), (287, 10), (287, 200), (10, 200)], "图框", width=0.7)
    s.polyline([(187, 10), (287, 10), (287, 40), (187, 40)], "图框", width=0.5)
    s.lines([((187, 25), (287, 25)), ((227, 10), (227, 40))], "图框", width=0.25)
    s.text((190, 30), "安装板  1:2", 5)
    s.text((190, 15), "材料 Q235  厚 10", 3.5)
    s.text((230, 15), "图号 AZB-001", 3.5)

    # Plate 200 x 120 model mm -> 100 x 60 paper mm, corner at (40, 70)
    x0, y0, w, h = 40, 70, 100, 60
    s.polyline([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)], "轮廓线", width=0.5)

    holes = [(x0 + 10, y0 + 10), (x0 + w - 10, y0 + 10), (x0 + w - 10, y0 + h - 10), (x0 + 10, y0 + h - 10)]
    for c in holes:
        bolt_hole(s, c)

    # Central slot: two straight edges joined by semicircular ends
    cx, cy, half, r = x0 + w / 2, y0 + h / 2, 15, 6
    s.lines([((cx - half, cy - r), (cx + half, cy - r)), ((cx - half, cy + r), (cx + half, cy + r))], "轮廓线", width=0.5)
    s.arc((cx + half, cy), r, -90, 90, "轮廓线", width=0.5)
    s.arc((cx - half, cy), r, 90, 270, "轮廓线", width=0.5)
    s.lines([((cx - half - r - 3, cy), (cx + half + r + 3, cy))], "中心线", width=0.18,
            dashes="[6 1.5 1 1.5] 0", color=(1, 0, 0))

    # Section view A-A below the frame line: rectangle 100 x 5 hatched at 45 degrees
    sx0, sy0, sw, sh_ = 40, 40, 100, 5
    s.polyline([(sx0, sy0), (sx0 + sw, sy0), (sx0 + sw, sy0 + sh_), (sx0, sy0 + sh_)], "轮廓线", width=0.5)
    hatch = s.shape()
    spacing = 2.0
    k = -sh_
    while k < sw:
        a = (sx0 + max(k, 0), sy0 + max(0, -k))
        b_x = min(k + sh_, sw)
        b = (sx0 + b_x, sy0 + (b_x - k))
        if b[0] - a[0] > 0.2:
            hatch.draw_line(P(*a), P(*b))
        k += spacing * math.sqrt(2)
    hatch.finish(color=(0, 0, 0), width=0.13 * MM, closePath=False, oc=s.layers["剖面线"])
    hatch.commit()
    s.text((80, 33), "A-A", 3.5)

    s.linear_dim((x0, y0 + h), (x0 + w, y0 + h), y0 + h + 12, "200")
    s.linear_dim((holes[3][0], y0 + h), (holes[2][0], y0 + h), y0 + h + 6, "160")
    s.linear_dim((x0, y0), (x0, y0 + h), x0 - 12, "120", horizontal=False)
    s.linear_dim((x0 + w, holes[1][1]), (x0 + w, holes[2][1]), x0 + w + 8, "80", horizontal=False)

    # Diameter leader for the bolt holes
    c = holes[2]
    ang = math.radians(45)
    tip = (c[0] + 3 * math.cos(ang), c[1] + 3 * math.sin(ang))
    knee = (c[0] + 14 * math.cos(ang), c[1] + 14 * math.sin(ang))
    s.lines([(tip, knee), (knee, (knee[0] + 16, knee[1]))], "标注", width=0.18)
    s.arrow(tip, (-math.cos(ang), -math.sin(ang)))
    s.text((knee[0] + 1, knee[1] + 1), "4×Φ12", 3.5, layer="标注")

    s.text((150, 120), "技术要求:", 3.5)
    s.text((150, 113), "1. 未注倒角 C1;", 3.5)
    s.text((150, 106), "2. 去毛刺,锐边倒钝。", 3.5)

    return s.doc


def sample_pdf_bytes() -> bytes:
    doc = build()
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    with open(out, "wb") as f:
        f.write(sample_pdf_bytes())
    print(out)
