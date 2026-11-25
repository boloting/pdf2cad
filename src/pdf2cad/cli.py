"""Command line: pdf2cad drawing.pdf [-o drawing.dxf] [--scale 100] [--all-pages]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pymupdf

from .convert import Options, convert_page


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pdf2cad",
        description="把矢量 PDF 图纸转换成 DXF，并还原标注、图块、填充和文字。",
    )
    p.add_argument("pdf", type=Path, help="输入的矢量 PDF 文件")
    p.add_argument("-o", "--output", type=Path, help="输出 DXF 路径（默认与 PDF 同名）")
    p.add_argument("-p", "--page", type=int, default=1, help="要转换的页码，从 1 开始（默认 1）")
    p.add_argument("--all-pages", action="store_true", help="转换所有页，每页一个 DXF")
    p.add_argument("-s", "--scale", type=float, help="比例：1 图纸毫米等于多少模型单位，如 1:100 填 100。不填则按标注自动识别")
    p.add_argument("--no-dims", action="store_true", help="不识别标注")
    p.add_argument("--no-hatch", action="store_true", help="不识别填充")
    p.add_argument("--no-blocks", action="store_true", help="不识别图块")
    p.add_argument("--text-height-factor", type=float, default=0.7, help="字高 = PDF 字号 × 该系数（默认 0.7）")
    p.add_argument("--json", action="store_true", help="以 JSON 输出识别报告")
    args = p.parse_args(argv)

    if not args.pdf.exists():
        print(f"找不到文件: {args.pdf}", file=sys.stderr)
        return 2
    try:
        doc = pymupdf.open(args.pdf)
    except Exception as e:  # noqa: BLE001
        print(f"无法打开 PDF: {e}", file=sys.stderr)
        return 2

    pages = range(doc.page_count) if args.all_pages else [args.page - 1]
    reports = []
    with doc:
        for i in pages:
            opts = Options(
                page=i, scale=args.scale, dimensions=not args.no_dims, hatches=not args.no_hatch,
                blocks=not args.no_blocks, text_height_factor=args.text_height_factor,
            )
            try:
                result = convert_page(doc, opts)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 2
            out = args.output or args.pdf.with_suffix(".dxf")
            if args.all_pages and doc.page_count > 1:
                out = out.with_name(f"{out.stem}_p{i + 1}{out.suffix}")
            out.write_bytes(result.dxf_bytes())
            rep = result.report.to_dict()
            rep["output"] = str(out)
            reports.append(rep)
            if not args.json:
                _print_report(rep)

    if args.json:
        print(json.dumps(reports if args.all_pages else reports[0], ensure_ascii=False, indent=2))
    return 0


def _print_report(rep: dict) -> None:
    src = {"dimensions": "按标注识别", "manual": "手动指定", "default": "未识别，按 1:1"}[rep["scale_source"]]
    print(f"第 {rep['page'] + 1}/{rep['page_count']} 页 -> {rep['output']}")
    print(f"  比例 1:{rep['scale']:g}（{src}），图层 {len(rep['layers'])} 个")
    found = [f"{k} {v}" for k, v in rep["recognized"].items() if v]
    print("  识别: " + ("，".join(found) if found else "无"))
    for w in rep["warnings"]:
        print(f"  注意: {w}")


if __name__ == "__main__":
    raise SystemExit(main())
