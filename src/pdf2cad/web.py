"""Web UI: upload a vector PDF, preview the recognition, download the DXF."""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .convert import Options, convert_page
from .preview import dxf_svg, pdf_page_svg
from .sample import sample_pdf_bytes

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 40 * 1024 * 1024
MAX_RESULTS = 12
RESULT_TTL = 30 * 60
MAX_PATHS = 250_000


@dataclass
class Stored:
    created: float
    filename: str
    dxf: bytes
    source_svg: str
    preview_svg: str
    highlight_svg: str


class ResultStore:
    def __init__(self) -> None:
        self._items: OrderedDict[str, Stored] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, item: Stored) -> str:
        key = secrets.token_urlsafe(16)
        with self._lock:
            self._items[key] = item
            self._evict()
        return key

    def get(self, key: str) -> Stored:
        with self._lock:
            self._evict()
            item = self._items.get(key)
        if item is None:
            raise HTTPException(404, "结果已过期，请重新转换。")
        return item

    def _evict(self) -> None:
        now = time.time()
        for k in [k for k, v in self._items.items() if now - v.created > RESULT_TTL]:
            del self._items[k]
        while len(self._items) > MAX_RESULTS:
            self._items.popitem(last=False)


store = ResultStore()
busy = threading.BoundedSemaphore(2)
app = FastAPI(title="PDF2CAD", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/api/sample")
def sample() -> Response:
    return Response(
        sample_pdf_bytes(), media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="sample-mounting-plate.pdf"'},
    )


def _flag(v: str | None, default: bool = True) -> bool:
    if v is None:
        return default
    return v.lower() in ("1", "true", "on", "yes")


@app.post("/api/convert")
def convert(
    file: UploadFile = File(...),
    page: int = Form(1),
    scale: str = Form(""),
    dimensions: str = Form("true"),
    hatches: str = Form("true"),
    blocks: str = Form("true"),
) -> JSONResponse:
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "文件超过 40 MB。")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "这不是 PDF 文件。")
    scale_v: float | None = None
    if scale.strip():
        try:
            scale_v = float(scale)
        except ValueError:
            raise HTTPException(400, "比例需要是数字，例如 1:100 填 100。") from None
        if not 1e-4 <= scale_v <= 1e5:
            raise HTTPException(400, "比例超出范围。")

    if not busy.acquire(timeout=60):
        raise HTTPException(503, "服务器正忙，请稍后再试。")
    try:
        try:
            pdf = pymupdf.open(stream=data, filetype="pdf")
        except Exception:
            raise HTTPException(400, "无法打开这个 PDF，文件可能已损坏或被加密。") from None
        with pdf:
            if pdf.needs_pass:
                raise HTTPException(400, "PDF 有密码保护，请先解除密码。")
            if not 1 <= page <= pdf.page_count:
                raise HTTPException(400, f"页码超出范围：共 {pdf.page_count} 页。")
            n_paths = len(pdf[page - 1].get_cdrawings())
            if n_paths > MAX_PATHS:
                raise HTTPException(413, "这一页的图形太多，网页版暂不处理，请用命令行版本。")
            opts = Options(
                page=page - 1, scale=scale_v, dimensions=_flag(dimensions),
                hatches=_flag(hatches), blocks=_flag(blocks),
            )
            result = convert_page(pdf, opts)
        w, h = result.report.page_size_mm
        s = result.report.scale
        stem = Path(file.filename or "drawing.pdf").stem or "drawing"
        stored = Stored(
            created=time.time(),
            filename=f"{stem}.dxf" if result.report.page_count == 1 else f"{stem}_p{page}.dxf",
            dxf=result.dxf_bytes(),
            source_svg=pdf_page_svg(data, page - 1),
            preview_svg=dxf_svg(result.doc, w, h, s),
            highlight_svg=dxf_svg(result.doc, w, h, s, highlight=True),
        )
    finally:
        busy.release()
    key = store.put(stored)
    return JSONResponse({
        "id": key,
        "filename": stored.filename,
        "dxf_size": len(stored.dxf),
        "report": result.report.to_dict(),
    })


_SVG_HEADERS = {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'", "Cache-Control": "private, max-age=600"}


@app.get("/api/results/{key}/{kind}.svg")
def result_svg(key: str, kind: str) -> Response:
    item = store.get(key)
    svg = {"source": item.source_svg, "preview": item.preview_svg, "highlight": item.highlight_svg}.get(kind)
    if svg is None:
        raise HTTPException(404)
    return Response(svg, media_type="image/svg+xml", headers=_SVG_HEADERS)


@app.get("/api/results/{key}/dxf")
def result_dxf(key: str) -> Response:
    item = store.get(key)
    name = quote(item.filename)
    return Response(
        item.dxf, media_type="application/dxf",
        headers={"Content-Disposition": f"attachment; filename=\"drawing.dxf\"; filename*=UTF-8''{name}"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run("pdf2cad.web:app", host=host, port=port, reload=bool(os.environ.get("RELOAD")))


if __name__ == "__main__":
    main()
