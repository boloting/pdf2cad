import json

from fastapi.testclient import TestClient

from pdf2cad.cli import main
from pdf2cad.sample import sample_pdf_bytes
from pdf2cad.web import app


def test_cli_writes_dxf(tmp_path, capsys):
    pdf = tmp_path / "plate.pdf"
    pdf.write_bytes(sample_pdf_bytes())
    assert main([str(pdf), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert (tmp_path / "plate.dxf").exists()
    assert rep["recognized"]["线性标注"] == 4


def test_cli_missing_file(tmp_path):
    assert main([str(tmp_path / "nope.pdf")]) == 2


def test_web_convert_and_download():
    client = TestClient(app)
    res = client.post("/api/convert", files={"file": ("plate.pdf", sample_pdf_bytes(), "application/pdf")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["filename"] == "plate.dxf"
    key = body["id"]
    dxf = client.get(f"/api/results/{key}/dxf")
    assert dxf.status_code == 200 and b"DIMENSION" in dxf.content
    for kind in ("source", "preview", "highlight"):
        svg = client.get(f"/api/results/{key}/{kind}.svg")
        assert svg.status_code == 200 and svg.text.lstrip().startswith("<")
        assert "default-src 'none'" in svg.headers["content-security-policy"]


def test_web_rejects_bad_input():
    client = TestClient(app)
    assert client.post("/api/convert", files={"file": ("x.pdf", b"hello", "application/pdf")}).status_code == 400
    res = client.post(
        "/api/convert", files={"file": ("x.pdf", sample_pdf_bytes(), "application/pdf")}, data={"page": "3"},
    )
    assert res.status_code == 400 and "页码" in res.json()["detail"]
    assert client.get("/api/results/missing/dxf").status_code == 404


def test_index_served():
    client = TestClient(app)
    assert "PDF2CAD" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
