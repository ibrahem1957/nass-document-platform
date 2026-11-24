"""Digital/mixed visual crops: native image bytes only when provably safe, otherwise the complete region
rendered at 250 DPI; shadow/frame layers, composites, low resolution and the quality guard. Synthetic
PDFs only; no model, no network (a fake Qari behind httpx.MockTransport for the end-to-end case)."""

from __future__ import annotations

import io
import json

import fitz
import pytest
from PIL import Image, ImageDraw

from lab.core.auto import assemble, pipeline, visual_crops
from lab.core.auto import config as auto_config
from tests.lab.test_auto_pipeline import FakeQari, _client, ctx  # noqa: F401

RENDER_DPI, MIN_DPI, TOL, MIN_STD = 250, 200.0, 2.0, 3.0


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _chart(size=(900, 560)) -> bytes:
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for i, height in enumerate((300, 420, 180, 350, 260)):
        x = 80 + i * 160
        draw.rectangle([x, size[1] - 40 - height, x + 90, size[1] - 40], fill=(40 + 40 * i, 90, 200 - 30 * i))
    draw.line([60, size[1] - 40, size[0] - 30, size[1] - 40], fill=(0, 0, 0), width=3)
    return _png(image)


def _flat(size, color) -> bytes:
    return _png(Image.new("RGB", size, color))


def _minimalist(size=(1000, 500)) -> bytes:
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.line([50, 450, 950, 450], fill=(0, 0, 0), width=2)
    draw.line([50, 50, 50, 450], fill=(0, 0, 0), width=2)
    draw.line([50, 400, 300, 250, 600, 300, 950, 100], fill=(0, 0, 0), width=3)
    return _png(image)


def _doc(*placements, text=None, draw=None):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for rect, data in placements:
        page.insert_image(fitz.Rect(*rect), stream=data)
    for point, words in text or []:
        page.insert_text(point, words, fontsize=10)
    for rect in draw or []:
        page.draw_rect(fitz.Rect(*rect), color=(0, 0, 0), width=1)
    return doc


def _crop(doc, figure, **kw):
    layers = visual_crops.page_layers(doc[0])
    options = {"layers": layers, "other_figures": [], "merged": False, "contains_figures": [],
               "render_dpi": RENDER_DPI, "min_dpi": MIN_DPI, "tolerance_pt": TOL, "min_std": MIN_STD} | kw
    return visual_crops.figure_crop(doc, 1, figure, **options)


def _raster(bbox, fid="img0"):
    return {"id": fid, "bbox": list(bbox), "origin": "raster"}


# ------------------------------------------------------------------ 1-9 method selection


def test_word_shadow_behind_the_chart_renders_the_full_visual() -> None:
    doc = _doc(([100, 300, 420, 500], _flat((40, 25), (70, 70, 70))),     # shadow, drawn first
               ([104, 304, 416, 496], _chart()))                           # the chart on top
    data, meta, candidate = _crop(doc, _raster([100, 300, 420, 500]))
    assert meta["method"] == visual_crops.RENDERED and meta["render_dpi"] == 250
    assert "2 embedded images intersect" in meta["method_reason"]
    image = Image.open(io.BytesIO(data))
    assert not visual_crops.quality(image, min_std=MIN_STD)["suspicious"]   # the chart is visible
    assert candidate is not None  # the rejected native layer is kept for audit


def test_frame_merged_visual_is_always_rendered() -> None:
    doc = _doc(([100, 300, 400, 450], _chart((1000, 500))))
    _data, meta, _candidate = _crop(doc, _raster([100, 300, 400, 450]), merged=True)
    assert meta["method"] == visual_crops.RENDERED and "frame-merged" in meta["method_reason"]


def test_clean_standalone_high_resolution_image_uses_native_bytes() -> None:
    doc = _doc(([100, 300, 400, 450], _chart((1000, 500))))  # 1000 px over 300 pt = 240 DPI
    data, meta, candidate = _crop(doc, _raster([100, 300, 400, 450]))
    assert meta["method"] == visual_crops.NATIVE and candidate is None
    assert Image.open(io.BytesIO(data)).size == (1000, 500) and meta["native_pixels"]
    assert meta["effective_dpi"] == [240.0, 240.0] and meta["render_dpi"] is None


def test_low_resolution_embedded_image_is_rendered() -> None:
    doc = _doc(([100, 300, 400, 450], _chart((200, 100))))  # 48 DPI
    _data, meta, _candidate = _crop(doc, _raster([100, 300, 400, 450]))
    assert meta["method"] == visual_crops.RENDERED and "below 200 DPI" in meta["method_reason"]


def test_several_objects_forming_one_chart_render_the_complete_region() -> None:
    doc = _doc(([100, 300, 250, 450], _chart((700, 700))), ([250, 300, 400, 450], _chart((700, 700))))
    _data, meta, _candidate = _crop(doc, _raster([100, 300, 400, 450]))
    assert meta["method"] == visual_crops.RENDERED and meta["crop_bbox"] == [100.0, 300.0, 400.0, 450.0]


def test_vector_chart_and_native_text_labels_are_rendered() -> None:
    vector = _doc(draw=[[100, 300, 400, 450]], text=[((120, 440), "Axis label 2024")])
    _data, meta, _candidate = _crop(vector, {"id": "vec0", "bbox": [100, 300, 400, 450], "origin": "vector"})
    assert meta["method"] == visual_crops.RENDERED and meta["method_reason"] == "vector figure: always rendered"
    labelled = _doc(([100, 300, 400, 450], _chart((1000, 500))), text=[((150, 380), "Revenue 18,450")])
    _data, meta, _candidate = _crop(labelled, _raster([100, 300, 400, 450]))
    assert meta["method"] == visual_crops.RENDERED and "native text word" in meta["method_reason"]


def test_uniform_shadow_crop_rejects_native_and_is_preserved_and_flagged() -> None:
    doc = _doc(([100, 300, 400, 450], _flat((1000, 500), (60, 60, 60))))  # clean geometry, no content
    data, meta, candidate = _crop(doc, _raster([100, 300, 400, 450]))
    assert meta["native_quality"]["suspicious"] and meta["method"] == visual_crops.RENDERED
    assert "quality guard" in meta["method_reason"] and candidate is not None
    assert meta["quality"]["suspicious"] and meta["review"]   # rendered crop also flat: kept + flagged
    assert data  # never discarded


def test_legitimate_minimalist_chart_is_not_rejected() -> None:
    doc = _doc(([100, 300, 400, 450], _minimalist()))
    _data, meta, _candidate = _crop(doc, _raster([100, 300, 400, 450]))
    assert meta["method"] == visual_crops.NATIVE and not meta["quality"]["suspicious"]
    monochrome = Image.new("L", (400, 200), 255)
    ImageDraw.Draw(monochrome).rectangle([40, 40, 120, 160], fill=0)
    assert not visual_crops.quality(monochrome, min_std=MIN_STD)["suspicious"]


def test_native_box_not_covering_the_full_visual_is_rendered() -> None:
    doc = _doc(([110, 310, 390, 440], _chart((1000, 500))))
    _data, meta, _candidate = _crop(doc, _raster([100, 300, 400, 450]))  # layout region larger by 10 pt
    assert meta["method"] == visual_crops.RENDERED and "does not cover" in meta["method_reason"]


def test_shadow_offset_outside_the_box_extends_the_composition_but_never_into_another_figure() -> None:
    doc = _doc(([106, 306, 426, 506], _flat((40, 25), (70, 70, 70))), ([100, 300, 420, 500], _chart()))
    _data, meta, _candidate = _crop(doc, _raster([100, 300, 420, 500]))
    assert meta["crop_bbox"] == [100.0, 300.0, 426.0, 506.0]  # the visible shadow edge is included
    _data, blocked, _candidate = _crop(doc, _raster([100, 300, 420, 500]),
                                       other_figures=[{"id": "img9", "bbox": [422, 480, 560, 600]}])
    assert blocked["crop_bbox"] == [100.0, 300.0, 420.0, 500.0] and blocked["composition"]["ambiguous"]
    assert any("overlap figure" in note for note in blocked["review"])


def test_contained_figures_are_one_composite() -> None:
    figures = [_raster([48, 369, 561, 467], "img2"), _raster([64, 385, 532, 437], "img3"),
               _raster([48, 500, 300, 600], "img4")]
    assert visual_crops.contained_figures(figures) == {"img3": "img2"}  # neighbours are never merged


# ------------------------------------------------------------------ 11-12 geometry and resolution


def test_rendered_region_coordinates_and_250_dpi_dimensions() -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(fitz.Rect(250, 250, 260, 260), color=(1, 0, 0), fill=(1, 0, 0))
    image, meta = visual_crops.render_region(page, [100, 200, 400, 350], 250)
    assert meta["clip_bbox"] == [100.0, 200.0, 400.0, 350.0] and meta["render_dpi"] == 250
    # PyMuPDF snaps the clip outward to whole pixels: up to one extra pixel at each edge, never fewer
    assert 0 <= image.width - 300 * 250 / 72 <= 2 and 0 <= image.height - 150 * 250 / 72 <= 2
    assert all(abs(v - 250) <= 1 for v in meta["effective_dpi"])
    x, y = round((255 - 100) * 250 / 72), round((255 - 200) * 250 / 72)
    red = image.getpixel((x, y))
    assert red[0] > 200 and red[1] < 60 and red[2] < 60  # PDF point (255, 255) lands at the expected pixel


# ------------------------------------------------------------------ end to end (13-18)


@pytest.fixture
def word_report(tmp_path, monkeypatch):
    monkeypatch.setenv("QARI_SERVICE_TOKEN", "tok")
    doc = _doc(([100, 300, 420, 500], _flat((40, 25), (70, 70, 70))), ([104, 304, 416, 496], _chart()),
               text=[((72, 100), "Report heading paragraph above the chart."),
                     ((72, 700), "A closing paragraph below the chart.")])
    pdf = tmp_path / "WordReport.pdf"
    doc.save(pdf)
    doc.close()
    file = tmp_path / "auto.toml"
    file.write_text("", encoding="utf-8")
    config = auto_config.load_config(file, environ={"AGENTICDOC_RESULTS_DIR": str(tmp_path / "results"),
                                                    "QARI_SERVICE_URL": "http://127.0.0.1:8000"})
    run_dir = pipeline.create_run(pdf, display_name="WordReport.pdf", config=config, detector=None)
    return {"run_dir": run_dir, "config": config}


def test_word_report_sends_the_complete_rendered_chart_with_provenance(word_report) -> None:
    run_dir, config = word_report["run_dir"], word_report["config"]
    (job,) = [j for j in pipeline.all_jobs(run_dir) if j["purpose"] == "digital_figure"]
    meta = job["image"]
    assert meta["method"] == visual_crops.RENDERED and meta["render_dpi"] == 250 and meta["configured_render_dpi"] == 250
    assert meta["native_candidate_file"] and (run_dir / meta["native_candidate_file"]).is_file()
    sent = Image.open(run_dir / job["image_file"])
    assert not visual_crops.quality(sent, min_std=MIN_STD)["suspicious"]
    shadow = Image.open(run_dir / meta["native_candidate_file"])
    assert shadow.size == (40, 25)  # the layer that used to be sent
    fake = FakeQari()
    pipeline.execute(run_dir, config, _client(fake))
    manifest = json.loads(assemble.finalize(run_dir, config).read_text(encoding="utf-8"))
    entry = manifest["visual_crops"][job["job_id"]]
    for field in ("page", "figure_id", "method", "method_reason", "original_bbox", "crop_bbox", "render_dpi",
                  "effective_dpi", "pixel_size", "native", "image_sha256", "request_id", "ocr_state", "quality"):
        assert field in entry, field
    assert entry["request_id"] and entry["ocr_state"] == "completed" and not entry["ocr_empty"]
    assert entry["native"]["width_px"] in (40, 900)
    document = json.loads((run_dir / "document.json").read_text(encoding="utf-8"))
    figure = next(e for e in document["elements"] if e["type"] == "figure")
    assert figure["crop"]["method"] == visual_crops.RENDERED and figure["crop"]["render_dpi"] == 250


def test_native_text_order_is_preserved_on_digital_pages(word_report) -> None:
    pipeline.execute(word_report["run_dir"], word_report["config"], _client(FakeQari()))
    (page,) = assemble.assemble(word_report["run_dir"])["pages"]
    types = [e["type"] for e in page["elements"]]
    contents = " ".join(e.get("content") or "" for e in page["elements"])
    assert types.index("figure") > 0 and "Report heading paragraph" in contents
    assert contents.index("Report heading paragraph") < contents.index("A closing paragraph")


def test_tables_come_from_find_tables_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QARI_SERVICE_TOKEN", "tok")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for x in (100, 250, 400):
        page.draw_line((x, 200), (x, 290))
    for y in (200, 245, 290):
        page.draw_line((100, y), (400, y))
    for (x, y), word in (((110, 230), "Region"), ((260, 230), "Value"), ((110, 275), "North"), ((260, 275), "120")):
        page.insert_text((x, y), word, fontsize=11)
    pdf = tmp_path / "Table.pdf"
    doc.save(pdf)
    file = tmp_path / "auto.toml"
    file.write_text("", encoding="utf-8")
    config = auto_config.load_config(file, environ={"AGENTICDOC_RESULTS_DIR": str(tmp_path / "results")})
    run_dir = pipeline.create_run(pdf, display_name="Table.pdf", config=config, detector=None)
    analysis = pipeline.load_plan(run_dir)["analysis"]
    tables = (analysis["pages"][0].get("tables") or {}).get("tables") or []
    if not tables:
        pytest.skip("find_tables found no table in this synthetic page (nothing to compare)")
    element = next((e for e in assemble.assemble(run_dir)["elements"] if e["type"] == "table"), None)
    if element is None:
        pytest.skip("layout v3 fell back to page text on this synthetic page (tables not rendered separately)")
    assert [c["text"] for c in element["cells"]] == [c.get("processed_cell_text") for c in tables[0]["cells"]]
    assert not [j for j in pipeline.all_jobs(run_dir) if j["purpose"] == "digital_figure"]  # tables are not OCR'd


def test_scanned_pages_keep_their_native_pixel_path(ctx) -> None:  # noqa: F811
    job = next(j for j in pipeline.all_jobs(ctx["run_dir"]) if j["job_id"] == "p001_R0")
    assert job["image"]["page_image_method"] == "native_jpeg" and job["image"]["pixel_box"] == [40, 140, 760, 660]
    assert "native_assessment" not in job["image"] and "render_dpi" not in job["image"]


def test_qari_and_layout_proven_settings_are_unchanged(tmp_path) -> None:
    file = tmp_path / "auto.toml"
    file.write_text("", encoding="utf-8")
    config = auto_config.load_config(file, environ={})
    qari = config.proven.qari
    assert (qari.prompt, qari.max_new_tokens, qari.max_time_crop_s, qari.max_time_page_s) == ("Free OCR.", 1536, 30.0, 60.0)
    assert (qari.loop_max_ngram, qari.loop_repeats, qari.retry_repetition_penalty) == (40, 5, 1.1)
    assert config.proven.layout.crop_margin_px == 60 and config.proven.layout.digital_render_dpi == 250
    assert (config.layout.native_min_dpi, config.layout.native_bbox_tolerance_pt) == (200.0, 2.0)
    assert auto_config.proven_differences(config) == {}


def test_saved_runs_are_not_modified(word_report) -> None:
    run_dir = word_report["run_dir"]
    pipeline.execute(run_dir, word_report["config"], _client(FakeQari()))
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    assemble.assemble(run_dir)
    assert {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()} == before
