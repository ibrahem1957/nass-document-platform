"""Region sets from layout boxes on scanned pages: px->pt mapping, duplicates/overlaps, small
boxes, margin clamping, native pixel crops and provenance. Synthetic scanned PDFs only; no request."""

from __future__ import annotations

import io
import json
import sys
import zipfile

import fitz
import pytest
from PIL import Image, ImageDraw

from lab.core.e2.api import documents
from lab.core.hybrid import analysis as hybrid_analysis
from lab.core.hybrid import enrich, layout_regions
from lab.core.hybrid import run as hybrid
from lab.core.ingest import IngestService
from lab.core.workspace import WorkspaceManager

PAGE_PT = [0, 0, 595, 842]
IMAGE_PX = [1190, 1684]  # 144 DPI scan


def _scan_jpeg() -> bytes:
    image = Image.new("RGB", tuple(IMAGE_PX), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([100, 200, 700, 600], fill=(30, 90, 200))    # a "chart"
    draw.rectangle([150, 900, 1050, 1300], fill=(200, 60, 40))  # a "table"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


@pytest.fixture
def scanned(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setattr(documents, "DOCUMENT_REGISTRY", tmp_path / "no_registry.toml")
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(tmp_path / "results"))
    pdf = tmp_path / "Synthetic_Scan.pdf"
    doc = fitz.open()
    doc.new_page(width=595, height=842).insert_image(fitz.Rect(*PAGE_PT), stream=_scan_jpeg())
    digital = doc.new_page(width=595, height=842)
    digital.insert_text((72, 72), "digital page", fontsize=12)
    doc.save(pdf)
    doc.close()
    manager = WorkspaceManager(settings)
    workspace_id = manager.create().workspace_id
    ingest = IngestService(manager, settings)
    with pdf.open("rb") as handle:
        document = ingest.ingest(workspace_id, handle, pdf.name)
    source = ingest.original_path(workspace_id, document.doc_id)
    root = tmp_path / "results"
    run_dir = hybrid.create_run(root=root, document=document, pdf_path=source,
                                analysis=hybrid_analysis.analyze_pdf(source), model=hybrid.LIBRARY_ONLY, vlm_pages={})
    hybrid.apply_layout(run_dir, None, root, by="Tester", statement=hybrid.LAYOUT_STATEMENT)
    return {"run_dir": run_dir, "source": source, "root": root}


def _block(label, bbox_px, index, page=1, warnings=()):
    return {"page": page, "label": label, "bbox_px": list(bbox_px), "image_px_size": IMAGE_PX,
            "region_bbox_pt": PAGE_PT, "source_id": "test/source", "block_index": index, "warnings": list(warnings)}


BLOCKS = [_block("chart", [100, 200, 700, 600], 0), _block("table", [150, 900, 1050, 1300], 1),
          _block("chart", [100, 200, 700, 600], 2),           # exact duplicate of 0
          _block("image", [1000, 50, 1100, 120], 3),          # small (about 50 x 35 pt)
          _block("text", [100, 1400, 1000, 1500], 4)]         # label not selected


def test_px_to_pt_maps_through_the_analysed_image() -> None:
    assert layout_regions.px_to_pt([119, 168.4, 595, 842], IMAGE_PX, PAGE_PT) == [59.5, 84.2, 297.5, 421.0]
    assert layout_regions.px_to_pt([0, 0, 100, 50], [200, 100], [100, 200, 300, 300]) == [100, 200, 200, 250]


def test_classification_removes_only_exact_duplicates() -> None:
    boxes = [{"label": "chart", "bbox_pt": [0, 0, 300, 200], "block_index": 0},
             {"label": "chart", "bbox_pt": [0, 0, 300, 200], "block_index": 1},
             {"label": "chart", "bbox_pt": [1, 1, 301, 201], "block_index": 2},
             {"label": "table", "bbox_pt": [250, 150, 500, 400], "block_index": 3},
             {"label": "image", "bbox_pt": [0, 600, 50, 650], "block_index": 4},
             {"label": "image", "bbox_pt": [100, 500, 400, 800], "block_index": 5}]
    status = {item["block_index"]: item for item in layout_regions.classify(boxes)}
    assert status[1]["status"] == "duplicate removed"
    # Near-identical boxes are containment, not a partial overlap: the outer one is preferred.
    assert status[2]["status"] == "contained" and not status[2]["default_selected"]
    assert "inside block 0" in " ".join(status[2]["reasons"])
    assert status[0]["status"] == "uncertain"  # it genuinely overlaps the table (block 3)
    assert status[3]["status"] == "uncertain" and "partially overlaps" in " ".join(status[3]["reasons"])
    assert status[4]["status"] == "small" and not status[4]["default_selected"]
    assert status[5]["status"] == "selected" and status[5]["default_selected"]


def test_margin_is_clamped_to_the_embedded_image() -> None:
    assert layout_regions.with_margin([5, 5, 100, 100], 15, [200, 200]) == [0, 0, 115, 115]
    assert layout_regions.with_margin([150, 150, 195, 198], 15, [200, 200]) == [135, 135, 200, 200]


def test_proposal_on_a_scanned_page(scanned) -> None:
    metadata, _plan, _library = hybrid.load(scanned["run_dir"])
    with fitz.open(hybrid.layout_pdf(scanned["run_dir"], metadata, scanned["source"])[0]) as document:
        candidates = layout_regions.propose(document, BLOCKS + [_block("chart", [10, 10, 500, 500], 9, page=2)],
                                            labels=["chart", "image", "table"], margin_px=15)
    by_index = {(c["page"], c["block_index"]): c for c in candidates}
    assert by_index[(1, 0)]["region_id"] == "p1_L0" and by_index[(1, 0)]["pixel_box"] == [85, 185, 715, 615]
    assert by_index[(1, 2)]["status"] == "duplicate removed" and by_index[(1, 2)]["region_id"] is None
    assert by_index[(1, 3)]["status"] == "small" and not by_index[(1, 3)]["default_selected"]
    assert by_index[(1, 4)]["status"] == "label not selected"
    assert by_index[(2, 9)]["status"] == "unavailable" and "not a scanned" in " ".join(by_index[(2, 9)]["reasons"])


def test_saved_set_has_native_pixel_crops_and_provenance(scanned) -> None:
    existing = enrich.create_region_set(scanned["run_dir"],
                                        [r["region_id"] for r in enrich.candidate_regions(scanned["run_dir"])][:1],
                                        pdf_path=scanned["source"], by="Tester")
    before = {p: p.read_bytes() for p in existing.rglob("*") if p.is_file()}
    source = {"kind": "saved_provider_output", "provider_id": "test", "set_id": "set_x", "run": "run_y"}
    folder = layout_regions.create_layout_region_set(
        scanned["run_dir"], blocks=BLOCKS, labels=["chart", "table"], margin_px=15, selected=["p1_L0", "p1_L1"],
        source=source, pdf_path=scanned["source"], by="Tester")
    data = enrich.load_set(folder)
    first = data["regions"][0]
    assert first["crop_file"] == "crops/p1_L0.png" and first["crop_options"]["resolution_mode"] == "native_subregion"
    assert first["crop_options"]["pixel_box"] == [85, 185, 715, 615] and first["layout_box"]["margin_px"] == 15
    assert first["layout_box"]["source"] == source and first["layout_box"]["block_index"] == 0
    native = Image.open(io.BytesIO(_scan_jpeg())).convert("RGB")  # same encoder settings -> same decoded pixels
    with fitz.open(scanned["source"]) as pdf:
        xref = pdf[0].get_images(full=True)[0][0]
        embedded = Image.open(io.BytesIO(pdf.xref_stream_raw(xref))).convert("RGB")
    crop = Image.open(folder / first["crop_file"]).convert("RGB")
    assert crop.size == (630, 430) and list(crop.getdata()) == list(embedded.crop((85, 185, 715, 615)).getdata())
    assert native.size == tuple(IMAGE_PX)
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(
        enrich.export_crops_zip(folder, provider="Qari", model="Qari", by="Tester"))).read("manifest.json"))
    assert manifest["regions"][0]["layout_box"]["label"] == "chart"
    assert data["region_source"]["kind"] == "layout_boxes"
    assert {p: p.read_bytes() for p in existing.rglob("*") if p.is_file()} == before  # older set unchanged
    with pytest.raises(enrich.EnrichError):
        layout_regions.create_layout_region_set(scanned["run_dir"], blocks=BLOCKS, labels=["chart"], margin_px=15,
                                                selected=["p1_L9"], source=source, pdf_path=scanned["source"],
                                                by="Tester")


def test_blocks_from_a_saved_structured_output() -> None:
    structured = {"res": {"width": 1511, "height": 472,
                          "parsing_res_list": [{"block_label": "chart", "block_bbox": [2, 4, 1511, 380]},
                                               {"block_label": "text"}],
                          "doc_preprocessor_res": {"model_settings": {"use_doc_unwarping": True}, "angle": 0}}}
    blocks, size, warnings = layout_regions.blocks_from_structured(structured)
    assert blocks == [{"label": "chart", "bbox_px": [2.0, 4.0, 1511.0, 380.0], "block_index": 0}]
    assert size == [1511, 472] and "unwarping" in warnings[0]


def test_local_layout_source_is_a_placeholder() -> None:
    with pytest.raises(layout_regions.LayoutSourceError):
        layout_regions.pp_doclayout_local()


def test_outputs_of_older_sets_have_no_layout_box(scanned) -> None:
    folder = enrich.create_region_set(scanned["run_dir"],
                                      [r["region_id"] for r in enrich.candidate_regions(scanned["run_dir"])][:1],
                                      pdf_path=scanned["source"], by="Tester")
    rid = enrich.load_set(folder)["regions"][0]["region_id"]
    export = enrich.export_crops_zip(folder, provider="Qari", model="Qari", by="Tester")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", zipfile.ZipFile(io.BytesIO(export)).read("manifest.json"))
        archive.writestr(f"{rid}.txt", "text")
    enrich.import_results(folder, buffer.getvalue(), declared_model="Qari", reviewer="Tester",
                          statement=enrich.IMPORT_STATEMENT)
    (output,) = enrich.collect_outputs(folder, lambda _id: None, scanned["root"])[rid]
    assert output["layout_box"] is None and output["crop_options"]["resolution_mode"] == "native"
