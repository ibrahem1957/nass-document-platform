"""Local PP-DocLayout layout source with a FAKE predictor (no model, no download, no network):
coordinates and native image dimensions, title attachment, ambiguity, missing setup, failures,
provenance and older region sets."""

from __future__ import annotations

import json

import pytest

from lab.core.hybrid import enrich, layout_regions
from tests.lab.test_hybrid_layout_regions import IMAGE_PX, scanned  # noqa: F401 - synthetic scanned run


class FakePredictor:
    def __init__(self, boxes=None, fail=False):
        self.boxes = boxes or []
        self.fail = fail
        self.inputs = []

    def predict(self, array, batch_size=1):
        self.inputs.append(array)
        if self.fail:
            raise RuntimeError("synthetic model failure")
        return [{"res": {"input_path": None, "page_index": None, "boxes": [dict(box) for box in self.boxes]}}]


def _box(label, coordinate, score=0.9, cls_id=0):
    return {"label": label, "coordinate": coordinate, "score": score, "cls_id": cls_id}


def _detect(scanned_ctx, predictor):  # noqa: F811
    return layout_regions.pp_doclayout_local(scanned_ctx["run_dir"], pdf_path=scanned_ctx["source"], pages=None,
                                             model_dir="unused-with-fake", predictor=predictor)


def test_detection_runs_on_the_native_image_and_keeps_pixel_coordinates(scanned) -> None:  # noqa: F811
    predictor = FakePredictor([_box("chart", [100.0, 200.0, 700.0, 600.0])])
    blocks, report = _detect(scanned, predictor)
    assert len(predictor.inputs) == 1  # page 2 is a digital page: skipped, not analysed
    array = predictor.inputs[0]
    assert array.shape == (IMAGE_PX[1], IMAGE_PX[0], 3)
    blue_bgr = array[400, 400]  # inside the RGB (30, 90, 200) rectangle -> BGR (200, 90, 30)
    assert abs(int(blue_bgr[0]) - 200) < 12 and abs(int(blue_bgr[2]) - 30) < 12
    (block,) = blocks
    assert block["bbox_px"] == [100.0, 200.0, 700.0, 600.0] and block["image_px_size"] == IMAGE_PX
    assert block["region_bbox_pt"] == [0, 0, 595, 842] and block["score"] == 0.9 and block["warnings"] == []
    pages = {item["page"]: item for item in report["pages"]}
    assert pages[1]["status"] == "detected" and pages[2]["status"] == "skipped"
    assert report["provenance"]["geometry_changing_preprocessing"].startswith("none")


def test_boxes_outside_the_image_are_flagged_not_trusted(scanned) -> None:  # noqa: F811
    blocks, _report = _detect(scanned, FakePredictor([_box("table", [100.0, 200.0, 2000.0, 600.0])]))
    assert "outside" in blocks[0]["warnings"][0]


def test_failures_are_reported_and_nothing_is_guessed(scanned) -> None:  # noqa: F811
    blocks, report = _detect(scanned, FakePredictor(fail=True))
    assert blocks == [] and report["pages"][0]["status"] == "failed"
    assert "synthetic model failure" in report["pages"][0]["reason"]


def test_missing_setup_refuses_before_loading_anything(tmp_path, scanned) -> None:  # noqa: F811
    empty = tmp_path / "model"
    empty.mkdir()
    setup = layout_regions.doclayout_setup(empty)
    assert not setup["ok"] and any("inference.yml" in problem for problem in setup["problems"])
    with pytest.raises(layout_regions.LayoutSourceError):
        layout_regions.pp_doclayout_local(scanned["run_dir"], pdf_path=scanned["source"], pages=None,
                                          model_dir=str(empty))
    assert not layout_regions.doclayout_setup(None)["ok"]


def test_labels_and_model_name_are_read_from_the_local_model_files(tmp_path) -> None:
    model = tmp_path / "PP-DocLayout-L_infer"
    model.mkdir()
    (model / "inference.yml").write_text("Global:\n  model_name: PP-DocLayout-L\nPostProcess:\n  label_list:\n"
                                         "  - text\n  - figure_title\n  - chart\n  - table\n  - image\n",
                                         encoding="utf-8")
    (model / "inference.pdiparams").write_bytes(b"x")
    (model / "inference.json").write_text("{}", encoding="utf-8")
    setup = layout_regions.doclayout_setup(model, "PP-DocLayout_plus-L")
    assert setup["labels"] == ["text", "figure_title", "chart", "table", "image"]
    assert setup["title_attachment_available"] and setup["model_name_in_files"] == "PP-DocLayout-L"
    assert any("not 'PP-DocLayout_plus-L'" in problem for problem in setup["problems"])


def _items(*specs):
    return [{"label": label, "bbox_pt": list(box), "block_index": index} for index, (label, box) in enumerate(specs)]


def test_title_directly_above_is_attached_and_its_box_recorded() -> None:
    items = _items(("chart", (50, 150, 350, 300)), ("figure_title", (60, 125, 340, 145)))
    layout_regions.attach_titles(items)
    chart, title = items
    assert chart["bbox_pt"] == [50, 125, 350, 300] and chart["bbox_pt_without_title"] == [50, 150, 350, 300]
    assert chart["title_attachments"][0]["bbox_pt"] == [60, 125, 340, 145]
    assert chart["title_attachments"][0]["gap_pt"] == 5 and title["title_status"].startswith("attached to block 0")


def test_far_narrow_or_ambiguous_titles_are_not_attached() -> None:
    far = _items(("chart", (50, 200, 350, 300)), ("figure_title", (60, 125, 340, 145)))  # 55 pt gap
    layout_regions.attach_titles(far)
    assert "title_attachments" not in far[0]
    shifted = _items(("chart", (50, 150, 350, 300)), ("figure_title", (300, 125, 500, 145)))  # 25% overlap
    layout_regions.attach_titles(shifted)
    assert "title_attachments" not in shifted[0]
    ambiguous = _items(("chart", (50, 150, 350, 300)), ("figure_title", (60, 125, 340, 145)),
                       ("figure_title", (60, 100, 340, 120)))  # two titles compete for one chart
    layout_regions.attach_titles(ambiguous)
    assert ambiguous[1]["title_status"] == ambiguous[2]["title_status"] == "ambiguous"
    assert ambiguous[0].get("title_ambiguous") and "title_attachments" not in ambiguous[0]
    assert "Needs Review" in ambiguous[0]["reasons_extra"][0]


def test_captions_below_are_optional_and_off_by_default() -> None:
    items = _items(("table", (50, 150, 350, 300)), ("figure_title", (60, 310, 340, 330)))
    layout_regions.attach_titles(items)
    assert "title_attachments" not in items[0]
    items = _items(("table", (50, 150, 350, 300)), ("figure_title", (60, 310, 340, 330)))
    layout_regions.attach_titles(items, captions_below=True)
    assert items[0]["title_attachments"][0]["position"] == "below"


def test_region_set_from_local_layout_records_model_provenance(scanned) -> None:  # noqa: F811
    predictor = FakePredictor([_box("chart", [100.0, 300.0, 700.0, 600.0], 0.93),
                               _box("figure_title", [120.0, 250.0, 680.0, 290.0], 0.88)])
    blocks, report = _detect(scanned, predictor)
    older = enrich.create_region_set(scanned["run_dir"],
                                     [r["region_id"] for r in enrich.candidate_regions(scanned["run_dir"])][:1],
                                     pdf_path=scanned["source"], by="Tester")
    before = {p: p.read_bytes() for p in older.rglob("*") if p.is_file()}
    source = {"kind": "pp_doclayout_local", "model_name": "PP-DocLayout_plus-L", "provenance": report["provenance"]}
    folder = layout_regions.create_layout_region_set(
        scanned["run_dir"], blocks=blocks, labels=["chart", "table", "image"], margin_px=15, selected=["p1_L0"],
        source=source, pdf_path=scanned["source"], by="Tester", detection_report=report)
    region = enrich.load_set(folder)["regions"][0]
    box = region["layout_box"]
    assert box["score"] == 0.93 and box["model_name"] == "PP-DocLayout_plus-L"
    assert box["title_attachments"][0]["bbox_px"] == [120.0, 250.0, 680.0, 290.0]
    assert box["bbox_pt_without_title"] == [50.0, 150.0, 350.0, 300.0]
    assert region["crop_options"]["pixel_box"] == [85, 235, 715, 615]  # title included, plus the margin
    saved = json.loads((folder / "layout_detection.json").read_text(encoding="utf-8"))
    assert saved["source"]["kind"] == "pp_doclayout_local" and len(saved["blocks"]) == 2
    assert {p: p.read_bytes() for p in older.rglob("*") if p.is_file()} == before  # older set unchanged
    assert not (older / "layout_detection.json").exists()
