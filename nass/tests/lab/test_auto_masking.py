"""Automatic Extraction quality fixes: ambiguous titles no longer block visuals, visual masking for the
page OCR, order-insensitive duplicate detection, and scanned-page reading order. Synthetic inputs
only; no model, no network."""

from __future__ import annotations

import hashlib
import io
import json

import httpx
import pytest
from PIL import Image, ImageDraw

from lab.core.auto import dedup, duplicates, masking, order, pipeline, regions
from lab.core.auto import validate as validation
from tests.lab.test_auto_pipeline import FakeQari, _client, ctx  # noqa: F401

IMAGE_PX, PAGE_PT = [1190, 1684], [0, 0, 595, 842]


def _box(label, px, index):
    return {"label": label, "bbox_px": [float(v) for v in px], "block_index": index, "warnings": [], "score": 0.9}


def _select(boxes, **options):
    return regions.select_regions(1, boxes, image_px=IMAGE_PX, image_bbox_pt=PAGE_PT,
                                  labels=("chart", "image", "table"), margin_px=60, **options)


def _by_index(selection):
    return {r["block_index"]: r for r in selection["regions"]}


# ------------------------------------------------------------------ 1-4 region selection


def test_ambiguous_caption_does_not_block_visual_ocr() -> None:
    # a figure_title between two charts at similar distances: title or caption is unclear
    selection = _select([_box("chart", [100, 200, 700, 500], 0), _box("figure_title", [120, 530, 680, 570], 1),
                         _box("chart", [100, 600, 700, 900], 2)])
    found = _by_index(selection)
    assert found[0]["decision"] == found[2]["decision"] == regions.AUTO_FLAGGED
    assert found[0]["status"] == found[2]["status"] == "selected"
    assert found[0]["title_ambiguity"] and "title left unattached" in " ".join(found[0]["reasons"])


def test_ambiguous_title_stays_unattached_and_is_reported() -> None:
    selection = _select([_box("chart", [100, 200, 700, 500], 0), _box("figure_title", [120, 530, 680, 570], 1),
                         _box("chart", [100, 600, 700, 900], 2)])
    found = _by_index(selection)
    assert "title_attachments" not in found[0] and "title_attachments" not in found[2]
    assert found[0]["box_pt_before_margin"] == [50.0, 100.0, 350.0, 250.0]  # original box kept
    assert found[0]["pixel_box"] == [40, 140, 760, 560]  # original 60 px crop kept
    title = next(o for o in selection["others"] if o["block_index"] == 1)
    assert title["title_status"] == "ambiguous"
    codes = {f["code"] for f in validation.region_flags(found[0])}
    assert "ambiguous_title_unattached" in codes


def test_actual_partial_overlap_still_requires_review() -> None:
    found = _by_index(_select([_box("chart", [100, 200, 700, 600], 0), _box("table", [500, 500, 1000, 900], 1)]))
    assert found[0]["decision"] == found[1]["decision"] == regions.REVIEW
    assert found[0]["status"] == "uncertain"


def test_kpi_cards_with_an_ambiguous_title_are_processed_automatically() -> None:
    selection = _select([_box("image", [100, 200, 700, 500], 0), _box("figure_title", [120, 530, 680, 570], 1),
                         _box("image", [100, 600, 700, 900], 2)])
    found = _by_index(selection)
    assert {found[0]["decision"], found[2]["decision"]} == {regions.AUTO_FLAGGED}
    assert regions.overlapping_auto(selection["regions"]) == []


def test_nested_and_container_rules_are_unchanged() -> None:
    nested = _by_index(_select([_box("chart", [100, 200, 700, 600], 0), _box("image", [150, 250, 400, 450], 1)]))
    assert nested[0]["decision"] == regions.AUTO and nested[1]["decision"] == regions.REVIEW
    split = _by_index(_select([_box("image", [0, 0, 1190, 800], 0), _box("chart", [20, 20, 580, 780], 1),
                               _box("chart", [610, 20, 1170, 780], 2)]))
    assert split[0]["decision"] == regions.REVIEW and split[0]["status"] == "container"
    assert split[1]["decision"] in regions.PROCESSED and split[2]["decision"] in regions.PROCESSED


# ------------------------------------------------------------------ 5-10 masking


def _page_image():
    image = Image.new("RGB", tuple(IMAGE_PX), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle([120, 250, 679, 289], fill=(20, 20, 20))     # figure title above the chart
    draw.rectangle([100, 300, 699, 599], fill=(30, 90, 200))    # chart
    draw.rectangle([100, 900, 1000, 1000], fill=(60, 60, 60))   # body paragraph
    return image


def _title_and_chart():
    return _select([_box("chart", [100, 300, 700, 600], 0), _box("figure_title", [120, 250, 680, 290], 1),
                    _box("text", [100, 900, 1000, 1000], 2)])


def test_masking_covers_exactly_the_detected_visual_box() -> None:
    selection = _title_and_chart()
    (region,) = selection["regions"]
    assert region["title_attachments"]  # the title is part of the crop ...
    image = _page_image()
    masked, record = masking.plan(selection, image)
    (mask,) = record["masks"]
    assert mask["box_px"] == [100, 300, 700, 600] and mask["crop_margin_applied"] is False
    assert masking.masked_fraction(masked, [100, 300, 700, 600]) == 1.0
    assert masked.getpixel((99, 450)) == image.getpixel((99, 450))   # just outside: unchanged
    assert masked.getpixel((700, 450)) == image.getpixel((700, 450))


def test_original_page_image_is_never_modified() -> None:
    image = _page_image()
    before = image.tobytes()
    masked, _record = masking.plan(_title_and_chart(), image)
    assert image.tobytes() == before and masked is not image


def test_figure_title_outside_the_detected_box_stays_visible() -> None:
    image = _page_image()
    masked, record = masking.plan(_title_and_chart(), image)
    assert masked.getpixel((400, 270)) == (20, 20, 20)       # ... but it is not masked
    assert masked.getpixel((500, 950)) == (60, 60, 60)       # body paragraph not masked
    assert record["text_under_masks"] == []


def test_crop_margin_never_expands_the_mask() -> None:
    selection = _title_and_chart()
    (region,) = selection["regions"]
    assert region["pixel_box"][0] == 100 - 60  # the crop has the margin ...
    masked, _record = masking.plan(selection, _page_image())
    for x, y in ((70, 450), (730, 450), (400, 630)):  # ... the mask does not
        assert masked.getpixel((x, y)) == (240, 240, 240)


def _manual_selection(boxes):
    regions_ = []
    for index, (box, decision) in enumerate(boxes):
        regions_.append({"region_id": f"p1_R{index}", "label": "chart", "decision": decision, "bbox_px": box,
                         "box_pt_before_margin": list(box), "reasons": []})
    return {"image_px": [200, 100], "image_bbox_pt": [0, 0, 200, 100], "regions": regions_,
            "text_blocks": [], "attached_titles": []}


def test_nested_overlapping_and_duplicate_masks_are_safe() -> None:
    selection = _manual_selection([([10, 10, 60, 60], regions.AUTO), ([40, 40, 90, 90], regions.AUTO_FLAGGED),
                                   ([20, 20, 30, 30], regions.AUTO), ([10, 10, 60, 60], regions.AUTO),
                                   ([100, 10, 150, 60], regions.REVIEW), ([160, 10, 190, 60], regions.AUTO_REVIEW)])
    masks = {m["region_id"]: m for m in masking.mask_boxes(selection)}
    assert set(masks) == {"p1_R0", "p1_R1", "p1_R2", "p1_R3", "p1_R5"}  # review region never masked
    assert masks["p1_R2"]["covered_by"] == "p1_R0" and masks["p1_R3"]["covered_by"] == "p1_R0"
    image = Image.new("RGB", (200, 100), (0, 0, 0))
    masked = masking.apply_masks(image, list(masks.values()))
    assert masked.getpixel((35, 35)) == masking.FILL and masked.getpixel((85, 85)) == masking.FILL
    assert masked.getpixel((65, 15)) == (0, 0, 0)    # outside the union: not enlarged
    assert masked.getpixel((95, 95)) == (0, 0, 0)
    assert masked.getpixel((120, 30)) == (0, 0, 0)   # review region visible to the page OCR
    assert masked.getpixel((175, 30)) == masking.FILL  # legacy "auto_review" counts as processed


def test_text_inside_a_mask_is_reported() -> None:
    selection = _select([_box("chart", [100, 300, 700, 600], 0), _box("text", [150, 400, 650, 450], 1)])
    _masked, record = masking.plan(selection, _page_image())
    (item,) = record["text_under_masks"]
    assert item["block_index"] == 1 and item["share_masked"] == 1.0 and item["region_id"] == "p1_R0"


def test_masked_page_provenance_and_checksums_are_recorded(ctx) -> None:  # noqa: F811
    run_dir = ctx["run_dir"]
    job = next(j for j in pipeline.all_jobs(run_dir) if j["job_id"] == "p001_page")
    record = job["masking"]
    assert record["applied"] and record["policy"] == masking.MASK_POLICY and record["version"] == masking.MASK_VERSION
    assert record["masked_region_ids"] == ["p1_R0"] and record["not_masked_review_region_ids"] == ["p1_R1"]
    sent = (run_dir / job["image_file"]).read_bytes()
    assert record["masked_image_file"] == job["image_file"] and record["masked_image_sha256"] == hashlib.sha256(sent).hexdigest()
    original = (run_dir / record["original_image_file"]).read_bytes()
    assert record["original_image_sha256"] == hashlib.sha256(original).hexdigest()
    assert record["masks"][0]["box_px"] == [100, 200, 700, 600]
    chart = Image.open(io.BytesIO(original)).getpixel((400, 400))
    assert chart != masking.FILL and Image.open(io.BytesIO(sent)).getpixel((400, 400)) == masking.FILL


def test_page_ocr_gets_the_masked_image_and_crop_ocr_the_original(ctx) -> None:  # noqa: F811
    fake = FakeQari()
    pipeline.execute(ctx["run_dir"], ctx["config"], _client(fake))
    import base64

    sent = {r["request_id"].split(":")[1]: Image.open(io.BytesIO(base64.b64decode(r["image_png_b64"])))
            for r in fake.requests}
    assert masking.masked_fraction(sent["p001_page"], [100, 200, 700, 600]) == 1.0
    assert sent["p001_page"].getpixel((500, 1000)) != masking.FILL  # outside the visual: original pixels
    crop = sent["p001_R0"].convert("RGB")
    assert crop.getpixel((360, 260)) != masking.FILL  # the crop is the unmasked original (blue chart)
    assert masking.masked_fraction(crop, [60, 60, 660, 460]) < 0.01


# ------------------------------------------------------------------ 14-15 duplicate detection


def test_order_insensitive_value_and_label_duplicate_is_detected() -> None:
    found = duplicates.candidates(2, "مقدمة\nالمسار الأحمر 18,450\n", {"p2_R0": "18,450 المسار الأحمر"}, source="raw")
    (candidate,) = found
    assert candidate["value"] == "18450" and candidate["region_id"] == "p2_R0" and not candidate["same_order"]
    assert duplicates.candidates(2, "المسار الأحمر ١٨٬٤٥٠", {"r": "18,450 المسار الأحمر"}, source="raw")
    english = duplicates.candidates(1, "Red route 18,450", {"r": "18,450 red route"}, source="raw")
    assert len(english) == 1


def test_similar_values_units_and_separators_are_not_confused() -> None:
    crop = {"r": "18,450 المسار الأحمر\n3.01% نسبة\n٦,٤ مؤشر"}
    assert duplicates.candidates(1, "المسار الأحمر 18,451", crop, source="raw") == []
    assert duplicates.candidates(1, "نسبة 3.01‰", crop, source="raw") == []        # % vs ‰
    assert duplicates.candidates(1, "مؤشر 64", crop, source="raw") == []           # ambiguous 6,4 is not 64
    assert duplicates.candidates(1, "مؤشر ٦,٤", crop, source="raw")                 # same written form
    assert duplicates.candidates(1, "المسار الأزرق 18,450", crop, source="raw") == []  # other label


def test_a_number_alone_is_never_a_duplicate_candidate() -> None:
    assert duplicates.candidates(1, "18,450", {"r": "18,450"}, source="raw") == []


def test_duplicate_check_is_report_only_and_repeated_rows_are_kept() -> None:
    page = "| North | 120 |\n| North | 120 |\nTotal 120 units"
    crop = {"p1_R0": "North 120"}
    before = page
    found = duplicates.candidates(1, page, crop, source="raw")
    assert page == before and len(found) == 2  # both repeated rows reported, nothing removed
    kept = dedup.remove_figure_spans(page, {"p1_R0": "| North | 120 |\n| North | 120 |"})
    assert kept["text"] == "Total 120 units"  # only the span matching the crop rows; distinct line kept
    summary = duplicates.summary(found, [], {"p1_R0"})
    assert summary["raw_page_ocr"] == {"total": 2, "masked_regions": 2, "unmasked_regions": 0}


# ------------------------------------------------------------------ 16 reading order


def _visual(rid, box):
    return {"region_id": rid, "bbox_pt": box}


def _text(index, box):
    return {"block_index": index, "label": "text", "bbox_pt": box}


def test_body_paragraphs_are_placed_between_figures() -> None:
    visuals = [_visual("v1", [50, 200, 550, 300])]
    texts = [_text(0, [50, 50, 550, 150]), _text(1, [50, 350, 550, 450])]
    placed = order.place_body("first paragraph\n\nsecond paragraph", visuals, texts, rtl=False)
    assert placed["method"] == "paragraphs_aligned_to_text_blocks"
    assert [(s["kind"], s.get("text")) for s in placed["sequence"]] == [
        ("body", "first paragraph"), ("visual", None), ("body", "second paragraph")]


def test_arabic_rows_are_read_right_to_left() -> None:
    visuals = [_visual("left", [50, 100, 250, 300]), _visual("right", [300, 100, 550, 300])]
    placed = order.place_body(None, visuals, [], rtl=True)
    assert [s["item"]["region_id"] for s in placed["sequence"]] == ["right", "left"]
    english = order.place_body(None, visuals, [], rtl=False)
    assert [s["item"]["region_id"] for s in english["sequence"]] == ["left", "right"]


def test_fallbacks_are_deterministic_and_recorded() -> None:
    visuals = [_visual("v1", [50, 200, 550, 300])]
    first = order.place_body("only text", visuals, [], rtl=False)
    assert first["method"] == "page_text_first" and first["sequence"][0]["kind"] == "body" and first["limitation"]
    single = order.place_body("a\n\nb", visuals, [_text(0, [50, 400, 550, 500])], rtl=False)
    assert single["method"] == "single_text_segment" and [s["kind"] for s in single["sequence"]] == ["visual", "body"]
    texts = [_text(0, [50, 50, 550, 150]), _text(1, [50, 155, 550, 190]), _text(2, [50, 350, 550, 450])]
    approx = order.place_body("p1\n\np2\n\np3\n\np4", visuals, texts, rtl=False)
    assert approx["method"] == "proportional_to_text_block_area" and approx["limitation"]
    bodies = [s["text"] for s in approx["sequence"] if s["kind"] == "body"]
    assert "\n\n".join(bodies) == "p1\n\np2\n\np3\n\np4"  # nothing lost, order preserved
    assert approx == order.place_body("p1\n\np2\n\np3\n\np4", visuals, texts, rtl=False)  # deterministic


# ------------------------------------------------------------------ 13, 17-18 compatibility


def test_process_now_never_reruns_the_page_ocr(ctx) -> None:  # noqa: F811
    fake = FakeQari()
    pipeline.execute(ctx["run_dir"], ctx["config"], _client(fake))
    pipeline.process_now(ctx["run_dir"], "p1_R1")
    pipeline.execute(ctx["run_dir"], ctx["config"], _client(fake))
    sent = [r["request_id"].split(":")[1] for r in fake.requests]
    assert sent.count("p001_page") == 1 and sent[-1] == "p001_R1"
    from lab.core.auto import assemble

    document = assemble.assemble(ctx["run_dir"])
    flag = next(f for f in document["validation"]["flags"] if f["code"] == "unmasked_region_possible_duplicate")
    assert flag["page"] == 1 and flag["region_id"] == "p1_R1"
    assert "duplicate_check" in document["validation"]
    element = next(e for e in document["elements"] if e.get("region_id") == "p1_R1")
    assert element["content"] and element["masked_in_page_ocr"] is False  # both versions kept


def test_digital_pages_and_validation_report_remain_compatible(ctx) -> None:  # noqa: F811
    from lab.core.auto import assemble

    pipeline.execute(ctx["run_dir"], ctx["config"], _client(FakeQari()))
    document = assemble.assemble(ctx["run_dir"])
    digital = document["pages"][1]
    assert digital["route"]["route"] == "DIGITAL" and digital["reading_order"]["method"] in {"layout_v3_xy_cut", "layout_fallback"}
    assert any("Digital page heading" in (e.get("content") or "") for e in digital["elements"])
    report = document["validation"]
    assert {"summary", "status", "flags", "numbers", "duplicate_check"} <= set(report)
    assert set(report["duplicate_check"]["final_document"]) == {"total", "masked_regions", "unmasked_regions"}


def test_saved_runs_are_not_modified_by_reassembly(ctx) -> None:  # noqa: F811
    from lab.core.auto import assemble

    pipeline.execute(ctx["run_dir"], ctx["config"], _client(FakeQari()))
    before = {p: p.read_bytes() for p in ctx["run_dir"].rglob("*") if p.is_file()}
    assemble.assemble(ctx["run_dir"])
    assemble.assemble(ctx["run_dir"])
    assert {p: p.read_bytes() for p in ctx["run_dir"].rglob("*") if p.is_file()} == before
    assert json.dumps(assemble.assemble(ctx["run_dir"])["validation"]["duplicate_check"])  # serializable
    assert httpx  # imported for the shared fixtures


def test_compare_is_read_only_and_reports_both_runs(ctx) -> None:  # noqa: F811
    from lab.core.auto import compare

    pipeline.execute(ctx["run_dir"], ctx["config"], _client(FakeQari()))
    before = {p: p.read_bytes() for p in ctx["run_dir"].rglob("*") if p.is_file()}
    result = compare.compare(ctx["run_dir"], ctx["run_dir"])
    assert result["delta"]["visuals_processed"] == 0 and result["notes"] == []
    assert result["new"]["masked_regions"] == 1 and result["new"]["visuals"]["processed"] == 1
    assert {p: p.read_bytes() for p in ctx["run_dir"].rglob("*") if p.is_file()} == before


# ------------------------------------------------------------------ mixed page with an embedded scan


class EmbeddedPredictor:
    def predict(self, array, batch_size=1):
        return [{"res": {"boxes": [{"label": "chart", "coordinate": [100, 100, 600, 500], "score": 0.9},
                                   {"label": "text", "coordinate": [100, 700, 1000, 800], "score": 0.9}]}}]


def test_mixed_page_embedded_scan_is_masked_and_assembled(tmp_path, monkeypatch) -> None:
    import fitz

    from lab.core.auto import assemble, router
    from lab.core.auto import config as auto_config

    monkeypatch.setenv("QARI_SERVICE_TOKEN", "tok")
    scan = Image.new("RGB", (1110, 1300), (245, 245, 245))
    ImageDraw.Draw(scan).rectangle([100, 100, 599, 499], fill=(30, 90, 200))
    buffer = io.BytesIO()
    scan.save(buffer, format="JPEG", quality=90)
    pdf = tmp_path / "Mixed.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 60), "Native heading text on a mixed page", fontsize=12)
    page.insert_image(fitz.Rect(20, 100, 575, 750), stream=buffer.getvalue())
    page.insert_text((72, 800), "Native closing sentence below the scan.", fontsize=11)
    doc.save(pdf)
    doc.close()
    file = tmp_path / "auto.toml"
    file.write_text("", encoding="utf-8")
    config = auto_config.load_config(file, environ={"AGENTICDOC_RESULTS_DIR": str(tmp_path / "results"),
                                                    "QARI_SERVICE_URL": "http://127.0.0.1:8000"})
    detector = regions.LayoutDetector(model_name="PP-DocLayout_plus-L", model_dir=None, device="cpu",
                                      threshold=None, predictor=EmbeddedPredictor())
    run_dir = pipeline.create_run(pdf, display_name="Mixed.pdf", config=config, detector=detector)
    plan = pipeline.load_plan(run_dir)
    route = plan["routing"]["pages"][0]
    assert route["route"] == router.MIXED and route["embedded_scan_regions"]
    jobs = {j["job_id"]: j for j in plan["jobs"]["jobs"]}
    assert set(jobs) == {"p001_E0R0", "p001_E0_body"}  # the embedded scan replaces the whole-figure job
    body_job = jobs["p001_E0_body"]
    assert body_job["masking"]["masked_region_ids"] == ["p1_E0R0"]
    sent_body = Image.open(run_dir / body_job["image_file"])
    assert masking.masked_fraction(sent_body, [100, 100, 600, 500]) == 1.0
    original = Image.open(run_dir / body_job["masking"]["original_image_file"])
    assert original.convert("RGB").getpixel((350, 300)) != masking.FILL
    crop = Image.open(run_dir / jobs["p001_E0R0"]["image_file"]).convert("RGB")
    assert masking.masked_fraction(crop, [60, 60, 560, 460]) < 0.01  # crop is unmasked
    fake = FakeQari()
    assert pipeline.execute(run_dir, config, _client(fake))["completed"] == 2
    document = assemble.assemble(run_dir)
    (page_doc,) = document["pages"]
    text = "\n".join(e.get("content") or "" for e in page_doc["elements"])
    assert "Native heading text" in text and "Native closing sentence" in text  # native text kept
    assert text.count("North 120") == 1  # figure content once (safety net removed the fake page copy)
    assert page_doc["duplicates"]["masked_ids"] == ["p1_E0R0"]
    assert document["validation"]["duplicate_check"]["final_document"]["total"] == 0
    manifest = json.loads(assemble.finalize(run_dir, config).read_text(encoding="utf-8"))
    assert manifest["masking"]["p001_E0_body"]["masked_region_ids"] == ["p1_E0R0"]
