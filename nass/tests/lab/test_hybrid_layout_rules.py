"""Layout-box selection rules: nested containment, split visuals, titles overlapping the top
boundary, captions of a previous figure, and regions that may cut content. Synthetic boxes only."""

from __future__ import annotations

import fitz

from lab.core.hybrid import layout_regions
from lab.core.hybrid import run as hybrid
from tests.lab.test_hybrid_layout_regions import IMAGE_PX, PAGE_PT, scanned  # noqa: F401


def _items(*specs):
    return [{"label": label, "bbox_pt": list(box), "block_index": index} for index, (label, box) in enumerate(specs)]


def _by_index(items):
    return {item["block_index"]: item for item in items}


# ------------------------------------------------------------ containment


def test_single_nested_box_prefers_the_complete_outer_visual() -> None:
    status = _by_index(layout_regions.classify(_items(("chart", (0, 0, 400, 300)), ("image", (40, 40, 200, 200)))))
    assert status[0]["status"] == "selected" and status[0]["default_selected"]
    assert status[1]["status"] == "contained" and not status[1]["default_selected"]
    assert status[0]["decisions"][0]["decision"] == "outer visual preferred"
    assert "partially overlaps" not in " ".join(status[0]["reasons"])  # containment is not a partial overlap


def test_distinct_inner_visuals_split_the_container() -> None:
    status = _by_index(layout_regions.classify(_items(("image", (0, 0, 600, 400)),
                                                       ("chart", (10, 10, 290, 390)), ("chart", (310, 10, 590, 390)))))
    assert status[0]["status"] == "container" and not status[0]["default_selected"]
    assert status[1]["status"] == status[2]["status"] == "selected"
    assert status[0]["decisions"][0]["decision"] == "split into inner visuals"


def test_unclear_inner_boxes_keep_the_outer_and_ask_for_review() -> None:
    # two small inner boxes (legend-like): not distinct visuals
    status = _by_index(layout_regions.classify(_items(("chart", (0, 0, 600, 400)),
                                                       ("image", (10, 10, 80, 60)), ("image", (100, 10, 170, 60)))))
    assert status[0]["status"] == "selected" and "Needs Review" in " ".join(status[0]["reasons"])
    assert status[1]["status"] == "contained" and status[2]["status"] == "contained"


def test_genuine_partial_overlap_is_still_uncertain() -> None:
    status = _by_index(layout_regions.classify(_items(("chart", (0, 0, 300, 300)), ("table", (200, 200, 500, 500)))))
    assert status[0]["status"] == status[1]["status"] == "uncertain"


# ------------------------------------------------------------------ titles


def test_title_overlapping_the_top_boundary_is_attached_in_full() -> None:
    items = _items(("chart", (50, 150, 350, 300)), ("figure_title", (60, 140, 340, 165)))  # 15 pt into the chart
    layout_regions.attach_titles(items)
    chart = items[0]
    assert chart["title_attachments"][0]["overlaps_boundary"] and chart["bbox_pt"][1] == 140
    deep = _items(("chart", (50, 150, 350, 300)), ("figure_title", (60, 200, 340, 220)))  # inside, not at the top
    layout_regions.attach_titles(deep)
    assert "title_attachments" not in deep[0]


def test_caption_of_the_previous_figure_is_not_attached_to_the_next() -> None:
    items = _items(("chart", (50, 100, 350, 250)), ("figure_title", (60, 255, 340, 270)),
                   ("chart", (50, 300, 350, 450)))
    layout_regions.attach_titles(items)
    assert "title_attachments" not in items[2] and "caption of block 0" in items[1]["title_decision"]
    assert "title_attachments" not in items[0]  # captions below are off by default
    items = _items(("chart", (50, 100, 350, 250)), ("figure_title", (60, 255, 340, 270)),
                   ("chart", (50, 300, 350, 450)))
    layout_regions.attach_titles(items, captions_below=True)
    assert items[0]["title_attachments"][0]["position"] == "below" and "title_attachments" not in items[2]


def test_text_between_two_figures_at_similar_distance_is_left_for_review() -> None:
    items = _items(("chart", (50, 100, 350, 250)), ("figure_title", (60, 265, 340, 285)),
                   ("chart", (50, 300, 350, 450)))
    layout_regions.attach_titles(items)
    assert items[1]["title_status"] == "ambiguous"
    assert items[0].get("title_ambiguous") and items[2].get("title_ambiguous")


# ------------------------------------------------------------- incomplete


def test_region_cutting_another_block_is_flagged_with_a_proposed_expansion() -> None:
    regions = layout_regions.classify(_items(("table", (50, 100, 350, 300))))
    others = _items(("table", (50, 100, 350, 300)), ("text", (300, 150, 450, 200)))
    layout_regions.flag_cut_content(regions, others)
    table = regions[0]
    assert table["bbox_pt"] == [50, 100, 350, 300]  # the detected box is not changed
    assert table["proposed_expansion_pt"] == [50, 100, 450, 300]
    assert "Needs Review" in " ".join(table["reasons"]) and table["suspected_missing_content"][0]["block_index"] == 1


def test_expansion_is_applied_only_when_chosen_and_the_detected_box_is_kept(scanned) -> None:  # noqa: F811
    blocks = [{"page": 1, "label": "table", "bbox_px": [100, 200, 700, 600], "image_px_size": IMAGE_PX,
               "region_bbox_pt": PAGE_PT, "source_id": "t", "block_index": 0, "warnings": []},
              {"page": 1, "label": "text", "bbox_px": [600, 300, 900, 400], "image_px_size": IMAGE_PX,
               "region_bbox_pt": PAGE_PT, "source_id": "t", "block_index": 1, "warnings": []}]
    metadata, _plan, _library = hybrid.load(scanned["run_dir"])
    with fitz.open(hybrid.layout_pdf(scanned["run_dir"], metadata, scanned["source"])[0]) as document:
        plain = {c["block_index"]: c for c in layout_regions.propose(document, blocks, labels=["table"])}
        grown = {c["block_index"]: c for c in layout_regions.propose(document, blocks, labels=["table"],
                                                                     expansions=["p1_L0"])}
    assert plain[0]["proposed_expansion_pt"] == [50.0, 100.0, 450.0, 300.0] and not plain[0].get("expansion_applied")
    assert plain[0]["bbox_pt"] == plain[0]["bbox_pt_detected"] == [50.0, 100.0, 350.0, 300.0]
    assert grown[0]["expansion_applied"] and grown[0]["bbox_pt"] == [50.0, 100.0, 450.0, 300.0]
    assert grown[0]["bbox_pt_detected"] == [50.0, 100.0, 350.0, 300.0]
    assert grown[0]["pixel_box"][2] > plain[0]["pixel_box"][2]


# ------------------------------------------------------- caption trimming


def _caption(box, decision="caption of block 9 (closer to the visual above)", status=None):
    item = {"label": "figure_title", "bbox_pt": list(box), "block_index": 9, "title_decision": decision}
    if status:
        item["title_status"] = status
    return item


def test_foreign_caption_in_the_top_of_a_kept_box_is_trimmed_off() -> None:
    regions = layout_regions.classify(_items(("image", (50, 100, 350, 400))))
    caption = _caption((60, 105, 340, 120))
    layout_regions.trim_foreign_captions(regions, [caption], margin_pt=5.0)
    region = regions[0]
    assert region["bbox_pt"] == [50, 125.0, 350, 400]
    trim = region["decisions"][-1]
    assert trim["type"] == "caption_trim" and trim["applied"] and trim["original_bbox_pt"] == [50, 100, 350, 400]
    assert "title_attachments" not in region  # the caption is never attached


def test_ambiguous_caption_is_trimmed_but_never_attached() -> None:
    regions = layout_regions.classify(_items(("image", (50, 100, 350, 400))))
    layout_regions.trim_foreign_captions(regions, [_caption((60, 105, 340, 120), decision=None, status="ambiguous")],
                                         margin_pt=5.0)
    assert regions[0]["bbox_pt"][1] == 125.0 and "title_attachments" not in regions[0]


def test_trimming_that_would_cut_the_visual_is_not_applied() -> None:
    regions = layout_regions.classify(_items(("image", (50, 100, 350, 400))))
    others = [_caption((60, 105, 200, 120)), {"label": "text", "bbox_pt": [210, 104, 340, 140], "block_index": 5}]
    layout_regions.trim_foreign_captions(regions, others, margin_pt=5.0)
    assert regions[0]["bbox_pt"] == [50, 100, 350, 400]
    assert not regions[0]["decisions"][-1]["applied"] and "block 5" in regions[0]["decisions"][-1]["reason"]


def test_caption_lower_in_the_box_is_not_trimmed() -> None:
    regions = layout_regions.classify(_items(("image", (50, 100, 350, 400))))
    layout_regions.trim_foreign_captions(regions, [_caption((60, 250, 340, 265))], margin_pt=5.0)
    assert regions[0]["bbox_pt"] == [50, 100, 350, 400]


# ------------------------------------------------------- margin end-to-end


def test_margin_from_the_preview_is_used_for_the_saved_crops(scanned) -> None:  # noqa: F811
    import io
    import json
    import zipfile

    from PIL import Image

    from lab.core.hybrid import enrich
    from tests.lab.test_hybrid_layout_regions import BLOCKS

    metadata, _plan, _library = hybrid.load(scanned["run_dir"])
    with fitz.open(hybrid.layout_pdf(scanned["run_dir"], metadata, scanned["source"])[0]) as document:
        preview60 = {c["region_id"]: c["pixel_box"] for c in layout_regions.propose(document, BLOCKS, labels=["chart"],
                                                                                    margin_px=60) if c.get("region_id")}
        preview15 = {c["region_id"]: c["pixel_box"] for c in layout_regions.propose(document, BLOCKS, labels=["chart"],
                                                                                    margin_px=15) if c.get("region_id")}
    assert preview60["p1_L0"] == [40, 140, 760, 660]
    source = {"kind": "test"}
    import pytest

    with pytest.raises(enrich.EnrichError, match="differ from the preview"):
        layout_regions.create_layout_region_set(scanned["run_dir"], blocks=BLOCKS, labels=["chart"], margin_px=60,
                                                selected=["p1_L0"], source=source, pdf_path=scanned["source"],
                                                by="Tester", expected_pixel_boxes=preview15)
    folder = layout_regions.create_layout_region_set(scanned["run_dir"], blocks=BLOCKS, labels=["chart"], margin_px=60,
                                                     selected=["p1_L0"], source=source, pdf_path=scanned["source"],
                                                     by="Tester", expected_pixel_boxes=preview60)
    data = json.loads((folder / "regions.json").read_text(encoding="utf-8"))
    region = data["regions"][0]
    assert data["region_source"]["margin_px"] == 60 and data["region_source"]["preview_verified"]
    assert region["layout_box"]["margin_px"] == 60 and region["crop_options"]["pixel_box"] == [40, 140, 760, 660]
    assert Image.open(folder / region["crop_file"]).size == (720, 520)
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(
        enrich.export_crops_zip(folder, provider="Qari", model="Qari", by="Tester"))).read("manifest.json"))
    assert manifest["regions"][0]["layout_box"]["margin_px"] == 60
    assert manifest["regions"][0]["crop_options"]["margin_px"] == 60
