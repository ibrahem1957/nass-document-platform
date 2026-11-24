"""Crop-margin caption leakage diagnostic (dedup Rule 3). Diagnostic only: text is never changed. Synthetic boxes
(pixel and point coordinates are given the same values for readability)."""

from __future__ import annotations

import copy

from lab.core.auto import dedup


def _region(rid, detected, margin=60, attachments=None):
    x0, y0, x1, y1 = detected
    return {"region_id": rid, "bbox_px": list(detected), "bbox_pt_detected": list(detected), "margin_px": margin,
            "pixel_box": [x0 - margin, y0 - margin, x1 + margin, y1 + margin],
            "title_attachments": attachments or []}


def _block(index, box, label="figure_title", **extra):
    return {"block_index": index, "label": label, "bbox_px": list(box), "bbox_pt": list(box), **extra}


TABLE = (300, 100, 1400, 880)     # visual above
FORM = (300, 965, 1400, 1530)     # visual below
TITLE = (700, 901, 1000, 931)     # between them: 21 px below the table, 34 px above the form


def _selection(regions, others=(), attached=()):
    return {"regions": list(regions), "others": list(others), "attached_titles": list(attached)}


def test_page7_pattern_contested_attached_title_in_margin_is_flagged_not_removed():
    form = _region("p7_R1", FORM, attachments=[{"block_index": 1}])
    table = _region("p7_R0", TABLE)
    title = _block(1, TITLE, title_status="attached to block 4 (above)")
    text = "الشكل 13: سجل الصيانة طلب اشتراك شهري الاسم"
    before = copy.deepcopy(text)
    found = dedup.suspected_margin_leakage(form, _selection([table, form], attached=[title]), text)
    assert len(found) == 1
    record = found[0]
    assert record["reason_code"] == "attached_title_with_competing_visual"
    # crop starts at 965 - 60 = 905 px: 26 of the title's 30 px rows are inside the margin
    assert record["block_share_inside_crop"] == 0.867 and record["block_rows_inside_crop_px"] == 26.0
    assert record["side"] == "above" and record["outside_detected_visual"] is True
    assert record["crop_caption_labels"][0]["text"] == "الشكل 13:" and record["label_attribution"] == "unattributed"
    assert record["evidence_level"] == "geometric_risk_with_caption_label_in_crop_text"
    assert record["confirmed"] is False and record["units"] == "px"
    assert text == before  # diagnostic only


def test_title_fully_inside_the_margin_is_measured_as_full_overlap():
    form = _region("R1", FORM)
    title = _block(1, (700, 925, 1000, 955), title_status="ambiguous")
    found = dedup.suspected_margin_leakage(form, _selection([_region("R0", TABLE), form], others=[title]), None)
    assert found[0]["block_share_inside_crop"] == 1.0 and found[0]["crop_caption_labels"] == []


def test_partial_overlap_without_caption_label_is_geometric_risk_only():
    form = _region("R1", FORM)
    title = _block(1, (700, 880, 1000, 910), title_status="ambiguous")  # 5 of 30 px inside the crop (905..)
    found = dedup.suspected_margin_leakage(form, _selection([_region("R0", TABLE), form], others=[title]),
                                           "Monthly subscription request Name Card number")
    assert found[0]["reason_code"] == "ambiguous_title_in_crop_margin"
    assert 0 < found[0]["block_share_inside_crop"] < 1
    assert found[0]["evidence_level"] == "geometric_risk_only" and found[0]["crop_caption_labels"] == []


def test_title_entirely_outside_the_crop_is_not_reported():
    form = _region("R1", FORM)
    title = _block(1, (700, 820, 1000, 850))
    assert dedup.suspected_margin_leakage(form, _selection([form], others=[title]), "Figure 2: x") == []


def test_internal_chart_title_inside_the_detected_box_is_not_reported():
    chart = _region("R0", TABLE)
    internal = _block(3, (700, 110, 1000, 140))  # inside the detected visual
    found = dedup.suspected_margin_leakage(chart, _selection([chart], others=[internal]),
                                           "Route performance by week Figure 3: Route performance 12 14")
    assert found == []


def test_own_unambiguous_attached_title_is_not_reported():
    chart = _region("R0", FORM, attachments=[{"block_index": 2}])
    own = _block(2, (700, 925, 1000, 955), title_status="attached to block 0 (above)")
    assert dedup.suspected_margin_leakage(chart, _selection([chart], attached=[own]), "Figure 1: x") == []


def test_multiple_nearby_blocks_each_recorded_and_labels_unattributed():
    form = _region("R1", FORM)
    above = _block(1, (700, 920, 1000, 950), title_status="ambiguous")
    below = _block(5, (700, 1545, 1000, 1575), label="text")  # a caption detected as plain text
    found = dedup.suspected_margin_leakage(form, _selection([_region("R0", TABLE), form], others=[above, below]),
                                           "Figure 4: Payment methods Smart card 41.0%")
    assert {(r["block_index"], r["side"]) for r in found} == {(1, "above"), (5, "below")}
    assert all(r["label_attribution"] == "unattributed" and r["crop_caption_labels"] for r in found)
    assert {r["reason_code"] for r in found} == {"ambiguous_title_in_crop_margin", "text_block_in_crop_margin"}


def test_no_title_metadata_or_no_pixels_gives_nothing():
    form = _region("R1", FORM)
    assert dedup.suspected_margin_leakage(form, {"regions": [form]}, "Figure 1: x") == []
    assert dedup.suspected_margin_leakage({"region_id": "R"}, _selection([], others=[_block(1, TITLE)]), "x") == []


def test_arabic_and_english_labels_with_arabic_digits():
    labels = dedup.crop_caption_labels("الشكل ٩: مؤشرات التشغيل ... Figure 10: Fleet Table 3 - costs")
    assert [label["text"] for label in labels] == ["الشكل ٩:", "Figure 10:", "Table 3 -"]
    assert dedup.crop_caption_labels("يشكل 30% من الرحلات") == []  # a word that merely contains the letters
