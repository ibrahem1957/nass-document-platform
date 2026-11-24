"""Structured `title_association` / `title_candidates` evidence: geometry and policy outcome are preserved for
every figure title, without changing any box, crop or attachment decision. Synthetic boxes only."""

from __future__ import annotations

import json

from lab.core.hybrid import layout_regions


def _items(*specs):
    return [{"label": label, "bbox_pt": list(box), "block_index": index} for index, (label, box) in enumerate(specs)]


def _attach(items, **kwargs):
    layout_regions.attach_titles(items, **kwargs)
    return items


def test_caption_below_with_policy_off_is_recorded_as_a_candidate_without_attaching():
    items = _attach(_items(("chart", (50, 100, 350, 250)), ("figure_title", (60, 262, 340, 275))))
    chart, title = items
    association = title["title_association"]
    assert association["status"] == "candidate_not_attached_by_policy"
    assert association["policy"] == {"attach_titles": True, "captions_below": False}
    assert association["candidates"][0]["position"] == "title_below_visual"
    assert association["candidates"][0]["gap_pt"] == 12.0
    assert "title_attachments" not in chart and chart["bbox_pt"] == [50, 100, 350, 250]  # crop untouched
    assert chart["title_candidates"][0]["block_index"] == 1
    assert chart["title_candidates"][0]["association_status"] == "candidate_not_attached_by_policy"
    json.dumps(items)  # the evidence is plain JSON (no object references)


def test_attached_title_keeps_existing_behaviour_and_records_status():
    items = _attach(_items(("chart", (50, 150, 350, 300)), ("figure_title", (60, 125, 340, 140))))
    chart, title = items
    assert chart["title_attachments"][0]["block_index"] == 1 and chart["bbox_pt"][1] == 125
    assert title["title_association"]["status"] == "attached"
    assert title["title_association"]["attached_block_index"] == 0
    assert "title_candidates" not in chart  # an attached title is not also a loose candidate


def test_title_between_two_visuals_records_both_measured_candidates_as_ambiguous():
    items = _attach(_items(("chart", (50, 100, 350, 250)), ("figure_title", (60, 265, 340, 285)),
                           ("chart", (50, 300, 350, 450))))
    association = items[1]["title_association"]
    assert association["status"] == "ambiguous"
    assert {(c["block_index"], c["position"]) for c in association["candidates"]} == {
        (0, "title_below_visual"), (2, "title_above_visual")}
    assert items[0]["title_candidates"] and items[2]["title_candidates"]


def test_title_far_from_any_visual_has_no_candidate():
    items = _attach(_items(("chart", (50, 100, 350, 250)), ("figure_title", (60, 400, 340, 415))))
    assert items[1]["title_association"]["status"] == "no_candidate_within_limits"
    assert items[1]["title_association"]["candidates"] == []


def _between(upper_bottom: float, title_top: float, title_bottom: float, lower_top: float):
    return _items(("table", (50, 100, 350, upper_bottom)), ("figure_title", (60, title_top, 340, title_bottom)),
                  ("image", (50, lower_top, 350, lower_top + 200)))


def test_title_only_weakly_nearer_to_the_lower_visual_is_not_attached():
    # Page-7 failure pattern: 18 pt below the upper visual, 12 pt above the lower one (5 pt margin met, < 2x).
    items = _attach(_between(300, 318, 328, 340))
    upper, title, lower = items
    assert title["title_status"] == "ambiguous"
    assert "title_attachments" not in lower and lower["bbox_pt"] == [50, 340, 350, 540]  # crop not widened
    assert upper.get("title_ambiguous") and lower.get("title_ambiguous")
    codes = title["title_association"]["reason_codes"]
    assert "weak_distance_preference" in codes and "not_attached_without_decisive_evidence" in codes
    assert title["title_association"]["status"] == "ambiguous"
    assert {c["gap_pt"] for c in title["title_association"]["candidates"]} == {18.0, 12.0}


def test_title_decisively_nearer_to_the_lower_visual_is_still_attached():
    items = _attach(_between(300, 325, 335, 340))  # 25 pt vs 5 pt
    title, lower = items[1], items[2]
    assert lower["title_attachments"][0]["block_index"] == 1 and lower["bbox_pt"][1] == 325
    assert "decisive_distance" in title["title_association"]["reason_codes"]


def test_title_overlapping_the_lower_visual_top_boundary_is_still_attached():
    items = _attach(_between(300, 320, 345, 340))  # 20 pt below the upper visual, overlaps the lower one
    assert items[2]["title_attachments"][0]["overlaps_boundary"]


def test_caption_preference_for_the_upper_visual_attaches_nothing_and_is_labelled_weak():
    items = _attach(_between(300, 316, 328, 349))  # page-1 pattern: 16 pt vs 21 pt
    upper, title, lower = items
    assert "title_attachments" not in upper and "title_attachments" not in lower
    assert "caption of block 0" in title["title_decision"]
    assert title["title_association"]["status"] == "candidate_not_attached_by_policy"
    assert "weak_distance_preference" in title["title_association"]["reason_codes"]


def test_similar_distances_stay_ambiguous_with_reason_code():
    items = _attach(_between(300, 314, 326, 340))  # 14 pt vs 14 pt
    assert items[1]["title_association"]["reason_codes"] == ["between_two_visuals", "similar_distances"]


def test_attachment_switched_off_is_reported_as_disabled():
    items = _attach(_items(("chart", (50, 150, 350, 300)), ("figure_title", (60, 125, 340, 140))), attach=False)
    assert items[1]["title_association"]["status"] == "attachment_disabled"
    assert "title_attachments" not in items[0] and items[0]["bbox_pt"] == [50, 150, 350, 300]
