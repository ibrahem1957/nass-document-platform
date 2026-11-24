"""Typography association and saved layout handling (no PDF needed for these unit tests)."""

from __future__ import annotations

from docprep.config import LayoutConfig, TypographyConfig
from docprep.layout import LayoutBox, load_layout, match_bbox, segment_boxes
from docprep.typography import (
    PdfSpan,
    associate_page,
    body_profile,
    color_distance,
    is_bold,
)


def span(text, bbox, size=11.0, font="Helvetica", flags=0, color=0):
    return PdfSpan(1, tuple(bbox), text, size, font, flags, color, is_bold(font, flags), False)


def test_spans_attach_only_to_the_matching_neighbour():
    elements = [("h", [72, 90, 300, 110], "Overview"), ("b", [72, 112, 540, 126], "Body text follows here.")]
    spans = [span("Overview", [72, 91, 160, 109], size=16, font="Helvetica-Bold"),
             span("Body text follows here.", [72, 113, 260, 125])]
    result = associate_page(elements, spans, TypographyConfig())
    assert result["h"].available and result["h"].size == 16 and result["h"].bold_share == 1.0
    assert result["b"].available and result["b"].size == 11 and result["b"].bold_share == 0.0


def test_spans_with_different_text_are_not_attached():
    elements = [("x", [72, 90, 300, 110], "Completely different words")]
    result = associate_page(elements, [span("Overview", [72, 91, 160, 109])], TypographyConfig())
    assert not result["x"].available and "do not match" in result["x"].reason


def test_presentation_forms_match_logical_arabic_text():
    # NFKC folds Arabic presentation forms; the character order of PDF spans is irrelevant.
    elements = [("a", [300, 90, 540, 110], "لا")]
    result = associate_page(elements, [span("ﻻ", [480, 92, 500, 108])], TypographyConfig())
    assert result["a"].available


def test_body_profile_is_character_weighted():
    elements = [("t", [72, 80, 300, 100], "Title"), ("b1", [72, 110, 540, 124], "A" * 200),
                ("b2", [72, 130, 540, 144], "B" * 150)]
    spans = [span("Title", [72, 81, 150, 99], size=20), span("A" * 200, [72, 111, 530, 123], size=10.5),
             span("B" * 150, [72, 131, 530, 143], size=10.5)]
    profile = body_profile(associate_page(elements, spans, TypographyConfig()).values())
    assert profile.size == 10.5 and profile.chars == 355


def test_bold_and_colour_helpers():
    assert is_bold("Arial-BoldMT", 0) and is_bold("Anything", 16) and not is_bold("Helvetica", 0)
    assert color_distance("#000000", "#cc0000") > 60 and color_distance("#000000", "#101010") < 60


LAYOUT = {"pages": [{"page": 1, "page_image": {"px": [1700, 2200], "bbox_pt": [0.0, 0.0, 612.0, 792.0]},
                     "boxes": [{"label": "paragraph_title", "bbox_px": [100, 100, 900, 160], "score": 0.9,
                                "block_index": 0},
                               {"label": "text", "bbox_px": [100, 100, 900, 165], "score": 0.8, "block_index": 1},
                               {"label": "figure_title", "bbox_px": [100, 1000, 900, 1040], "score": 0.4,
                                "block_index": 2}]}]}


def test_layout_pixels_are_converted_to_points():
    boxes, sizes, warnings = load_layout(LAYOUT)
    assert sizes[1] == (612.0, 792.0) and not warnings
    first = boxes[1][0]
    assert abs(first.bbox[0] - 100 * 612 / 1700) < 1e-6 and abs(first.bbox[3] - 160 * 792 / 2200) < 1e-6


def test_overlapping_regions_with_different_roles_are_ambiguous():
    boxes, _, _ = load_layout(LAYOUT)
    match = match_bbox([36.5, 36.5, 323.0, 57.5], boxes[1], LayoutConfig())
    assert match is not None and match.ambiguous and match.note.startswith("overlapping")


def test_no_match_without_enough_overlap():
    boxes, _, _ = load_layout(LAYOUT)
    assert match_bbox([400, 600, 600, 700], boxes[1], LayoutConfig()) is None


def test_missing_page_image_geometry_is_reported():
    _, _, warnings = load_layout({"pages": [{"page": 1, "boxes": [{"label": "text", "bbox_px": [0, 0, 1, 1]}]}]})
    assert warnings and "without page image geometry" in warnings[0]


def test_segment_boxes_follow_the_recorded_text_blocks():
    boxes = [LayoutBox(1, "text", 0.9, [0, 0, 10, 10], 4), LayoutBox(1, "paragraph_title", 0.6, [0, 0, 5, 5], 5)]
    selected = segment_boxes({"body_segment": {"segment": 0, "text_blocks": [5, 4]}}, boxes)
    assert [b.block_index for b in selected] == [5, 4]
    assert segment_boxes({}, boxes) == []
