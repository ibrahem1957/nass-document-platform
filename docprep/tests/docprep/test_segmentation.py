"""Derived segmentation of a single page_text element (scanned pages)."""

from __future__ import annotations

from docprep.config import SegmentationConfig
from docprep.layout import LayoutBox
from docprep.segmentation import check_partition, segment_text

PAGE = ("Network Performance Overview (synthetic) Network Performance Overview - synthetic data for testing. "
        "This chapter reviews volumes. Figure 1: Daily trips per line Figure 2: Waiting time")


def test_spans_partition_the_original_text_exactly():
    spans = segment_text("p1_e0", PAGE, [], SegmentationConfig())
    assert "".join(s.text for s in spans) == PAGE
    assert check_partition(PAGE, spans) == []
    assert [s.span_id for s in spans] == [f"p1_e0_span_{i:03d}" for i in range(1, len(spans) + 1)]
    assert all(PAGE[s.start:s.end] == s.text for s in spans)


def test_title_echo_captions_and_sentences():
    spans = segment_text("p1_e0", PAGE, [], SegmentationConfig())
    texts = [s.text.strip() for s in spans]
    assert texts == ["Network Performance Overview (synthetic)",
                     "Network Performance Overview - synthetic data for testing.",
                     "This chapter reviews volumes.", "Figure 1: Daily trips per line", "Figure 2: Waiting time"]
    assert [s.role_hint for s in spans] == ["heading", "paragraph", "paragraph", "caption", "caption"]
    assert "text:title_echoed_by_following_sentence" in spans[0].basis
    assert spans[0].uncertain and spans[0].support == "text"
    assert all(s.geometry is None for s in spans)  # no layout: no coordinates invented


def test_numbered_heading_inside_page_text():
    text = "Intro sentence one. 2) Second topic: Details about the second topic follow here. More details."
    spans = segment_text("p2_e0", text, [], SegmentationConfig())
    assert [s.text.strip() for s in spans] == ["Intro sentence one.", "2) Second topic:",
                                                "Details about the second topic follow here.", "More details."]
    assert spans[1].role_hint == "heading" and spans[1].cut_reason == "sentence_end"
    assert spans[2].cut_reason == "after_colon_label"


def test_long_sentence_ending_with_colon_is_not_cut():
    text = "This report answers several important questions about the network, including the following: delays."
    spans = segment_text("p3_e0", text, [], SegmentationConfig())
    assert len(spans) == 1 and spans[0].role_hint == "paragraph"


def test_single_layout_block_gives_only_a_container():
    box = LayoutBox(1, "text", 0.9, [50.0, 100.0, 550.0, 300.0], 3)
    spans = segment_text("p4_e0", "First sentence here. Second sentence here.", [box], SegmentationConfig())
    assert len(spans) == 2
    for s in spans:
        assert s.geometry["container_bbox_pt"] == [50.0, 100.0, 550.0, 300.0]
        assert "unknown" in s.geometry["precision"]
        assert any(b.startswith("layout:text") for b in s.basis) and s.support == "both"


def test_sequence_match_with_layout_blocks():
    boxes = [LayoutBox(1, "paragraph_title", 0.7, [100, 80, 500, 100], 5), LayoutBox(1, "text", 0.9, [60, 110, 550, 200], 1)]
    spans = segment_text("p5_e0", "Fleet summary: The fleet grew during the year.", boxes, SegmentationConfig())
    assert [s.role_hint for s in spans] == ["heading", "paragraph"]
    assert spans[0].geometry["layout_label"] == "paragraph_title" and "approximate" in spans[0].geometry["precision"]
    assert spans[0].support == "both" and "layout:sequence_match" in spans[0].basis


def test_no_boundary_keeps_the_whole_element():
    spans = segment_text("p6_e0", "a single run of words without any punctuation", [], SegmentationConfig())
    assert len(spans) == 1 and spans[0].support == "none" and spans[0].uncertain
    assert spans[0].notes and "whole element" in spans[0].notes[0]


def test_punctuation_only_remainder_is_not_dropped():
    text = "Sentence one. ... Sentence two."
    spans = segment_text("p7_e0", text, [], SegmentationConfig())
    assert "".join(s.text for s in spans) == text and check_partition(text, spans) == []


def test_empty_text_has_no_spans():
    assert segment_text("p8_e0", "   ", [], SegmentationConfig()) == []


def test_partition_check_detects_loss():
    spans = segment_text("p1_e0", PAGE, [], SegmentationConfig())
    spans[1].end -= 1
    assert check_partition(PAGE, spans)
