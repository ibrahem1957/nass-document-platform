"""Model-free semantic verification layer."""

from __future__ import annotations

from docprep.semantic import (
    HeuristicVerifier,
    NullVerifier,
    content_terms,
    make_verifier,
)


def test_related_following_text_supports_heading():
    ev = HeuristicVerifier().verify("معالجة وتنظيف البيانات:",
                                    ["تم التعامل مع القيم المفقودة وإزالة السجلات المكررة وتنظيف البيانات قبل التحليل."])
    assert ev.score > 0
    assert any("colon" in s for s in ev.supporting)
    assert "inconclusive" in ev.limitations


def test_shared_words_alone_are_limited_and_not_proof():
    ev = HeuristicVerifier().verify("Network", ["Network"])
    assert ev.score <= 0.3  # overlap is capped; a very short following text weakens it
    assert any("not proof" in s for s in ev.supporting)
    assert any("very short" in c for c in ev.conflicting)


def test_no_overlap_is_not_a_rejection():
    ev = HeuristicVerifier().verify("Background:", ["The project started after several delays in the previous year."])
    assert ev.score > 0
    assert any("not counted against" in s for s in ev.supporting)


def test_continuation_and_sentence_like_candidate_conflict():
    ev = HeuristicVerifier().verify("The values were checked against the reference tables.", ["and more."])
    assert ev.score < 0
    assert ev.conflicting


def test_no_following_content():
    assert HeuristicVerifier().verify("Annex", []).score < 0


def test_null_verifier_and_factory():
    assert NullVerifier().verify("x", ["y"]).score == 0.0
    assert make_verifier("heuristic").name == "heuristic"
    assert make_verifier("heuristic", enabled=False).name == "none"
    assert make_verifier("none").name == "none"


def test_light_stemming_matches_arabic_affixes():
    assert content_terms("البيانات") & content_terms("بيانات")
