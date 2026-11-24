"""Hybrid PDF Extraction: text checks, per-page combination, write-once runs (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz
import pytest

from lab.core.hybrid import analysis as hybrid_analysis
from lab.core.hybrid import qari
from lab.core.hybrid import run as hybrid
from lab.core.hybrid import textcheck
from lab.core.ingest import IngestService
from lab.core.workspace import WorkspaceManager


# ------------------------------------------------------------------ rejoin


def test_rejoin_only_removes_spaces_confirmed_by_the_vlm() -> None:
    library = "ا خ ت بار النموذج\nفي ال سنة ٢٠٢٥"
    vlm = "اختبار النموذج\nفي السنة ٢٠٢٥"
    result = textcheck.rejoin(library, vlm)
    assert result["text"] == "اختبار النموذج\nفي السنة ٢٠٢٥"
    assert result["valid"] and result["join_count"] == 2
    assert all("evidence" in join for join in result["joins"])


def test_rejoin_keeps_diacritics_and_ignores_them_only_for_matching() -> None:
    result = textcheck.rejoin("كِ تَابٌ", "كتاب")
    assert result["text"] == "كِتَابٌ" and result["valid"]


def test_rejoin_never_joins_without_vlm_evidence_or_across_digits_and_punctuation() -> None:
    assert textcheck.rejoin("في ال بيت", "منزل")["join_count"] == 0
    assert textcheck.rejoin("ال ، بيت", "البيت")["join_count"] == 0
    assert textcheck.rejoin("عام 20 25", "عام 2025")["join_count"] == 0
    result = textcheck.rejoin("ا خ ت بار.", "اختبار.")
    assert result["text"] == "اختبار." and result["valid"]


# ----------------------------------------------------------------- numbers


def test_number_audit_normalizes_digits_and_counts_occurrences() -> None:
    audit = textcheck.number_audit("القيمة ١٬٢٨٤٬٣٣٠ و ٤٩٫٥٨٪ و 7 و 7", "1,284,330 and 49.58% and 7")
    assert audit["matched_occurrences"] == 3
    assert audit["library_only"] == {"7": 1} and not audit["needs_review"]
    audit = textcheck.number_audit("12", "12 13")
    assert audit["needs_review"] and audit["vlm_only"] == {"13": 1}
    assert audit["conflicts"][0]["vlm_originals"] == ["13"]


# ----------------------------------------------------------------- quality


def test_quality_flags_runs_and_phrases_but_not_table_rows() -> None:
    assert textcheck.vlm_quality("٠" * 20)["flagged"]
    assert textcheck.vlm_quality("ـ" * 12)["flagged"]
    assert textcheck.vlm_quality(" ".join(["النتائج كانت جيدة"] * 6))["flagged"]
    table = "\n".join("| 0 | 0 | 0 |" for _ in range(6)) + "\n|---|---|---|"
    assert not textcheck.vlm_quality(table)["flagged"]
    assert "max_tokens" in " ".join(textcheck.vlm_quality("text", finish_reason="length")["reasons"])


# ----------------------------------------------------------------- combine


def _page(category: str, *, raw: str = "raw", processed: str = "processed", status: str = "not_needed",
          images: str | None = None, share: float = 0.0, decision: str | None = None) -> dict:
    return {"page": 1, "raw_text": raw, "processed_text": processed,
            "reorder": {"reorder_status": status, "decision": decision, "config": {"min_arabic_letter_share": 0.30}},
            "category": category, "category_reason": f"category {category}", "arabic_letter_share": share,
            "images": {"warning": images}}


def _vlm(text: str, state: str = "COMPLETED", status: str = "Success", **extra) -> dict:
    return {"text": text, "job_state": state, "status": status, "finish_reason": "stop",
            "truncated_at_token_limit": False} | extra


def test_category_a_keeps_library_text_and_never_merges_vlm() -> None:
    result = hybrid.combine_page(_page("A", images=hybrid_analysis.IMAGE_WARNING), _vlm("chart 99"), None)
    assert result["final_text"] == "raw" and result["status"] == hybrid.NEEDS_REVIEW  # 99 is VLM-only
    assert hybrid_analysis.IMAGE_WARNING in result["warnings"]


def test_category_b_accepts_rejoined_candidate() -> None:
    page = _page("B", raw="٥٢٠٢ جذومن ال راب ت خ ا", processed="ا خ ت بار ال نموذج ٢٠٢٥", status="uncertain",
                 share=0.9)
    result = hybrid.combine_page(page, _vlm("اختبار النموذج 2025"), None)
    assert result["final_source"] == hybrid.REJOINED_SOURCE and result["status"] == hybrid.ACCEPTED
    assert result["rejoined"]["built_from"] == "pymupdf_processed (uncertain)"
    assert result["final_text"] == "اختبار النموذج ٢٠٢٥"  # original digits kept


def test_category_c_uses_vlm_but_never_marks_it_accepted() -> None:
    result = hybrid.combine_page(_page("C", raw="", processed=""), _vlm("نص الصفحة"), None)
    assert result["final_text"] == "نص الصفحة" and result["status"] == hybrid.NEEDS_REVIEW
    truncated = hybrid.combine_page(_page("C", raw="", processed=""),
                                    _vlm("نص", finish_reason="length", truncated_at_token_limit=True), None)
    assert truncated["status"] == hybrid.UNCERTAIN


def test_failed_model_keeps_existing_text_and_invents_nothing() -> None:
    failed = _vlm("", state="SUBMISSION_UNCONFIRMED", status="Uncertain")
    assert hybrid.combine_page(_page("C", raw="", processed=""), failed, None)["status"] == hybrid.FAILED
    kept = hybrid.combine_page(_page("C", raw="garbled", processed="garbled", status="uncertain"), failed, None)
    assert kept["final_text"] == "garbled" and kept["status"] == hybrid.UNCERTAIN


# ------------------------------------------------------------- evaluation


def test_evaluation_per_page_and_total() -> None:
    evaluation = hybrid.evaluate("=== page 1 ===\nabc\n=== page 2 ===\ndef",
                                 {"final_hybrid": {1: "abc", 2: "dxf"}}, 2)
    total = evaluation["sources"]["final_hybrid"]["total"]["none"]
    assert evaluation["level"] == "page" and total["cer"] == pytest.approx(1 / 6)


# ------------------------------------------------------------------- qari


def test_qari_endpoint_rules() -> None:
    assert qari.parse_endpoint("http://localhost:8000")["loopback"]
    assert qari.parse_endpoint("https://abc-8000.proxy.runpod.net/v1")["base_path"] == ""
    with pytest.raises(qari.QariError):
        qari.parse_endpoint("http://abc-8000.proxy.runpod.net")
    with pytest.raises(qari.QariError):
        qari.parse_endpoint("https://user:pw@example.com")
    assert qari.check_ready("not a url")["status"] == "Unavailable"


def test_qari_unconfirmed_page_is_never_resent(tmp_path, monkeypatch) -> None:
    calls = []

    def fake(endpoint, token, png, **_kw):
        calls.append(png)
        return {"state": qari.UNCONFIRMED, "ok": False, "elapsed_ms": 1.0,
                "error": {"code": "no_response", "message": "timeout"}}

    monkeypatch.setattr(qari, "transcribe", fake)
    vlm = tmp_path / "vlm"
    qari.create_plan(vlm, endpoint=qari.parse_endpoint("http://localhost:8000"), model=qari.QARI_MODEL,
                     pages=[{"page": 1, "png": b"PNG1"}], document_sha256="a" * 64, approved_by="Tester")
    record = qari.send_page(vlm, 1, token=None, document_authorized=False)
    assert record["job_state"] == qari.UNCONFIRMED and record["status"] == "Uncertain"
    with pytest.raises(qari.QariError):  # remote servers need the document authorization
        remote = tmp_path / "remote"
        qari.create_plan(remote, endpoint=qari.parse_endpoint("https://x.example.com"), model=qari.QARI_MODEL,
                         pages=[{"page": 1, "png": b"PNG1"}], document_sha256="a" * 64, approved_by="Tester")
        qari.send_page(remote, 1, token=None, document_authorized=False)
    assert len(calls) == 1


# ------------------------------------------------------------ library run


@pytest.fixture
def doc(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(tmp_path / "results"))
    pdf = tmp_path / "Hybrid_Test.pdf"
    document = fitz.open()
    for number in (1, 2):
        document.new_page().insert_text((72, 72), f"English page {number} with 1,284 items", fontsize=12)
    document.save(pdf)
    document.close()
    manager = WorkspaceManager(settings)
    workspace_id = manager.create().workspace_id
    ingest = IngestService(manager, settings)
    with pdf.open("rb") as handle:
        lab_document = ingest.ingest(workspace_id, handle, pdf.name)
    return lab_document, ingest.original_path(workspace_id, lab_document.doc_id), tmp_path / "results"


def test_library_only_run_is_write_once_and_complete(doc) -> None:
    document, source, root = doc
    analysis = hybrid_analysis.analyze_pdf(source)
    assert [page["category"] for page in analysis["pages"]] == ["A", "A"]
    assert hybrid_analysis.default_vlm_pages(analysis) == {}
    run_dir = hybrid.create_run(root=root, document=document, pdf_path=source, analysis=analysis,
                                model=hybrid.LIBRARY_ONLY, vlm_pages={},
                                ground_truth=("=== page 1 ===\nEnglish page 1 with 1,284 items\n", ["gt.txt"]))
    hybrid.finalize(run_dir, None, root)
    assert run_dir.parent.name == "hybrid_pdf_extraction"
    for name in ("metadata.json", "plan.json", "library_raw.json", "tables.json", "page_001.txt", "page_002.json",
                 "full_text.txt", "results.json", "report.html", "evaluation.json"):
        assert (run_dir / name).exists(), name
    assert not (run_dir / "vlm").exists() and not (run_dir / "vlm_raw.json").exists()
    record = json.loads((run_dir / "page_001.json").read_text(encoding="utf-8"))
    assert record["status"] == hybrid.ACCEPTED and record["final_source"].startswith("pymupdf")
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert {"pymupdf_raw", "pymupdf_processed", "final_hybrid"} <= set(evaluation["sources"])
    with pytest.raises(hybrid.HybridError):
        hybrid.finalize(run_dir, None, root)


def test_model_run_requires_approval_and_selected_pages(doc) -> None:
    document, source, root = doc
    analysis = hybrid_analysis.analyze_pdf(source)
    with pytest.raises(hybrid.HybridError):
        hybrid.create_run(root=root, document=document, pdf_path=source, analysis=analysis, model=hybrid.QARI,
                          vlm_pages={1: "selected by the user"}, approved_by="Tester", statement="yes",
                          qari_endpoint="http://localhost:8000", qari_pages=[{"page": 1, "png": b"x"}])
    with pytest.raises(hybrid.HybridError):
        hybrid.create_run(root=root, document=document, pdf_path=source, analysis=analysis, model=hybrid.QARI,
                          vlm_pages={}, approved_by="Tester", statement=hybrid.APPROVAL_STATEMENT,
                          qari_endpoint="http://localhost:8000", qari_pages=[])
    assert not Path(root, "hybrid_pdf_extraction").exists() or not list(Path(root, "hybrid_pdf_extraction").iterdir())


def test_arabic_pages_use_processed_text_even_when_uncertain() -> None:
    page = _page("B", raw="٥٢٠٢ raw", processed="٢٠٢٥ processed", status="uncertain", share=0.8)
    result = hybrid.combine_page(page, None, None)
    assert result["final_text"] == "٢٠٢٥ processed" and result["library_source"] == "pymupdf_processed (uncertain)"
    assert result["status"] == hybrid.NEEDS_REVIEW
    assert "processed text used (logical order and numbers repaired), no model output" in " ".join(result["reasons"])
    ltr = _page("A", raw="raw", processed="processed", status="not_needed", share=0.1, decision="mainly_ltr")
    assert hybrid.combine_page(ltr, None, None)["library_source"] == "pymupdf_raw"
    empty = _page("C", raw="raw", processed="", status="uncertain", share=0.9)
    assert hybrid.combine_page(empty, None, None)["final_text"] == "raw"


def test_rejoin_is_never_built_from_raw_text() -> None:
    page = _page("B", raw="ا خ ت بار", processed="ا خ ت بار", status="not_needed", share=0.1, decision="mainly_ltr")
    result = hybrid.combine_page(page, _vlm("اختبار"), None)
    assert result["rejoined"] is None and result["library_source"] == "pymupdf_raw"


# ------------------------------------------------- decision logic v2 / recompute


def test_rejoined_improved_but_above_threshold_is_needs_review() -> None:
    rule = {"single_letter_threshold": 0.10, "single_letter_min_arabic_tokens": 1}
    processed = "ا خ ت بار ك ت ا ب ق ل م"  # 10/11 single letters -> 3/5 after rejoining
    page = _page("B", raw="x", processed=processed, status="uncertain", share=0.9)
    result = hybrid.combine_page(page, _vlm("اختبار كتاب"), rule)
    candidate = result["rejoined"]
    assert result["final_source"] == hybrid.REJOINED_SOURCE and result["status"] == hybrid.NEEDS_REVIEW
    assert candidate["single_letter_ratio_after"] < candidate["single_letter_ratio_before"]
    assert candidate["relative_improvement"] >= hybrid.REJOIN_MIN_RELATIVE_REDUCTION


def test_failed_rejoin_with_genuine_number_conflicts_keeps_processed_text() -> None:
    page = _page("B", raw="x", processed="ا خ ت بار ٢٠٢٥", status="uncertain", share=0.9)
    result = hybrid.combine_page(page, _vlm("بسم الله ٢٠٢٠"), None)
    assert result["final_text"] == "ا خ ت بار ٢٠٢٥" and result["status"] == hybrid.NEEDS_REVIEW
    assert "VLM numbers conflict with the text layer" in result["reasons"][-1]
    assert result["number_audit"]["conflict_kind"] == "genuine"


def test_number_conflicts_on_image_pages_do_not_reject_the_vlm() -> None:
    page = _page("B", raw="x", processed="ا خ ت بار ٢٠٢٥", status="uncertain", share=0.9,
                 images=hybrid_analysis.IMAGE_WARNING)
    result = hybrid.combine_page(page, _vlm("بسم الله ٢٠٢٠"), None)
    assert result["final_source"].startswith("vlm_raw")
    assert result["number_audit"]["conflict_kind"] == "may come from images"


def test_version_1_logic_is_kept_for_existing_runs() -> None:
    page = _page("B", raw="x", processed="ا خ ت بار ٢٠٢٥", status="uncertain", share=0.9)
    old = hybrid.combine_page(page, _vlm("بسم الله ٢٠٢٠"), None, "1")
    assert old["final_source"].startswith("vlm_raw") and old["decision_logic_version"] == "1"


def test_recompute_previews_then_appends_a_revision_and_refuses_finalized_runs(doc) -> None:
    document, source, root = doc
    analysis = hybrid_analysis.analyze_pdf(source)
    run_dir = hybrid.create_run(root=root, document=document, pdf_path=source, analysis=analysis,
                                model=hybrid.LIBRARY_ONLY, vlm_pages={})
    before = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    preview = hybrid.preview_recompute(run_dir, None, root, logic_version="1")
    assert len(preview["pages"]) == 2 and preview["proposed_logic_version"] == "1"
    with pytest.raises(hybrid.HybridError):
        hybrid.apply_recompute(run_dir, None, root, by="Tester", statement="ok", logic_version="1")
    hybrid.apply_recompute(run_dir, None, root, by="Tester", statement=hybrid.RECOMPUTE_STATEMENT,
                           logic_version="1")
    assert hybrid.effective_logic_version(run_dir) == "1"
    assert all(path.read_bytes() == data for path, data in before.items())  # nothing overwritten
    hybrid.finalize(run_dir, None, root)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["decision_logic_version"] == "1" and results["decision_revision"]
    with pytest.raises(hybrid.HybridError):
        hybrid.preview_recompute(run_dir, None, root)


# ------------------------------------------------------------ layout (phase 1)

from lab.core.hybrid import layout as hybrid_layout  # noqa: E402


def _block(block_id, bbox, text, size=10.0, lines=1):
    return {"id": block_id, "bbox": bbox, "raw_text": text, "processed_text": text, "max_font_size": size,
            "line_count": lines}


def _elements(blocks, figures=(), page=1):
    return {"page": page, "width": 600, "height": 800, "body_font_size": 10.0, "blocks": list(blocks),
            "figures": list(figures), "skipped_images": [], "settings": {}}


def test_xy_cut_reads_columns_right_to_left_on_arabic_pages() -> None:
    items = [{"id": "left", "bbox": [50, 100, 250, 300]}, {"id": "right", "bbox": [320, 100, 550, 300]},
             {"id": "title", "bbox": [50, 40, 550, 70]}]
    rtl = hybrid_layout.xy_cut(items, rtl=True, settings=hybrid_layout.DEFAULTS, ambiguous=[])
    ltr = hybrid_layout.xy_cut(items, rtl=False, settings=hybrid_layout.DEFAULTS, ambiguous=[])
    assert [i["id"] for i in rtl] == ["title", "right", "left"]
    assert [i["id"] for i in ltr] == ["title", "left", "right"]


def test_overlapping_elements_are_marked_ambiguous_not_ordered_silently() -> None:
    ambiguous = []
    items = [{"id": "a", "bbox": [50, 100, 300, 200]}, {"id": "b", "bbox": [60, 110, 310, 210]}]
    hybrid_layout.xy_cut(items, rtl=False, settings=hybrid_layout.DEFAULTS, ambiguous=ambiguous)
    assert ambiguous and set(ambiguous[0]["elements"]) == {"a", "b"}


def test_assembly_reorders_whole_elements_dedups_tables_and_places_figures() -> None:
    table = {"table_index": 0, "bbox": [50, 300, 550, 400],
             "cells": [{"row": 0, "col": 0, "processed_cell_text": "Year", "reorder_status": "not_needed"},
                       {"row": 0, "col": 1, "processed_cell_text": "Value", "reorder_status": "not_needed"},
                       {"row": 1, "col": 0, "processed_cell_text": "2025", "reorder_status": "not_needed"},
                       {"row": 1, "col": 1, "processed_cell_text": "12", "reorder_status": "not_needed"}]}
    blocks = [_block("b0", [50, 40, 550, 70], "Report Title", size=18), _block("b1", [50, 100, 550, 200], "Body text."),
              _block("b2", [60, 310, 540, 390], "Year Value 2025 12"),
              _block("b3", [50, 560, 550, 580], "Figure 1: Sales")]
    figure = {"id": "img0", "origin": "raster", "bbox": [50, 420, 550, 550], "area_share": 0.13}
    page_text = "Body text.\nReport Title\nYear Value 2025 12\nFigure 1: Sales"  # different order, same characters
    decision = {"final_source": "pymupdf_raw", "final_text": page_text}
    library_page = {"reorder": {"reorder_status": "not_applicable"}, "arabic_letter_share": 0.0}
    out = hybrid_layout.assemble_page(_elements(blocks, [figure]), decision=decision, library_page=library_page,
                                      tables_doc={"tables": [table]}, vlm_text="", reorder_config={})
    md, record = out["markdown"], out["record"]
    assert record["validation"]["text_elements_match_page_text"] and record["fallback_reason"] is None
    assert md.startswith("## Report Title") and md.count("2025") == 1  # table text appears once
    assert record["table_dedup"] == [{"block": "b2", "table_index": 0}]
    assert md.index("| Year | Value |") < md.index("🖼️ **Image (not read)**") < md.index("Figure 1: Sales")
    assert record["captions"][0]["attached_to"] == "img0"


def test_assembly_falls_back_to_page_text_when_characters_differ() -> None:
    blocks = [_block("b0", [50, 40, 550, 70], "abc")]
    out = hybrid_layout.assemble_page(_elements(blocks), decision={"final_source": "pymupdf_raw", "final_text": "abd"},
                                      library_page={"reorder": {}, "arabic_letter_share": 0.0}, tables_doc=None,
                                      vlm_text="", reorder_config={})
    assert out["record"]["fallback_reason"] and out["markdown"].strip() == "abd"


def test_vlm_final_text_is_one_region_with_figure_placeholders() -> None:
    figure = {"id": "vec0", "origin": "vector", "bbox": [50, 420, 550, 550], "area_share": 0.1,
              "confidence": "suspected"}
    out = hybrid_layout.assemble_page(_elements([], [figure]),
                                      decision={"final_source": "vlm_raw (Success)", "final_text": "نص"},
                                      library_page={"reorder": {}, "arabic_letter_share": 1.0}, tables_doc=None,
                                      vlm_text="نص", reorder_config={})
    assert out["markdown"].startswith("نص") and "📊 **Figure (not read)**" in out["markdown"]
    assert "suspected chart region" in out["markdown"]


def test_finalize_saves_layout_revision_and_page_documents(doc) -> None:
    document, source, root = doc
    analysis = hybrid_analysis.analyze_pdf(source)
    run_dir = hybrid.create_run(root=root, document=document, pdf_path=source, analysis=analysis,
                                model=hybrid.LIBRARY_ONLY, vlm_pages={})
    assert (run_dir / "source.pdf").read_bytes() == Path(source).read_bytes()
    hybrid.finalize(run_dir, None, root)
    assert (run_dir / "page_001.md").exists() and (run_dir / "full_document.md").exists()
    assert "--- Page 2 ---" in (run_dir / "full_document.md").read_text(encoding="utf-8")
    assert len(hybrid.layout_revisions(run_dir)) == 1
    finalized = {path: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    hybrid.apply_layout(run_dir, None, root, by="Tester", statement=hybrid.LAYOUT_STATEMENT)
    assert len(hybrid.layout_revisions(run_dir)) == 2 and hybrid.latest_layout(run_dir)["revision"] == 2
    assert {path: path.read_bytes() for path in finalized} == finalized  # finalized files untouched
