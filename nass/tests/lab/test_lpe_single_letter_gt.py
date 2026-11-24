"""Single-letter Arabic token ratio, 'Needs OCR?' rule, ground truth with a new run,
final-status timing, and the duplicate "No table detected" message.

Synthetic data in pytest's tmp_path only; no real PDF and no existing run is used.
"""

from __future__ import annotations

import json

import pymupdf
import pytest

from lab.local_pdf_extraction import diagnostics, exports, runner, storage, tables
from tests.lab import lpe_pdf_factory as factory

SPLIT = "ا خ ت بار ا ل ب ي ا ن ا ت ن ت ي ج ة ا ل ت ح ل ي ل"  # Chrome/Skia-like split words
CORRECT = "اختبار البيانات نتيجة التحليل في التقرير السنوي للسلامة المرورية"
RULE = {"single_letter_threshold": 0.10, "single_letter_min_arabic_tokens": 10}


# ------------------------------------------------------------- the indicator
def test_single_letter_ratio_on_split_and_correct_text() -> None:
    split = diagnostics.single_letter_stats(SPLIT)
    assert split["arabic_tokens"] == len(SPLIT.split())
    assert split["single_letter_arabic_ratio"] > 0.5
    assert diagnostics.single_letter_stats(CORRECT)["single_letter_arabic_ratio"] == 0.0
    assert diagnostics.single_letter_stats("بِ و، (ه) abc 5")["single_letter_arabic_tokens"] == 3  # marks/punct ignored
    assert diagnostics.single_letter_stats("English only")["single_letter_arabic_ratio"] == 0.0


def test_needs_ocr_rule_applies_only_when_given() -> None:
    flagged = diagnostics.page_indicators(SPLIT, RULE)
    assert flagged["needs_ocr"] and diagnostics.SPLIT_WORDS_REASON in flagged["needs_ocr_reason"]
    assert flagged["needs_ocr_rule"] == RULE
    assert not diagnostics.page_indicators(SPLIT)["needs_ocr"]  # raw methods: indicator only
    assert not diagnostics.page_indicators(CORRECT, RULE)["needs_ocr"]
    few = "ا خ ت"  # fewer Arabic tokens than the minimum
    assert not diagnostics.page_indicators(few, RULE)["needs_ocr"]
    assert diagnostics.page_indicators(SPLIT, {**RULE, "single_letter_threshold": 0.99})["needs_ocr"] is False


def test_needs_ocr_settings_from_manifest() -> None:
    assert diagnostics.needs_ocr_settings({}) is None  # runs created before the rule
    assert diagnostics.needs_ocr_settings({"diagnostics": {"single_letter_threshold": 0.2}}) == {
        "single_letter_threshold": 0.2,
        "single_letter_min_arabic_tokens": diagnostics.DEFAULT_SINGLE_LETTER_MIN_TOKENS}


def test_aggregate_ratio_is_token_weighted() -> None:
    records = [{"status": "COMPLETED", "diagnostics": diagnostics.page_indicators(SPLIT)},
               {"status": "COMPLETED", "diagnostics": diagnostics.page_indicators(CORRECT)}]
    totals = diagnostics.aggregate(records)
    expected = totals["single_letter_arabic_tokens"] / totals["arabic_tokens"]
    assert totals["single_letter_arabic_ratio"] == round(expected, 4)


# ---------------------------------------------------------- runs (tmp only)
def _split_words_pdf(path) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    css = f"@font-face {{font-family: F; src: url({factory.ARIAL.name});}} * {{font-family: F; font-size: 14px;}}"
    page.insert_htmlbox(pymupdf.Rect(50, 50, 545, 800), f"<p dir='rtl'>{SPLIT}</p>", css=css,
                        archive=pymupdf.Archive(str(factory.FONT_DIR)))
    doc.save(path)
    doc.close()
    return path.read_bytes()


@pytest.mark.skipif(not factory.ARIAL.exists(), reason="needs C:/Windows/Fonts/arial.ttf")
def test_rule_flags_processed_method_only_and_threshold_is_in_manifest(tmp_path) -> None:
    data = _split_words_pdf(tmp_path / "split.pdf")
    settings = {"reorder": {}, "diagnostics": {"single_letter_threshold": 0.10, "single_letter_min_arabic_tokens": 10}}
    run_path = storage.create_run(tmp_path / "root", [("split.pdf", data)],
                                  runner.method_entries(["pymupdf", "pymupdf_ar_reordered"], {}), settings)
    runner.run(run_path)
    assert storage.load_manifest(run_path)["settings"]["diagnostics"]["single_letter_threshold"] == 0.10
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    processed, _ = storage.read_page(run_path, key, "pymupdf_ar_reordered", 1)
    raw, _ = storage.read_page(run_path, key, "pymupdf", 1)
    assert processed["diagnostics"]["needs_ocr"]
    assert diagnostics.SPLIT_WORDS_REASON in processed["diagnostics"]["needs_ocr_reason"]
    assert raw["diagnostics"]["single_letter_arabic_ratio"] > 0.10
    assert not raw["diagnostics"]["needs_ocr"]
    collected = exports.collect(run_path)
    csv_text = exports.to_csv(collected).decode("utf-8-sig")
    assert "single_letter_arabic_ratio" in csv_text.splitlines()[0]
    report = exports.to_html(collected, lambda k, p: None, embed_images=False).decode("utf-8")
    assert "Single-letter Arabic token ratio per document" in report and diagnostics.SPLIT_WORDS_REASON in report
    assert exports.comparison(collected)["per_document"][key]["pymupdf"]["single_letter_arabic_ratio"] > 0.10


def test_old_records_get_the_ratio_on_load_without_rewriting(tmp_path) -> None:
    data = factory.small_pdf(tmp_path / "s.pdf", pages=1).read_bytes()
    run_path = storage.create_run(tmp_path / "root", [("s.pdf", data)], runner.method_entries(["pymupdf"], {}),
                                  {"reorder": {}})
    runner.run(run_path)
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    _, json_path = storage.page_paths(run_path, key, "pymupdf", 1)
    record = json.loads(json_path.read_text(encoding="utf-8"))
    for name in ("single_letter_arabic_ratio", "single_letter_arabic_tokens"):
        record["diagnostics"].pop(name)
    json_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")  # simulate an older run
    before = json_path.read_bytes()
    output = exports.collect(run_path)["documents"][0]["pages"][0]["outputs"]["pymupdf"]
    assert output["record"]["diagnostics"]["single_letter_computed_on_load"] is True
    assert "single_letter_arabic_ratio" in output["record"]["diagnostics"]
    assert json_path.read_bytes() == before


# ----------------------------------------------- final status (was RUNNING)
def test_state_and_summary_are_final_after_the_run(tmp_path) -> None:
    data = factory.small_pdf(tmp_path / "s.pdf", pages=2).read_bytes()
    run_path = storage.create_run(tmp_path / "root", [("s.pdf", data)], runner.method_entries(["pymupdf"], {}),
                                  {"reorder": {}})
    runner.run(run_path)
    assert storage.load_state(run_path)["status"] == "COMPLETED"
    assert json.loads((run_path / "summary.json").read_text(encoding="utf-8"))["status"] == "COMPLETED"
    assert storage.load_manifest(run_path)["status"] == "COMPLETED"


def test_derive_status_reports_running_while_another_worker_holds_the_lock(tmp_path) -> None:
    data = factory.small_pdf(tmp_path / "s.pdf", pages=1).read_bytes()
    run_path = storage.create_run(tmp_path / "root", [("s.pdf", data)], runner.method_entries(["pymupdf"], {}),
                                  {"reorder": {}})
    token = storage.acquire_lock(run_path)
    try:
        assert storage.derive_status(run_path)["status"] == "RUNNING"
        assert storage.derive_status(run_path, worker_finished=True)["status"] != "RUNNING"
    finally:
        storage.release_lock(run_path, token)


# ------------------------------------------------------------ ground truth
def test_ground_truth_uploaded_with_a_new_run_is_saved_and_scored(tmp_path) -> None:
    data = factory.small_pdf(tmp_path / "s.pdf", pages=2).read_bytes()
    gt = diagnostics.combine_ground_truth_files([("page_1.txt", "Page 1: Road safety 45.5%".encode()),
                                                 ("page_2.txt", "Page 2: Road safety 45.5%".encode())])
    run_path = storage.create_run(tmp_path / "root", [("s.pdf", data), ("t.pdf", data)],
                                  runner.method_entries(["pymupdf"], {}), {"reorder": {}},
                                  ground_truth=[(gt, ["page_1.txt", "page_2.txt"]), None])
    manifest = storage.load_manifest(run_path)
    first, second = manifest["documents"]
    assert first["ground_truth_attached"] and not second["ground_truth_attached"]
    assert storage.load_ground_truth(run_path, first["doc_key"]) == gt
    runner.run(run_path)
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["ground_truth"][first["doc_key"]]["pymupdf"]["level"] == "page"
    assert summary["ground_truth_status"][first["doc_key"]]["computed"] is True
    assert summary["ground_truth_status"][second["doc_key"]] == {
        "attached": False, "level": None, "computed": False, "scored_methods": [],
        "message": "No ground truth attached: CER/WER not computed."}


def test_ground_truth_attached_later_updates_summary(tmp_path) -> None:
    data = factory.small_pdf(tmp_path / "s.pdf", pages=1).read_bytes()
    run_path = storage.create_run(tmp_path / "root", [("s.pdf", data)], runner.method_entries(["pymupdf"], {}),
                                  {"reorder": {}})
    runner.run(run_path)
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    assert json.loads((run_path / "summary.json").read_text(encoding="utf-8"))["ground_truth"] == {}
    storage.save_ground_truth(run_path, key, "Road safety", ["gt.txt"])
    exports.write_summary(run_path)  # what the Results tab now does after attaching
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["ground_truth"][key]["pymupdf"]["level"] == "document"
    assert summary["ground_truth_status"][key]["computed"] is True


def test_ground_truth_list_must_match_uploads(tmp_path) -> None:
    data = factory.small_pdf(tmp_path / "s.pdf", pages=1).read_bytes()
    with pytest.raises(storage.StorageError):
        storage.create_run(tmp_path / "root", [("s.pdf", data)], runner.method_entries(["pymupdf"], {}), {},
                           ground_truth=[None, None])


def test_combine_and_describe_ground_truth() -> None:
    assert diagnostics.combine_ground_truth_files([("doc.txt", "\ufeffنص".encode())]) == "نص"
    combined = diagnostics.combine_ground_truth_files([("p2.txt", b"b"), ("p1.txt", b"a")])
    assert combined == "=== page 1 ===\na\n=== page 2 ===\nb"
    with pytest.raises(ValueError):
        diagnostics.combine_ground_truth_files([("x.txt", b"a"), ("y.txt", b"b")])  # no page numbers
    with pytest.raises(ValueError):
        diagnostics.combine_ground_truth_files([("p1.txt", b"a"), ("page1.txt", b"b")])  # duplicate page
    with pytest.raises(ValueError):
        diagnostics.combine_ground_truth_files([("doc.txt", b"\xff\xfe\x00")])  # not UTF-8
    described = diagnostics.describe_ground_truth("=== page 1 ===\na\n=== page 9 ===\nb", page_count=3)
    assert described["level"] == "page" and described["outside_document"] == [9]
    assert diagnostics.describe_ground_truth("whole document", 3)["level"] == "document"


# ------------------------------------------------ "No table detected" once
def test_no_table_message_appears_once_on_pages_with_images(tmp_path) -> None:
    path = tmp_path / "img.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    pixmap.clear_with(200)
    page.insert_image(pymupdf.Rect(50, 50, 250, 250), pixmap=pixmap)
    doc.save(path)
    doc.close()
    with pymupdf.open(path) as document:
        result = tables.page_tables(document, 1)
    assert result["page_has_images"]
    assert tables.to_markdown(result).count("No table detected") == 1
    assert exports.tables_section_html(result).count("No table detected") == 1
    old_note = "No table detected. The page contains images: find_tables() cannot read tables inside images."
    assert tables.clean_note(old_note).startswith("The page contains images")
    assert exports.tables_section_html({**result, "notes": [old_note]}).count("No table detected") == 1
