"""Phase 2 (image regions) on small synthetic PDFs built in the tests. No network: hosted
calls use an in-process fake executor; offline paths assert that no provider is called."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import fitz
import pytest

import lab.core.e2.api.runner as api_runner
from lab.core.e2.api import documents
from lab.core.e2.api.authorization import REQUIRED_STATEMENT, record_enablement
from lab.core.e2.api.providers import load_providers
from lab.core.e2.api.runner import APIEvaluationRunner
from lab.core.e2.store import ResultStore
from lab.core.hybrid import analysis as hybrid_analysis
from lab.core.hybrid import enrich, qari
from lab.core.hybrid import run as hybrid
from lab.core.ingest import IngestService
from lab.core.workspace import WorkspaceManager

KEY = "SECRET-phase2-key"


def _pixmap(width, height, value=120):
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    pix.clear_with(value)
    return pix


@pytest.fixture
def ctx(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setattr(documents, "DOCUMENT_REGISTRY", tmp_path / "no_registry.toml")
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(tmp_path / "results"))
    pdf = tmp_path / "Synthetic_Charts.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 60), "Quarterly report with two charts and a KPI card", fontsize=12)
    page.insert_image(fitz.Rect(60, 100, 290, 300), pixmap=_pixmap(460, 400, 90), keep_proportion=False)   # chart 1
    page.insert_image(fitz.Rect(310, 100, 540, 300), pixmap=_pixmap(460, 400, 150), keep_proportion=False)  # chart 2
    page.insert_image(fitz.Rect(100, 340, 500, 431), pixmap=_pixmap(976, 208, 200), keep_proportion=False)  # KPI card
    page.insert_text((120, 380), "Sales 2025", fontsize=11)  # text-layer text over the KPI image
    doc.save(pdf)
    doc.close()
    manager = WorkspaceManager(settings)
    workspace_id = manager.create().workspace_id
    ingest = IngestService(manager, settings)
    with pdf.open("rb") as handle:
        document = ingest.ingest(workspace_id, handle, pdf.name)
    source = ingest.original_path(workspace_id, document.doc_id)
    root = tmp_path / "results"
    analysis = hybrid_analysis.analyze_pdf(source)
    run_dir = hybrid.create_run(root=root, document=document, pdf_path=source, analysis=analysis,
                                model=hybrid.LIBRARY_ONLY, vlm_pages={})
    hybrid.apply_layout(run_dir, None, root, by="Tester", statement=hybrid.LAYOUT_STATEMENT)
    return {"manager": manager, "settings": settings, "workspace_id": workspace_id, "document": document,
            "source": source, "root": root, "run_dir": run_dir, "monkeypatch": monkeypatch, "tmp": tmp_path}


def _set(ctx):
    regions = [item["region_id"] for item in enrich.candidate_regions(ctx["run_dir"]) if item["parent"] is None]
    return enrich.create_region_set(ctx["run_dir"], regions, pdf_path=ctx["source"], by="Tester")


def _results_zip(export: bytes, texts: dict[str, str], *, extra: dict[str, bytes] | None = None) -> bytes:
    manifest = zipfile.ZipFile(io.BytesIO(export)).read("manifest.json")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", manifest)
        for rid, text in texts.items():
            archive.writestr(f"{rid}.txt", text.encode("utf-8"))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _import(ctx, set_dir, texts, model="Qari-OCR 0.4.0"):
    export = enrich.export_crops_zip(set_dir, provider="Qari (local script)", model=model, by="Tester")
    return enrich.import_results(set_dir, _results_zip(export, texts), declared_model=model, reviewer="Tester",
                                 statement=enrich.IMPORT_STATEMENT)


# ------------------------------------------------------------------ regions


def test_multiple_charts_and_kpi_become_distinct_regions_with_hashed_crops(ctx) -> None:
    set_dir = _set(ctx)
    data = enrich.load_set(set_dir)
    ids = [r["region_id"] for r in data["regions"]]
    assert len(ids) == 3 and len(set(ids)) == 3 and all(r["page"] == 1 for r in data["regions"])
    for region in data["regions"]:
        assert hashlib.sha256((set_dir / region["crop_file"]).read_bytes()).hexdigest() == region["crop_sha256"]
        assert region["crop_dpi"] == enrich.CROP_DPI
    assert any("wide image" in r["detected_type"] for r in data["regions"])
    assert _set(ctx) != set_dir  # a new, independent set


# ------------------------------------------------------- manual import (Qari)


def test_manual_qari_import_round_trip_keeps_text_unchanged(ctx) -> None:
    set_dir = _set(ctx)
    ids = [r["region_id"] for r in enrich.load_set(set_dir)["regions"]]
    texts = {ids[0]: "توزيع الحوادث\nالرياض 42%\nجدة 31%", ids[1]: "Screenshot: Login تسجيل الدخول",
             ids[2]: "3.01%\n05 PM"}
    folder = _import(ctx, set_dir, texts)
    record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    assert record["provenance"] == "manual_import" and "NOT independently verified" in record["verification"]
    outputs = enrich.collect_outputs(set_dir, lambda _id: None, ctx["root"])
    for rid, text in texts.items():
        (output,) = outputs[rid]
        assert output["text"] == text and output["provenance"] == "manual_import" and output["usable"]
    kpi = outputs[ids[2]][0]["assessment"]["numbers"]
    assert [n["original"] for n in kpi] == ["3.01%", "05"]


def test_import_rejects_unknown_duplicate_and_modified_manifest(ctx) -> None:
    set_dir = _set(ctx)
    rid = enrich.load_set(set_dir)["regions"][0]["region_id"]
    export = enrich.export_crops_zip(set_dir, provider="Qari", model="Qari", by="Tester")
    unknown = enrich.check_import(set_dir, _results_zip(export, {rid: "x"}, extra={"p9_img9.txt": b"y"}))
    assert not unknown["ok"] and unknown["unknown"]
    duplicate = enrich.check_import(set_dir, _results_zip(export, {rid: "x"}, extra={f"sub/{rid}.txt": b"y"}))
    assert not duplicate["ok"] and duplicate["duplicates"] == [rid]
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(export)).read("manifest.json"))
    manifest["regions"][0]["crop_sha256"] = "0" * 64
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr(f"{rid}.txt", "x")
    assert not enrich.check_import(set_dir, buffer.getvalue())["ok"]
    partial = enrich.check_import(set_dir, _results_zip(export, {rid: "x"}))
    assert partial["ok"] and len(partial["missing"]) == 2  # missing regions are reported, not invented


# ------------------------------------------------ comparison and documents


def test_two_providers_for_the_same_region_and_unified_markdown(ctx, monkeypatch) -> None:
    set_dir = _set(ctx)
    ids = [r["region_id"] for r in enrich.load_set(set_dir)["regions"]]
    first = _import(ctx, set_dir, {ids[0]: "<table><tr><td>المدينة</td><td>النسبة</td></tr>"
                                            "<tr><td>الرياض</td><td>42%</td></tr></table>"}, model="Paddle (import)")
    second = _import(ctx, set_dir, {ids[0]: "المدينة النسبة الرياض 42%"}, model="Qari (import)")
    outputs = enrich.collect_outputs(set_dir, lambda _id: None, ctx["root"])
    assert {o["output_key"] for o in outputs[ids[0]]} == {first.name, second.name}
    enrich.save_selection(set_dir, {ids[0]: {"output_key": first.name, "reason": "table structure kept"}},
                          by="Tester", statement=enrich.SELECTION_STATEMENT)
    monkeypatch.setattr(api_runner.APIEvaluationRunner, "run", lambda *a, **k: pytest.fail("API called"))
    monkeypatch.setattr(qari, "transcribe", lambda *a, **k: pytest.fail("Qari called"))
    computed = enrich.compute_enriched(ctx["run_dir"], set_dir, lambda _id: None, ctx["root"], run_spec=None,
                                       pdf_path=ctx["source"])
    markdown = computed["pages"][0]["markdown"]
    assert "| المدينة | النسبة |" in markdown and "read by Paddle (import) (manual import, declared)" in markdown
    assert markdown.count("Image (not read)") == 2  # the other two regions keep their placeholders
    extraction = computed["pages"][0]["record"]["image_extraction"][0]
    assert extraction["placed"] and extraction["sources"][0]["provenance"] == "manual_import"


def test_bad_outputs_are_flagged_and_never_corrected() -> None:
    assert enrich.assess("٠" * 20)["needs_review"]
    assert "max_tokens" in " ".join(enrich.assess("partial", finish_reason="length")["reasons"])
    assert enrich.assess(" ".join(["نفس العبارة المكررة"] * 6))["needs_review"]
    assert enrich.assess("Value: [unreadable]")["needs_review"]
    assert "labels" in " ".join(enrich.assess("12 15 18", region_type="vector chart")["reasons"])
    text = "Revenue 1,284 SAR"
    assert enrich.assess(text)["numbers"][0]["original"] == "1,284"  # unchanged, only listed


def test_image_based_table_html_to_markdown_only_when_reliable() -> None:
    simple, notes = enrich.html_tables_to_markdown("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>")
    assert "| A | B |\n| --- | --- |\n| 1 | 2 |" in simple and notes[0]["converted"]
    merged = "<table><tr><td colspan='2'>A</td></tr><tr><td>1</td><td>2</td></tr></table>"
    kept, notes = enrich.html_tables_to_markdown(merged)
    assert kept == merged and not notes[0]["converted"]


def test_extracted_text_overlapping_the_text_layer_is_flagged_not_deleted(ctx) -> None:
    set_dir = _set(ctx)
    kpi = next(r["region_id"] for r in enrich.load_set(set_dir)["regions"] if "wide" in r["detected_type"])
    folder = _import(ctx, set_dir, {kpi: "Sales 2025"})
    enrich.save_selection(set_dir, {kpi: {"output_key": folder.name, "reason": "only output"}}, by="Tester",
                          statement=enrich.SELECTION_STATEMENT)
    computed = enrich.compute_enriched(ctx["run_dir"], set_dir, lambda _id: None, ctx["root"], run_spec=None,
                                       pdf_path=ctx["source"])
    page = computed["pages"][0]
    assert any("largely repeats the PDF text layer" in note for note in page["record"]["needs_review"])
    assert "Sales 2025" in page["markdown"]


# ------------------------------------------------------------ references


def test_image_reference_matching_requires_confirmation_and_metrics_use_it_only(ctx) -> None:
    set_dir = _set(ctx)
    ids = [r["region_id"] for r in enrich.load_set(set_dir)["regions"]]
    reference = {"pages": [{"page_number": 1, "visual_elements": {"charts": [
        {"id": "c1", "title": "Accidents by city", "x_axis": {"label": "City", "categories": ["Riyadh", "Jeddah"]},
         "series": [{"name": "Share", "data": [{"category": "Riyadh", "label_shown": "42%"},
                                               {"category": "Jeddah", "label_shown": "31%"}]}]},
        {"id": "c2", "title": "Second chart", "series": []}], "images": []}, "tables": []}]}
    ref_dir = enrich.import_reference(set_dir, json.dumps(reference).encode("utf-8"), filename="gt.json", by="Tester")
    proposals = enrich.propose_matches(set_dir, ref_dir)
    assert all(item["proposed"] is None for item in proposals)  # 3 regions, 2 charts: ambiguous, never auto-assigned
    folder = _import(ctx, set_dir, {ids[0]: "Accidents by city\nRiyadh 42%\nJeddah 31%\nDammam 9%"})
    outputs = enrich.collect_outputs(set_dir, lambda _id: None, ctx["root"])
    assert enrich.evaluate(set_dir, outputs, ref_dir)["regions"] == {}  # nothing confirmed yet
    enrich.save_matches(set_dir, ref_dir, {ids[0]: "p1:charts:c1"}, by="Tester", statement=enrich.MATCH_STATEMENT)
    result = enrich.evaluate(set_dir, outputs, ref_dir)
    metrics = result["regions"][ids[0]]["outputs"][folder.name]
    assert metrics["numbers"]["matched"] == 2 and metrics["numbers"]["not_in_reference"] == {"9%": 1}
    assert metrics["labels"]["found"] >= 3
    assert json.loads((ref_dir / "original.json").read_bytes()) == reference  # stored unchanged


# ------------------------------------------------------------- hosted API


def test_hosted_provider_receives_the_crop_not_the_page(ctx) -> None:
    root = ctx["root"]
    spec = next(item for item in load_providers() if item.id == "oxen-qwen3-vl-4b-instruct")
    documents.register_document(ctx["source"], display_name="Synthetic_Charts.pdf", registered_by="Tester",
                                statement=documents.REGISTRATION_STATEMENT, root=root)
    entry = documents.load_registry(root=root)[ctx["document"].sha256]
    documents.authorize_document(entry, file_path=ctx["source"], reviewer="Tester",
                                 declaration=documents.USER_OWNED_DECLARATION, days_valid=1, root=root)
    record_enablement(spec, enabled=True, enabled_by="Tester", statement=REQUIRED_STATEMENT, max_pages_total=10,
                      max_pages_per_day=10, days_valid=1, root=root)
    set_dir = _set(ctx)
    region = enrich.load_set(set_dir)["regions"][0]
    sent = []

    def factory():
        def execute(spec_, credential, image_path, filename, job_id, result_hosts, on_event):
            sent.append(hashlib.sha256(Path(image_path).read_bytes()).hexdigest())
            return {"phase": "done", "ok": True, "job_id": "chatcmpl-1",
                    "result": {"text": "Chart: 42%", "blocks": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                               "finish_reason": "stop", "truncated": False, "exchanges": []}}
        return execute

    ctx["monkeypatch"].setattr(api_runner, "subprocess_executor", factory)
    with pytest.raises(enrich.EnrichError):
        enrich.create_api_run(set_dir, spec, [region["region_id"]], by="Tester", statement="ok", root=root)
    run_dir = enrich.create_api_run(set_dir, spec, [region["region_id"]], by="Tester",
                                    statement=enrich.APPROVAL_STATEMENT, root=root)
    runner = APIEvaluationRunner(ctx["manager"], ctx["settings"], ResultStore(root))
    enrich.execute_api_run(run_dir, runner, spec, workspace_id=ctx["workspace_id"], document=ctx["document"],
                           pdf_path=ctx["source"], credential=KEY, root=root)
    assert sent == [region["crop_sha256"]]
    enrich.execute_api_run(run_dir, runner, spec, workspace_id=ctx["workspace_id"], document=ctx["document"],
                           pdf_path=ctx["source"], credential=KEY, root=root)
    assert len(sent) == 1  # never resent
    (output,) = enrich.collect_outputs(set_dir, {spec.id: spec}.get, root)[region["region_id"]]
    assert output["text"] == "Chart: 42%" and output["provenance"] == "hosted_api"
    assert KEY.encode() not in b"".join(p.read_bytes() for p in root.rglob("*") if p.is_file())


# ------------------------------------------------------------ immutability


def test_finalized_run_stays_unchanged_and_documents_are_a_new_revision(ctx) -> None:
    hybrid.finalize(ctx["run_dir"], None, ctx["root"])
    before = {p: p.read_bytes() for p in ctx["run_dir"].rglob("*") if p.is_file()}
    set_dir = _set(ctx)
    rid = enrich.load_set(set_dir)["regions"][0]["region_id"]
    folder = _import(ctx, set_dir, {rid: "نص من الصورة"})
    enrich.save_selection(set_dir, {rid: {"output_key": folder.name, "reason": "test"}}, by="Tester",
                          statement=enrich.SELECTION_STATEMENT)
    revision = enrich.apply_enriched(ctx["run_dir"], set_dir, lambda _id: None, ctx["root"], run_spec=None, by="Tester",
                                     statement=enrich.DOCUMENT_STATEMENT, pdf_path=ctx["source"])
    assert all(p.read_bytes() == data for p, data in before.items())  # nothing existing changed
    assert (revision / "full_document.md").exists()
    assert enrich.latest_enriched(ctx["run_dir"])["revision"] == "enrichment 1"
    names = zipfile.ZipFile(io.BytesIO(hybrid.zip_bytes(ctx["run_dir"], None, ctx["root"]))).namelist()
    assert any(name.endswith("LATEST_DOCUMENTS.txt") for name in names)


# --------------------------------------------- saved outputs export / image refs


PADDLE_IMAGE_ONLY = ('<div style="text-align: center;"><img src="imgs/img_in_chart_box_2_4_1511_380.jpg" '
                     'alt="Image" width="99%" /></div>\n')


def test_html_image_reference_is_not_extracted_content_or_numbers() -> None:
    kind = enrich.content_kind(PADDLE_IMAGE_ONLY)
    assert kind["kind"] == "html_image_reference_only"
    assert kind["image_references"] == ["imgs/img_in_chart_box_2_4_1511_380.jpg"]
    assessment = enrich.assess(PADDLE_IMAGE_ONLY, region_type="raster image")
    assert assessment["numbers"] == [] and assessment["visible_text"] == ""
    assert "only an HTML image reference" in assessment["reasons"][0]
    table = enrich.assess("<table><tr><td>الرياض</td><td>42%</td></tr></table>")
    assert table["content"]["kind"] == "text" and [n["original"] for n in table["numbers"]] == ["42%"]


def test_saved_outputs_export_is_read_only_and_complete(ctx) -> None:
    set_dir = _set(ctx)
    rid = enrich.load_set(set_dir)["regions"][0]["region_id"]
    folder = _import(ctx, set_dir, {rid: PADDLE_IMAGE_ONLY})
    before = {p: p.read_bytes() for p in ctx["run_dir"].rglob("*") if p.is_file()}
    entries = enrich.outputs_export(set_dir, lambda _id: None, ctx["root"])
    prefix = f"phase2_outputs/{set_dir.name}/{folder.name}/{rid}/"
    assert entries[prefix + "parsed_text.txt"].decode("utf-8") == PADDLE_IMAGE_ONLY  # unchanged
    status = json.loads(entries[prefix + "status.json"])
    assert status["content"]["kind"] == "html_image_reference_only" and status["usable"] is False
    assert status["provenance"] == "manual_import" and status["numbers_in_visible_text"] == []
    assert {p: p.read_bytes() for p in ctx["run_dir"].rglob("*") if p.is_file()} == before  # nothing written
    names = zipfile.ZipFile(io.BytesIO(hybrid.zip_bytes(ctx["run_dir"], None, ctx["root"]))).namelist()
    assert any(name.endswith(prefix + "status.json") for name in names)


# ------------------------------------------------------ Paddle extraction modes


def test_paddle_modes_overlay_only_documented_fields_and_keep_the_registry() -> None:
    spec = next(item for item in load_providers() if item.id == enrich.PADDLE_PROVIDER_ID)
    before = dict(spec.optional_payload)
    chart = enrich.mode_spec(spec, "chart_recognition")
    assert chart.optional_payload == before | {"useChartRecognition": True}
    assert "useOcrForImageBlock" not in chart.optional_payload and "use_ocr_for_image_block" not in chart.optional_payload
    assert enrich.mode_spec(spec, "standard") is spec and dict(spec.optional_payload) == before
    with pytest.raises(enrich.EnrichError):
        enrich.mode_spec(spec, "image_block_ocr")


def test_chart_review_flags_empty_chart_blocks_and_never_reads_html_values() -> None:
    empty_chart = {"extraction_mode": "chart_recognition", "assessment": enrich.assess(PADDLE_IMAGE_ONLY),
                   "structured_blocks": [{"label": "chart", "content": "", "visible_text": ""}]}
    assert "no content" in enrich.chart_review(empty_chart, "raster image")[0]
    standard = dict(empty_chart, extraction_mode="standard")
    assert "chart recognition was off" in enrich.chart_review(standard, "raster image")[0]
    table = "<table><tr><td>المدينة</td><td>النسبة</td></tr><tr><td>الرياض</td><td>42%</td></tr></table>"
    filled = {"extraction_mode": "chart_recognition", "assessment": enrich.assess(table),
              "structured_blocks": [{"label": "chart", "content": table, "visible_text": enrich.visible_text(table)}]}
    assert enrich.chart_review(filled, "raster image") == []
    assert [n["original"] for n in filled["assessment"]["numbers"]] == ["42%"]


def test_chart_mode_run_records_and_sends_the_exact_options(ctx) -> None:
    root = ctx["root"]
    spec = next(item for item in load_providers() if item.id == enrich.PADDLE_PROVIDER_ID)
    documents.register_document(ctx["source"], display_name="Synthetic_Charts.pdf", registered_by="Tester",
                                statement=documents.REGISTRATION_STATEMENT, root=root)
    entry = documents.load_registry(root=root)[ctx["document"].sha256]
    documents.authorize_document(entry, file_path=ctx["source"], reviewer="Tester",
                                 declaration=documents.USER_OWNED_DECLARATION, days_valid=1, root=root)
    record_enablement(spec, enabled=True, enabled_by="Tester", statement=REQUIRED_STATEMENT, max_pages_total=10,
                      max_pages_per_day=10, days_valid=1, root=root)
    set_dir = _set(ctx)
    region = enrich.load_set(set_dir)["regions"][0]
    seen = []

    def factory():
        def execute(spec_, credential, image_path, filename, job_id, result_hosts, on_event):
            seen.append(dict(spec_.optional_payload))
            return {"phase": "error", "ok": False, "job_id": None, "resumable": False,
                    "error": {"code": "http_400", "message": "synthetic refusal"}, "exchanges": []}
        return execute

    ctx["monkeypatch"].setattr(api_runner, "subprocess_executor", factory)
    run_dir = enrich.create_api_run(set_dir, spec, [region["region_id"]], by="Tester",
                                    statement=enrich.APPROVAL_STATEMENT, root=root, mode="chart_recognition")
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["extraction_mode"] == "chart_recognition" and plan["optional_payload"]["useChartRecognition"] is True
    assert plan["unsupported_options"] and plan["identity"]["optional_payload"]["useChartRecognition"] is True
    runner = APIEvaluationRunner(ctx["manager"], ctx["settings"], ResultStore(root))
    enrich.execute_api_run(run_dir, runner, spec, workspace_id=ctx["workspace_id"], document=ctx["document"],
                           pdf_path=ctx["source"], credential=KEY, root=root)
    assert seen == [plan["optional_payload"]]
    (output,) = enrich.collect_outputs(set_dir, {spec.id: spec}.get, root)[region["region_id"]]
    assert output["extraction_mode"] == "chart_recognition" and output["optional_payload"]["useChartRecognition"]
    assert "[chart recognition]" in enrich.output_label(output)
