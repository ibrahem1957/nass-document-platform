from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from studio.adapters import (
    AdapterError,
    CommandResult,
    DocPrepAdapter,
    NassAdapter,
)
from studio.config import ExistingDocument, StudioConfig
from studio.jobs import JobError, JobManager
from studio.renderers import (
    nass_markdown,
    nass_reading_view,
    prepared_markdown,
    prepared_reading_view,
)
from studio.server import create_server

PDF = b"%PDF-1.4\nsynthetic studio fixture\n%%EOF\n"


def nass_document(text: str = "نسبة ٣٣٫٨٤٪ and 12.50%") -> dict:
    return {
        "schema_version": 1, "kind": "automatic_extraction_document", "run_id": "run_mock",
        "document": {"display_name": "fixture.pdf"},
        "pages": [{"page": 1, "element_ids": ["e1", "e2"]}, {"page": 2, "element_ids": ["e3"]}],
        "elements": [
            {"id": "e1", "page": 1, "reading_order": 0, "type": "heading", "content": "عنوان",
             "validation_state": "ok"},
            {"id": "e2", "page": 1, "reading_order": 1, "type": "text", "content": text,
             "validation_state": "needs_review"},
            {"id": "e3", "page": 2, "reading_order": 0, "type": "figure", "content": "Figure value 42",
             "validation_state": "ok"},
        ],
    }


def prepared_document(text: str = "نسبة ٣٣٫٨٤٪ and 12.50%") -> dict:
    elements = [
        {"element_id": "e1", "original_type": "heading", "text": "عنوان", "page": 1, "document_order": 0,
         "validation_state": "ok", "quality_finding_ids": []},
        {"element_id": "e2", "original_type": "text", "text": text, "page": 1, "document_order": 1,
         "validation_state": "needs_review", "quality_finding_ids": ["q1"]},
        {"element_id": "e3", "original_type": "figure", "text": "Figure value 42", "page": 2,
         "document_order": 2, "validation_state": "ok", "quality_finding_ids": []},
        {"element_id": "e4", "original_type": "table", "text": "H1 | H2\n1 | 2", "page": 2,
         "document_order": 3, "validation_state": "ok", "quality_finding_ids": []},
        {"element_id": "e5", "original_type": "table", "text": "3 | 4", "page": 3,
         "document_order": 4, "validation_state": "ok", "quality_finding_ids": []},
    ]
    return {
        "schema_version": "1.0.0", "kind": "docprep_prepared_document", "document_id": "doc",
        "source_run_id": "run_mock", "page_count": 3,
        "metadata": {"title": "عنوان من المصدر"},
        "elements": elements,
        "sections": [
            {"section_id": "s1", "title": "عنوان", "level": 0, "parent": None, "children": ["s2"],
             "heading_element_ids": ["e1"], "element_ids_in_reading_order": ["e1", "e2"],
             "page_start": 1, "page_end": 3, "quality_finding_ids": []},
            {"section_id": "s2", "title": "Results", "level": 1, "parent": "s1", "children": [],
             "heading_element_ids": [], "element_ids_in_reading_order": ["e3", "e4", "e5"],
             "page_start": 2, "page_end": 3, "quality_finding_ids": []},
        ],
        "lists": [],
        "visuals": [{"visual_id": "e3", "visual_type": "figure", "review_flags": [],
                     "description_paragraphs": ["e2"]}],
        "logical_tables": [{"logical_table_id": "lt1", "source_part_ids": ["e4", "e5"], "pages": [2, 3],
                            "quality_finding_ids": []}],
        "quality": {"findings": []},
    }


def config(tmp_path: Path, existing: tuple[ExistingDocument, ...] = ()) -> StudioConfig:
    root = tmp_path / "studio"
    root.mkdir(exist_ok=True)
    nass_root = tmp_path / "nass-root"; nass_root.mkdir(exist_ok=True)
    docprep_root = tmp_path / "docprep-root"; docprep_root.mkdir(exist_ok=True)
    python = tmp_path / "python.exe"; python.touch(exist_ok=True)
    return StudioConfig(root, root / "jobs", nass_root, python, docprep_root, python, 2, 60, existing)


def write_nass_run(root: Path, document: dict | None = None) -> Path:
    run = root / "automatic_extraction" / "run_mock"
    run.mkdir(parents=True)
    doc = document or nass_document()
    (run / "document.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    manifest = {"schema_version": 1, "kind": "automatic_extraction_manifest", "run_id": "run_mock",
                "document": {"bytes": len(PDF), "sha256": hashlib.sha256(PDF).hexdigest()}}
    (run / "execution_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def write_docprep_outputs(root: Path, prepared: dict | None = None) -> dict[str, Path]:
    outputs = {}
    for name in ("structure", "relationships", "quality", "prepared"):
        path = root / name / ("run_" + name)
        path.mkdir(parents=True)
        outputs[name] = path
    doc = prepared or prepared_document()
    (outputs["prepared"] / "prepared_document.json").write_text(
        json.dumps(doc, ensure_ascii=False), encoding="utf-8",
    )
    validation = {"schema_version": 1, "kind": "docprep_prepared_document_validation",
                  "summary": {"ok": 15, "error": 0}}
    (outputs["prepared"] / "prepared_document_validation.json").write_text(
        json.dumps(validation), encoding="utf-8",
    )
    return outputs


class MockNass:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def readiness(self):
        return {"ready": True, "external_processing": True, "deployment_mode": "test",
                "service_display": "https://example.invalid", "token_present": True}

    def run(self, job_id, pdf_path, display_name, output_root, cancel):
        self.calls.append((job_id, pdf_path, display_name, output_root))
        if self.fail:
            raise AdapterError("mock Nass failure")
        return write_nass_run(output_root)


class MockDocPrep:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def run_sequence(self, job_id, nass_run, output_base, cancel, on_stage):
        self.calls.append((job_id, nass_run, output_base))
        on_stage("structure")
        if self.fail:
            raise AdapterError("mock DocPrep failure")
        on_stage("relationships"); on_stage("quality"); on_stage("prepared")
        return write_docprep_outputs(output_base)


def run_synchronously(manager: JobManager, job_id: str):
    manager._process(manager.config.jobs_root / job_id, threading.Event())


def all_view_text(view: dict) -> list[str]:
    found = []

    def blocks(items):
        for block in items:
            if block["kind"] == "list":
                def list_items(values):
                    for item in values:
                        found.append(item["text"]); list_items(item.get("children", []))
                list_items(block["items"])
            elif block["kind"] == "logical_table":
                found.extend(part["text"] for part in block["parts"])
            else:
                found.append(block["text"])

    def section(value):
        blocks(value["blocks"])
        for child in value["children"]: section(child)

    for item in view["sections"]: section(item)
    blocks(view["remaining_blocks"])
    return found


def test_pdf_upload_validation_and_filename_safety(tmp_path):
    manager = JobManager(config(tmp_path), nass_adapter=MockNass(), docprep_adapter=MockDocPrep())
    with pytest.raises(JobError, match="not a PDF"):
        manager.create(b"not pdf", "bad.pdf", start=False)
    with pytest.raises(JobError, match="only .pdf"):
        manager.create(PDF, "../../payload.txt", start=False)
    status = manager.create(PDF, "../../safe name.pdf", start=False)
    assert status["original_filename"] == "safe name.pdf"
    assert (manager.config.jobs_root / status["job_id"] / "original" / "source.pdf").read_bytes() == PDF


def test_automatic_pipeline_propagates_exact_paths_and_outputs(tmp_path):
    nass, docprep = MockNass(), MockDocPrep()
    manager = JobManager(config(tmp_path), nass_adapter=nass, docprep_adapter=docprep)
    status = manager.create(PDF, "fixture.pdf", start=False)
    run_synchronously(manager, status["job_id"])
    final = manager.status(status["job_id"])
    assert final["state"] == "ready"
    assert [call[0] for call in nass.calls] == [status["job_id"]]
    assert docprep.calls[0][1].name == "run_mock"
    assert docprep.calls[0][2] == manager.config.jobs_root / status["job_id"] / "upstream" / "docprep"
    paths = manager.result_paths(status["job_id"])
    assert paths["nass_json"].is_file() and paths["prepared_json"].is_file()


def test_docprep_failure_preserves_completed_nass_result(tmp_path):
    manager = JobManager(config(tmp_path), nass_adapter=MockNass(), docprep_adapter=MockDocPrep(fail=True))
    status = manager.create(PDF, "fixture.pdf", start=False)
    run_synchronously(manager, status["job_id"])
    final = manager.status(status["job_id"])
    assert final["state"] == "failed"
    assert final["nass_available"] is True and final["docprep_available"] is False
    assert manager.result_paths(status["job_id"])["nass_json"].is_file()
    assert (manager.config.jobs_root / status["job_id"] / "diagnostics" / "docprep_structure.json").is_file()


def test_retry_and_concurrent_jobs_use_new_isolated_directories(tmp_path):
    manager = JobManager(config(tmp_path), nass_adapter=MockNass(), docprep_adapter=MockDocPrep())
    first = manager.create(PDF, "first.pdf", start=False)
    second = manager.create(PDF, "second.pdf", start=False)
    retry = manager.retry(first["job_id"], start=False)
    assert len({first["job_id"], second["job_id"], retry["job_id"]}) == 3
    assert retry["retry_of"] == first["job_id"]
    assert all((manager.config.jobs_root / item["job_id"] / "original" / "source.pdf").is_file()
               for item in (first, second, retry))


def test_restart_marks_ambiguous_job_failed_without_retrying(tmp_path):
    cfg = config(tmp_path)
    first_nass = MockNass()
    manager = JobManager(cfg, nass_adapter=first_nass, docprep_adapter=MockDocPrep())
    status = manager.create(PDF, "interrupted.pdf", start=False)
    assert status["state"] == "queued"
    second_nass = MockNass()
    restarted = JobManager(cfg, nass_adapter=second_nass, docprep_adapter=MockDocPrep())
    recovered = restarted.status(status["job_id"])
    assert recovered["state"] == "failed"
    assert "not retried automatically" in recovered["error"]
    assert second_nass.calls == []


def test_nass_page_rendering_preserves_order_text_and_uncertainty():
    source = nass_document()
    view = nass_reading_view(source)
    assert [page["page"] for page in view["pages"]] == [1, 2]
    assert [block["text"] for block in view["pages"][0]["blocks"]] == ["عنوان", source["elements"][1]["content"]]
    assert view["pages"][0]["blocks"][1]["requires_review"] is True
    assert "٣٣٫٨٤٪" in nass_markdown(source)


def test_composite_member_is_only_suppressed_from_reading_view():
    source = nass_document()
    source["elements"][2]["validation_state"] = "included_in_composite"
    view = nass_reading_view(source)
    assert view["omitted_composite_members"] == ["e3"]
    assert source["elements"][2]["content"] not in [block["text"] for page in view["pages"] for block in page["blocks"]]
    assert source["elements"][2]["content"] in json.dumps(source)


def test_prepared_order_nested_sections_figures_and_logical_table():
    source = prepared_document()
    view = prepared_reading_view(source)
    assert view["represented_source_count"] == len(source["elements"])
    assert view["sections"][0]["title"] == "عنوان"
    assert view["sections"][0]["children"][0]["title"] == "Results"
    child_blocks = view["sections"][0]["children"][0]["blocks"]
    assert child_blocks[0]["kind"] == "visual"
    assert child_blocks[1]["kind"] == "logical_table"
    assert [part["page"] for part in child_blocks[1]["parts"]] == [2, 3]
    texts = all_view_text(view)
    assert texts.count("Figure value 42") == 1
    assert texts.count("H1 | H2\n1 | 2") == 1
    assert texts.count("3 | 4") == 1


def test_nested_list_uses_saved_text_without_fabrication():
    source = prepared_document()
    source["elements"].extend([
        {"element_id": "e6", "original_type": "text", "text": "Parent item", "page": 3,
         "document_order": 5, "validation_state": "ok", "quality_finding_ids": []},
        {"element_id": "e7", "original_type": "text", "text": "Child item", "page": 3,
         "document_order": 6, "validation_state": "ok", "quality_finding_ids": []},
    ])
    source["sections"][1]["element_ids_in_reading_order"].extend(["e6", "e7"])
    source["lists"] = [{"list_id": "l1", "nested_items": [
        {"element_id": "e6", "text": "Parent item", "page": 3, "children": [
            {"element_id": "e7", "text": "Child item", "page": 3, "children": []},
        ]},
    ], "structure": {"pages": [3]}, "quality_finding_ids": []}]
    view = prepared_reading_view(source)
    list_block = view["sections"][0]["children"][0]["blocks"][-1]
    assert list_block["kind"] == "list"
    assert list_block["items"][0]["children"][0]["text"] == "Child item"
    assert view["represented_source_count"] == len(source["elements"])


def test_markdown_exports_are_deterministic_and_faithful():
    nass = nass_document()
    prepared = prepared_document()
    assert nass_markdown(nass) == nass_markdown(nass)
    assert prepared_markdown(prepared) == prepared_markdown(prepared)
    for value in ("نسبة ٣٣٫٨٤٪ and 12.50%", "Figure value 42", "H1 | H2\n1 | 2", "3 | 4"):
        assert value in prepared_markdown(prepared)


class FakeCommandRunner:
    def __init__(self):
        self.commands = []

    def run(self, job_id, command_name, command, *, cwd, timeout_s, cancel):
        self.commands.append((job_id, command_name, command, cwd))
        root = Path(command[command.index("--out") + 1])
        output = root / ("run_" + command_name.split()[-1])
        output.mkdir(parents=True)
        return CommandResult(command_name, 0, f"written: {output}\n", "")


def test_docprep_adapter_runs_supported_cli_sequence(tmp_path):
    cfg = config(tmp_path)
    runner = FakeCommandRunner()
    adapter = DocPrepAdapter(cfg, runner)
    nass_run = tmp_path / "run_nass"; nass_run.mkdir()
    stages = []
    outputs = adapter.run_sequence("job_" + "a" * 20, nass_run, tmp_path / "outputs",
                                   threading.Event(), stages.append)
    assert stages == ["structure", "relationships", "quality", "prepared"]
    assert [call[2][2] for call in runner.commands] == [
        "docprep.structure", "docprep.relations", "docprep.quality", "docprep.prepare",
    ]
    assert runner.commands[1][2][runner.commands[1][2].index("--structure") + 1] == str(outputs["structure"])
    assert runner.commands[3][2][runner.commands[3][2].index("--quality") + 1] == str(outputs["quality"])


def test_nass_adapter_invocation_carries_exact_job_output_without_consent_flag(tmp_path):
    cfg = config(tmp_path)
    expected = tmp_path / "nass-output" / "automatic_extraction" / "run_mock"
    expected.mkdir(parents=True)
    (expected / "document.json").write_text("{}", encoding="utf-8")

    class Runner:
        command = None

        def run(self, job_id, command_name, command, **kwargs):
            self.command = command
            return CommandResult(command_name, 0, "STUDIO_RESULT=" + json.dumps({"run_dir": str(expected)}), "")

    runner = Runner()
    adapter = NassAdapter(cfg, runner)
    pdf = tmp_path / "source.pdf"; pdf.write_bytes(PDF)
    result = adapter.run("job_" + "a" * 20, pdf, "source.pdf", tmp_path / "nass-output",
                         threading.Event())
    assert result == expected.resolve()
    assert "--authorize-external" not in runner.command
    assert runner.command[runner.command.index("--output-root") + 1] == str(tmp_path / "nass-output")


class WebManager:
    def __init__(self, readiness):
        self._readiness = readiness
        self.created = []

    def readiness(self): return self._readiness
    def list_jobs(self): return []

    def create(self, pdf_bytes, filename):
        self.created.append((pdf_bytes, filename))
        return {"job_id": "job_" + "a" * 20, "state": "queued", "original_filename": filename}


def library_config(tmp_path: Path) -> StudioConfig:
    nass_dir = tmp_path / "saved-nass"; nass_dir.mkdir()
    nass = nass_document()
    nass_path = nass_dir / "document.json"
    nass_path.write_text(json.dumps(nass, ensure_ascii=False), encoding="utf-8")
    pdf_path = nass_dir / "source.pdf"; pdf_path.write_bytes(PDF)
    manifest = {"schema_version": 1, "kind": "automatic_extraction_manifest", "run_id": "run_mock",
                "document": {"bytes": len(PDF), "sha256": hashlib.sha256(PDF).hexdigest()}}
    (nass_dir / "execution_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    prepared_dir = tmp_path / "saved-prepared"; prepared_dir.mkdir()
    prepared_path = prepared_dir / "prepared_document.json"
    prepared_path.write_text(json.dumps(prepared_document(), ensure_ascii=False), encoding="utf-8")
    prepared_validation = {"schema_version": 1, "kind": "docprep_prepared_document_validation",
                           "summary": {"ok": 15, "error": 0}}
    (prepared_dir / "prepared_document_validation.json").write_text(
        json.dumps(prepared_validation), encoding="utf-8",
    )
    existing = (ExistingDocument("saved", "Saved", nass_path, prepared_path, pdf_path),)
    return config(tmp_path, existing)


@contextmanager
def server_context(cfg, manager):
    server = create_server(cfg, manager, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_address[1])
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def get(url):
    with urlopen(url, timeout=5) as response:
        return response.status, dict(response.headers), response.read()


def test_previously_processed_loading_and_exact_distinct_json_downloads(tmp_path):
    cfg = library_config(tmp_path)
    manager = WebManager({"ready": True, "external_processing": False})
    with server_context(cfg, manager) as base:
        _, _, listing = get(base + "/api/library")
        assert json.loads(listing)["documents"][0]["id"] == "saved"
        _, nass_headers, nass_bytes = get(base + "/api/items/library/saved/download/nass-json")
        _, prepared_headers, prepared_bytes = get(base + "/api/items/library/saved/download/prepared-json")
    assert nass_bytes == cfg.existing_documents[0].nass_document.read_bytes()
    assert prepared_bytes == cfg.existing_documents[0].prepared_document.read_bytes()
    assert nass_bytes != prepared_bytes
    assert "nass_document.json" in nass_headers["Content-Disposition"]
    assert "prepared_document.json" in prepared_headers["Content-Disposition"]


def test_json_view_sources_are_complete_saved_files_with_large_values_and_untrusted_text(tmp_path):
    cfg = library_config(tmp_path)
    nass_path = cfg.existing_documents[0].nass_document
    prepared_path = cfg.existing_documents[0].prepared_document
    nass = json.loads(nass_path.read_text(encoding="utf-8"))
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    unsafe_text = '<img src=x onerror="alert(1)"> نص عربي English'
    nass["elements"][0]["content"] = unsafe_text
    nass["large_numeric_value"] = 123456789012345678901234567890
    prepared["elements"][0]["text"] = unsafe_text
    prepared["large_numeric_value"] = 123456789012345678901234567890
    nass_path.write_text(json.dumps(nass, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    prepared_path.write_text(
        json.dumps(prepared, ensure_ascii=False, separators=(",", ":")), encoding="utf-8",
    )
    manager = WebManager({"ready": True, "external_processing": False})
    with server_context(cfg, manager) as base:
        _, _, nass_bytes = get(base + "/api/items/library/saved/download/nass-json")
        _, _, prepared_bytes = get(base + "/api/items/library/saved/download/prepared-json")
    assert nass_bytes == nass_path.read_bytes()
    assert prepared_bytes == prepared_path.read_bytes()
    assert json.loads(nass_bytes)["elements"][0]["content"] == unsafe_text
    assert json.loads(prepared_bytes)["elements"][0]["text"] == unsafe_text
    assert b"123456789012345678901234567890" in nass_bytes
    assert b"123456789012345678901234567890" in prepared_bytes
    assert manager.created == []


def test_saved_results_and_markdown_do_not_modify_upstream(tmp_path):
    cfg = library_config(tmp_path)
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in tmp_path.rglob("*") if path.is_file()}
    with server_context(cfg, WebManager({"ready": True, "external_processing": False})) as base:
        get(base + "/api/items/library/saved/nass")
        get(base + "/api/items/library/saved/docprep")
        get(base + "/api/items/library/saved/download/nass-markdown")
        get(base + "/api/items/library/saved/download/prepared-markdown")
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_external_processing_upload_starts_without_confirmation(tmp_path):
    cfg = library_config(tmp_path)
    manager = WebManager({"ready": True, "external_processing": True})
    with server_context(cfg, manager) as base:
        boundary = "studioBoundary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"pdf\"; filename=\"x.pdf\"\r\n"
            "Content-Type: application/pdf\r\n\r\n"
        ).encode() + PDF + f"\r\n--{boundary}--\r\n".encode()
        request = Request(base + "/api/jobs", data=body, method="POST",
                          headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        with urlopen(request, timeout=5) as response:
            assert response.status == 202
            result = json.loads(response.read())
        assert result["state"] == "queued"
        assert manager.created == [(PDF, "x.pdf")]


@pytest.mark.parametrize("path", [
    "/../config/studio.json", "/%2e%2e/config/studio.json",
    "/api/items/library/saved/%5c..%5cstudio.json",
])
def test_path_traversal_is_rejected(tmp_path, path):
    cfg = library_config(tmp_path)
    with server_context(cfg, WebManager({"ready": True, "external_processing": False})) as base:
        with pytest.raises(HTTPError) as error:
            get(base + path)
        assert error.value.code == 404


def test_ui_uses_safe_text_rendering_and_two_primary_tabs():
    root = Path(__file__).resolve().parents[1]
    script = (root / "studio" / "static" / "app.js").read_text(encoding="utf-8")
    html = (root / "studio" / "static" / "index.html").read_text(encoding="utf-8")
    assert ".innerHTML" not in script
    assert "textContent" in script and '.dir="auto"' in script
    assert html.count('data-tab="') == 2
    assert "Nass — Extraction" in html and "DocPrep — Organized Document" in html
    assert "authorizeExternal" not in script and "authorizeExternal" not in html
    assert "External OCR may use RunPod" in html


def test_both_result_tabs_have_default_readable_and_lazy_complete_json_views():
    root = Path(__file__).resolve().parents[1]
    script = (root / "studio" / "static" / "app.js").read_text(encoding="utf-8")
    html = (root / "studio" / "static" / "index.html").read_text(encoding="utf-8")
    css = (root / "studio" / "static" / "app.css").read_text(encoding="utf-8")
    assert html.count('data-view="readable" class="active"') == 2
    assert html.count('data-view="json"') == 2
    assert 'data-view-group="nass"' in html and 'data-view-group="docprep"' in html
    assert "copyNassJson" in html and "copyDocprepJson" in html
    assert 'state.jsonUrls={nass:base+"/download/nass-json",docprep:base+"/download/prepared-json"}' in script
    assert 'selectView("nass","readable")' in script
    assert 'selectView("docprep","readable")' in script
    assert "if(showJson)loadJsonView(group)" in script
    assert "navigator.clipboard.writeText(value)" in script
    assert "controls.pre.textContent=formatted" in script
    assert ".innerHTML" not in script
    assert "overflow:auto" in css and "font:13px/1.55 Consolas" in css
    assert "/api/jobs" not in script[script.index("function selectView"):script.index("function selectTab")]
