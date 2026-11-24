"""Automatic Extraction end to end on a synthetic PDF (scanned page + digital page), with a fake
PP-DocLayout predictor and a fake Qari service behind httpx.MockTransport. No model, no network."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile

import fitz
import httpx
import pytest
from PIL import Image, ImageDraw

from lab.core.auto import assemble, pipeline, qari_client, regions, router
from lab.core.auto import config as auto_config

PAGE_PT = [0, 0, 595, 842]
IMAGE_PX = [1190, 1684]
CHART_TEXT = "Revenue by region North 120 South 95 West 70"
BODY_TEXT = "Annual report body text\n" + CHART_TEXT + "\nThe end of the scanned page"


def _scan_jpeg() -> bytes:
    image = Image.new("RGB", tuple(IMAGE_PX), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([100, 200, 700, 600], fill=(30, 90, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class Predictor:
    """PP-DocLayout stand-in: a chart, an inner image (contained), and two text blocks."""

    calls = 0

    def predict(self, array, batch_size=1):
        Predictor.calls += 1
        return [{"res": {"boxes": [
            {"label": "text", "coordinate": [100, 60, 1000, 160], "score": 0.9},
            {"label": "chart", "coordinate": [100, 200, 700, 600], "score": 0.95},
            {"label": "image", "coordinate": [150, 250, 400, 450], "score": 0.6},
            {"label": "text", "coordinate": [100, 700, 1000, 800], "score": 0.9},
        ]}}]


class FakeQari:
    """Mimics runpod/qari_service.py responses; records every transmitted request."""

    def __init__(self, mode="ok"):
        self.mode, self.requests = mode, []

    def __call__(self, request):
        if request.method == "GET":
            return httpx.Response(404)
        body = json.loads(request.content)
        if self.mode == "down":
            raise httpx.ConnectError("refused", request=request)
        self.requests.append(body)
        if self.mode == "silent":
            raise httpx.ReadTimeout("no answer", request=request)
        text = BODY_TEXT if body["kind"] == "page" else CHART_TEXT
        flags = {"loop_stopped": False, "hit_limit": False, "timed_out": False}
        return httpx.Response(200, json={
            "request_id": body["request_id"], "kind": body["kind"], "service_version": "1", "recipe": "fake",
            "model": {"base": "Qwen/Qwen3-VL-4B-Instruct", "adapter": "NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct"},
            "proven_overrides": {}, "image": {"sha256": body["image_sha256"], "resized": False},
            "final_text": text, "final_attempt": 1, "retried": False, "final_flags": flags,
            "attempts": [{"attempt": 1, "text": text, **flags}], "settings": {"kind": body["kind"]}})


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("QARI_SERVICE_TOKEN", "s3cret-test-token-value")
    pdf = tmp_path / "Synthetic.pdf"
    doc = fitz.open()
    doc.new_page(width=595, height=842).insert_image(fitz.Rect(*PAGE_PT), stream=_scan_jpeg())
    digital = doc.new_page(width=595, height=842)
    digital.insert_text((72, 72), "Digital page heading", fontsize=18)
    digital.insert_text((72, 120), "A plain paragraph of digital text on the second page.", fontsize=11)
    doc.save(pdf)
    doc.close()
    file = tmp_path / "auto.toml"
    file.write_text("", encoding="utf-8")
    config = auto_config.load_config(file, environ={"AGENTICDOC_RESULTS_DIR": str(tmp_path / "results"),
                                                    "QARI_SERVICE_URL": "http://127.0.0.1:8000"})
    detector = regions.LayoutDetector(model_name="PP-DocLayout_plus-L", model_dir=None, device="cpu",
                                      threshold=None, predictor=Predictor())
    run_dir = pipeline.create_run(pdf, display_name="Synthetic.pdf", config=config, detector=detector)
    return {"run_dir": run_dir, "config": config, "pdf": pdf}


def _client(fake):
    return qari_client.QariClient("http://127.0.0.1:8000", "tok", transport=httpx.MockTransport(fake))


def test_plan_routes_pages_and_prepares_native_pixel_jobs(ctx) -> None:
    plan = pipeline.load_plan(ctx["run_dir"])
    routes = {p["page"]: p for p in plan["routing"]["pages"]}
    assert routes[1]["route"] == router.SCANNED and routes[1]["rule"] == "S1"
    assert routes[2]["route"] == router.DIGITAL
    jobs = {j["job_id"]: j for j in plan["jobs"]["jobs"]}
    assert set(jobs) == {"p001_R0", "p001_R1", "p001_page"}
    chart = jobs["p001_R0"]
    assert chart["automatic"] and chart["image"]["pixel_box"] == [40, 140, 760, 660]
    assert chart["image"]["native_pixels"] and chart["image"]["page_image_method"] == "native_jpeg"
    assert Image.open(ctx["run_dir"] / chart["image_file"]).size == (720, 520)
    assert not jobs["p001_R1"]["automatic"]  # contained inner image -> review queue, not processed
    data = (ctx["run_dir"] / chart["image_file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == chart["image_sha256"]
    run = plan["run"]
    assert run["proven_differences"] == {} and run["image_resizing"] == "none (native pixels)"
    assert "s3cret-test-token-value" not in json.dumps(run) and run["config"]["qari"]["token_present"] is True


def test_extraction_runs_without_approval_name_or_document_authorization(ctx, monkeypatch) -> None:
    from lab.core.e2.api import documents

    def forbidden(*args, **kwargs):
        raise AssertionError("Automatic Extraction must not consult per-document authorization")

    monkeypatch.setattr(documents, "authorization_state", forbidden)
    fake = FakeQari()
    summary = pipeline.execute(ctx["run_dir"], ctx["config"], _client(fake))  # no approve(), no name
    assert summary["completed"] == 2 and summary["stopped"] is None and len(fake.requests) == 2
    assert not (ctx["run_dir"] / "approval.json").exists()
    assert not hasattr(pipeline, "approve") and not hasattr(pipeline, "APPROVAL_STATEMENT")
    (record,) = pipeline.service_records(ctx["run_dir"])
    assert record["endpoint"] == "http://127.0.0.1:8000" and record["deployment_mode"] == "local"
    assert record["token_present"] is True and "s3cret-test-token-value" not in json.dumps(record)
    path = assemble.finalize(ctx["run_dir"], ctx["config"])  # no name
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["qari_service"]["deployment_modes_used"] == ["local"]
    assert "finalized_by" not in manifest and "s3cret-test-token-value" not in path.read_text(encoding="utf-8")


def test_untrusted_or_disabled_service_sends_nothing_and_the_run_stays_resumable(ctx, tmp_path) -> None:
    file = tmp_path / "auto2.toml"
    file.write_text("", encoding="utf-8")
    base = {"AGENTICDOC_RESULTS_DIR": str(tmp_path / "results")}
    untrusted = auto_config.load_config(file, environ=base | {"QARI_SERVICE_URL": "https://qari.example.com"})
    fake = FakeQari()
    with pytest.raises(qari_client.QariUntrusted):  # local mode: a non-loopback URL is refused up front
        qari_client.from_config(untrusted, transport=httpx.MockTransport(fake))
    disabled = auto_config.load_config(file, environ=base | {"QARI_SERVICE_URL": "http://127.0.0.1:8000",
                                                            "QARI_AUTOMATIC_ENABLED": "false"})
    summary = pipeline.execute(ctx["run_dir"], disabled, _client(fake))
    assert summary["stopped"].startswith("not sent") and "QARI_AUTOMATIC_ENABLED" in summary["stopped"]
    assert fake.requests == [] and len(pipeline.pending_jobs(ctx["run_dir"])) == 2
    assert pipeline.execute(ctx["run_dir"], ctx["config"], _client(fake))["completed"] == 2  # resume


def test_trusted_development_endpoint_is_used_automatically(ctx, tmp_path) -> None:
    file = tmp_path / "auto3.toml"
    file.write_text("", encoding="utf-8")
    config = auto_config.load_config(file, environ={
        "AGENTICDOC_RESULTS_DIR": str(tmp_path / "results"), "QARI_DEPLOYMENT_MODE": "development_external",
        "QARI_SERVICE_URL": "https://pod123-8000.proxy.example.net",
        "QARI_TRUSTED_DEV_HOSTS": "pod123-8000.proxy.example.net"})
    fake = FakeQari()
    client = qari_client.from_config(config, transport=httpx.MockTransport(fake))
    assert pipeline.execute(ctx["run_dir"], config, client)["completed"] == 2
    record = pipeline.service_records(ctx["run_dir"])[0]
    assert record["deployment_mode"] == "development_external" and "QARI_TRUSTED_DEV_HOSTS" in record["trust_basis"]


def test_a_client_for_another_endpoint_is_refused(ctx) -> None:
    other = qari_client.QariClient("http://127.0.0.1:9999", "tok", transport=httpx.MockTransport(FakeQari()))
    with pytest.raises(pipeline.PipelineError, match="configured Qari service"):
        pipeline.execute(ctx["run_dir"], ctx["config"], other)


def test_full_run_review_queue_process_now_and_finalize(ctx) -> None:
    run_dir, config = ctx["run_dir"], ctx["config"]
    fake = FakeQari()
    summary = pipeline.execute(run_dir, config, _client(fake))
    assert summary["completed"] == 2 and summary["stopped"] is None
    assert {r["request_id"].split(":")[1] for r in fake.requests} == {"p001_R0", "p001_page"}
    queue = pipeline.review_queue(run_dir)
    assert [q["region_id"] for q in queue] == ["p1_R1"] and queue[0]["image_file"]
    pipeline.process_now(run_dir, "p1_R1")
    pipeline.execute(run_dir, config, _client(fake))
    assert len(fake.requests) == 3
    document = assemble.assemble(run_dir)
    page1 = document["pages"][0]
    # body lines aligned to the two detected text blocks: text above the chart, chart, then text below
    assert page1["reading_order_method"] == "lines_aligned_to_text_blocks"
    kinds = [(e["type"], e.get("region_id")) for e in page1["elements"]]
    assert kinds == [("page_text", None), ("figure", "p1_R0"), ("figure", "p1_R1"), ("page_text", None)]
    assert page1["elements"][3]["content"] == "The end of the scanned page"
    assert page1["elements"][3]["body_segment"]["continues"] == page1["elements"][0]["id"]
    # p1_R1 was processed with Process now after the (masked) page OCR: not masked -> warned, page OCR not rerun
    codes = {(f["code"], f.get("region_id")) for f in document["validation"]["flags"]}
    assert ("unmasked_region_possible_duplicate", "p1_R1") in codes
    assert ("unmasked_region_possible_duplicate", "p1_R0") not in codes
    assert sum(1 for r in fake.requests if r["request_id"].split(":")[1] == "p001_page") == 1
    assert [e["type"] for e in page1["elements"]][0] == "page_text"
    body = page1["elements"][0]
    assert "North 120" not in body["content"] and "Annual report body text" in body["content"]
    assert body["raw_content"] == BODY_TEXT  # raw output unchanged
    assert body["dedup"]["removed"][0]["rule"] == "figure_span"
    figure = next(e for e in page1["elements"] if e.get("region_id") == "p1_R0")
    assert figure["content"] == CHART_TEXT and figure["crop"]["sha256"]
    assert figure["provenance"]["job_id"] == "p001_R0" and figure["model"]["adapter"].startswith("NAMAA-Space")
    assert document["pages"][1]["route"]["route"] == router.DIGITAL
    manifest_path = assemble.finalize(run_dir, config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("document.md", "document.json", "validation_report.json", "rag_chunks.jsonl"):
        assert (run_dir / name).is_file() and manifest["files_sha256"][name]
    assert manifest["qari_service"]["endpoints_used"] == ["http://127.0.0.1:8000"]
    assert manifest["document"]["sha256"] == hashlib.sha256(ctx["pdf"].read_bytes()).hexdigest()  # provenance
    document_json = json.loads((run_dir / "document.json").read_text(encoding="utf-8"))
    crop = next(e for e in document_json["elements"] if e.get("region_id") == "p1_R0")["crop"]
    assert crop["sha256"] == hashlib.sha256((run_dir / crop["file"]).read_bytes()).hexdigest()
    assert manifest["reading_order"]["1"] == "lines_aligned_to_text_blocks"
    assert manifest["reading_order_provenance"]["1"]["source"] == "PP-DocLayout text-block and visual boxes"
    assert manifest["masking"]["p001_page"]["masked_region_ids"] == ["p1_R0"]
    assert manifest["duplicate_check"]["final_document"]["total"] == 0
    assert manifest["proven_settings"]["qari"]["max_new_tokens"] == 1536 and manifest["proven_differences"] == {}
    assert manifest["jobs"]["processed_on_request"] == 1
    with pytest.raises(pipeline.PipelineError):
        assemble.finalize(run_dir, config)  # write-once
    with pytest.raises(pipeline.PipelineError):
        pipeline.process_now(run_dir, "p1_R1")
    names = zipfile.ZipFile(io.BytesIO(assemble.zip_bytes(run_dir))).namelist()
    assert "execution_manifest.json" in names and "crops/p001_R0.png" in names and "source.pdf" not in names
    assert any(name.startswith("raw/p001_R0/attempt_01") for name in names)


def test_unavailable_service_pauses_and_the_run_resumes(ctx) -> None:
    run_dir, config = ctx["run_dir"], ctx["config"]
    summary = pipeline.execute(run_dir, config, _client(FakeQari("down")))
    assert summary["stopped"].startswith("Qari service unavailable") and summary["not_transmitted"] == 1
    assert len(pipeline.pending_jobs(run_dir)) == 2  # nothing was sent: still pending
    with pytest.raises(pipeline.PipelineError, match="not completed"):
        assemble.finalize(run_dir, config)
    fake = FakeQari()
    assert pipeline.execute(run_dir, config, _client(fake))["completed"] == 2
    assert pipeline.pending_jobs(run_dir) == [] and pipeline.status(run_dir)["counts"]["completed"] == 2


def test_unconfirmed_requests_are_never_resent_automatically(ctx) -> None:
    run_dir, config = ctx["run_dir"], ctx["config"]
    silent = FakeQari("silent")
    pipeline.execute(run_dir, config, _client(silent))
    assert len(silent.requests) == 2
    again = FakeQari()
    pipeline.execute(run_dir, config, _client(again))
    assert again.requests == []  # unconfirmed jobs are not pending
    assert pipeline.status(run_dir)["counts"]["unconfirmed"] == 2
    with pytest.raises(pipeline.PipelineError, match="not completed"):
        assemble.finalize(run_dir, config)
    pipeline.retry_job(run_dir, "p001_R0", config, _client(again))  # explicit retry
    assert len(again.requests) == 1
    assert pipeline.recover_job(run_dir, "p001_page", _client(FakeQari())) is None  # 404: nothing saved
    path = assemble.finalize(run_dir, config, allow_incomplete=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["incomplete"] and manifest["open_jobs"] == 1
    report = json.loads((run_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert any(f["code"] == "ocr_unconfirmed" for f in report["flags"])


def test_offline_export_and_hash_checked_import(ctx) -> None:
    run_dir = ctx["run_dir"]
    archive = zipfile.ZipFile(io.BytesIO(pipeline.export_offline(run_dir)))
    jobs = [json.loads(line) for line in archive.read("jobs.jsonl").decode().splitlines()]
    assert {j["job_id"] for j in jobs} == {"p001_R0", "p001_page"}
    assert "README.txt" in archive.namelist() and not any(n.endswith(".pdf") for n in archive.namelist())
    lines = []
    for job in jobs:
        image = archive.read(job["image"])
        digest = hashlib.sha256(image).hexdigest()
        response = {"final_text": "offline " + job["job_id"], "attempts": [{"attempt": 1}], "final_attempt": 1,
                    "retried": False, "final_flags": {}, "image": {"sha256": digest}}
        lines.append(json.dumps({"job_id": job["job_id"], "image_sha256": digest, "response": response}))
    lines.append(json.dumps({"job_id": "p001_page", "image_sha256": "0" * 64, "response": {}}))
    record = pipeline.import_offline(run_dir, ("\n".join(lines)).encode())
    assert sorted(record["imported"]) == ["p001_R0", "p001_page"] and len(record["skipped"]) == 1
    again = pipeline.import_offline(run_dir, lines[0].encode())
    assert again["imported"] == [] and "already completed" in again["skipped"][0]["reason"]
    assert pipeline.latest(run_dir, "p001_R0")["source"] == "offline_import"


def test_planning_needs_a_layout_detector_for_scanned_content(tmp_path, ctx) -> None:
    with pytest.raises(pipeline.PipelineError, match="layout detector"):
        pipeline.create_run(ctx["pdf"], display_name="x.pdf", config=ctx["config"], detector=None)


def test_existing_runs_are_never_modified(ctx) -> None:
    run_dir = ctx["run_dir"]
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    pipeline.execute(run_dir, ctx["config"], _client(FakeQari()))
    assemble.assemble(run_dir)
    for path, data in before.items():
        assert path.read_bytes() == data
    with pytest.raises(FileExistsError):
        pipeline.write_once(run_dir / "plan" / "jobs.json", {"x": 1})


def test_service_payload_is_base64_png_of_the_saved_crop(ctx) -> None:
    fake = FakeQari()
    pipeline.execute(ctx["run_dir"], ctx["config"], _client(fake))
    request = next(r for r in fake.requests if r["request_id"].split(":")[1] == "p001_R0")
    sent = base64.b64decode(request["image_png_b64"])
    assert sent == (ctx["run_dir"] / "crops" / "p001_R0.png").read_bytes() and sent.startswith(b"\x89PNG")


class NoTextPredictor:
    def predict(self, array, batch_size=1):
        return [{"res": {"boxes": [
            {"label": "table", "coordinate": [150, 900, 1050, 1300], "score": 0.9},
            {"label": "chart", "coordinate": [100, 200, 700, 600], "score": 0.95},
        ]}}]


def test_scanned_reading_order_without_text_blocks_is_page_text_first(ctx) -> None:
    detector = regions.LayoutDetector(model_name="PP-DocLayout_plus-L", model_dir=None, device="cpu",
                                      threshold=None, predictor=NoTextPredictor())
    run_dir = pipeline.create_run(ctx["pdf"], display_name="Synthetic.pdf", config=ctx["config"], detector=detector)
    pipeline.execute(run_dir, ctx["config"], _client(FakeQari()))
    page = assemble.assemble(run_dir)["pages"][0]
    assert page["reading_order_method"] == "page_text_first"
    order = [(e["type"], e.get("label")) for e in page["elements"]]
    assert order == [("page_text", None), ("figure", "chart"), ("table", "table")]  # then box order, top to bottom
    assert [e["reading_order"] for e in page["elements"]] == [0, 1, 2]


def test_missing_layout_model_dir_fails_clearly_without_loading_anything(ctx, tmp_path) -> None:
    detector = regions.LayoutDetector(model_name="PP-DocLayout_plus-L", model_dir=str(tmp_path / "no_model"),
                                      device="cpu", threshold=None)
    with pytest.raises(pipeline.PipelineError, match="PP-DocLayout is not ready") as error:
        pipeline.create_run(ctx["pdf"], display_name="Synthetic.pdf", config=ctx["config"], detector=detector)
    assert "LAYOUT_MODEL_DIR" in str(error.value) and detector._predictor is None  # never loaded
