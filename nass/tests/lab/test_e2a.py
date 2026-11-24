from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from lab.core.e2 import REMOTE_INFERENCE_ENABLED
from lab.core.e2 import backends as backends_module
from lab.core.e2.backends import (
    InferenceOutput,
    InferenceError,
    PaddleOCRVLFullPageSession,
    RemoteInferenceDisabled,
    build_ollama_payload,
    check_readiness,
    hf_snapshot_dir,
    local_http_json,
    missing_snapshot_files,
    normalize_official_pipeline_result,
    open_session,
)
from lab.core.e2.e2_bench import render_report
from lab.core.e2.hardware import HardwareProfile, peak_rss_mb, probe_hardware
from lab.core.e2.metrics_e2 import (
    MARKUP_VARIANT,
    derive_markup_strip,
    page_extras,
    parse_hypothesis_tables,
    repetition_stats,
    score_tables,
    strip_markup,
    summarize_run,
)
from lab.core.e2.prompts import PROMPTS, get_prompt
from lab.core.e2.render import RenderError, render_pdf_page
from lab.core.e2.runner import E2Runner, E2RunRefused, adapter_info_for, is_mock_run
from lab.core.e2.specs import ModelSpec, SpecError, is_loopback_url, load_model_specs, mock_spec
from lab.core.groundtruth import GroundTruthError, GroundTruthStore
from lab.core.ingest import IngestService
from lab.core.models import GroundTruthPage, GroundTruthTable
from lab.core.workspace import WorkspaceManager

DATA = Path(__file__).resolve().parent / "data"
CANARY = "CANARY-GT-٤٢٤٢-LEAK"


def _hardware(ram: float = 64.0, cuda: bool = False) -> HardwareProfile:
    return HardwareProfile(
        cpu="test",
        logical_cpus=4,
        ram_total_gb=ram,
        ram_available_gb=ram,
        gpus=[{"name": "gpu", "vram_gb": 24.0}] if cuda else [],
        cuda_available=cuda,
    )


# ------------------------------------------------------------- registry


def test_registry_pins_exact_revisions_and_adapter_base() -> None:
    specs = {spec.id: spec for spec in load_model_specs()}
    assert {"qari-ocr-0.4.0", "paddleocr-vl-1.6", "qwen3-vl-4b"} <= set(specs)
    for spec in specs.values():
        if spec.backend == "transformers":
            assert len(spec.revision) == 40 and int(spec.revision, 16) >= 0
    qari = specs["qari-ocr-0.4.0"]
    assert qari.weights_kind == "peft_lora_adapter"
    assert qari.base_repo_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert qari.base_revision == specs["qwen3-vl-4b"].revision
    assert all(spec.backend != "remote" for spec in specs.values())
    assert REMOTE_INFERENCE_ENABLED is False


def test_official_full_page_pipeline_is_a_separate_pinned_adapter() -> None:
    specs = {spec.id: spec for spec in load_model_specs()}
    element = specs["paddleocr-vl-1.6"]
    pipeline = specs["paddleocr-vl-1.6-full-page"]
    assert element.pipeline_mode == "element"
    assert pipeline.pipeline_mode == "official_full_page"
    assert pipeline.backend == "paddleocr_vl"
    assert pipeline.revision == element.revision
    assert pipeline.layout_model_name == "PP-DocLayoutV3"
    assert pipeline.layout_repo_id == "PaddlePaddle/PP-DocLayoutV3_safetensors"
    assert pipeline.layout_revision == "e4f489dc536556fc1be02b973ba06756e0d4a2ac"
    assert pipeline.layout_download_bytes == 133_289_670
    assert pipeline.storage_revision != element.storage_revision
    # Baseline = vendor defaults: nothing overridden. The project settings live in a
    # separately identified experimental configuration.
    assert pipeline.configuration == "vendor_default" and pipeline.generation == {}
    experiment = specs["paddleocr-vl-1.6-full-page-exp-px4m"]
    assert experiment.configuration == "experimental"
    assert experiment.generation == {
        "max_new_tokens": 4096, "max_pixels": 4_000_000, "repetition_penalty": 1.1
    }
    assert experiment.official_overrides["requested_but_ignored_locally"] == {
        "repetition_penalty": 1.1
    }
    assert f"e2.{experiment.id}" != f"e2.{pipeline.id}"
    adapter = adapter_info_for(pipeline)
    assert adapter.params["prompt_used_by_backend"] is False
    assert adapter.params["prompt_id"] is None
    assert adapter.params["registry_prompt_id"] == "paddle_ocr_v1"


def test_installed_ram_blocks_and_low_available_ram_only_warns(tmp_path, monkeypatch) -> None:
    spec = {item.id: item for item in load_model_specs()}["paddleocr-vl-1.6"]
    monkeypatch.setattr(backends_module, "package_version", lambda _name: "99.0")
    _write_snapshot(tmp_path, spec.repo_id, spec.revision, ["config.json", "model.safetensors"])
    # Transient: plenty installed, little free right now -> warn, never "unsupported".
    busy = replace(_hardware(ram=64.0), ram_available_gb=1.0)
    state = check_readiness(spec, busy, cache=tmp_path)
    assert state.status == "ready"
    assert any("Available RAM is 1.0 GB now" in reason for reason in state.reasons)
    # Structural: the machine cannot host the model at all.
    small = replace(_hardware(ram=2.0), ram_available_gb=2.0)
    state = check_readiness(spec, small, cache=tmp_path)
    assert state.status == "insufficient_hardware"
    assert any("Installed RAM 2.0 GB" in reason for reason in state.reasons)


def test_low_available_ram_does_not_hide_missing_weights(tmp_path) -> None:
    spec = {item.id: item for item in load_model_specs()}["qwen3-vl-4b"]
    busy = replace(_hardware(ram=64.0), ram_available_gb=0.5)
    assert check_readiness(spec, busy, cache=tmp_path).status in {"not_downloaded", "not_installed"}


def test_adapter_without_base_and_non_loopback_ollama_are_rejected() -> None:
    base = mock_spec().model_dump()
    with pytest.raises(ValueError, match="base"):
        ModelSpec.model_validate(
            base | {"backend": "transformers", "weights_kind": "peft_lora_adapter", "mock_response": None}
        )
    with pytest.raises(ValueError, match="loopback"):
        ModelSpec.model_validate(
            base
            | {
                "backend": "ollama",
                "weights_kind": "quantized_gguf",
                "mock_response": None,
                "ollama_tag": "x",
                "endpoint": "http://192.168.1.5:11434",
            }
        )


def test_mock_specs_are_not_allowed_in_the_real_registry(tmp_path: Path) -> None:
    config = tmp_path / "models.toml"
    config.write_text(
        '[[models]]\nid="m"\nlabel="m"\nbackend="mock"\nrepo_id="r"\nrevision="r"\n'
        'weights_kind="mock"\nlicense="x"\nlanguages="x"\nprompt_id="vlm_transcribe_v1"\n'
        "download_gb=0\nmin_ram_gb=0\nmin_vram_gb=0\ncpu_feasible=true\n",
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="Mock"):
        load_model_specs(config)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:11434", True),
        ("http://[::1]:8080", True),
        ("http://127.0.0.1.evil.com", False),
        ("https://huggingface.co", False),
        ("http://10.0.0.1", False),
        ("file:///etc/passwd", False),
    ],
)
def test_loopback_detection(url: str, expected: bool) -> None:
    assert is_loopback_url(url) is expected


def test_prompts_are_static_and_referenced() -> None:
    for template in PROMPTS.values():
        assert "{" not in template.text and "}" not in template.text
        assert len(template.sha256) == 64
    for spec in load_model_specs():
        assert get_prompt(spec.prompt_id).text


# ------------------------------------------------------------- hardware / readiness


def test_hardware_probe_is_local_and_well_formed() -> None:
    profile = probe_hardware()
    assert profile.logical_cpus >= 1
    assert profile.ram_total_gb is None or profile.ram_total_gb > 0
    assert isinstance(profile.gpus, list)
    import sys

    peak = peak_rss_mb()
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        assert peak is not None and peak > 10  # a Python process uses > 10 MiB
    else:
        assert peak is None or peak > 0


def _write_snapshot(cache: Path, repo: str, revision: str, files: list[str]) -> Path:
    snapshot = hf_snapshot_dir(repo, revision, cache)
    snapshot.mkdir(parents=True)
    for name in files:
        (snapshot / name).write_text("{}" if name.endswith(".json") else "x", encoding="utf-8")
    return snapshot


def test_sharded_snapshot_requires_every_shard(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path, "o/m", "r1", ["config.json", "a.safetensors"])
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}), encoding="utf-8"
    )
    assert missing_snapshot_files(snapshot, "full") == ["b.safetensors"]


def test_transformers_readiness_states(tmp_path: Path, monkeypatch) -> None:
    specs = {spec.id: spec for spec in load_model_specs()}
    qari = specs["qari-ocr-0.4.0"]
    monkeypatch.setattr(backends_module, "package_version", lambda name: None)
    assert check_readiness(qari, _hardware(), cache=tmp_path).status == "not_installed"
    monkeypatch.setattr(backends_module, "package_version", lambda name: "9.0.0")
    state = check_readiness(qari, _hardware(), cache=tmp_path)
    assert state.status == "not_downloaded"
    assert any("huggingface-cli download" in step for step in state.next_steps)
    _write_snapshot(tmp_path, qari.repo_id, qari.revision, ["adapter_config.json", "adapter_model.safetensors"])
    # The adapter alone is not enough: the pinned base model must be present too.
    assert check_readiness(qari, _hardware(), cache=tmp_path).status == "not_downloaded"
    _write_snapshot(tmp_path, qari.base_repo_id, qari.base_revision, ["config.json", "model.safetensors"])
    ready = check_readiness(qari, _hardware(), cache=tmp_path)
    assert ready.status == "ready" and ready.runnable
    assert any("CPU" in reason for reason in ready.reasons)
    assert check_readiness(qari, _hardware(ram=8.0), cache=tmp_path).status == "insufficient_hardware"
    disabled = qari.model_copy(update={"enabled": False})
    assert check_readiness(disabled, _hardware(), cache=tmp_path).status == "disabled"


def test_full_pipeline_readiness_requires_layout_without_downloading(tmp_path, monkeypatch) -> None:
    spec = {item.id: item for item in load_model_specs()}["paddleocr-vl-1.6-full-page"]
    monkeypatch.setattr(backends_module, "package_version", lambda _name: "9.0.0")
    _write_snapshot(tmp_path, spec.repo_id, spec.revision, ["config.json", "model.safetensors"])
    state = check_readiness(spec, _hardware(), cache=tmp_path)
    assert state.status == "not_downloaded"
    assert f"{spec.layout_download_bytes:,} bytes" in " ".join(state.reasons)
    assert spec.layout_revision in " ".join(state.next_steps)
    layout = hf_snapshot_dir(spec.layout_repo_id, spec.layout_revision, tmp_path)
    layout.mkdir(parents=True)
    for name in spec.layout_required_files:
        (layout / name).write_text("x", encoding="utf-8")
    assert check_readiness(spec, _hardware(), cache=tmp_path).status == "ready"


def test_official_pipeline_result_preserves_layout_tables_and_markdown() -> None:
    class Result:
        json = {
            "res": {
                "parsing_res_list": [
                    {
                        "block_bbox": [10, 20, 30, 40],
                        "block_label": "text",
                        "block_content": "مرحبا English",
                        "block_id": 7,
                        "block_order": 0,
                    },
                    {
                        "block_bbox": [[1, 2], [9, 2], [9, 8], [1, 8]],
                        "block_label": "table",
                        "block_content": "| A | B |\n|---|---|\n| 1 | 2 |",
                        "block_id": 8,
                        "block_order": 1,
                    },
                ]
            }
        }
        markdown = {"markdown_texts": "مرحبا English\n\n| A | B |\n|---|---|\n| 1 | 2 |"}

    config = {"max_new_tokens": 4096, "repetition_penalty": 1.1, "max_pixels": 4_000_000}
    output = normalize_official_pipeline_result(Result(), inference_config=config)
    assert output.text == Result.markdown["markdown_texts"]
    assert [block["order"] for block in output.blocks] == [0, 1]
    assert output.blocks[1]["bbox"] == [1.0, 2.0, 9.0, 8.0]
    assert len(output.raw["layout_regions"]) == 2
    assert output.raw["structured_tables"][0]["block_id"] == 8
    assert output.raw["truncated_at_max_new_tokens"] is None
    assert output.raw["truncation_status"].startswith("unknown")
    assert output.raw["inference_config"] == config


def test_official_result_never_invents_missing_reading_order() -> None:
    result = {
        "res": {
            "parsing_res_list": [
                {"block_content": "first", "block_id": 1},
                {"block_content": "second", "block_id": 2, "block_order": 1},
            ]
        }
    }
    output = normalize_official_pipeline_result(result)
    assert output.blocks is None
    assert output.text == ""
    assert output.raw["reading_order_status"] == "incomplete_or_unknown"
    assert output.raw["assembly_warning"]


def test_full_pipeline_session_uses_only_explicit_local_assets(tmp_path, monkeypatch) -> None:
    import sys
    import types

    spec = {item.id: item for item in load_model_specs()}["paddleocr-vl-1.6-full-page"]
    _write_snapshot(tmp_path, spec.repo_id, spec.revision, ["config.json", "model.safetensors"])
    layout = hf_snapshot_dir(spec.layout_repo_id, spec.layout_revision, tmp_path)
    layout.mkdir(parents=True)
    for name in spec.layout_required_files:
        (layout / name).write_text("x", encoding="utf-8")
    # Registered with monkeypatch so the session's setdefault is undone after the test.
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", "unset-by-test")
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME")
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(tmp_path / "results"))
    received: dict = {}

    class FakeResult:
        json = {"res": {"parsing_res_list": []}}
        markdown = {"markdown_texts": "RAW"}

    class FakePipeline:
        def __init__(self, **kwargs):
            received["init"] = kwargs

        def predict(self, **kwargs):
            received["predict"] = kwargs
            return [FakeResult()]

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCRVL=FakePipeline))
    image = tmp_path / "page.png"
    image.write_bytes(b"PNG")
    session = PaddleOCRVLFullPageSession(spec, tmp_path)
    output = session.infer(image, "CANARY-GT-MUST-NOT-REACH-PIPELINE")
    assert output.text == "RAW"
    assert received["init"]["layout_detection_model_dir"] == str(layout)
    assert received["init"]["vl_rec_model_dir"] == str(
        hf_snapshot_dir(spec.repo_id, spec.revision, tmp_path)
    )
    assert received["init"]["use_layout_detection"] is True
    # Vendor baseline: only identity and local paths are passed; every other option
    # keeps the PaddleOCR-VL-1.6 default.
    assert set(received["init"]) == {
        "pipeline_version", "engine", "layout_detection_model_name",
        "layout_detection_model_dir", "vl_rec_model_name", "vl_rec_model_dir",
        "use_layout_detection",
    }
    assert received["predict"] == {"input": str(image)}
    effective = output.raw["inference_config"]
    assert effective["configuration"] == "vendor_default"
    assert effective["vendor_defaults_reference"]["region_max_pixels"] == 1003520
    assert "CANARY" not in json.dumps(received)
    assert os.environ["PADDLE_PDX_CACHE_HOME"].endswith("paddlex_home")

    experiment = {item.id: item for item in load_model_specs()}[
        "paddleocr-vl-1.6-full-page-exp-px4m"
    ]
    received.clear()
    output = PaddleOCRVLFullPageSession(experiment, tmp_path).infer(image, "x")
    assert received["predict"] == {
        "input": str(image), "max_new_tokens": 4096, "max_pixels": 4_000_000,
        "repetition_penalty": 1.1,
    }
    effective = output.raw["inference_config"]
    assert effective["configuration"] == "experimental"
    assert effective["requested_but_ignored_by_local_model"] == {"repetition_penalty": 1.1}


def test_open_session_dispatch_reaches_official_full_page_backend(monkeypatch) -> None:
    spec = {item.id: item for item in load_model_specs()}["paddleocr-vl-1.6-full-page"]
    sentinel = object()
    monkeypatch.setattr(
        backends_module,
        "PaddleOCRVLFullPageSession",
        lambda received, cache: sentinel if received is spec and cache is None else None,
    )
    assert open_session(spec, page_timeout_s=10) is sentinel


def test_worker_can_load_and_run_official_backend_dispatch(monkeypatch, tmp_path) -> None:
    from lab.core.e2 import worker as worker_module

    spec = {item.id: item for item in load_model_specs()}["paddleocr-vl-1.6-full-page"]
    image = tmp_path / "page.png"
    image.write_bytes(b"PNG")

    class FakeSession:
        def metadata(self):
            return {"backend": "paddleocr_vl", "pipeline_mode": "official_full_page"}

        def infer(self, _image, _prompt):
            return InferenceOutput("RAW", {"truncation_status": "unknown"})

        def close(self):
            return None

    messages = []

    class Connection:
        def send(self, value):
            messages.append(value)

        def close(self):
            return None

    monkeypatch.setattr(worker_module, "enforce_offline_runtime", lambda: None)
    monkeypatch.setattr(worker_module, "open_session", lambda *_args, **_kwargs: FakeSession())
    worker_module.run_worker(
        Connection(), spec.model_dump(mode="json"), [{"page": 1, "image_path": str(image)}],
        "unused", 10.0, None,
    )
    assert [message["phase"] for message in messages] == [
        "loaded", "page_start", "page", "done"
    ]
    assert messages[2]["text"] == "RAW"


def test_ollama_readiness_without_network(monkeypatch) -> None:
    spec = {item.id: item for item in load_model_specs()}["qwen3-vl-4b-ollama-q4km"]
    calls: list[str] = []

    def fake(url, payload=None, *, timeout=5.0):
        calls.append(url)
        if url.endswith("/api/version"):
            return {"version": "0.33.1"}
        return {"models": []}

    monkeypatch.setattr(backends_module, "local_http_json", fake)
    state = check_readiness(spec, _hardware())
    assert state.status == "not_downloaded"
    assert any("ollama pull" in step for step in state.next_steps)
    assert all(url.startswith("http://127.0.0.1:11434") for url in calls)

    def fake_ready(url, payload=None, *, timeout=5.0):
        if url.endswith("/api/version"):
            return {"version": "0.33.1"}
        return {"models": [{"name": spec.ollama_tag, "digest": "abc", "size": 1}]}

    monkeypatch.setattr(backends_module, "local_http_json", fake_ready)
    ready = check_readiness(spec, _hardware())
    assert ready.status == "ready" and ready.details["digest"] == "abc"

    def down(url, payload=None, *, timeout=5.0):
        raise OSError("connection refused")

    monkeypatch.setattr(backends_module, "local_http_json", down)
    assert check_readiness(spec, _hardware()).status == "server_unavailable"


def test_non_loopback_http_is_refused_before_connecting(monkeypatch) -> None:
    def no_network():
        raise AssertionError("an opener must not be created for a refused URL")

    monkeypatch.setattr(backends_module, "_local_opener", no_network)
    for url in ("https://huggingface.co/api/models", "http://8.8.8.8/", "http://127.0.0.1.nip.io/"):
        with pytest.raises(InferenceError, match="non-loopback"):
            local_http_json(url, {"image": "x"})


def test_ollama_payload_contains_only_image_prompt_and_options() -> None:
    payload = build_ollama_payload("tag", "OCR:", b"\x89PNG", {"temperature": 0})
    assert set(payload) == {"model", "messages", "stream", "options"}
    message = payload["messages"][0]
    assert set(message) == {"role", "content", "images"}
    assert message["content"] == "OCR:" and len(message["images"]) == 1
    assert payload["stream"] is False


def test_remote_backend_is_refused_everywhere(settings) -> None:
    remote = mock_spec().model_copy(update={"backend": "remote", "weights_kind": "full", "mock_response": None})
    assert check_readiness(remote, _hardware()).status == "remote_disabled"
    with pytest.raises(RemoteInferenceDisabled):
        open_session(remote, page_timeout_s=1)
    manager = WorkspaceManager(settings)
    runner = E2Runner(manager, settings, hardware=_hardware())
    with pytest.raises(E2RunRefused, match="Remote"):
        runner.run("ws_x", None, Path("x.pdf"), remote, [1])  # type: ignore[arg-type]


# ------------------------------------------------------------- rendering


def test_render_is_deterministic_and_respects_dpi_and_cap(tmp_path: Path) -> None:
    pdf = DATA / "text.pdf"
    first = render_pdf_page(pdf, 1, tmp_path / "a", dpi=100)
    second = render_pdf_page(pdf, 1, tmp_path / "b", dpi=100)
    assert first.sha256 == second.sha256
    assert first.dpi == 100
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf)
    width_pt, height_pt = document[0].get_size()
    document.close()
    assert abs(first.width - width_pt * 100 / 72) <= 1
    assert abs(first.height - height_pt * 100 / 72) <= 1
    capped = render_pdf_page(pdf, 1, tmp_path / "c", dpi=600, max_long_edge=500)
    assert max(capped.width, capped.height) <= 500
    with pytest.raises(RenderError):
        render_pdf_page(pdf, 999, tmp_path / "d")


# ------------------------------------------------------------- runner (mock)


def _workspace_with_gt(settings):
    manager = WorkspaceManager(settings)
    workspace = manager.create()
    ingest = IngestService(manager, settings)
    with (DATA / "text.pdf").open("rb") as handle:
        document = ingest.ingest(workspace.workspace_id, handle, "text.pdf")
    payload = {
        "pages": [
            {"page_number": page, "full_text_in_reading_order": f"{CANARY} page {page}"}
            for page in range(1, document.page_count + 1)
        ]
    }
    ground_truth = GroundTruthStore(manager).import_json(
        workspace.workspace_id, document, json.dumps(payload, ensure_ascii=False), verified=True, reviewer="t"
    )
    source = ingest.original_path(workspace.workspace_id, document.doc_id)
    return manager, workspace.workspace_id, document, source, ground_truth


def test_mock_run_receives_only_image_and_static_prompt(settings) -> None:
    manager, workspace_id, document, source, ground_truth = _workspace_with_gt(settings)
    response = "  نص ١٢٣ exact\n| a | b |  "
    spec = mock_spec(response=response)
    runner = E2Runner(manager, settings, hardware=_hardware())
    with pytest.raises(E2RunRefused, match="Mock"):
        runner.run(workspace_id, document, source, spec, [1])
    run = runner.run(workspace_id, document, source, spec, [1], allow_mock=True)
    assert run.status == "ok"
    assert run.pages[0].text == response  # exact RAW response, not stripped
    e2 = run.environment["e2"]
    assert is_mock_run(run) and e2["is_mock"] is True
    received = e2["page_raw"]["1"]["response"]["received"]
    assert received["prompt"] == get_prompt(spec.prompt_id).text
    assert received["image_sha256"] == e2["images"][0]["sha256"]
    assert CANARY not in json.dumps(run.model_dump(mode="json"), ensure_ascii=False)
    assert run.adapter.task == "ocr" and run.adapter.model_id == "mock/none@mock"
    assert run.adapter.params["prompt_sha256"] == get_prompt(spec.prompt_id).sha256
    assert e2["input_policy"].startswith("rendered page image")


def test_page_timeout_kills_worker_and_reports_structured_error(settings) -> None:
    manager, workspace_id, document, source, _ = _workspace_with_gt(settings)
    fast = replace(settings, e2_page_timeout_s=1, e2_load_timeout_s=60)
    runner = E2Runner(manager, fast, hardware=_hardware())
    run = runner.run(workspace_id, document, source, mock_spec(delay_s=30), [1], allow_mock=True)
    assert run.pages[0].status == "error"
    assert run.pages[0].error.startswith("page_timeout")
    assert run.status == "error"
    assert run.environment["e2"]["errors"][0]["code"] == "page_timeout"


def test_backend_failure_is_reported_per_page(settings) -> None:
    manager, workspace_id, document, source, _ = _workspace_with_gt(settings)
    runner = E2Runner(manager, settings, hardware=_hardware())
    run = runner.run(workspace_id, document, source, mock_spec(fail=True), [1], allow_mock=True)
    assert run.pages[0].status == "error"
    assert run.pages[0].error.startswith("backend_error")
    assert run.environment["e2"]["errors"][0]["type"] == "InferenceError"


def test_unready_model_is_refused_with_readiness(settings, monkeypatch, tmp_path) -> None:
    manager, workspace_id, document, source, _ = _workspace_with_gt(settings)
    monkeypatch.setattr(backends_module, "package_version", lambda name: None)
    runner = E2Runner(manager, settings, hardware=_hardware(), cache=tmp_path)
    spec = {item.id: item for item in load_model_specs()}["paddleocr-vl-1.6"]
    with pytest.raises(E2RunRefused) as error:
        runner.run(workspace_id, document, source, spec, [1])
    assert error.value.readiness.status == "not_installed"


# ------------------------------------------------------------- metrics


TABLE = GroundTruthTable(
    table_id="t1",
    columns=["Region", "Revenue > H1", "Revenue > H2"],
    rows=[["Gulf", "9.84", "11.27"], ["Levant", "5.12", "5.86"]],
)


def _gt_page(text: str = "Intro line", tables=None) -> GroundTruthPage:
    return GroundTruthPage(
        page=1, text=text, status="verified", seeded_from=None, verified_by="t",
        verified_at=None, notes=None, tables=tables if tables is not None else [TABLE],
    )


def test_markdown_table_associates_every_data_cell() -> None:
    output = "| Region | H1 | H2 |\n|---|---|---|\n| Gulf | 9.84 | 11.27 |\n| Levant | 5.12 | 5.86 |"
    assert parse_hypothesis_tables(output)[0][1] == ["Gulf", "9.84", "11.27"]
    scores = score_tables([TABLE], output)
    assert (scores.cells_exact, scores.cells_reference) == (6, 6)
    assert (scores.data_cells_associated, scores.data_cells_reference) == (4, 4)


def test_exact_cells_without_structure_are_not_association() -> None:
    output = "Gulf 9.84 11.27\nLevant 5.12 5.86"
    scores = score_tables([TABLE], output)
    assert scores.cells_exact == 6
    assert scores.data_cells_associated == 0 and scores.tables_detected == 0


def test_swapped_columns_are_caught_by_header_mapping() -> None:
    output = "| Region | H2 | H1 |\n|---|---|---|\n| Gulf | 11.27 | 9.84 |\n| Levant | 5.86 | 5.12 |"
    assert score_tables([TABLE], output).data_cells_associated == 4
    wrong = "| Region | H1 | H2 |\n|---|---|---|\n| Gulf | 11.27 | 9.84 |\n| Levant | 5.12 | 5.86 |"
    assert score_tables([TABLE], wrong).data_cells_associated == 2


def test_visual_rtl_row_order_and_html_tables_are_parsed() -> None:
    rtl = "| 11.27 | 9.84 | Gulf |\n| 5.86 | 5.12 | Levant |"
    assert score_tables([TABLE], rtl).data_cells_associated == 4
    html = (
        "<table><tr><th>Region</th><th>H1</th><th>H2</th></tr>"
        "<tr><td>Gulf</td><td>9.84</td><td>11.27</td></tr></table>"
    )
    assert score_tables([TABLE], html).data_cells_associated == 2


def test_mixed_direction_and_missing_extra_words() -> None:
    page = _gt_page("مرحبا CER 0.05 هنا\nEnglish only\nعربي فقط", tables=[])
    extras = page_extras(page, "مرحبا CER 0.05 هنا\nEnglish only\nhallucinated")
    assert (extras.mixed_lines_exact, extras.mixed_lines_reference) == (1, 1)
    assert extras.missing_words == 2  # "عربي" "فقط"
    assert extras.extra_words == 1  # "hallucinated"


def test_repetition_metric_flags_loops_without_changing_raw() -> None:
    raw = "بداية " + "٢٥،٥ " * 20
    before = raw[:]
    stats = repetition_stats(raw)
    assert stats["flagged"] is True
    assert stats["max_consecutive_identical_tokens"] == 20
    assert raw == before
    assert repetition_stats("Arabic text and English text 123")["flagged"] is False
    legitimate = "\n".join(["Quarterly report", "value 1", "Quarterly report", "value 2"] * 3)
    assert repetition_stats(legitimate)["flagged"] is False


def test_markup_strip_is_a_separate_run_and_raw_is_unchanged(settings) -> None:
    manager, workspace_id, document, source, ground_truth = _workspace_with_gt(settings)
    raw = E2Runner(manager, settings, hardware=_hardware()).run(
        workspace_id, document, source, mock_spec(response="# Title\n| **a** | b |\n|---|---|"), [1],
        allow_mock=True,
    )
    before = raw.model_dump_json()
    derived = derive_markup_strip(raw)
    assert raw.model_dump_json() == before
    assert derived.variant == MARKUP_VARIANT and derived.raw_run_id == raw.run_id
    assert derived.pages[0].text == "Title\na b"
    with pytest.raises(ValueError):
        derive_markup_strip(derived)
    summary = summarize_run(derived, ground_truth, table_source=raw)
    assert summary["is_mock"] is True
    assert CANARY not in json.dumps(summary, ensure_ascii=False)
    assert summary["metrics"]["ref_chars"] > 0


def test_strip_markup_keeps_content_characters() -> None:
    assert strip_markup("<b>٣٣٫٨٤٪</b> ≥ 5") == " ٣٣٫٨٤٪  ≥ 5"


# ------------------------------------------------------------- ground truth tables


def test_ground_truth_tables_import_and_ragged_rows_are_rejected(settings) -> None:
    manager = WorkspaceManager(settings)
    workspace = manager.create()
    ingest = IngestService(manager, settings)
    with (DATA / "text.pdf").open("rb") as handle:
        document = ingest.ingest(workspace.workspace_id, handle, "text.pdf")
    table = {"id": "t4_1", "columns": ["A", "B"], "rows": [["x", "1"]]}
    payload = {"pages": [{"page_number": 1, "full_text_in_reading_order": "x", "tables": [table]}]}
    store = GroundTruthStore(manager)
    ground_truth = store.import_json(workspace.workspace_id, document, json.dumps(payload))
    assert ground_truth.pages[0].tables[0].rows == [["x", "1"]]
    table["rows"] = [["x"]]
    with pytest.raises(GroundTruthError, match="one cell per column"):
        store.import_json(workspace.workspace_id, document, json.dumps(payload))


# ------------------------------------------------------------- report


def test_report_labels_mock_rows_and_never_scores_unready_models() -> None:
    row = {
        "label": "MOCK m", "variant": "raw", "kind": "MOCK architecture test", "is_mock": True,
        "metrics": {
            "char_edits": 1, "ref_chars": 2, "cer": 0.5, "word_edits": 1, "ref_words": 1, "wer": 1.0,
            "num_correct": 0, "num_ref_count": 0, "curated_num_text_correct": 0,
            "curated_num_text_ref": 0, "curated_num_table_correct": 0, "curated_num_table_ref": 0,
            "date_correct": 0, "date_ref_count": 0, "identifier_correct": 0,
            "identifier_ref_count": 0, "operator_correct": 0, "operator_ref_count": 0,
            "reading_order_tau": None, "reading_order_matched_lines": 0, "reading_order_ref_lines": 0,
        },
        "tables": {"cells_exact": 0, "cells_reference": 0, "data_cells_associated": 0,
                   "data_cells_reference": 0, "tables_detected": 0},
        "mixed_direction": {"exact": 0, "reference": 0, "char_edits": 0, "ref_chars": 0},
        "words": {"missing": 0, "reference": 0, "extra": 0},
        "timing": {"page_ms_mean": None}, "memory": {"peak_rss_mb": None},
    }
    result = {
        "language": "English", "pages": [1], "verified_pages": 1, "rows": [row],
        "not_run": [{"label": "Qwen", "status": "not_downloaded", "reasons": ["Weights missing"],
                     "next_steps": ["ollama pull x"]}],
    }
    report = render_report([result], _hardware().to_dict())
    assert "1/2=0.5000" in report
    assert "MOCK rows are architecture tests only" in report
    assert "**Qwen** — `not_downloaded`" in report and "Next step: ollama pull x" in report


# ------------------------------------------------------------- UI


def test_e2_page_shows_status_and_never_enables_unready_inference(settings, monkeypatch) -> None:
    import sys

    from streamlit.testing.v1 import AppTest

    # AppTest replaces sys.modules["__main__"] with the page script and leaves it
    # there; spawn-based workers in later tests would then re-run the page.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setenv("LAB_TMP_DIR", settings.tmp_dir)
    manager = WorkspaceManager(settings)
    workspace = manager.create()
    ingest = IngestService(manager, settings)
    with (DATA / "text.pdf").open("rb") as handle:
        ingest.ingest(workspace.workspace_id, handle, "text.pdf")
    page = Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "7_OCR_VLM_Evaluation.py"
    app = AppTest.from_file(page, default_timeout=60)
    app.session_state["workspace_id"] = workspace.workspace_id
    app.session_state["e2_hardware"] = _hardware(ram=16.0)
    app.run()
    assert not app.exception
    # The simple full-PDF view is the default; local inference is in the advanced view.
    next(item for item in app.radio if item.label == "View").set_value(
        "Advanced: single page, local models, manual review").run()
    assert not app.exception
    warnings = " ".join(item.value for item in app.warning)
    assert "Missing Ground Truth" in warnings
    # The button is enabled exactly when the selected model's readiness allows it.
    first = load_model_specs()[0]
    button = next(item for item in app.button if item.label == "Run local inference")
    assert button.disabled is (not check_readiness(first, _hardware(ram=16.0)).runnable)
    assert "MOCK" not in " ".join(option for option in app.selectbox[1].options)
    mode = next(item for item in app.selectbox if item.label == "Inference mode")
    assert "Official full-page pipeline" in " ".join(mode.options)
    mode.select("Official full-page pipeline - vendor-default configuration (baseline)")
    app.run()
    assert not app.exception
    button = next(item for item in app.button if item.label == "Run local inference")
    assert button.disabled is True
    assert any("133,289,670 bytes" in item.value for item in app.text)


def test_ollama_session_sends_one_image_and_keeps_exact_response(monkeypatch, tmp_path) -> None:
    from lab.core.e2.backends import OllamaSession

    spec = {item.id: item for item in load_model_specs()}["qwen3-vl-4b-ollama-q4km"]
    image = tmp_path / "p.png"
    image.write_bytes(b"\x89PNG-bytes")
    sent: list[tuple[str, dict | None]] = []

    def fake(url, payload=None, *, timeout=5.0):
        sent.append((url, payload))
        if url.endswith("/api/version"):
            return {"version": "0.33.1"}
        if url.endswith("/api/tags"):
            return {"models": [{"name": spec.ollama_tag, "digest": "sha256:d",
                                "details": {"quantization_level": "Q4_K_M"}}]}
        if url.endswith("/api/ps"):
            return {"models": [{"name": spec.ollama_tag, "size": 10, "size_vram": 0}]}
        return {"message": {"role": "assistant", "content": "  ٣٣٫٨٤٪ ≥ x  "},
                "prompt_eval_count": 7, "eval_count": 9, "done_reason": "stop"}

    monkeypatch.setattr(backends_module, "local_http_json", fake)
    session = OllamaSession(spec, timeout_s=5)
    assert session.metadata()["digest"] == "sha256:d"
    assert session.metadata()["quantization"] == "Q4_K_M"
    output = session.infer(image, get_prompt(spec.prompt_id).text)
    assert output.text == "  ٣٣٫٨٤٪ ≥ x  "
    assert (output.prompt_tokens, output.completion_tokens) == (7, 9)
    assert output.raw["server_memory"] == [{"size": 10, "size_vram": 0}]
    chat = [payload for url, payload in sent if url.endswith("/api/chat")]
    assert len(chat) == 1
    assert chat[0]["messages"][0]["content"] == get_prompt(spec.prompt_id).text
    assert len(chat[0]["messages"][0]["images"]) == 1
    assert all(url.startswith("http://127.0.0.1:11434/") for url, _ in sent)


# ------------------------------------------------------------- persistent library


def test_store_is_write_once_and_keeps_incomplete_runs(settings, tmp_path) -> None:
    from lab.core.e2.library import build_comparison
    from lab.core.e2.store import ResultStore, StoreError

    manager, workspace_id, document, source, ground_truth = _workspace_with_gt(settings)
    store = ResultStore(tmp_path / "results")
    runner = E2Runner(manager, settings, hardware=_hardware())
    spec = mock_spec(response="MOCK")
    saved: list[int] = []
    handle = store.begin(adapter_id="e2.mock", revision="mock", run_id="run_a", started={"language": "English"})

    def on_page(page, record):
        saved.append(page)
        handle.write_page(page, record)

    run = runner.run(workspace_id, document, source, spec, [1], allow_mock=True, run_id="run_a", on_page=on_page)
    assert saved == [1] and run.run_id == "run_a"
    with pytest.raises(StoreError):
        handle.write_page(1, {"text": "overwrite"})
    with pytest.raises(StoreError):
        store.begin(adapter_id="e2.mock", revision="mock", run_id="run_a", started={})
    # Before finalize the run is listed as incomplete, never silently dropped.
    assert store.runs()[0]["final"]["status"] == "incomplete"
    summary = summarize_run(run, ground_truth)
    directory = handle.finalize(run, summaries=[summary], language="English", status="completed")
    final = json.loads((directory / "final.json").read_text(encoding="utf-8"))
    assert {"run.json", "started.json", "pages/p0001.json", "summary.json"} <= set(final["files_sha256"])
    assert json.loads((directory / "pages" / "p0001.json").read_text(encoding="utf-8"))["text"] == "MOCK"
    with pytest.raises(StoreError):
        handle.finalize(run, summaries=[summary], language="English", status="completed")
    index = (store.root / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index) == 1 and json.loads(index[0])["status"] == "completed"
    table = build_comparison(store, load_model_specs(), None)
    for spec_item in load_model_specs():
        assert f"| {spec_item.label} | Arabic | not yet tested |" in table
    first = store.write_report("comparison", table)
    second = store.write_report("comparison", table)
    assert first != second and first.is_file() and second.is_file()


def test_storage_revision_keys_keep_collision_resistant_identity(tmp_path) -> None:
    from lab.core.e2.store import ResultStore

    store = ResultStore(tmp_path / "results")
    common = "revision-" + "a" * 100
    first = store.begin(
        adapter_id="e2.model", revision=common + "-one", run_id="run_one", started={}
    )
    second = store.begin(
        adapter_id="e2.model", revision=common + "-two", run_id="run_two", started={}
    )
    assert first.directory.parent != second.directory.parent
    first_started = json.loads((first.directory / "started.json").read_text(encoding="utf-8"))
    assert first_started["storage_identity"]["revision"] == common + "-one"
    assert len(first.directory.parent.name) <= 80


def test_manual_reviews_are_append_only_and_link_exact_run_page_and_config(tmp_path) -> None:
    from lab.core.e2.store import ResultStore

    store = ResultStore(tmp_path / "results")
    review = {
        "run_id": "run_exact",
        "page": 2,
        "decision": "FAIL",
        "notes": "numeric value differs",
        "error_categories": ["numeric_error"],
        "adapter_id": "e2.model",
        "model_revision": "org/model@" + "a" * 40,
        "inference_config": {"max_new_tokens": 4096},
    }
    first = store.write_manual_review(review)
    second = store.write_manual_review(review | {"decision": "NEEDS_REVIEW"})
    assert first != second and first.is_file() and second.is_file()
    history = store.manual_reviews(run_id="run_exact", page=2)
    assert [item["decision"] for item in history] == ["FAIL", "NEEDS_REVIEW"]
    assert all(item["review_type"] == "manual_user_review" for item in history)
    assert all(item["automated_score"] is False for item in history)


def test_manual_review_order_survives_identical_timestamps(tmp_path, monkeypatch) -> None:
    from datetime import datetime as real_datetime

    import lab.core.e2.store as store_module

    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return real_datetime(2026, 11, 17, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(store_module, "datetime", FrozenClock)
    store = store_module.ResultStore(tmp_path / "results")
    base = {
        "run_id": "run_tick",
        "page": 1,
        "notes": "",
        "error_categories": [],
        "adapter_id": "e2.model",
        "model_revision": "org/model@" + "b" * 40,
        "inference_config": {},
    }
    decisions = ["FAIL", "NEEDS_REVIEW", "PASS", "FAIL", "PASS"]
    paths = [store.write_manual_review(base | {"decision": item}) for item in decisions]
    assert len(set(paths)) == len(decisions)
    history = store.manual_reviews(run_id="run_tick", page=1)
    assert [item["decision"] for item in history] == decisions
    assert [item["sequence"] for item in history] == [1, 2, 3, 4, 5]


def test_runner_preflight_rejects_stale_or_failed_evidence(settings, monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone

    from lab.core.e2 import runner as runner_module

    spec = {item.id: item for item in load_model_specs()}["paddleocr-vl-1.6"]
    manifest = {
        "manifest_id": "m",
        "_manifest_sha256": "h",
        "components": [{"component_id": "vl", "repo_id": spec.repo_id, "revision": spec.revision}],
    }
    monkeypatch.setattr(runner_module, "load_manifest", lambda: manifest)
    cache = Path(settings.tmp_dir) / "verified-cache"
    runner = E2Runner(WorkspaceManager(settings), settings, hardware=_hardware(), cache=cache)
    verified_root = str(hf_snapshot_dir(spec.repo_id, spec.revision, cache))
    base = {
        "deployment_id": spec.id,
        "manifest_id": "m",
        "manifest_sha256": "h",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "DEPENDENCIES_VERIFIED",
        "errors": [],
        "warnings": ["wheel_hashes_unverified: expected"],
        "components": [
            {"component_id": "vl", "status": "HASH_VERIFIED", "errors": [], "root": verified_root}
        ],
        "runtime": {"ok": True},
        "hardware": {"ok": True},
    }
    assert runner.verified_preflight(spec, base) is base
    stale = base | {
        "checked_at": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    }
    with pytest.raises(E2RunRefused, match="preflight did not pass"):
        runner.verified_preflight(spec, stale)
    corrupt = base | {"errors": ["hash_mismatch"]}
    with pytest.raises(E2RunRefused, match="hash_mismatch"):
        runner.verified_preflight(spec, corrupt)
    # The preflight passed, but for a different directory than the one that would load.
    elsewhere = base | {
        "components": [base["components"][0] | {"root": str(Path(settings.tmp_dir) / "other")}]
    }
    with pytest.raises(E2RunRefused, match="did not hash"):
        runner.verified_preflight(spec, elsewhere)


def test_library_shows_unknown_truncation_instead_of_zero(tmp_path) -> None:
    from lab.core.e2.library import _truncation_cell, truncated_pages

    run_dir = tmp_path / "official"
    (run_dir / "pages").mkdir(parents=True)
    record = {"page": 1, "text": "x", "raw": {"truncated_at_max_new_tokens": None}}
    (run_dir / "pages" / "p0001.json").write_text(json.dumps(record), encoding="utf-8")
    assert truncated_pages(run_dir) is None
    assert _truncation_cell(str(run_dir)) == "unknown (1 pages)"


def test_library_reports_truncated_pages(tmp_path) -> None:
    from lab.core.e2.library import truncated_pages

    run_dir = tmp_path / "run"
    (run_dir / "pages").mkdir(parents=True)
    for page, flag in ((1, False), (2, True), (3, True)):
        record = {"page": page, "text": "x", "raw": {"truncated_at_max_new_tokens": flag}}
        (run_dir / "pages" / f"p{page:04d}.json").write_text(json.dumps(record), encoding="utf-8")
    assert truncated_pages(run_dir) == [2, 3]
    text_layer = tmp_path / "text"
    (text_layer / "pages").mkdir(parents=True)
    (text_layer / "pages" / "p0001.json").write_text(json.dumps({"page": 1, "raw": {}}), encoding="utf-8")
    assert truncated_pages(text_layer) is None
