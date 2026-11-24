"""Automatic Extraction: configuration, router, region selection, reading order, de-duplication and
validation. Pure functions and synthetic inputs only (no model, no network)."""

from __future__ import annotations

import pytest

from lab.core.auto import config as auto_config
from lab.core.auto import dedup, regions, router
from lab.core.auto import validate as validation
from lab.core.hybrid import layout_regions

# ------------------------------------------------------------------ configuration


def test_defaults_are_the_proven_benchmark_settings(tmp_path) -> None:
    config = auto_config.load_config(environ={auto_config.CONFIG_ENV: str(_write(tmp_path / "empty.toml", ""))})
    qari = config.proven.qari
    assert (qari.base_model, qari.adapter, qari.prompt) == (
        "Qwen/Qwen3-VL-4B-Instruct", "NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct", "Free OCR.")
    assert (qari.max_new_tokens, qari.max_time_crop_s, qari.max_time_page_s) == (1536, 30.0, 60.0)
    assert (qari.loop_max_ngram, qari.loop_repeats, qari.retry_repetition_penalty) == (40, 5, 1.1)
    assert qari.do_sample is False and qari.retries_after_loop_timeout_or_limit == 1
    layout = config.proven.layout
    assert (layout.model_name, layout.crop_margin_px, layout.crop_format, layout.crop_pixels) == (
        "PP-DocLayout_plus-L", 60, "png", "native")
    assert not layout.unwarping and not layout.orientation_correction
    assert config.qari.max_image_pixels is None  # no resizing by default
    assert config.qari.concurrency == 1 and config.qari.service_url is None
    assert auto_config.proven_differences(config) == {}


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_environment_overrides_the_file_and_is_typed(tmp_path) -> None:
    file = _write(tmp_path / "auto.toml", '[qari]\nconcurrency = 2\nservice_url = "http://127.0.0.1:9000"\n')
    config = auto_config.load_config(file, environ={"QARI_CONCURRENCY": "3", "LAYOUT_DEVICE": "gpu:0",
                                                    "QARI_DEPLOYMENT_MODE": "internal",
                                                    "QARI_AUTOMATIC_ENABLED": "true"})
    assert config.qari.concurrency == 3 and config.qari.service_url == "http://127.0.0.1:9000"
    assert config.layout.device == "gpu:0" and config.qari.deployment_mode == "internal"
    with pytest.raises(auto_config.ConfigError):
        auto_config.load_config(file, environ={"QARI_AUTOMATIC_ENABLED": "maybe"})
    with pytest.raises(auto_config.ConfigError):
        auto_config.load_config(file, environ={"QARI_DEPLOYMENT_MODE": "public"})
    with pytest.raises(auto_config.ConfigError):
        auto_config.load_config(file, environ={"LAYOUT_DEVICE": "toaster"})


def test_proven_overrides_need_explicit_permission_and_are_recorded(tmp_path) -> None:
    file = _write(tmp_path / "auto.toml", "[proven_overrides.qari]\nmax_new_tokens = 2048\n")
    with pytest.raises(auto_config.ConfigError, match="AUTO_ALLOW_PROVEN_OVERRIDES"):
        auto_config.load_config(file, environ={})
    config = auto_config.load_config(file, environ={"AUTO_ALLOW_PROVEN_OVERRIDES": "true"})
    assert config.proven.qari.max_new_tokens == 2048
    assert auto_config.proven_differences(config) == {"qari.max_new_tokens": {"proven": 1536, "configured": 2048}}


def test_tokens_never_come_from_files_or_appear_in_snapshots(tmp_path, monkeypatch) -> None:
    with pytest.raises(auto_config.ConfigError, match="tokens must not be stored"):
        auto_config.load_config(_write(tmp_path / "bad.toml", '[qari]\ntoken = "abc"\n'), environ={})
    monkeypatch.setenv("QARI_SERVICE_TOKEN", "secret-value-123")
    config = auto_config.load_config(_write(tmp_path / "ok.toml", ""), environ={})
    snapshot = config.snapshot()
    assert config.qari_token() == "secret-value-123" and snapshot["qari"]["token_present"] is True
    assert "secret-value-123" not in str(snapshot)


def test_missing_explicit_config_file_is_an_error(tmp_path) -> None:
    with pytest.raises(auto_config.ConfigError, match="not found"):
        auto_config.load_config(environ={auto_config.CONFIG_ENV: str(tmp_path / "nope.toml")})


# ------------------------------------------------------------------ router


def _page(category="A", images=(), raw_text="some text", page=1):
    return {"page": page, "category": category, "category_reason": f"category {category}", "raw_text": raw_text,
            "images": {"images": [{"bbox": [0, 0, 595, 842 * share], "area_share": share} for share in images]},
            "table_count": 0, "arabic_letter_share": 0.0, "library_error": None}


VISIBLE = {"visible_chars": 100, "invisible_chars": 0}
INVISIBLE = {"visible_chars": 0, "invisible_chars": 100}


@pytest.mark.parametrize("page, visibility, route, rule", [
    (_page("C", images=(1.0,), raw_text=""), VISIBLE, router.SCANNED, "S1"),
    (_page("C", images=(1.0,)), VISIBLE, router.SCANNED, "S2"),
    (_page("B", images=(0.95,)), VISIBLE, router.SCANNED, "S2"),
    (_page("A", images=(1.0,)), INVISIBLE, router.SCANNED, "S3"),
    (_page("C"), VISIBLE, router.SCANNED, "S4"),
    (_page("A", images=(1.0,)), VISIBLE, router.MIXED, "M1"),
    (_page("A", images=(0.3,)), VISIBLE, router.MIXED, "M2"),
    (_page("A", images=(0.01,)), VISIBLE, router.DIGITAL, "D1"),
    (_page("B"), VISIBLE, router.DIGITAL, "D1"),
])
def test_router_rules(page, visibility, route, rule) -> None:
    result = router.classify_page(page, visibility)
    assert (result["route"], result["rule"]) == (route, rule)
    assert result["reason"] and result["evidence"]["phase1_category"] == page["category"]


def test_category_b_keeps_native_text_with_rejoin_guidance_and_embedded_scans_are_listed() -> None:
    assert router.classify_page(_page("B"), VISIBLE)["category_b_rejoin"] is True
    assert router.classify_page(_page("B", images=(1.0,)), VISIBLE)["category_b_rejoin"] is False  # scanned
    mixed = router.classify_page(_page("A", images=(0.6, 0.1)), VISIBLE)
    assert mixed["route"] == router.MIXED and len(mixed["embedded_scan_regions"]) == 1


# ------------------------------------------------------------------ regions


IMAGE_PX, PAGE_PT = [1190, 1684], [0, 0, 595, 842]


def _box(label, px, index):
    return {"label": label, "bbox_px": list(px), "block_index": index, "warnings": [], "score": 0.9}


def _select(boxes, **options):
    return regions.select_regions(1, boxes, image_px=IMAGE_PX, image_bbox_pt=PAGE_PT, labels=("chart", "image", "table"),
                                  margin_px=60, **options)


def test_nested_box_outer_is_automatic_and_inner_goes_to_review() -> None:
    selection = _select([_box("chart", [100, 200, 700, 600], 0), _box("image", [150, 250, 400, 450], 1)])
    by_index = {r["block_index"]: r for r in selection["regions"]}
    assert by_index[0]["decision"] == regions.AUTO
    assert by_index[1]["decision"] == regions.REVIEW and by_index[1]["status"] == "contained"
    assert regions.overlapping_auto(selection["regions"]) == []


def test_partial_overlap_and_small_boxes_go_to_review_and_duplicates_are_recorded() -> None:
    selection = _select([_box("chart", [100, 200, 700, 600], 0), _box("table", [500, 500, 1000, 900], 1),
                         _box("chart", [100, 200, 700, 600], 2), _box("image", [1000, 50, 1100, 120], 3)])
    decisions = {r["block_index"]: r["decision"] for r in selection["regions"]}
    assert decisions == {0: regions.REVIEW, 1: regions.REVIEW, 2: regions.REMOVED, 3: regions.REVIEW}


def test_crop_uses_the_proven_60px_margin_on_native_pixels_and_keeps_the_detected_box() -> None:
    (region,) = _select([_box("chart", [100, 200, 700, 600], 0)])["regions"]
    assert region["pixel_box"] == [40, 140, 760, 660] and region["margin_px"] == 60
    assert region["box_pt_before_margin"] == [50.0, 100.0, 350.0, 300.0]


def test_title_overlapping_the_top_is_attached_and_policy_review_all() -> None:
    selection = _select([_box("chart", [100, 300, 700, 600], 0), _box("figure_title", [120, 270, 680, 320], 1)])
    (region,) = selection["regions"]
    assert region["title_attachments"] and region["box_pt_before_margin"][1] == 135.0
    assert selection["attached_titles"][0]["block_index"] == 1
    assert _select([_box("chart", [100, 200, 700, 600], 0)], policy="review_all")["regions"][0]["decision"] == regions.REVIEW


def test_reading_order_rows_and_direction() -> None:
    items = [{"id": "right", "bbox_pt": [300, 100, 500, 200]}, {"id": "left", "bbox_pt": [50, 110, 250, 190]},
             {"id": "below", "bbox_pt": [50, 300, 500, 400]}]
    assert [i["id"] for i in regions.reading_order(items, rtl=True)] == ["right", "left", "below"]
    assert [i["id"] for i in regions.reading_order(items, rtl=False)] == ["left", "right", "below"]


def test_detector_maps_predictor_boxes_and_flags_out_of_bounds() -> None:
    from PIL import Image

    class Predictor:
        def predict(self, array, batch_size=1):
            assert array.shape == (100, 200, 3)
            return [{"res": {"boxes": [{"label": "chart", "coordinate": [10, 10, 50, 50], "score": 0.9},
                                       {"label": "table", "coordinate": [10, 10, 500, 50], "score": 0.8}]}}]

    detector = regions.LayoutDetector(model_name="PP-DocLayout_plus-L", model_dir=None, device="cpu",
                                      threshold=None, predictor=Predictor())
    boxes = detector.detect(Image.new("RGB", (200, 100)))
    assert boxes[0]["bbox_px"] == [10.0, 10.0, 50.0, 50.0] and not boxes[0]["warnings"]
    assert "outside" in boxes[1]["warnings"][0]
    assert layout_regions.PP_DOCLAYOUT_DEFAULT_MODEL == "PP-DocLayout_plus-L"


# ------------------------------------------------------------------ de-duplication


def test_arabic_aware_normalization() -> None:
    assert dedup.normalize("إجمالي الإيرادات ١٬٢٨٤") == dedup.normalize("اجمالي الايرادات 1,284")
    assert dedup.normalize("الْعَرَبِيَّة") == dedup.normalize("العربيه")
    assert dedup.normalize("3.01%") != dedup.normalize("3.01‰")


def test_margin_line_removed_only_when_it_matches_an_outside_block() -> None:
    crop = "Figure 2: Revenue by region\nNorth 120\nSouth 95\nSource: annual report 2024"
    outside = [{"id": "b7", "text": "Source: annual report 2024", "bbox_pt": [0, 0, 1, 1], "source": "native"}]
    result = dedup.remove_margin_lines(crop, outside)
    assert "Source" not in result["text"] and "North 120" in result["text"]
    assert result["removed"][0]["matched_block"] == "b7" and result["removed"][0]["line_number"] == 4
    assert dedup.remove_margin_lines(crop, [])["text"] == crop  # unknown block text: nothing removed
    short = dedup.remove_margin_lines("North 120", [{"id": "b1", "text": "North 120 South 95 West 70",
                                                     "bbox_pt": [0, 0, 1, 1]}])
    assert short["text"] == "North 120"  # a 2-token partial match is not enough


def test_outside_blocks_are_those_outside_the_detected_box_but_inside_the_crop() -> None:
    region = {"box_pt_before_margin": [100, 100, 300, 300], "crop_bbox_pt": [70, 70, 330, 330]}
    blocks = [{"id": "inside", "bbox_pt": [120, 120, 280, 280]}, {"id": "margin", "bbox_pt": [100, 305, 300, 325]},
              {"id": "far", "bbox_pt": [100, 400, 300, 420]}]
    assert [b["id"] for b in dedup.outside_blocks_for(region, blocks)] == ["margin"]


def test_figure_spans_of_three_or_more_tokens_are_removed_from_page_text() -> None:
    page = "Annual results\nRevenue grew in 2024.\nNorth 120 South 95 West 70\nThe end of the report."
    result = dedup.remove_figure_spans(page, {"p1_R0": "North 120 South 95 West 70"})
    assert "North 120" not in result["text"] and "Revenue grew" in result["text"]
    assert result["removed"][0]["matched_region"] == "p1_R0" and result["removed"][0]["tokens"] == 6
    two = dedup.remove_figure_spans("Total 95 is high", {"p1_R0": "Total 95"})
    assert two["text"] == "Total 95 is high" and not two["removed"]


def test_repeated_rows_and_distinct_numbers_are_kept() -> None:
    page = "Q1 sales 120 units\nnote\nQ1 sales 120 units\nQ1 sales 121 units"
    result = dedup.remove_figure_spans(page, {"p1_R0": "Q1 sales 120 units"})
    assert len(result["removed"]) == 1  # the crop has the row once: only one page occurrence is removed
    assert result["text"].count("Q1 sales 120 units") == 1 and "121" in result["text"]


def test_mixed_body_and_figure_line_is_ambiguous_and_kept() -> None:
    page = "As shown North 120 South 95 the north region leads clearly this year"
    result = dedup.remove_figure_spans(page, {"p1_R0": "North 120 South 95"})
    assert result["text"] == page and result["ambiguous_kept"] and not result["removed"]


# ------------------------------------------------------------------ validation


def _result(text="ok text 12%", **final):
    flags = {"loop_stopped": False, "hit_limit": False, "timed_out": False} | final
    attempts = [{"attempt": 1, **flags}]
    return {"state": "completed", "response": {"final_text": text, "final_flags": flags, "attempts": attempts,
                                               "retried": False}}


def test_job_flags() -> None:
    job = {"job_id": "p001_R0", "page": 1, "region_id": "p1_R0"}
    assert validation.job_flags(job, _result()) == []
    assert validation.job_flags(job, _result(text=" "))[0]["code"] == "empty_ocr"
    assert {f["code"] for f in validation.job_flags(job, _result(loop_stopped=True, timed_out=True))} == {
        "loop_stopped", "timed_out"}
    assert validation.job_flags(job, _result(hit_limit=True))[0]["code"] == "hit_limit"
    assert validation.job_flags(job, None)[0]["code"] == "not_processed"
    assert validation.job_flags(job, {"state": "unconfirmed"})[0]["code"] == "ocr_unconfirmed"
    retried = _result()
    retried["response"].update(retried=True, attempts=[{"attempt": 1, "loop_stopped": True}, {"attempt": 2}])
    assert validation.job_flags(job, retried)[0]["code"] == "retried"


def test_region_and_number_flags_never_change_content() -> None:
    region = {"page": 1, "region_id": "p1_R0", "decision": "auto_review", "reasons": ["Needs Review: cut"],
              "suspected_missing_content": [{"block_index": 2}], "proposed_expansion_pt": [0, 0, 1, 1]}
    codes = {f["code"] for f in validation.region_flags(region)}
    assert codes == {"incomplete_region", "processed_needs_review"}
    pending = validation.region_flags({"page": 1, "region_id": "p1_R1", "decision": "review", "reasons": ["small"]})
    assert pending[0]["code"] == "pending_review"
    element = {"id": "p1_e0", "page": 1, "type": "text", "content": "rate ٦,٤ and 3.01‰"}
    report = validation.report([element], [], {}, [], [])
    assert report["flags"][0]["code"] == "number_format" and element["content"] == "rate ٦,٤ and 3.01‰"


def test_legitimately_repeated_numbers_in_body_text_are_kept() -> None:
    page = "Total 120 units sold\nNorth 120 South 95 West 70\nWe sold 120 again and 95 later"
    result = dedup.remove_figure_spans(page, {"p1_R0": "North 120 South 95 West 70"})
    assert result["text"] == "Total 120 units sold\nWe sold 120 again and 95 later"
    assert len(result["removed"]) == 1


def test_identical_table_rows_inside_a_crop_are_never_collapsed() -> None:
    table = "| Q1 | 120 |\n| Q1 | 120 |\n| Q2 | 95 |"
    page = "Body text here\n| Q1 | 120 |\n| Q1 | 120 |\n| Q2 | 95 |"
    result = dedup.remove_figure_spans(page, {"p1_R0": table})
    assert result["text"] == "Body text here" and len(result["removed"]) == 1  # one span covering all rows
    kept = dedup.remove_margin_lines(table, [])
    assert kept["text"] == table  # the crop's own rows are never de-duplicated against each other


# ------------------------------------------------------------------ deployment modes and model location


def test_deployment_mode_defaults_to_local_and_dev_external_needs_admin_hosts(tmp_path) -> None:
    file = _write(tmp_path / "modes.toml", "")
    config = auto_config.load_config(file, environ={})
    assert config.qari.deployment_mode == "local" and config.qari.automatic_enabled is True
    with pytest.raises(auto_config.ConfigError, match="QARI_TRUSTED_DEV_HOSTS"):
        auto_config.load_config(file, environ={"QARI_DEPLOYMENT_MODE": "development_external"})
    dev = auto_config.load_config(file, environ={"QARI_DEPLOYMENT_MODE": "development_external",
                                                 "QARI_TRUSTED_DEV_HOSTS": " Pod-1.example.net , pod-2.example.net"})
    assert dev.qari.trusted_development_hosts == ("pod-1.example.net", "pod-2.example.net")
    with pytest.raises(auto_config.ConfigError, match="production"):
        auto_config.load_config(file, environ={"QARI_DEPLOYMENT_MODE": "development_external",
                                               "QARI_TRUSTED_DEV_HOSTS": "pod-1.example.net",
                                               "AGENTICDOC_ENV": "production"})
    production = auto_config.load_config(file, environ={"AGENTICDOC_ENV": "production",
                                                        "QARI_DEPLOYMENT_MODE": "internal"})
    assert production.qari.deployment_mode == "internal"


def test_retired_trust_settings_fail_loudly(tmp_path) -> None:
    file = _write(tmp_path / "retired.toml", "")
    for name in ("QARI_TRUSTED_INTERNAL", "QARI_REQUIRE_DOCUMENT_AUTHORIZATION"):
        with pytest.raises(auto_config.ConfigError, match="QARI_DEPLOYMENT_MODE"):
            auto_config.load_config(file, environ={name: "true"})
    with pytest.raises(auto_config.ConfigError):
        auto_config.load_config(_write(tmp_path / "old.toml", "[qari]\ntrusted_internal = true\n"), environ={})


def test_windows_layout_model_dir_from_environment_dotenv_and_toml(tmp_path) -> None:
    from pathlib import Path

    windows = r"C:\Users\User\models\PP-DocLayout_plus-L_infer"
    file = _write(tmp_path / "w.toml", "")
    assert auto_config.load_config(file, environ={"LAYOUT_MODEL_DIR": windows}).layout.model_dir == str(Path(windows))
    toml = _write(tmp_path / "lit.toml", "[layout]\nmodel_dir = '" + windows + "'\n")  # TOML literal string
    assert auto_config.load_config(toml, environ={}).layout.model_dir == str(Path(windows))
    dotenv = _write(tmp_path / ".env", "# comment\nLAYOUT_MODEL_DIR=" + windows + "\n"
                    'QARI_SERVICE_URL="http://127.0.0.1:8000"\nQARI_DEPLOYMENT_MODE=local   # loopback\n'
                    "QARI_SERVICE_TOKEN=<placeholder>\n")
    values = auto_config.read_dotenv(dotenv)
    assert values["LAYOUT_MODEL_DIR"] == windows  # backslashes are kept as written
    assert values["QARI_SERVICE_URL"] == "http://127.0.0.1:8000" and values["QARI_DEPLOYMENT_MODE"] == "local"


def test_dotenv_never_overrides_the_real_environment_and_skips_placeholders(tmp_path, monkeypatch) -> None:
    dotenv = _write(tmp_path / ".env", "LAYOUT_DEVICE=gpu:0\nQARI_CONCURRENCY=4\nQARI_SERVICE_TOKEN=<placeholder>\n")
    monkeypatch.setenv(auto_config.DOTENV_ENV, str(dotenv))
    monkeypatch.setenv("LAYOUT_DEVICE", "cpu")
    monkeypatch.delenv("QARI_CONCURRENCY", raising=False)
    monkeypatch.delenv("QARI_SERVICE_TOKEN", raising=False)
    loaded = auto_config.load_dotenv_into_process()
    import os

    assert loaded == ["QARI_CONCURRENCY"] and os.environ["LAYOUT_DEVICE"] == "cpu"
    assert "QARI_SERVICE_TOKEN" not in os.environ
    monkeypatch.delenv("QARI_CONCURRENCY")


def test_missing_layout_model_dir_is_reported_clearly(tmp_path) -> None:
    file = _write(tmp_path / "m.toml", "")
    unset = auto_config.layout_model_status(auto_config.load_config(file, environ={}))
    assert not unset["ok"] and "LAYOUT_MODEL_DIR is not set" in unset["problems"][0]
    missing = auto_config.layout_model_status(auto_config.load_config(
        file, environ={"LAYOUT_MODEL_DIR": str(tmp_path / "no_such_model")}))
    assert not missing["ok"] and "does not exist" in missing["problems"][0]


def test_benchmark_mode_is_unchanged() -> None:
    from pathlib import Path

    from lab.core.hybrid import run as hybrid

    page = (Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "10_Hybrid_PDF_Extraction.py").read_text(
        encoding="utf-8")
    assert '"Benchmark Mode"' in page and "hybrid.APPROVAL_STATEMENT" in page  # its approval rules remain
    assert hybrid.APPROVAL_STATEMENT and hybrid.LAYOUT_STATEMENT
    auto_dir = Path(__file__).resolve().parents[2] / "lab" / "core" / "auto"
    auto_source = "\n".join(p.read_text(encoding="utf-8") for p in auto_dir.glob("*.py"))
    assert "authorization_state" not in auto_source and "approval.json" not in auto_source
