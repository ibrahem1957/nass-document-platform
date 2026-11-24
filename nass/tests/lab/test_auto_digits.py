"""Arabic-Indic digit checks, the report-only second read, and suppression of provably redundant review
placeholders. Synthetic inputs; fake Qari behind httpx.MockTransport; no model, no network."""

from __future__ import annotations

import json

import httpx
import pytest
from PIL import Image

from lab.core.auto import assemble, digits, pipeline, redundancy
from lab.core.auto import config as auto_config
from tests.lab.test_auto_pipeline import BODY_TEXT, _client, ctx  # noqa: F401

KPI = "إجمالي الإيرادات ٣٬٤٥٠"


def _outcome(text):
    return digits.compare(*text)["values"][0]["outcome"]


# ------------------------------------------------------------------ 1-4 second-read comparison


def test_kpi_misread_is_a_digit_disagreement() -> None:
    assert _outcome(("إجمالي الإيرادات ٥٠٣٤٣", KPI)) == "digit_disagreement"
    assert _outcome((KPI, "إجمالي الإيرادات ٥٠٣٤٣")) == "digit_disagreement"


def test_reversed_digit_sequence_is_a_possible_reversal() -> None:
    assert _outcome(("العملاء ٣٤٥٠", "العملاء ٠٥٤٣")) == "possible_digit_reversal"
    assert _outcome(("العملاء ١٢٢١", "العملاء ١٢٢١")) == "second_read_agreement"  # a palindrome is not a reversal


def test_equal_readings_are_agreement_not_proof() -> None:
    result = digits.compare(KPI, "إجمالي الإيرادات ٣٬٤٥٠")
    assert result["counts"]["second_read_agreement"] == 1
    assert "not proof of correctness" in result["note"]
    assert _outcome((KPI, "إجمالي الإيرادات ۳٬۴۵۰")) == "second_read_agreement"  # Persian shapes, same value
    assert _outcome(("النسبة ٤٥٪", "النسبة ٤٥%")) == "second_read_agreement"      # same unit, other glyph


def test_separator_only_differences_stay_visible() -> None:
    assert _outcome((KPI, "إجمالي الإيرادات ٣٤٥٠")) == "separator_difference"
    assert _outcome((KPI, "إجمالي الإيرادات 3,450")) == "separator_difference"  # ٬ (thousands) vs ambiguous ,
    assert _outcome(("النسبة ٤٥٪", "النسبة ٤٥‰")) == "separator_difference"      # unit differs


def test_unalignable_values_are_inconclusive() -> None:
    result = digits.compare("القيمة ١٢٣\nالقيمة ٤٥٦", "١٢٣")
    assert result["counts"]["verification_inconclusive"] == 2


# ------------------------------------------------------------------ 5-7 static checks


def _codes(text):
    return [f["code"] for f in digits.static_flags(text)]


def test_long_ungrouped_arabic_integer_asks_for_verification() -> None:
    assert _codes("الإيرادات ٥٠٣٤٣") == ["arabic_digits_ungrouped"]
    assert _codes(KPI) == [] and _codes("Revenue 50343") == []  # grouped / Western: no alarm
    year = digits.static_flags("عام ١٤٤٥")
    assert [f["code"] for f in year] == ["possible_year"] and year[0]["severity"] == "info"


def test_mixed_digit_scripts_are_flagged() -> None:
    assert "mixed_digit_scripts" in _codes("القيمة ١2٣٤")


def test_irregular_grouping_is_flagged() -> None:
    assert "irregular_grouping" in _codes("القيمة ٣٬٤٥")
    assert "irregular_grouping" in _codes("Value 12,34,567")
    assert "irregular_grouping" in _codes("القيمة ٣٬٤٥٠,٥")  # mixed separator scripts


def test_dates_times_versions_and_identifiers_raise_no_value_alarms() -> None:
    for text in ("التاريخ ٢٠٢٥/٠٣/٠١", "الوقت ١٠:٤٥", "v2.1.3", "الإصدار ٢٫١٫٣", "ISO-9001", "A١٢٣٤٥",
                 "رقم الطلب #١٢٣٤٥٦"):
        assert [f for f in digits.static_flags(text) if f["severity"] == "warning"] == [], text
    kinds = {t["original"]: t["kind"] for t in digits.tokens("v2.1.3 ISO-9001 ١٠:٤٥")}
    assert kinds["2.1.3"] == "version" and kinds["-9001"] == "identifier" and kinds["١٠:٤٥".split(":")[0]] == "time"


# ------------------------------------------------------------------ 8-9 verification requests


class ArabicQari:
    """Fake service: Arabic-digit KPI text for crops; `second` is returned for verification requests."""

    def __init__(self, second=KPI, mode="ok"):
        self.second, self.mode, self.requests = second, mode, []

    def __call__(self, request):
        if request.method == "GET":
            return httpx.Response(404)
        body = json.loads(request.content)
        if self.mode == "down":
            raise httpx.ConnectError("refused", request=request)
        self.requests.append(body)
        verify = ":verify" in body["request_id"]
        text = (self.second if verify else KPI) if body["kind"] == "crop" else BODY_TEXT
        flags = {"loop_stopped": False, "hit_limit": False, "timed_out": False}
        return httpx.Response(200, json={
            "request_id": body["request_id"], "kind": body["kind"], "model": {"adapter": "NAMAA-Space/Qari"},
            "image": {"sha256": body["image_sha256"]}, "final_text": text, "final_attempt": 1, "retried": False,
            "final_flags": flags, "attempts": [{"attempt": 1, "text": text, **flags}], "settings": {}})


def _mock(fake):
    return qari_client_for(fake)


def qari_client_for(fake):
    from lab.core.auto import qari_client

    return qari_client.QariClient("http://127.0.0.1:8000", "tok", transport=httpx.MockTransport(fake))


def _config(ctx, **env):  # noqa: F811
    file = ctx["run_dir"].parent.parent.parent / "verify.toml"
    file.write_text("", encoding="utf-8")
    return auto_config.load_config(file, environ={"AGENTICDOC_RESULTS_DIR": ctx["config"].app.results_dir,
                                                  "QARI_SERVICE_URL": "http://127.0.0.1:8000", **env})


def test_second_read_is_one_extra_request_with_an_upscaled_copy(ctx) -> None:  # noqa: F811
    run_dir, config = ctx["run_dir"], ctx["config"]
    fake = ArabicQari(second="إجمالي الإيرادات ٥٠٣٤٣")
    pipeline.execute(run_dir, config, _mock(fake))
    primary_before = (pipeline.latest(run_dir, "p001_R0") or {})["response"]["final_text"]
    summary = pipeline.verify_digits(run_dir, config, _mock(fake))
    assert summary["eligible"] == 1 and summary["sent"] == 1  # only the crop with Arabic digits (R0)
    verify_requests = [r for r in fake.requests if ":verify" in r["request_id"]]
    assert len(verify_requests) == 1 and verify_requests[0]["kind"] == "crop"
    again = pipeline.verify_digits(run_dir, config, _mock(fake))
    assert again["sent"] == 0 and again["already_verified"] == 1  # at most one extra request per crop
    records = pipeline.verification_records(run_dir, "p001_R0")
    meta = records["attempts"][0]["intent"]["input"]
    original = Image.open(run_dir / "crops" / "p001_R0.png")
    upscaled = Image.open(run_dir / meta["file"])
    assert upscaled.size == (round(original.width * 1.5), round(original.height * 1.5)) and meta["resampling"] == "LANCZOS"
    assert (pipeline.latest(run_dir, "p001_R0") or {})["response"]["final_text"] == primary_before  # untouched
    document = assemble.assemble(run_dir)
    report = document["validation"]["digit_verification"]
    assert report["requests_sent"] == 1 and report["outcomes"]["digit_disagreement"] == 1
    assert any(f["code"] == "digit_disagreement" for f in document["validation"]["flags"])
    element = next(e for e in document["elements"] if e.get("region_id") == "p1_R0")
    assert element["content"] == KPI and element["digit_verification"]["status"] == "compared"
    assert len(pipeline.all_jobs(run_dir)) == 3  # verification is not a primary job


def test_unavailable_service_keeps_primary_and_records_skipped(ctx) -> None:  # noqa: F811
    run_dir, config = ctx["run_dir"], ctx["config"]
    pipeline.execute(run_dir, config, _mock(ArabicQari()))
    summary = pipeline.verify_digits(run_dir, config, _mock(ArabicQari(mode="down")))
    assert summary["sent"] == 0 and "unavailable" in summary["stopped"]
    report = assemble.assemble(run_dir)["validation"]["digit_verification"]
    assert report["verification_skipped"] == 1 and report["requests_sent"] == 0
    assert report["jobs"]["p001_R0"]["status"] == "verification_skipped"
    assert (pipeline.latest(run_dir, "p001_R0") or {})["state"] == "completed"
    assert pipeline.verify_digits(run_dir, config, _mock(ArabicQari()))["sent"] == 1  # nothing was sent: retried


def test_disabled_or_limit_reached_is_recorded(ctx) -> None:  # noqa: F811
    run_dir = ctx["run_dir"]
    pipeline.execute(run_dir, ctx["config"], _mock(ArabicQari()))
    fake = ArabicQari()
    off = pipeline.verify_digits(run_dir, _config(ctx, VERIFY_DIGITS_ENABLED="false"), _mock(fake))
    assert off["sent"] == 0 and "disabled" in off["stopped"] and fake.requests == []
    limit = pipeline.verify_digits(run_dir, _config(ctx, VERIFY_DIGITS_MAX_REQUESTS="0"), _mock(fake))
    assert limit["sent"] == 0 and "limit" in limit["stopped"]
    entry = assemble.assemble(run_dir)["validation"]["digit_verification"]["jobs"]["p001_R0"]
    assert entry["status"] == "verification_skipped" and "limit" in entry["reason"]


def test_crops_without_eastern_digits_are_never_reread(ctx) -> None:  # noqa: F811
    from tests.lab.test_auto_pipeline import FakeQari

    pipeline.execute(ctx["run_dir"], ctx["config"], _client(FakeQari()))  # English chart text
    fake = ArabicQari()
    assert pipeline.verify_digits(ctx["run_dir"], ctx["config"], _mock(fake))["eligible"] == 0
    assert fake.requests == []


# ------------------------------------------------------------------ 10-15 redundant placeholders


def _region(rid, index, status, box, crop=None, decisions=None):
    return {"region_id": rid, "block_index": index, "status": status, "label": "chart", "box_pt_before_margin": box,
            "bbox_pt": box, "crop_bbox_pt": crop or box, "decisions": decisions or [], "reasons": []}


def _contained(outer_index):
    return [{"type": "containment", "decision": "contained; outer preferred", "contained_in": outer_index}]


def _split(inner):
    return [{"type": "containment", "decision": "split into inner visuals", "inner": inner}]


def test_contained_region_covered_by_a_processed_outer_is_redundant() -> None:
    outer = _region("p1_R0", 0, "selected", [50, 100, 350, 300], crop=[20, 70, 380, 330])
    inner = _region("p1_R1", 1, "contained", [75, 125, 200, 225], decisions=_contained(0))
    selection = {"regions": [outer, inner], "others": [], "attached_titles": []}
    verdict = redundancy.assess(inner, selection, {"p1_R0": "chart text"})
    assert verdict["redundant"] and verdict["covered_by"] == ["p1_R0"] and verdict["coverage"] == 1.0
    assert not redundancy.assess(inner, selection, {"p1_R0": "  "})["redundant"]  # empty OCR covers nothing
    assert not redundancy.assess(inner, selection, {})["redundant"]


def test_container_fully_covered_by_its_processed_inner_visuals_is_redundant() -> None:
    container = _region("p1_R0", 0, "container", [0, 0, 300, 200], decisions=_split([1, 2]))
    left = _region("p1_R1", 1, "selected", [10, 10, 140, 190], crop=[-20, -20, 160, 220])
    right = _region("p1_R2", 2, "selected", [160, 10, 290, 190], crop=[140, -20, 320, 220])
    selection = {"regions": [container, left, right], "others": [], "attached_titles": []}
    verdict = redundancy.assess(container, selection, {"p1_R1": "a", "p1_R2": "b"})
    assert verdict["redundant"] and verdict["covered_by"] == ["p1_R1", "p1_R2"]


def test_partially_covered_container_stays_visible() -> None:
    container = _region("p1_R0", 0, "container", [0, 0, 300, 200], decisions=_split([1, 2]))
    left = _region("p1_R1", 1, "selected", [10, 10, 140, 190], crop=[0, 0, 150, 200])
    right = _region("p1_R2", 2, "selected", [160, 10, 290, 190], crop=[160, 0, 300, 200])  # 10 pt gap uncovered
    selection = {"regions": [container, left, right], "others": [], "attached_titles": []}
    verdict = redundancy.assess(container, selection, {"p1_R1": "a", "p1_R2": "b"})
    assert not verdict["redundant"] and "only" in verdict["reason"]


def test_uncertain_or_uniquely_informative_regions_stay_visible() -> None:
    outer = _region("p1_R0", 0, "selected", [50, 100, 350, 300])
    uncertain = _region("p1_R1", 1, "uncertain", [75, 125, 200, 225], decisions=_contained(0))
    selection = {"regions": [outer, uncertain], "others": [], "attached_titles": []}
    assert not redundancy.assess(uncertain, selection, {"p1_R0": "x"})["redundant"]
    container = _region("p1_R2", 2, "container", [0, 0, 300, 200], decisions=_split([3, 4]))
    # the crops leave a 2 pt strip that holds a detected text block (e.g. a vertical legend): unique content
    a = _region("p1_R3", 3, "selected", [10, 10, 140, 190], crop=[-20, -20, 149, 220])
    b = _region("p1_R4", 4, "selected", [160, 10, 290, 190], crop=[151, -20, 320, 220])
    legend = {"label": "text", "block_index": 9, "bbox_pt": [149, 40, 151, 160]}
    selection = {"regions": [container, a, b], "others": [legend], "attached_titles": []}
    verdict = redundancy.assess(container, selection, {"p1_R3": "a", "p1_R4": "b"})
    assert not verdict["redundant"] and "unique content" in verdict["reason"]
    no_relation = _region("p1_R5", 5, "contained", [75, 125, 200, 225])
    assert not redundancy.assess(no_relation, selection, {"p1_R3": "a"})["redundant"]


def test_redundant_region_is_hidden_from_markdown_and_rag_but_kept_everywhere_else(ctx) -> None:  # noqa: F811
    from tests.lab.test_auto_pipeline import FakeQari

    run_dir, config = ctx["run_dir"], ctx["config"]
    pipeline.execute(run_dir, config, _client(FakeQari()))  # p1_R1 (inside the chart p1_R0) stays pending
    document = assemble.assemble(run_dir)
    element = next(e for e in document["elements"] if e.get("region_id") == "p1_R1")
    assert element["suppressed_redundant"] and element["covered_by"] == ["p1_R0"] and element["suppression_reason"]
    assert element["validation_state"] == "pending_review"
    assert "p1_R1" not in document["markdown"]
    assert all(c["element_id"] != element["id"] for c in assemble.rag_chunks(document))
    codes = {(f["code"], f.get("region_id")) for f in document["validation"]["flags"]}
    assert ("pending_review", "p1_R1") in codes and ("redundant_region_suppressed", "p1_R1") in codes
    assert [q["region_id"] for q in pipeline.review_queue(run_dir)] == ["p1_R1"]  # Process now still offered
    manifest = json.loads(assemble.finalize(run_dir, config, allow_incomplete=True).read_text(encoding="utf-8"))
    saved = json.loads((run_dir / "document.json").read_text(encoding="utf-8"))
    assert any(e.get("region_id") == "p1_R1" and e.get("suppressed_redundant") for e in saved["elements"])
    assert manifest["validation"]


def test_process_now_still_works_for_a_suppressed_region(ctx) -> None:  # noqa: F811
    from tests.lab.test_auto_pipeline import FakeQari

    run_dir, config = ctx["run_dir"], ctx["config"]
    fake = FakeQari()
    pipeline.execute(run_dir, config, _client(fake))
    pipeline.process_now(run_dir, "p1_R1")
    pipeline.execute(run_dir, config, _client(fake))
    element = next(e for e in assemble.assemble(run_dir)["elements"] if e.get("region_id") == "p1_R1")
    assert element["content"] and not element.get("suppressed_redundant")


# ------------------------------------------------------------------ 16 nothing historical changes


def test_reassembly_and_verification_never_modify_existing_files(ctx) -> None:  # noqa: F811
    run_dir, config = ctx["run_dir"], ctx["config"]
    pipeline.execute(run_dir, config, _mock(ArabicQari()))
    pipeline.verify_digits(run_dir, config, _mock(ArabicQari()))
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    assemble.assemble(run_dir)
    pipeline.verify_digits(run_dir, config, _mock(ArabicQari()))  # already verified: nothing new is sent
    after = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    assert all(after[p] == data for p, data in before.items())
    assemble.finalize(run_dir, config, allow_incomplete=True)
    assert pipeline.verify_digits(run_dir, config, _mock(ArabicQari()))["stopped"] == "the run is finalized"


def test_proven_settings_and_primary_jobs_are_unchanged(tmp_path) -> None:
    file = tmp_path / "auto.toml"
    file.write_text("", encoding="utf-8")
    config = auto_config.load_config(file, environ={})
    assert auto_config.proven_differences(config) == {}
    assert (config.verification.digits_enabled, config.verification.max_requests, config.verification.upscale) == (
        True, 30, 1.5)
    assert config.proven.qari.max_time_crop_s == 30.0 and config.proven.qari.prompt == "Free OCR."
    with pytest.raises(auto_config.ConfigError):
        auto_config.load_config(file, environ={"VERIFY_DIGITS_UPSCALE": "5"})


def test_per_run_limit_counts_only_verification_requests(ctx) -> None:  # noqa: F811
    run_dir = ctx["run_dir"]
    fake = ArabicQari()
    pipeline.execute(run_dir, ctx["config"], _mock(fake))
    pipeline.process_now(run_dir, "p1_R1")
    pipeline.execute(run_dir, ctx["config"], _mock(fake))  # now two crops with Arabic-Indic digits
    primary_requests = len(fake.requests)
    summary = pipeline.verify_digits(run_dir, _config(ctx, VERIFY_DIGITS_MAX_REQUESTS="1"), _mock(fake))
    assert summary["eligible"] == 2 and summary["sent"] == 1 and summary["skipped"] == 1
    assert "limit" in summary["stopped"] and len(fake.requests) == primary_requests + 1
    report = assemble.assemble(run_dir)["validation"]["digit_verification"]
    assert report["requests_sent"] == 1 and report["compared"] == 1 and report["verification_skipped"] == 1
    assert sum(1 for j in pipeline.all_jobs(run_dir) if pipeline.runnable(j)) == 3  # primary jobs unaffected
