"""Phase 2 crop options: native byte preservation, dimensions, fallbacks, provenance, timeouts
and compatibility with region sets made before crop options existed. Synthetic PDFs only."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import fitz
import pytest
from PIL import Image

from lab.core.e2.api.providers import load_providers
from lab.core.hybrid import crops, enrich, qari
from tests.lab.test_hybrid_phase2 import _import, ctx  # noqa: F401 - shared synthetic run fixture


def _jpeg(width=400, height=566, color=(180, 120, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _scanned_page(*, text: str | None = None, png: bool = False):
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    if png:
        buffer = io.BytesIO()
        Image.new("RGB", (400, 566), (10, 200, 10)).save(buffer, format="PNG")
        page.insert_image(page.rect, stream=buffer.getvalue())
    else:
        page.insert_image(page.rect, stream=_jpeg())
    if text:
        page.insert_text((72, 72), text, fontsize=12)
    return document, page


def test_native_full_page_keeps_the_embedded_jpeg_bytes() -> None:
    document, page = _scanned_page()
    data, meta = crops.make_crop(document, page, [0, 0, 595, 842], resolution="auto")
    xref = page.get_images(full=True)[0][0]
    assert data == document.xref_stream_raw(xref) and data[:3] == crops.JPEG_MAGIC
    assert meta["resolution_mode"] == "native" and meta["re_encoded"] is False and meta["format"] == "jpeg"
    assert meta["pixel_size"] == [400, 566] and meta["native_xref"] == xref
    assert meta["effective_dpi"] == round(400 / (595 / 72), 1)
    assert Image.open(io.BytesIO(data)).size == (400, 566)


def test_sub_region_of_a_scanned_page_is_cut_from_the_embedded_pixels() -> None:
    document, page = _scanned_page()
    data, meta = crops.make_crop(document, page, [0, 0, 297.5, 421], resolution="auto", fmt="png")
    assert meta["resolution_mode"] == "native_subregion" and meta["pixel_box"] == [0, 0, 200, 283]
    assert Image.open(io.BytesIO(data)).size == (200, 283) and meta["format"] == "png"


@pytest.mark.parametrize("kwargs, reason", [({"text": "visible text"}, "text layer"), ({"png": True}, "filter")])
def test_native_falls_back_to_rendering_with_a_recorded_reason(kwargs, reason) -> None:
    document, page = _scanned_page(**kwargs)
    data, meta = crops.make_crop(document, page, [0, 0, 595, 842], resolution="native")
    assert meta["resolution_mode"] == "rendered" and meta["effective_dpi"] == crops.DEFAULT_DPI
    assert reason in meta["fallback_reason"]
    # A full page (> 25% of the page) with format "auto" falls back to JPEG q90, with the reason recorded.
    assert data[:3] == crops.JPEG_MAGIC and meta["format"] == "jpeg" and meta["jpeg_quality"] == 90
    assert "JPEG" in meta["format_reason"]
    small, small_meta = crops.make_crop(document, page, [0, 0, 100, 100], resolution=250)
    assert small[:8] == b"\x89PNG\r\n\x1a\n" and "format_reason" not in small_meta
    explicit, explicit_meta = crops.make_crop(document, page, [0, 0, 595, 842], resolution=250, fmt="png")
    assert explicit[:8] == b"\x89PNG\r\n\x1a\n" and explicit_meta["format"] == "png"


def test_zero_off_diagonal_terms_are_compared_by_magnitude() -> None:
    assert crops._zero(-0.0) and crops._zero(0.0) and crops._zero(5e-7) and not crops._zero(1e-3)


def test_rendered_dimensions_and_jpeg_quality() -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 72), "digital page", fontsize=12)
    data, meta = crops.make_crop(document, page, [0, 0, 297.5, 421], resolution=150, fmt="jpeg")
    assert meta["resolution_mode"] == "rendered" and meta["format"] == "jpeg" and meta["jpeg_quality"] == 90
    assert data[:3] == crops.JPEG_MAGIC
    assert Image.open(io.BytesIO(data)).size == tuple(meta["pixel_size"])
    assert abs(meta["pixel_size"][0] - round(297.5 * 150 / 72)) <= 1
    assert meta["fallback_reason"] is None  # a numeric resolution was requested explicitly


def test_size_based_timeouts_apply_to_baidu_only() -> None:
    baidu = next(item for item in load_providers() if item.id == "baidu-aistudio-paddleocr-vl-1.6")
    oxen = next(item for item in load_providers() if item.id == "oxen-qwen3-vl-4b-instruct")
    small, large = {"region_id": "s", "area_share": 0.1}, {"region_id": "l", "area_share": 0.9}
    assert crops.default_timeout(small, baidu) == 60 and crops.default_timeout(large, baidu) == 180
    assert crops.default_timeout({}, baidu) == 180
    assert crops.default_timeout(small, oxen) == crops.default_timeout(large, oxen) == oxen.limits.request_timeout_s == 300
    assert crops.default_timeout(small, None, fallback_s=qari.QARI_TIMEOUT_S) == 300
    assert enrich.region_timeouts(None, [small, large], None, fallback_s=qari.QARI_TIMEOUT_S) == {"s": 300, "l": 300}
    assert enrich.region_timeouts(oxen, [small], 120) == {"s": 120.0}  # the override applies to every provider
    with pytest.raises(enrich.EnrichError):
        enrich.region_timeouts(oxen, [small], 400)  # above Oxen's 360 s compute-time billing bound
    with pytest.raises(ValueError):
        crops.checked_timeout(None, 5)
    longer = crops.with_timeout(baidu, 180)
    assert longer.limits.request_timeout_s == 180 and baidu.limits.request_timeout_s == 60


def test_qari_plan_keeps_jpeg_bytes_and_names_them_jpg(tmp_path) -> None:
    jpeg = _jpeg(50, 50)
    qari.create_plan(tmp_path / "vlm", endpoint=qari.parse_endpoint("http://localhost:8000"), model=qari.QARI_MODEL,
                     pages=[{"page": 1, "png": jpeg, "dpi": 200}], document_sha256="a" * 64, approved_by="Tester")
    plan = qari.load_plan(tmp_path / "vlm")
    assert plan["pages"][0]["file"].endswith(".jpg") and plan["pages"][0]["dpi"] == 200
    assert (tmp_path / "vlm" / plan["pages"][0]["file"]).read_bytes() == jpeg


def test_region_set_records_crop_provenance_in_manifest_and_outputs(ctx) -> None:  # noqa: F811
    regions = [r["region_id"] for r in enrich.candidate_regions(ctx["run_dir"]) if r["parent"] is None]
    default_set = enrich.create_region_set(ctx["run_dir"], regions, pdf_path=ctx["source"], by="Tester")
    jpeg_set = enrich.create_region_set(ctx["run_dir"], regions, pdf_path=ctx["source"], by="Tester",
                                        resolution=200, fmt="jpeg")
    first, second = enrich.load_set(default_set)["regions"][0], enrich.load_set(jpeg_set)["regions"][0]
    assert first["crop_sha256"] != second["crop_sha256"]  # other options -> a new crop
    options = second["crop_options"]
    assert second["crop_file"].endswith(".jpg") and options["format"] == "jpeg" and options["effective_dpi"] == 200
    assert options["crop_sha256"] == hashlib.sha256((jpeg_set / second["crop_file"]).read_bytes()).hexdigest()
    export = enrich.export_crops_zip(jpeg_set, provider="Qari", model="Qari", by="Tester")
    archive = zipfile.ZipFile(io.BytesIO(export))
    manifest = json.loads(archive.read("manifest.json"))
    assert manifest["regions"][0]["crop_options"]["format"] == "jpeg"
    assert archive.read(f"crops/{manifest['regions'][0]['crop_file']}") == (jpeg_set / second["crop_file"]).read_bytes()
    folder = _import(ctx, jpeg_set, {second["region_id"]: "نص"})
    (output,) = enrich.collect_outputs(jpeg_set, lambda _id: None, ctx["root"])[second["region_id"]]
    assert output["crop_options"]["format"] == "jpeg" and output["output_key"] == folder.name
    assert enrich.load_set(default_set)["regions"][0] == first  # the earlier set is unchanged


def test_region_sets_without_crop_options_still_load_and_export(ctx) -> None:  # noqa: F811
    regions = [r["region_id"] for r in enrich.candidate_regions(ctx["run_dir"]) if r["parent"] is None]
    new_set = enrich.create_region_set(ctx["run_dir"], regions[:1], pdf_path=ctx["source"], by="Tester")
    data = json.loads((new_set / "regions.json").read_text(encoding="utf-8"))
    legacy_dir = new_set.parent / "set_000900"
    (legacy_dir / "crops").mkdir(parents=True)
    region = data["regions"][0]
    (legacy_dir / region["crop_file"]).write_bytes((new_set / region["crop_file"]).read_bytes())
    for key in ("crop_options", "page_size_pt", "size_based_timeout_s"):
        region.pop(key)
    data["set_id"] = legacy_dir.name
    data.pop("crop_request")
    (legacy_dir / "regions.json").write_text(json.dumps(data), encoding="utf-8")
    legacy = enrich.load_set(legacy_dir)["regions"][0]
    assert crops.region_options(legacy)["legacy"] is True and crops.region_options(legacy)["format"] == "png"
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(
        enrich.export_crops_zip(legacy_dir, provider="Qari", model="Qari", by="Tester"))).read("manifest.json"))
    assert manifest["regions"][0]["crop_options"]["legacy"] is True
    baidu = next(item for item in load_providers() if item.id == "baidu-aistudio-paddleocr-vl-1.6")
    assert enrich.region_timeouts(baidu, [legacy], None)[legacy["region_id"]] in (60.0, 180.0)
