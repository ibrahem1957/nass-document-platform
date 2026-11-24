"""Region sets from layout boxes on scanned pages (Phase 2, Benchmark Mode).

Target pipeline: PDF -> layout detection (local PP-DocLayout) -> automatic cropping -> Qari OCR ->
validation -> structured document. The layout SOURCE is pluggable; everything after it (cropping,
export/import, sending, comparison) only sees the common block schema:

    {"page": int, "label": str, "bbox_px": [x0, y0, x1, y1], "image_px_size": [w, h],
     "region_bbox_pt": [x0, y0, x1, y1],   # the page area the pixel space covers
     "source_id": str, "block_index": int, "warnings": [str]}

Sources:
- "saved_provider_output": boxes from a SAVED PaddleOCR-VL output of an existing region set (no new
  request). A test of the cropping mechanism only, not a permanent part of the system.
- "pp_doclayout_local": local PP-DocLayout through PaddleOCR's `LayoutDetection` module, on the
  embedded page image at native resolution, with weights installed locally beforehand (no download,
  no hosted service; inference runs under the air-gap outbound network guard).

Crops are cut from the scanned page's embedded image pixels (no re-render) as lossless PNG. Only
exact duplicates are removed automatically; small, near-duplicate and overlapping boxes are kept
but never preselected, with the reason. Region ids p<page>_L<n> never collide with layout figure ids.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Callable

from lab.core.e2.api import full_pdf
from lab.core.e2.api.providers import ProviderSpec
from lab.core.hybrid import crops as crop_options

DEFAULT_LABELS = ("chart", "image", "table")
DEFAULT_MARGIN_PX = 15
MIN_LONG_SIDE_PT = 100.0  # the small-image rule of digital pages
NEAR_DUPLICATE_IOU = 0.9
LAYOUT_SOURCES = {
    "saved_provider_output": "Saved Paddle output (historical benchmark; no request)",
    "pp_doclayout_local": "Local PP-DocLayout (on this machine; no network)",
}
# Model name is configurable; the documented default of LayoutDetection is PP-DocLayout-L and the
# documented higher-accuracy model is PP-DocLayout_plus-L. Labels are verified from the local model files.
PP_DOCLAYOUT_DEFAULT_MODEL = "PP-DocLayout_plus-L"
PP_DOCLAYOUT_MODELS = ("PP-DocLayout_plus-L", "PP-DocLayout-L", "PP-DocLayout-M", "PP-DocLayout-S")
TITLE_LABELS = frozenset({"figure_title"})
TARGET_LABELS = frozenset({"chart", "table", "image"})
TITLE_MAX_GAP_PT = 30.0          # title may sit up to 30 pt above the visual ...
TITLE_MAX_TOP_OVERLAP_PT = 30.0  # ... or overlap its top boundary by up to 30 pt
TITLE_MIN_OVERLAP = 0.5
TITLE_DECISION_MARGIN_PT = 5.0   # title vs caption: the nearer visual must be closer by at least this
# Between two visuals, ATTACHING the text to the visual below (which also widens that visual's crop) needs a decisive
# geometric advantage: the visual above must be at least this many times farther away (or the text must overlap the
# lower visual's top boundary). A smaller advantage is only a preference: the text is left unattached for review.
TITLE_BETWEEN_VISUALS_ATTACH_RATIO = 2.0
CAPTION_LABELS = frozenset({"figure_caption", "table_caption", "chart_caption"})  # used only if the model has them
CONTAINMENT = 0.9                # a box >= 90% inside another is contained in it (not a partial overlap)
SPLIT_MIN_INNER_SHARE = 0.15     # split into inner visuals only if each covers >= 15% of the outer box ...
SPLIT_MIN_COVERAGE = 0.5         # ... they cover >= 50% together, do not overlap, and are not small
CUT_MIN_SHARE, CUT_MAX_SHARE = 0.1, 0.9  # another block 10-90% inside a region border -> content may be cut
CAPTION_TRIM_TOP_SHARE = 0.2     # a foreign caption inside the top 20% of a kept box is trimmed off
BOUNDS_TOLERANCE_PX = 2.0


class LayoutSourceError(ValueError):
    pass


# ------------------------------------------------------------------ sources


def _find_dict_with(value: Any, key: str) -> dict | None:
    if isinstance(value, dict):
        if key in value:
            return value
        for item in value.values():
            found = _find_dict_with(item, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_dict_with(item, key)
            if found is not None:
                return found
    return None


def _read_json(path: Path) -> Any:
    import json

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def blocks_from_structured(structured: Any) -> tuple[list[dict], list[int] | None, list[str]]:
    """(blocks with label/bbox_px, the image size the provider reports, warnings) from a saved
    PaddleOCR-VL structured output. Only fields actually present are used."""
    warnings = []
    holder = _find_dict_with(structured, "parsing_res_list")
    raw = (holder or {}).get("parsing_res_list")
    if raw is None:
        layout = _find_dict_with(structured, "layout_regions")
        raw = (layout or {}).get("layout_regions")
    size = None
    if holder and isinstance(holder.get("width"), (int, float)) and isinstance(holder.get("height"), (int, float)):
        size = [int(holder["width"]), int(holder["height"])]
    pre = _find_dict_with(structured, "doc_preprocessor_res")
    if pre:
        angle = (pre.get("doc_preprocessor_res") or {}).get("angle")
        if angle not in (None, 0, 0.0):
            warnings.append(f"the provider rotated the image by {angle} degrees before layout detection")
        settings = (pre.get("doc_preprocessor_res") or {}).get("model_settings") or {}
        if settings.get("use_doc_unwarping"):
            warnings.append("the provider's document unwarping was on; boxes may be slightly shifted (check visually)")
    blocks = []
    for index, block in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(block, dict) or not isinstance(block.get("block_bbox"), list) or len(block["block_bbox"]) != 4:
            continue
        blocks.append({"label": str(block.get("block_label") or "unknown"),
                       "bbox_px": [float(v) for v in block["block_bbox"]], "block_index": index})
    return blocks, size, warnings


def saved_provider_output(set_dir: Path, run_dir: Path, spec: ProviderSpec, root: Path,
                          loader: Callable[[dict], Any] | None = None) -> list[dict]:
    """Layout blocks from a saved PaddleOCR-VL run of a region set (read-only; no request)."""
    from lab.core.hybrid import enrich  # noqa: PLC0415 - avoid an import cycle

    data = enrich.load_set(set_dir)
    regions = {r["region_id"]: r for r in data["regions"]}
    plan = _read_json(Path(run_dir) / "plan.json") or {}
    if plan.get("provider_id") != spec.id:
        raise LayoutSourceError("The selected run is not a run of this provider")
    found = []
    for item in plan.get("regions") or []:
        region = regions.get(item["region_id"])
        record = full_pdf.derive_page(run_dir, spec, {"page": item["index"], "image_sha256": item["crop_sha256"]},
                                      root=root)
        if region is None or record.get("job_state") != "COMPLETED":
            continue
        locations = enrich.artifact_locations(spec, record.get("journal_id"), root)
        if loader is not None:
            structured = loader(locations)
        else:
            snapshot = Path(locations["page_snapshot_dir"]) if locations.get("page_snapshot_dir") else None
            structured = _read_json(snapshot / "structured_output.json") if snapshot else None
            if structured is None and locations.get("store_run_dir"):
                structured = _read_json(Path(locations["store_run_dir"]) / "run.json")
        blocks, size, warnings = blocks_from_structured(structured)
        crop_px = list(region["crop_px"])
        image_size = size or crop_px
        if size and size != crop_px:
            warnings.append(f"the provider reports {size[0]}x{size[1]} px for a {crop_px[0]}x{crop_px[1]} px crop; "
                            "boxes are mapped with the provider's size")
        for block in blocks:
            found.append({"page": region["page"], "label": block["label"], "bbox_px": block["bbox_px"],
                          "image_px_size": image_size, "region_bbox_pt": list(region["bbox"]),
                          "source_id": f"{data['set_id']}/{Path(run_dir).name}/{item['region_id']}",
                          "block_index": block["block_index"], "warnings": list(warnings),
                          "source_crop_sha256": region["crop_sha256"], "journal_id": record.get("journal_id")})
    return found


def _version(name: str) -> str | None:
    import importlib.metadata  # noqa: PLC0415

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _model_labels(model_dir: Path) -> list[str] | None:
    """`label_list` from the local model's inference.yml (read as text; no model load)."""
    import re  # noqa: PLC0415

    path = Path(model_dir) / "inference.yml"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        import yaml  # noqa: PLC0415

        found = _find_dict_with(yaml.safe_load(text), "label_list")
        if found and isinstance(found.get("label_list"), list):
            return [str(item) for item in found["label_list"]]
    except Exception:  # noqa: BLE001 - fall back to a plain-text read
        pass
    match = re.search(r"label_list:\s*\n((?:\s*-\s*.+\n?)+)", text)
    return [line.split("-", 1)[1].strip() for line in match.group(1).splitlines() if "-" in line] if match else None


def doclayout_setup(model_dir: str | Path | None, model_name: str = PP_DOCLAYOUT_DEFAULT_MODEL) -> dict:
    """Check the local setup WITHOUT loading the model or running inference."""
    import importlib.util  # noqa: PLC0415

    problems: list[str] = []
    versions = {name: _version(name) for name in ("paddleocr", "paddlex", "paddlepaddle", "paddlepaddle-gpu")}
    if importlib.util.find_spec("paddleocr") is None:
        problems.append("paddleocr is not installed in this environment (pip install paddleocr)")
    if not (versions["paddlepaddle"] or versions["paddlepaddle-gpu"]):
        problems.append("PaddlePaddle is not installed (paddlepaddle or paddlepaddle-gpu)")
    directory = Path(model_dir) if model_dir else None
    files = {}
    if directory is None or not directory.is_dir():
        problems.append("the local model directory does not exist; download and unpack the model first "
                        "(no automatic download is allowed)")
    else:
        for name in ("inference.yml", "inference.pdiparams", "inference.json", "inference.pdmodel"):
            if (directory / name).is_file():
                files[name] = (directory / name).stat().st_size
        if "inference.yml" not in files or "inference.pdiparams" not in files:
            problems.append("the model directory lacks inference.yml / inference.pdiparams (an *_infer package)")
        if "inference.json" not in files and "inference.pdmodel" not in files:
            problems.append("the model directory lacks the model graph (inference.json or inference.pdmodel)")
    labels = _model_labels(directory) if directory and directory.is_dir() else None
    configured_name = None
    if directory and (directory / "inference.yml").is_file():
        import re  # noqa: PLC0415

        found = re.search(r"model_name:\s*([^\s#]+)", (directory / "inference.yml").read_text(encoding="utf-8",
                                                                                              errors="replace"))
        configured_name = found.group(1) if found else None
        if configured_name and configured_name != model_name:
            problems.append(f"the model directory contains {configured_name!r}, not {model_name!r}")
    missing_targets = sorted(TARGET_LABELS - set(labels or [])) if labels else None
    return {"ok": not problems, "problems": problems, "versions": versions, "model_name": model_name,
            "model_dir": str(directory) if directory else None, "model_files": files,
            "model_name_in_files": configured_name, "labels": labels,
            "title_attachment_available": bool(labels and TITLE_LABELS & set(labels)),
            "labels_missing": missing_targets,
            "note": "setup check only: no model was loaded and nothing was downloaded or sent"}


def _load_predictor(model_name: str, model_dir: str, device: str | None, threshold: float | None):
    import os  # noqa: PLC0415

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")  # no model-hoster connectivity check
    from paddleocr import LayoutDetection  # noqa: PLC0415

    options = {"model_name": model_name, "model_dir": model_dir}
    if device:
        options["device"] = device
    if threshold is not None:
        options["threshold"] = float(threshold)
    return LayoutDetection(**options)


def _result_boxes(result) -> list[dict]:
    payload = getattr(result, "json", result)
    payload = payload() if callable(payload) else payload
    return list(((payload or {}).get("res") or {}).get("boxes") or [])


def pp_doclayout_local(run_dir: Path, *, pdf_path: Path | None, pages: list[int] | None, model_dir: str,
                       model_name: str = PP_DOCLAYOUT_DEFAULT_MODEL, device: str | None = None,
                       threshold: float | None = None, predictor=None) -> tuple[list[dict], dict]:
    """Local PP-DocLayout on each scanned page's embedded image at native resolution.

    No unwarping, orientation correction or resizing by this code (the layout module of PaddleOCR
    applies neither; only the document pipelines do). Boxes are expected in the input image's
    pixels and are checked against its bounds. Returns (blocks, report)."""
    import numpy as np  # noqa: PLC0415
    import pymupdf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from lab.core.e2.airgap import outbound_network_guard  # noqa: PLC0415
    from lab.core.hybrid import run as hybrid  # noqa: PLC0415

    setup = doclayout_setup(model_dir, model_name)
    if predictor is None and not setup["ok"]:
        raise LayoutSourceError("Local PP-DocLayout is not ready: " + "; ".join(setup["problems"]))
    metadata, _plan, _library = hybrid.load(run_dir)
    pdf, pdf_source = hybrid.layout_pdf(run_dir, metadata, pdf_path)
    provenance = {"kind": "pp_doclayout_local", "model_name": model_name, "model_dir": setup["model_dir"],
                  "model_files": setup["model_files"], "versions": setup["versions"], "labels": setup["labels"],
                  "device": device, "threshold": threshold,
                  "input": "embedded page image decoded with Pillow at native resolution, RGB -> BGR uint8 array",
                  "geometry_changing_preprocessing": "none (no unwarping, no orientation correction, no resize)",
                  "network": "outbound network guard active during model load and inference",
                  "pdf": pdf_source}
    blocks: list[dict] = []
    pages_report: list[dict] = []
    with pymupdf.open(pdf) as document:
        wanted = pages or list(range(1, len(document) + 1))
        with outbound_network_guard():
            model = predictor or _load_predictor(model_name, setup["model_dir"], device, threshold)
            for number in wanted:
                page = document[number - 1]
                native, reason = crop_options.native_page_raster(document, page)
                if native is None:
                    pages_report.append({"page": number, "status": "skipped", "reason": f"not a scanned page: {reason}"})
                    continue
                image = Image.open(io.BytesIO(native["bytes"])).convert("RGB")
                size = [image.width, image.height]
                if size != [native["width_px"], native["height_px"]]:
                    pages_report.append({"page": number, "status": "failed",
                                         "reason": f"decoded size {size} differs from the embedded image size"})
                    continue
                array = np.asarray(image)[:, :, ::-1].copy()  # BGR, as PaddleX expects for arrays
                try:
                    results = list(model.predict(array, batch_size=1))
                except Exception as exc:  # noqa: BLE001 - recorded; nothing is guessed
                    pages_report.append({"page": number, "status": "failed",
                                         "reason": f"layout detection failed: {type(exc).__name__}: {exc}"})
                    continue
                count = 0
                for result in results:
                    for index, box in enumerate(_result_boxes(result)):
                        coordinate = [float(v) for v in (box.get("coordinate") or [])][:4]
                        if len(coordinate) != 4:
                            continue
                        warnings = []
                        if (min(coordinate) < -BOUNDS_TOLERANCE_PX or coordinate[2] > size[0] + BOUNDS_TOLERANCE_PX
                                or coordinate[3] > size[1] + BOUNDS_TOLERANCE_PX):
                            warnings.append(f"box {coordinate} lies outside the {size[0]}x{size[1]} px image; check "
                                            "the coordinate mapping")
                        if coordinate[2] <= coordinate[0] or coordinate[3] <= coordinate[1]:
                            warnings.append("empty or inverted box")
                        blocks.append({"page": number, "label": str(box.get("label") or "unknown"),
                                       "bbox_px": coordinate, "image_px_size": size,
                                       "region_bbox_pt": list(native["bbox"]),
                                       "source_id": f"pp_doclayout_local/{model_name}/p{number}",
                                       "block_index": index, "score": box.get("score"), "cls_id": box.get("cls_id"),
                                       "warnings": warnings, "model_name": model_name})
                        count += 1
                pages_report.append({"page": number, "status": "detected", "blocks": count, "image_px_size": size})
    return blocks, {"provenance": provenance, "pages": pages_report, "setup": setup}


# ------------------------------------------------------------- geometry


def px_to_pt(bbox_px: list[float], image_px_size: list[int], region_bbox_pt: list[float]) -> list[float]:
    """Map a box in the pixel space of the image that was analysed to page points."""
    width_px, height_px = image_px_size
    x0, y0, x1, y1 = region_bbox_pt
    sx, sy = (x1 - x0) / float(width_px), (y1 - y0) / float(height_px)
    return [round(x0 + bbox_px[0] * sx, 2), round(y0 + bbox_px[1] * sy, 2),
            round(x0 + bbox_px[2] * sx, 2), round(y0 + bbox_px[3] * sy, 2)]


def pt_to_image_px(bbox_pt: list[float], image_bbox_pt: list[float], image_px: list[int]) -> list[float]:
    x0, y0, x1, y1 = image_bbox_pt
    sx, sy = image_px[0] / (x1 - x0), image_px[1] / (y1 - y0)
    return [(bbox_pt[0] - x0) * sx, (bbox_pt[1] - y0) * sy, (bbox_pt[2] - x0) * sx, (bbox_pt[3] - y0) * sy]


def with_margin(box_px: list[float], margin_px: int, image_px: list[int]) -> list[int]:
    """Expand by the margin and clamp to the embedded image."""
    return [max(0, int(round(box_px[0])) - margin_px), max(0, int(round(box_px[1])) - margin_px),
            min(image_px[0], int(round(box_px[2])) + margin_px), min(image_px[1], int(round(box_px[3])) + margin_px)]


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def _inside(inner, outer) -> float:
    area = _area(inner)
    return _intersection(inner, outer) / area if area else 0.0


def _long_side(box) -> float:
    return max(box[2] - box[0], box[3] - box[1])


def classify(boxes: list[dict]) -> list[dict]:
    """Status per box (same page). `boxes` items need `bbox_pt`; results add status/reasons/decisions.

    - exact duplicates are removed (recorded);
    - small boxes are kept but not preselected;
    - containment (a box >= 90% inside another) is NOT a partial overlap: by default the complete
      outer visual is preferred and the inner box is kept as "contained" (not preselected); the outer
      box is split into its inner visuals only when there are >= 2 inner boxes that do not overlap,
      are not small, each cover >= 15% of the outer box and together >= 50% of it;
    - genuine partial overlaps mark both boxes uncertain (never preselected)."""
    out: list[dict] = []
    seen: dict[tuple, str] = {}
    for box in boxes:
        key = (box["label"], tuple(round(v, 1) for v in box["bbox_pt"]))
        item = dict(box, status="selected", reasons=[], default_selected=True, decisions=[])
        if key in seen:
            item.update(status="duplicate removed", default_selected=False,
                        reasons=[f"exact duplicate of block {seen[key]}"],
                        decisions=[{"type": "duplicate", "of_block": seen[key]}])
            out.append(item)
            continue
        seen[key] = str(box.get("block_index"))
        if _long_side(box["bbox_pt"]) < MIN_LONG_SIDE_PT:
            item.update(status="small", default_selected=False,
                        reasons=[f"small — not preselected (longest side {_long_side(box['bbox_pt']):.0f} pt < "
                                 f"{MIN_LONG_SIDE_PT:.0f} pt)"])
        out.append(item)
    kept = [item for item in out if item["status"] != "duplicate removed"]

    # direct parent = the smallest box that contains it (>= 90%)
    parent: dict[int, dict] = {}
    rank = {id(item): (_area(item["bbox_pt"]), -position) for position, item in enumerate(kept)}  # equal size: earlier is outer
    for inner in kept:
        holders = [outer for outer in kept if outer is not inner and rank[id(outer)] > rank[id(inner)]
                   and _inside(inner["bbox_pt"], outer["bbox_pt"]) >= CONTAINMENT]
        if holders:
            parent[id(inner)] = min(holders, key=lambda box: _area(box["bbox_pt"]))
    children: dict[int, list[dict]] = {}
    for inner in kept:
        if id(inner) in parent:
            children.setdefault(id(parent[id(inner)]), []).append(inner)

    for outer in kept:
        inners = children.get(id(outer), [])
        if not inners:
            continue
        outer_area = _area(outer["bbox_pt"]) or 1.0
        shares = [_area(inner["bbox_pt"]) / outer_area for inner in inners]
        disjoint = all(_intersection(a["bbox_pt"], b["bbox_pt"]) == 0
                       for i, a in enumerate(inners) for b in inners[i + 1:])
        distinct = (len(inners) >= 2 and disjoint and min(shares) >= SPLIT_MIN_INNER_SHARE and sum(shares) >= SPLIT_MIN_COVERAGE
                    and all(_long_side(inner["bbox_pt"]) >= MIN_LONG_SIDE_PT for inner in inners))
        ids = [inner.get("block_index") for inner in inners]
        if distinct:
            outer.update(status="container", default_selected=False)
            outer["reasons"].append(f"contains {len(inners)} distinct visuals (blocks {ids}); the inner regions are "
                                    "preferred so no visual is merged into another")
            outer["decisions"].append({"type": "containment", "decision": "split into inner visuals", "inner": ids,
                                       "inner_area_shares": [round(v, 3) for v in shares]})
            for inner in inners:
                inner["decisions"].append({"type": "containment", "decision": "selected as a distinct visual",
                                           "contained_in": outer.get("block_index")})
            continue
        outer["decisions"].append({"type": "containment", "decision": "outer visual preferred", "inner": ids,
                                   "inner_area_shares": [round(v, 3) for v in shares]})
        if len(inners) >= 2:
            outer["reasons"].append(f"Needs Review: contains {len(inners)} inner boxes (blocks {ids}) that do not "
                                    "clearly form separate visuals; the outer visual is preselected")
        for inner in inners:
            if inner["status"] in {"selected", "small"}:
                inner["status"] = "contained"
            inner["default_selected"] = False
            inner["reasons"].append(f"inside block {outer.get('block_index')} "
                                    f"({_inside(inner['bbox_pt'], outer['bbox_pt']):.0%}); the complete outer visual "
                                    "is preferred (available for manual selection)")
            inner["decisions"].append({"type": "containment", "decision": "contained; outer preferred",
                                       "contained_in": outer.get("block_index")})

    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            if parent.get(id(a)) is b or parent.get(id(b)) is a or _intersection(a["bbox_pt"], b["bbox_pt"]) == 0:
                continue
            if parent.get(id(a)) is not None and parent.get(id(a)) is parent.get(id(b)):
                continue  # siblings inside one container: handled by the containment decision
            if _inside(a["bbox_pt"], b["bbox_pt"]) >= CONTAINMENT or _inside(b["bbox_pt"], a["bbox_pt"]) >= CONTAINMENT:
                continue  # nested deeper than the direct parent
            overlap = iou(a["bbox_pt"], b["bbox_pt"])
            for x, y in ((a, b), (b, a)):
                x["reasons"].append(f"partially overlaps block {y.get('block_index')} (IoU {overlap:.2f})")
                x["decisions"].append({"type": "partial_overlap", "with_block": y.get("block_index"),
                                       "iou": round(overlap, 3)})
                x["default_selected"] = False
                if x["status"] == "selected":
                    x["status"] = "uncertain"
    for item in kept:
        if item.get("warnings"):
            item["reasons"] += item["warnings"]
            if item["status"] == "selected":
                item["status"], item["default_selected"] = "uncertain", False
    return out


# ------------------------------------------------------------ titles


def _title_relations(title: dict, targets: list[dict]) -> tuple[list[tuple], list[tuple]]:
    """(visuals the text sits on top of, visuals the text sits under), each with the vertical gap
    (negative = overlap into the visual). Needs >= 50% horizontal overlap of the text width."""
    tx0, ty0, tx1, ty1 = title["bbox_pt"]
    width = max(tx1 - tx0, 1e-6)
    tops, bottoms = [], []
    for target in targets:
        gx0, gy0, gx1, gy1 = target["bbox_pt"]
        if max(0.0, min(tx1, gx1) - max(tx0, gx0)) / width < TITLE_MIN_OVERLAP:
            continue
        gap_top = gy0 - ty1  # > 0: text above the visual; < 0: overlaps its top boundary
        if -TITLE_MAX_TOP_OVERLAP_PT <= gap_top <= TITLE_MAX_GAP_PT and ty0 <= gy0 + TITLE_MAX_TOP_OVERLAP_PT:
            tops.append((target, gap_top))
        gap_bottom = ty0 - gy1  # > 0: text below the visual; < 0: overlaps its bottom boundary
        if -TITLE_MAX_TOP_OVERLAP_PT <= gap_bottom <= TITLE_MAX_GAP_PT and ty1 >= gy1 - TITLE_MAX_TOP_OVERLAP_PT:
            bottoms.append((target, gap_bottom))
    return tops, bottoms


TITLE_ASSOCIATION_SCHEMA = 1


def _measured_relation(title: dict, target: dict, gap: float, position: str) -> dict:
    """One measured title/visual relation from the saved detection boxes (no text, no inference)."""
    tx0, _, tx1, _ = title["bbox_pt"]
    gx0, _, gx1, _ = target["bbox_pt"]
    overlap = max(0.0, min(tx1, gx1) - max(tx0, gx0)) / max(tx1 - tx0, 1e-6)
    return {"block_index": target.get("block_index"), "label": target.get("label"),
            "visual_bbox_pt": [round(v, 2) for v in target["bbox_pt"]], "position": position,
            "gap_pt": round(gap, 2), "overlaps_boundary": gap < 0,
            "horizontal_overlap_of_title": round(overlap, 3)}


def _record_title_associations(titles: list[dict], measured: dict[int, list[tuple[dict, dict]]], *,
                               captions_below: bool, attach: bool) -> None:
    """Optional, backward-compatible `title_association` on every title block (and `title_candidates` on the
    visuals it was measured against). It preserves the geometric evidence and the policy outcome; it never
    changes a box or a crop, and it does not claim the title text belongs to a visual (no OCR is involved)."""
    for title in titles:
        pairs = measured.get(id(title), [])
        candidates = [record for _target, record in pairs]
        status_text = str(title.get("title_status") or "")
        decision = title.get("title_decision")
        attached_block = None
        if status_text.startswith("attached to block"):
            status = "attached"
            attached_block = next((c["block_index"] for c in candidates
                                   if f"block {c['block_index']} " in status_text + " "), None)
        elif status_text == "ambiguous":
            status = "ambiguous"
        elif not candidates:
            status = "no_candidate_within_limits"
        elif not attach:
            status = "attachment_disabled"
        elif any(c["position"] == "title_below_visual" for c in candidates) and not captions_below:
            status = "candidate_not_attached_by_policy"
        else:
            status = "unattached"
        preferred = None
        match = re.match(r"(?:caption|title) of block (\d+)", str(decision or ""))
        if match:
            preferred = int(match.group(1))
        title["title_association"] = {
            "schema": TITLE_ASSOCIATION_SCHEMA, "status": status, "attached_block_index": attached_block,
            "decision_preferred_block_index": preferred, "decision": decision,
            "candidates": candidates,
            "policy": {"attach_titles": attach, "captions_below": captions_below},
            "reason_codes": list(title.get("title_reason_codes") or []),
            "limits": {"max_gap_pt": TITLE_MAX_GAP_PT, "max_boundary_overlap_pt": TITLE_MAX_TOP_OVERLAP_PT,
                       "min_horizontal_overlap": TITLE_MIN_OVERLAP, "decision_margin_pt": TITLE_DECISION_MARGIN_PT,
                       "between_visuals_attach_ratio": TITLE_BETWEEN_VISUALS_ATTACH_RATIO},
            "limitations": ["geometric layout evidence only; the title text is not OCR-verified at this stage"],
        }
        if status == "attached":
            continue
        for target, candidate in pairs:
            target.setdefault("title_candidates", []).append({
                "block_index": title.get("block_index"), "label": title["label"],
                "bbox_pt": [round(v, 2) for v in title["bbox_pt"]], "position": candidate["position"],
                "gap_pt": candidate["gap_pt"], "horizontal_overlap_of_title": candidate["horizontal_overlap_of_title"],
                "association_status": status, "decision": decision})


def attach_titles(items: list[dict], *, captions_below: bool = False, attach: bool = True) -> list[dict]:
    """Attach title blocks to the visual they belong to; the full title is kept in the crop.

    A `figure_title` is a title of the visual below it when it sits above it or overlaps its top
    boundary (gap <= 30 pt, or <= 30 pt of overlap; >= 50% of the title width overlaps the visual).
    When the same text also sits under a previous visual, the nearer one decides (by >= 5 pt): nearer
    to the visual above -> it is that visual's caption and is NOT attached to the next visual
    (attached below only when captions are enabled); too close to call -> Needs Review. Caption labels
    (if the model has them) are only ever captions. Several titles for one visual, or one title for
    several visuals, are ambiguous and left for review."""
    titles = [item for item in items if item["label"] in TITLE_LABELS | CAPTION_LABELS]
    targets = [item for item in items if item["label"] in TARGET_LABELS]
    plans: dict[int, tuple] = {}
    measured: dict[int, list[dict]] = {}
    for title in titles:
        title.setdefault("reasons_extra", [])
        tops, bottoms = _title_relations(title, targets)
        # Measured before any attachment widens a target box; recorded as evidence only.
        measured[id(title)] = ([(target, _measured_relation(title, target, gap, "title_above_visual"))
                                for target, gap in tops]
                               + [(target, _measured_relation(title, target, gap, "title_below_visual"))
                                  for target, gap in bottoms])
        is_caption_label = title["label"] in CAPTION_LABELS
        if is_caption_label:
            tops = []
        nearest_top = min(tops, key=lambda t: max(t[1], 0.0)) if tops else None
        nearest_bottom = min(bottoms, key=lambda t: max(t[1], 0.0)) if bottoms else None
        if nearest_top and nearest_bottom and nearest_top[0] is not nearest_bottom[0]:
            top_gap, bottom_gap = max(nearest_top[1], 0.0), max(nearest_bottom[1], 0.0)
            decisive_below = top_gap <= 0.0 or bottom_gap >= TITLE_BETWEEN_VISUALS_ATTACH_RATIO * top_gap
            if bottom_gap + TITLE_DECISION_MARGIN_PT <= top_gap:
                tops = []  # it belongs to the visual above it: a caption, not the next visual's title
                title["title_decision"] = (f"caption of block {nearest_bottom[0].get('block_index')} (closer to the "
                                           f"visual above: {bottom_gap:.0f} pt vs {top_gap:.0f} pt)")
                decisive_above = bottom_gap <= 0.0 or top_gap >= TITLE_BETWEEN_VISUALS_ATTACH_RATIO * bottom_gap
                title["title_reason_codes"] = ["between_two_visuals", "geometric_preference_visual_above",
                                               "decisive_distance" if decisive_above else "weak_distance_preference"]
            elif top_gap + TITLE_DECISION_MARGIN_PT <= bottom_gap and decisive_below:
                bottoms = []
                title["title_decision"] = (f"title of block {nearest_top[0].get('block_index')} (closer to the "
                                           f"visual below: {top_gap:.0f} pt vs {bottom_gap:.0f} pt)")
                title["title_reason_codes"] = ["between_two_visuals", "geometric_preference_visual_below",
                                               "decisive_distance"]
            elif top_gap + TITLE_DECISION_MARGIN_PT <= bottom_gap:
                # Nearer to the visual below, but not decisively: geometry alone cannot tell a title of the lower
                # visual from a caption of the upper one. Nothing is attached; both visuals are flagged.
                title["title_status"] = "ambiguous"
                title["title_decision"] = (
                    f"between two visuals: nearer to the visual below ({top_gap:.0f} pt vs {bottom_gap:.0f} pt) but "
                    f"not decisively (needs {TITLE_BETWEEN_VISUALS_ATTACH_RATIO:g}x); title or caption is unclear")
                title["title_reason_codes"] = ["between_two_visuals", "geometric_preference_visual_below",
                                               "weak_distance_preference", "not_attached_without_decisive_evidence"]
                title["reasons_extra"].append("Needs Review: " + title["title_decision"])
                for target in (nearest_top[0], nearest_bottom[0]):
                    target.setdefault("reasons_extra", []).append(
                        f"Needs Review: block {title.get('block_index')} may be this visual's title or caption")
                    target["title_ambiguous"] = True
                continue
            else:
                title["title_status"] = "ambiguous"
                title["title_decision"] = "between two visuals at similar distances: title or caption is unclear"
                title["title_reason_codes"] = ["between_two_visuals", "similar_distances"]
                title["reasons_extra"].append("Needs Review: " + title["title_decision"])
                for target in (nearest_top[0], nearest_bottom[0]):
                    target.setdefault("reasons_extra", []).append(
                        f"Needs Review: block {title.get('block_index')} may be this visual's title or caption")
                    target["title_ambiguous"] = True
                continue
        if tops:
            plans[id(title)] = ("above", tops)
        elif bottoms and (captions_below or is_caption_label):
            if captions_below:
                plans[id(title)] = ("below", bottoms)
            else:
                title["title_decision"] = title.get("title_decision") or "caption below a visual; captions are not attached"
        elif bottoms:
            title["title_decision"] = title.get("title_decision") or (
                f"caption of block {bottoms[0][0].get('block_index')}; captions below are not attached (option off)")
    claims: dict[int, list[dict]] = {}
    for title in titles:
        if id(title) in plans:
            for target, _gap in plans[id(title)][1]:
                claims.setdefault(id(target), []).append(title)
    for title in titles:
        if id(title) not in plans:
            continue
        side, found = plans[id(title)]
        if len(found) > 1 or len(claims[id(found[0][0])]) > 1:
            title["title_status"] = "ambiguous"
            title["title_decision"] = "several possible visuals, or several titles for one visual"
            title["reasons_extra"].append("Needs Review: ambiguous title association (several possible targets "
                                          "or several titles for one target)")
            for target, _gap in found:
                target.setdefault("reasons_extra", []).append(
                    f"Needs Review: ambiguous title association with block {title.get('block_index')}")
                target["title_ambiguous"] = True
            continue
        target, gap = found[0]
        if not attach:  # title attachment switched off: the decision is kept, nothing is attached
            title.setdefault("title_decision", f"title of block {target.get('block_index')} (attachment off)")
            continue
        target.setdefault("title_attachments", []).append({
            "block_index": title.get("block_index"), "label": title["label"], "bbox_pt": list(title["bbox_pt"]),
            "bbox_px": list(title.get("bbox_px") or []), "position": side, "gap_pt": round(gap, 2),
            "overlaps_boundary": gap < 0, "decision": title.get("title_decision")})
        target["bbox_pt_without_title"] = target.get("bbox_pt_without_title") or list(target["bbox_pt"])
        # The union keeps the whole title inside the crop, also when it overlaps the visual's top boundary.
        target["bbox_pt"] = [min(target["bbox_pt"][0], title["bbox_pt"][0]), min(target["bbox_pt"][1], title["bbox_pt"][1]),
                             max(target["bbox_pt"][2], title["bbox_pt"][2]), max(target["bbox_pt"][3], title["bbox_pt"][3])]
        title["title_status"] = f"attached to block {target.get('block_index')} ({side})"
    _record_title_associations(titles, measured, captions_below=captions_below, attach=attach)
    return items


def trim_foreign_captions(regions: list[dict], others: list[dict], margin_pt: float) -> None:
    """Move a kept box's top edge below a caption of ANOTHER visual (or an ambiguous title/caption)
    that lies inside the box's top 20%, unless that would cut another block of the visual.
    The caption is never attached; the original and trimmed boxes and the reason are recorded."""
    captions = [o for o in others if o["label"] in TITLE_LABELS | CAPTION_LABELS
                and not o.get("title_status", "").startswith("attached")
                and (str(o.get("title_decision", "")).startswith("caption of") or o.get("title_status") == "ambiguous")]
    for region in regions:
        if region.get("status") == "duplicate removed" or not captions:
            continue
        x0, y0, x1, y1 = region["bbox_pt"]
        top_band = y0 + CAPTION_TRIM_TOP_SHARE * (y1 - y0)
        for caption in captions:
            cx0, cy0, cx1, cy1 = caption["bbox_pt"]
            width = max(cx1 - cx0, 1e-6)
            inside_x = max(0.0, min(cx1, x1) - max(cx0, x0)) / width
            if inside_x < TITLE_MIN_OVERLAP or not (y0 - 2.0 <= (cy0 + cy1) / 2 <= top_band) or cy1 >= y1:
                continue
            new_top = cy1 + margin_pt
            blockers = [o for o in others if o is not caption and o is not region
                        and not o.get("title_status", "").startswith("attached")
                        and _inside(o["bbox_pt"], region["bbox_pt"]) >= 0.5
                        and o["bbox_pt"][1] < new_top and o["bbox_pt"][3] > y0 and o["bbox_pt"] != region["bbox_pt"]]
            decision = {"type": "caption_trim", "caption_block": caption.get("block_index"),
                        "caption_decision": caption.get("title_decision") or caption.get("title_status"),
                        "caption_bbox_pt": list(caption["bbox_pt"]), "original_bbox_pt": list(region["bbox_pt"])}
            if blockers or new_top >= y1:
                decision.update(applied=False, reason="trimming would cut " + ", ".join(
                    f"block {b.get('block_index')} ({b['label']})" for b in blockers) if blockers else
                    "trimming would remove the whole box")
                region["reasons"].append(f"Needs Review: caption block {caption.get('block_index')} lies in the top "
                                         f"of this box but was not trimmed ({decision['reason']})")
            else:
                region["bbox_pt"] = [x0, round(new_top, 2), x1, y1]
                decision.update(applied=True, trimmed_bbox_pt=list(region["bbox_pt"]),
                                reason=f"caption of another visual in the top {CAPTION_TRIM_TOP_SHARE:.0%}; top edge "
                                       f"moved below it (+{margin_pt:.1f} pt margin)")
                region["reasons"].append(f"top trimmed below caption block {caption.get('block_index')} "
                                         "(it belongs to another visual)")
            region.setdefault("decisions", []).append(decision)
            x0, y0, x1, y1 = region["bbox_pt"]
            top_band = y0 + CAPTION_TRIM_TOP_SHARE * (y1 - y0)


def flag_cut_content(regions: list[dict], others: list[dict]) -> None:
    """Mark regions whose border cuts another block (10-90% of it inside): content may be missing.
    The detected box is kept; the union with the cut block is recorded as a proposed expansion."""
    for region in regions:
        if region.get("status") == "duplicate removed":
            continue
        proposal = list(region["bbox_pt"])
        cut = []
        for other in others:
            if other is region or other.get("title_status", "").startswith("attached"):
                continue
            share = _inside(other["bbox_pt"], region["bbox_pt"])
            if CUT_MIN_SHARE < share < CUT_MAX_SHARE:
                cut.append({"block_index": other.get("block_index"), "label": other["label"],
                            "inside_share": round(share, 3), "bbox_pt": list(other["bbox_pt"])})
                proposal = [min(proposal[0], other["bbox_pt"][0]), min(proposal[1], other["bbox_pt"][1]),
                            max(proposal[2], other["bbox_pt"][2]), max(proposal[3], other["bbox_pt"][3])]
        if cut:
            region["suspected_missing_content"] = cut
            region["proposed_expansion_pt"] = proposal
            region["reasons"].append("Needs Review: the region border cuts " + ", ".join(
                f"block {c['block_index']} ({c['label']}, {c['inside_share']:.0%} inside)" for c in cut)
                + "; a larger box is proposed (not applied)")
            region.setdefault("decisions", []).append({"type": "incomplete_region", "cut_blocks": cut,
                                                       "proposed_expansion_pt": proposal, "applied": False})


# --------------------------------------------------------------- proposal


def propose(document, blocks: list[dict], *, labels: list[str], margin_px: int = DEFAULT_MARGIN_PX,
            titles: bool = True, captions_below: bool = False, expansions: list[str] | None = None) -> list[dict]:
    """Candidates per block, with region ids, pixel boxes (margin, clamped) and crop availability.
    Figure titles directly above a chart/table/image are attached to it (see attach_titles)."""
    by_page: dict[int, list[dict]] = {}
    for block in blocks:
        by_page.setdefault(int(block["page"]), []).append(block)
    candidates = []
    for page_number in sorted(by_page):
        page = document[page_number - 1]
        native, reason = crop_options.native_page_raster(document, page)
        chosen, other = [], []
        items = []
        for block in by_page[page_number]:
            detected = px_to_pt(block["bbox_px"], block["image_px_size"], block["region_bbox_pt"])
            items.append(dict(block, bbox_pt=list(detected), bbox_pt_detected=detected))
        attach_titles([item for item in items if item["label"] in labels or item["label"] in TITLE_LABELS | CAPTION_LABELS],
                      captions_below=captions_below, attach=titles)
        attached_titles = []
        for item in items:
            if item.get("title_status", "").startswith("attached"):
                attached_titles.append(item)  # part of its target's crop; recorded, not a region of its own
                continue
            if item.get("reasons_extra"):
                item["warnings"] = list(item.get("warnings") or []) + item["reasons_extra"]
            (chosen if item["label"] in labels else other).append(item)
        classified = classify(chosen)
        if native is not None:
            margin_pt = margin_px * (native["bbox"][3] - native["bbox"][1]) / native["height_px"]
            trim_foreign_captions([item for item in classified if item["status"] != "duplicate removed"],
                                  [item for item in items if not item.get("title_status", "").startswith("attached")],
                                  margin_pt)
        flag_cut_content([item for item in classified if item["status"] != "duplicate removed"],
                         [item for item in items if not item.get("title_status", "").startswith("attached")])
        number = 0
        for item in classified:
            if item["status"] == "duplicate removed":  # removed automatically; recorded, never a region
                candidates.append(dict(item, region_id=None, pixel_box=None, margin_px=margin_px))
                continue
            item["region_id"] = f"p{page_number}_L{number}"
            number += 1
            if item.get("proposed_expansion_pt") and item["region_id"] in set(expansions or []):
                item["bbox_pt"] = list(item["proposed_expansion_pt"])  # applied by the user; detected box is kept
                item["expansion_applied"] = True
                for decision in item.get("decisions") or []:
                    if decision.get("type") == "incomplete_region":
                        decision["applied"] = True
            if native is None:
                item.update(status="unavailable", default_selected=False, pixel_box=None,
                            reasons=item["reasons"] + [f"not a scanned single-image page: {reason}"])
            else:
                box = pt_to_image_px(item["bbox_pt"], native["bbox"], [native["width_px"], native["height_px"]])
                item["pixel_box"] = with_margin(box, margin_px, [native["width_px"], native["height_px"]])
                item["native_image_px"] = [native["width_px"], native["height_px"]]
                item["native_image_bbox_pt"] = native["bbox"]
            item["margin_px"] = margin_px
            candidates.append(item)
        for item in other:
            candidates.append(dict(item, region_id=None, status="label not selected", default_selected=False,
                                   reasons=[f"label {item['label']!r} is not among {list(labels)}"]
                                   + ([item["title_decision"]] if item.get("title_decision") else [])
                                   + list(item.get("warnings") or []), margin_px=margin_px))
        for item in attached_titles:
            candidates.append(dict(item, region_id=None, status=item["title_status"], default_selected=False,
                                   reasons=["figure title included in its target's crop"]
                                   + ([item["title_decision"]] if item.get("title_decision") else []),
                                   margin_px=margin_px))
    return candidates


def preview_images(document, candidate: dict, *, page_width_px: int = 520) -> tuple[bytes, bytes] | None:
    """(page thumbnail with the crop box drawn, crop thumbnail) from the embedded image pixels."""
    from PIL import Image, ImageDraw

    if not candidate.get("pixel_box"):
        return None
    page = document[int(candidate["page"]) - 1]
    native, _reason = crop_options.native_page_raster(document, page)
    if native is None:
        return None
    image = Image.open(io.BytesIO(native["bytes"])).convert("RGB")
    crop = image.crop(tuple(candidate["pixel_box"]))
    scale = page_width_px / image.width
    thumb = image.resize((page_width_px, max(1, int(image.height * scale))))
    draw = ImageDraw.Draw(thumb)
    x0, y0, x1, y1 = (v * scale for v in candidate["pixel_box"])
    draw.rectangle([x0, y0, x1, y1], outline=(207, 34, 46), width=3)
    page_png, crop_png = io.BytesIO(), io.BytesIO()
    thumb.save(page_png, format="PNG")
    crop.thumbnail((page_width_px, page_width_px))
    crop.save(crop_png, format="PNG")
    return page_png.getvalue(), crop_png.getvalue()


# ------------------------------------------------------------------- save


def create_layout_region_set(run_dir: Path, *, blocks: list[dict], labels: list[str], margin_px: int,
                             selected: list[str], source: dict, pdf_path: Path | None, by: str,
                             titles: bool = True, captions_below: bool = False,
                             detection_report: dict | None = None, expansions: list[str] | None = None,
                             expected_pixel_boxes: dict[str, list[int]] | None = None) -> Path:
    """Write a new region set from layout boxes. The proposal is recomputed from the same inputs
    (never taken from the browser) and every crop is cut from the embedded page pixels."""
    import pymupdf

    from lab.core.hybrid import enrich  # noqa: PLC0415
    from lab.core.hybrid import run as hybrid  # noqa: PLC0415

    if not by.strip():
        raise enrich.EnrichError("Enter your name")
    if not 0 <= int(margin_px) <= 500:
        raise enrich.EnrichError("The margin must be between 0 and 500 px")
    metadata, _plan, _library = hybrid.load(run_dir)
    pdf, pdf_source = hybrid.layout_pdf(run_dir, metadata, pdf_path)
    regions = []
    with pymupdf.open(pdf) as document:
        candidates = {c["region_id"]: c for c in propose(document, blocks, labels=labels, margin_px=int(margin_px),
                                                         titles=titles, captions_below=captions_below,
                                                         expansions=expansions)
                      if c.get("region_id")}
        unknown = [rid for rid in selected if rid not in candidates]
        unusable = [rid for rid in selected if rid in candidates and not candidates[rid].get("pixel_box")]
        if not selected or unknown or unusable:
            raise enrich.EnrichError(f"Select available layout boxes (unknown: {unknown}; unavailable: {unusable})")
        if expected_pixel_boxes is not None:
            differ = [rid for rid in selected if list(candidates[rid]["pixel_box"]) != list(expected_pixel_boxes.get(rid) or [])]
            if differ:
                raise enrich.EnrichError(f"The crops would differ from the preview for {differ} (margin or options "
                                         "changed); review the preview again before saving")
        folder = enrich._new_dir(enrich.enrichment_dir(run_dir), "set")
        for index, rid in enumerate(selected, start=1):
            item = candidates[rid]
            page = document[int(item["page"]) - 1]
            data, options = crop_options.crop_native_pixels(document, page, item["pixel_box"])
            options.update(margin_px=int(margin_px), crop_sha256=enrich._sha(data))
            name = f"crops/{rid}.png"
            enrich._write_once(folder / name, data)
            page_area = (page.rect.width * page.rect.height) or 1.0
            native_px, image_bbox = item["native_image_px"], item["native_image_bbox_pt"]
            sx = (image_bbox[2] - image_bbox[0]) / native_px[0]
            sy = (image_bbox[3] - image_bbox[1]) / native_px[1]
            px = item["pixel_box"]
            crop_bbox = [round(image_bbox[0] + px[0] * sx, 2), round(image_bbox[1] + px[1] * sy, 2),
                         round(image_bbox[0] + px[2] * sx, 2), round(image_bbox[1] + px[3] * sy, 2)]
            region = {
                "region_id": rid, "page": int(item["page"]), "figure_id": rid.split("_", 1)[1], "origin": "layout_box",
                "bbox": crop_bbox, "area_share": round(_area(crop_bbox) / page_area, 4),
                "detected_type": f"layout box: {item['label']}", "layout_revision": None, "parent": None,
                "uncertain": item["status"] != "selected", "uncertainty": list(item["reasons"]),
                "default_selected": item["default_selected"], "layout_crop": None,
                "index": index, "crop_file": name, "crop_sha256": options["crop_sha256"], "crop_bytes": len(data),
                "crop_px": options["pixel_size"], "crop_dpi": options["effective_dpi"], "crop_options": options,
                "page_size_pt": [round(page.rect.width, 2), round(page.rect.height, 2)],
                "layout_box": {"label": item["label"], "source": source, "source_id": item["source_id"],
                               "block_index": item["block_index"], "bbox_px_in_source": item["bbox_px"],
                               "source_image_px": item["image_px_size"], "bbox_pt_without_margin": item["bbox_pt"],
                               "margin_px": int(margin_px), "pixel_box": item["pixel_box"],
                               "status_at_selection": item["status"], "reasons": item["reasons"],
                               "score": item.get("score"), "model_name": item.get("model_name"),
                               "title_attachments": item.get("title_attachments") or [],
                               "bbox_pt_without_title": item.get("bbox_pt_without_title"),
                               "bbox_pt_detected": item.get("bbox_pt_detected"),
                               "decisions": item.get("decisions") or [],
                               "suspected_missing_content": item.get("suspected_missing_content") or [],
                               "proposed_expansion_pt": item.get("proposed_expansion_pt"),
                               "expansion_applied": bool(item.get("expansion_applied"))},
            }
            region["size_based_timeout_s"] = crop_options.size_based_timeout(region)
            regions.append(region)
    enrich._write_once(folder / "regions.json", {
        "schema_version": 1, "kind": "hybrid_region_set", "enrich_version": enrich.ENRICH_VERSION,
        "set_id": folder.name, "run_id": Path(run_dir).name, "document": metadata["document"], "pdf_source": pdf_source,
        "crop_dpi": "native", "crop_request": {"resolution": "native", "format": "png", "jpeg_quality": None},
        "region_source": {"kind": "layout_boxes", "layout_source": source, "labels": list(labels),
                          "preview_verified": expected_pixel_boxes is not None,
                          "margin_px": int(margin_px), "titles_above_attached": bool(titles),
                          "captions_below_attached": bool(captions_below),
                          "expansions_applied": sorted(expansions or []), "rules": {
                              "exact_duplicates": "removed", "small_long_side_pt": MIN_LONG_SIDE_PT,
                              "near_duplicate_iou": NEAR_DUPLICATE_IOU,
                              "small_or_overlapping": "kept, not preselected, marked uncertain",
                              "title_attachment": f"figure_title above (gap <= {TITLE_MAX_GAP_PT} pt) or overlapping "
                                                  f"the top boundary (<= {TITLE_MAX_TOP_OVERLAP_PT} pt), horizontal "
                                                  f"overlap >= {TITLE_MIN_OVERLAP:.0%} of title width, unambiguous; "
                                                  "nearer visual above -> caption of that visual",
                              "containment": f">= {CONTAINMENT:.0%} inside -> contained; outer preferred unless >= 2 "
                                             f"disjoint, non-small inner visuals each >= {SPLIT_MIN_INNER_SHARE:.0%} "
                                             f"and together >= {SPLIT_MIN_COVERAGE:.0%} of the outer box",
                              "incomplete": f"another block {CUT_MIN_SHARE:.0%}-{CUT_MAX_SHARE:.0%} inside the border "
                                            "-> Needs Review; expansion proposed, applied only by the user"},
                          "note": "saved-provider boxes test the cropping mechanism only; the target layout source "
                                  "is local PP-DocLayout"},
        "created_at": enrich._now().isoformat(), "created_by": by.strip(), "regions": regions,
    })
    # The layout detection the set was made from, saved write-once next to it.
    enrich._write_once(folder / "layout_detection.json", {"source": source, "blocks": blocks,
                                                          "report": detection_report or {}})
    return folder
