"""E2-D analysis and reports for a two-stage hosted API batch.

Evidence classes are kept strictly apart and labelled in every output:

* HOSTED API output: what each provider returned (same model family as the local
  weights, revision unverified; never claimed identical to local weights).
* PDF TEXT LAYER: a *library extraction* (pypdfium2) of the PDF's own text. It is
  an unverified, indicative reference, NOT Ground Truth. It contains no chart
  images' text, and Word exports garble some Arabic (lam-alef order).
* GROUND TRUTH: official accuracy metrics are computed only from independently
  verified Ground Truth pages. None exists for the traffic reports, so this
  module produces qualitative comparisons and human-review tasks instead.

Nothing here is sent anywhere; analysis happens after inference, locally.
"""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from lab.core.e2.api import batch as batch_mod
from lab.core.e2.metrics_e2 import parse_hypothesis_tables, repetition_stats
from lab.core.metrics import (
    DATE_RE,
    IDENTIFIER_RE,
    NUMBER_RE,
    UnverifiedGroundTruthError,
    _line_order_tau,
    evaluate_page,
)
from lab.core.models import GroundTruth, PageResult

REFERENCE_LABEL = "PDF text layer: library extraction (pypdfium2), unverified, NOT Ground Truth"
API_LABEL = "Hosted API output: same model family, revision unverified; not local weights"
NO_GT_NOTE = (
    "No independently verified Ground Truth exists for these documents, so no accuracy "
    "score (CER/WER/F1) is reported. Every comparison below is indicative and must be "
    "confirmed by human review."
)
ARABIC_TEXT_LAYER_NOTE = (
    "The Arabic PDF is a Word export whose text layer swaps lam-alef pairs (e.g. "
    "'السالمة' for 'السلامة'). Comparisons fold alef variants and lam-alef order on BOTH "
    "sides so that this artefact is not counted against a model."
)

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟـ]")
_ALEF = re.compile(r"[أإآ]")
_LAM_ALEF = re.compile(r"[ال]{2,}")
_MARKUP = re.compile(r"<[^>]+>|[|#*_`>]+|^\s*[-:]+\s*$", re.MULTILINE)
_WORD = re.compile(r"\d(?:[\d.,٫٬/:]*\d)?[%٪]?|[^\W\d_]+", re.UNICODE)


# --------------------------------------------------------------- normalizing


def clean_output(text: str) -> str:
    """Model output without markdown/HTML markup (text kept)."""
    return _MARKUP.sub(" ", text or "")


def fold(text: str) -> str:
    """Comparison-only folding, applied identically to every side."""
    value = unicodedata.normalize("NFKC", text or "").translate(_DIGITS).lower()
    value = _ARABIC_DIACRITICS.sub("", value)
    value = _ALEF.sub("ا", value).replace("ى", "ي")
    # Order-insensitive within each run of alef/lam letters (the Word lam-alef swap).
    return _LAM_ALEF.sub(lambda match: "".join(sorted(match.group(0))), value)


def words(text: str) -> list[str]:
    """Letter runs and whole numbers (1,284 / ١٢٨٤ / 45% are one canonical token each)."""
    return [_canonical_number(token) if token[0].isdigit() else token for token in _WORD.findall(fold(text))]


def _canonical_number(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).translate(_DIGITS)
    value = value.replace("٫", ".").replace("٬", "").replace("−", "-").replace("٪", "%")
    value = re.sub(r"\s+", "", value)
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?%?", value):
        value = value.replace(",", "")
    return value


def values(text: str) -> dict[str, Counter]:
    """Numbers, percentages, dates and identifiers as canonical multisets."""
    normalized = unicodedata.normalize("NFKC", text or "")
    numbers = [_canonical_number(item.group(0)) for item in NUMBER_RE.finditer(normalized)]
    dates = Counter(_canonical_number(item.group(0)) for item in DATE_RE.finditer(normalized))
    return {
        "numbers": Counter(item for item in numbers if not item.endswith("%") and item not in dates),
        "percentages": Counter(item for item in numbers if item.endswith("%")),
        "dates": dates,
        "identifiers": Counter(
            item.group(0) for item in IDENTIFIER_RE.finditer(normalized)
            if any(char.isdigit() for char in item.group(0).translate(_DIGITS))
            and _canonical_number(item.group(0)) not in dates
        ),
    }


def _overlap(reference: Counter, hypothesis: Counter) -> dict[str, Any]:
    common = sum((reference & hypothesis).values())
    return {
        "reference": sum(reference.values()),
        "output": sum(hypothesis.values()),
        "matched": common,
        "recall": common / sum(reference.values()) if reference else None,
        "precision": common / sum(hypothesis.values()) if hypothesis else None,
        "missing": sorted((reference - hypothesis).elements())[:40],
        "unsupported": sorted((hypothesis - reference).elements())[:40],
    }


# --------------------------------------------------------------- per page


def analyze_output(text: str, reference: str, features: dict) -> dict[str, Any]:
    cleaned = clean_output(text)
    ref_words, out_words = Counter(words(reference)), Counter(words(cleaned))
    tables = parse_hypothesis_tables(text or "")
    ref_values, out_values = values(reference), values(cleaned)
    tau, matched_lines, ref_lines, out_lines = _line_order_tau(fold(reference), fold(cleaned))
    ref_long_lines = [line.strip() for line in reference.splitlines() if len(line.strip()) >= 25]
    out_folded_lines = [fold(line) for line in cleaned.splitlines() if line.strip()]
    missing_lines = []
    for line in ref_long_lines:
        target = fold(line)
        best = max((SequenceMatcher(None, target, other).ratio() for other in out_folded_lines), default=0.0)
        if best < 0.5 and not any(target in other for other in out_folded_lines):
            missing_lines.append(line[:160])
    has_images = bool(features.get("image_objects"))
    unsupported_note = (
        "not in the text layer: may be chart/image text (check the image) or hallucinated"
        if has_images else "not in the text layer: possible hallucination or misread"
    )
    return {
        "chars": len(text or ""),
        "lines": len([line for line in (text or "").splitlines() if line.strip()]),
        "words": sum(out_words.values()),
        "text_layer_word_recall": (sum((ref_words & out_words).values()) / sum(ref_words.values()))
        if ref_words else None,
        "text_layer_word_precision": (sum((ref_words & out_words).values()) / sum(out_words.values()))
        if out_words else None,
        "values": {name: _overlap(ref_values[name], out_values[name]) for name in ref_values},
        "unsupported_values_note": unsupported_note,
        "tables": {"count": len(tables), "rows": sum(len(table) for table in tables),
                   "cells": sum(len(row) for table in tables for row in table)},
        "reading_order_tau_vs_text_layer": tau,
        "reading_order_lines_matched": matched_lines,
        "possibly_missing_lines": missing_lines[:15],
        "possibly_missing_line_count": len(missing_lines),
        "repetition": repetition_stats(text or ""),
    }


def _pairwise(texts: dict[str, str]) -> dict[str, Any]:
    keys = sorted(texts)
    if len(keys) != 2:
        return {}
    left, right = (words(clean_output(texts[key])) for key in keys)
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    left_values, right_values = values(clean_output(texts[keys[0]])), values(clean_output(texts[keys[1]]))
    agreed = {name: sum((left_values[name] & right_values[name]).values()) for name in left_values}
    only = {name: {keys[0]: sorted((left_values[name] - right_values[name]).elements())[:20],
                   keys[1]: sorted((right_values[name] - left_values[name]).elements())[:20]}
            for name in left_values}
    return {"providers": keys, "word_sequence_similarity": matcher.ratio(),
            "values_agreed": agreed, "values_only_in": only}


# ----------------------------------------------------------------- loading


def _run_payload(run_dir: str | None) -> dict | None:
    if not run_dir:
        return None
    path = Path(run_dir) / "run.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _category(item: dict) -> str:
    features = item.get("page_features") or {}
    if features.get("path_objects", 0) >= 100:
        return "table"
    if features.get("image_objects"):
        return "chart/image"
    return "prose"


def analyze_batch(
    batch_dir: Path, root: Path, sources: dict[str, batch_mod.SourceDocument],
    *, ground_truths: dict[str, GroundTruth] | None = None,
) -> dict[str, Any]:
    plan = batch_mod.load_plan(batch_dir)
    status = batch_mod.batch_status(batch_dir, root)
    by_item = {item["item_id"]: item for item in status["items"]}
    pages: dict[tuple[str, int], dict] = {}
    rows: list[dict] = []
    for item in plan["items"]:
        state = by_item[item["item_id"]]
        source = sources.get(item["registry_id"])
        key = (item["registry_id"], item["page"])
        if key not in pages:
            reference = batch_mod.text_layer(source.pdf_path, item["page"]) if source else ""
            pages[key] = {"registry_id": item["registry_id"], "language": item["language"], "page": item["page"],
                          "page_role": item.get("page_role"), "category": _category(item),
                          "features": item.get("page_features"), "text_layer_chars": len(reference),
                          "_reference": reference, "outputs": {}}
        run = _run_payload(state.get("run_directory")) if state["status"] in batch_mod.DONE | {
            batch_mod.FAILED, batch_mod.RETRY_AUTHORIZED} else None
        page_result = (run or {}).get("pages", [{}])[0] if run else {}
        text = page_result.get("text") or ""
        usage = (run or {}).get("usage") or {}
        api = ((run or {}).get("environment") or {}).get("api") or {}
        raw = ((((run or {}).get("environment") or {}).get("e2") or {}).get("page_raw") or {}).get(str(item["page"])) or {}
        response = raw.get("response") or {}
        row = {
            "item_id": item["item_id"], "registry_id": item["registry_id"], "language": item["language"],
            "page": item["page"], "category": pages[key]["category"], "provider_id": item["provider_id"],
            "status": state["status"], "detail": state.get("detail"), "run_id": (run or {}).get("run_id"),
            "run_directory": state.get("run_directory"), "reused": state["status"] == batch_mod.REUSED,
            "latency_ms": (run or {}).get("wall_ms"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "cost_usd": usage.get("cost_usd"),
            "cost_source": usage.get("cost_source"), "finish_reason": response.get("finish_reason"),
            "truncated": response.get("truncated_at_max_new_tokens"),
            "reported_model": response.get("reported_model"), "job_id": api.get("job_id"),
            "error": (run or {}).get("error"), "image_sha256": item["image_sha256"],
            "evidence_class": API_LABEL,
        }
        if run and (run.get("status") == "ok"):
            row["analysis"] = analyze_output(text, pages[key]["_reference"], item.get("page_features") or {})
            row["official_metrics"] = _official(run, item, ground_truths)
            pages[key]["outputs"][item["provider_id"]] = text
        rows.append(row)
    comparisons = []
    for page in pages.values():
        comparisons.append({key: value for key, value in page.items() if key not in {"_reference", "outputs"}}
                           | {"cross_provider": _pairwise(page["outputs"])})
    analysis = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": plan["batch_id"], "stage": plan["stage"], "plan_sha256": plan["plan_sha256"],
        "definition_title": plan["definition_title"], "overall": status["overall"],
        "documents": plan["documents"], "providers": plan["providers"],
        "evidence_classes": {"api": API_LABEL, "reference": REFERENCE_LABEL,
                             "ground_truth": "official metrics only from independently verified Ground Truth"},
        "notes": [NO_GT_NOTE, ARABIC_TEXT_LAYER_NOTE] if not any(
            row.get("official_metrics") for row in rows) else [ARABIC_TEXT_LAYER_NOTE],
        "rows": rows, "pages": comparisons,
    }
    analysis["provider_summary"] = provider_summary(analysis)
    analysis["review_tasks"] = review_tasks(analysis)
    analysis["errors"] = [row for row in rows if row["status"] not in batch_mod.DONE]
    analysis["recommendation"] = recommendation(analysis)
    return analysis


def _official(run: dict, item: dict, ground_truths: dict[str, GroundTruth] | None) -> dict | None:
    """Accuracy only against verified Ground Truth for this exact document hash."""
    truth = (ground_truths or {}).get(item["document_sha256"])
    if truth is None or truth.doc_sha256 != item["document_sha256"]:
        return None
    page = next((entry for entry in truth.pages if entry.page == item["page"]), None)
    if page is None:
        return None
    try:
        metric = evaluate_page(run["run_id"], "raw", PageResult.model_validate(run["pages"][0]), page, truth.version)
    except UnverifiedGroundTruthError:
        return None
    return metric.model_dump(mode="json")


# ------------------------------------------------------------- summaries


def _mean(items: list[float | None]) -> float | None:
    present = [float(item) for item in items if isinstance(item, (int, float))]
    return sum(present) / len(present) if present else None


def provider_summary(analysis: dict) -> dict[str, Any]:
    summary = {}
    for provider in analysis["providers"]:
        provider_id = provider["provider_id"]
        rows = [row for row in analysis["rows"] if row["provider_id"] == provider_id]
        done = [row for row in rows if row.get("analysis")]
        by_category = {}
        for category in sorted({row["category"] for row in rows}):
            subset = [row for row in done if row["category"] == category]
            by_category[category] = {
                "pages": len(subset),
                "mean_text_layer_word_recall": _mean([row["analysis"]["text_layer_word_recall"] for row in subset]),
                "mean_number_recall": _mean([row["analysis"]["values"]["numbers"]["recall"] for row in subset]),
                "tables_found": sum(row["analysis"]["tables"]["count"] for row in subset),
            }
        summary[provider_id] = {
            "label": provider["label"],
            "pages_total": len(rows),
            "pages_with_output": len(done),
            "pages_reused": sum(row["reused"] for row in rows),
            "pages_not_completed": len(rows) - sum(row["status"] in batch_mod.DONE for row in rows),
            "mean_latency_ms": _mean([row["latency_ms"] for row in done if not row["reused"]]),
            "prompt_tokens": sum(row["prompt_tokens"] or 0 for row in done),
            "completion_tokens": sum(row["completion_tokens"] or 0 for row in done),
            "cost_usd": round(sum(float(row["cost_usd"] or 0) for row in rows if not row["reused"]), 6),
            "truncated_pages": sum(row["truncated"] is True for row in done),
            "repetition_flagged_pages": sum(bool(row["analysis"]["repetition"]["flagged"]) for row in done),
            "mean_text_layer_word_recall": _mean([row["analysis"]["text_layer_word_recall"] for row in done]),
            "mean_reading_order_tau": _mean([row["analysis"]["reading_order_tau_vs_text_layer"] for row in done]),
            "unsupported_numbers": sum(len(row["analysis"]["values"]["numbers"]["unsupported"]) for row in done),
            "missing_numbers": sum(len(row["analysis"]["values"]["numbers"]["missing"]) for row in done),
            "by_category": by_category,
        }
    return summary


def review_tasks(analysis: dict) -> list[dict[str, Any]]:
    """Concrete things a human must check; nothing here is a verdict."""
    tasks = []
    for row in analysis["rows"]:
        base = {"item_id": row["item_id"], "provider_id": row["provider_id"], "registry_id": row["registry_id"],
                "page": row["page"], "run_id": row["run_id"]}
        if row["status"] not in batch_mod.DONE:
            tasks.append(base | {"priority": "high", "task": f"Resolve item status {row['status']}: {row['detail']}"})
            continue
        result = row.get("analysis")
        if not result:
            continue
        tasks.append(base | {"priority": "normal",
                             "task": "Compare the RAW transcription with the page image; record PASS/FAIL/NEEDS_REVIEW"})
        numbers = result["values"]["numbers"]
        if numbers["missing"]:
            tasks.append(base | {"priority": "high", "task": "Check numbers present in the text layer but absent "
                                 f"from the output: {', '.join(numbers['missing'][:12])}"})
        if numbers["unsupported"]:
            tasks.append(base | {"priority": "high", "task": f"Check numbers {result['unsupported_values_note']}: "
                                 + ", ".join(numbers["unsupported"][:12])})
        for name in ("percentages", "dates", "identifiers"):
            entry = result["values"][name]
            if entry["missing"] or entry["unsupported"]:
                tasks.append(base | {"priority": "normal", "task": f"Check {name}: missing {entry['missing'][:8]}, "
                                     f"not in text layer {entry['unsupported'][:8]}"})
        if row["category"] == "table" and not result["tables"]["count"]:
            tasks.append(base | {"priority": "high", "task": "Table page but no table structure in the output; "
                                 "check whether rows/columns were flattened or lost"})
        if row["category"] == "chart/image":
            tasks.append(base | {"priority": "normal", "task": "Verify chart/indicator text against the image "
                                 "(the text layer has no chart text)"})
        if result["repetition"]["flagged"]:
            tasks.append(base | {"priority": "high", "task": "Repetition loop detected; check for duplicated content"})
        if row["truncated"]:
            tasks.append(base | {"priority": "high", "task": "Output truncated at max_tokens; content is missing"})
        if result["possibly_missing_line_count"]:
            tasks.append(base | {"priority": "normal", "task": f"{result['possibly_missing_line_count']} text-layer "
                                 "line(s) have no similar output line; check for omitted content"})
        tau = result["reading_order_tau_vs_text_layer"]
        if tau is not None and tau < 0.5:
            tasks.append(base | {"priority": "normal", "task": f"Reading order differs from the text layer "
                                 f"(tau {tau:.2f}); check column/RTL order"})
    return tasks


def recommendation(analysis: dict) -> dict[str, Any]:
    """Evidence-based guidance with explicit limits. Never a final winner from Stage 1."""
    summary = analysis["provider_summary"]
    stage = analysis["stage"]
    providers = list(summary)
    evidence: list[str] = []
    leaders: dict[str, str] = {}
    categories = sorted({category for item in summary.values() for category in item["by_category"]})
    for category in categories:
        scores = {pid: summary[pid]["by_category"].get(category, {}).get("mean_text_layer_word_recall")
                  for pid in providers}
        known = {pid: score for pid, score in scores.items() if score is not None}
        if len(known) == len(providers) == 2:
            (best, best_score), (other, other_score) = sorted(known.items(), key=lambda kv: -kv[1])
            if best_score - other_score >= 0.05:
                leaders[category] = best
                evidence.append(f"{category}: {best} recovers more text-layer words "
                                f"({best_score:.2f} vs {other_score:.2f}; indicative)")
            else:
                evidence.append(f"{category}: no clear difference ({best_score:.2f} vs {other_score:.2f})")
        else:
            evidence.append(f"{category}: not enough completed pages from both providers")
    for pid in providers:
        item = summary[pid]
        if item["pages_not_completed"]:
            evidence.append(f"{pid}: {item['pages_not_completed']} page(s) not completed")
        if item["repetition_flagged_pages"] or item["truncated_pages"]:
            evidence.append(f"{pid}: {item['repetition_flagged_pages']} repetition-flagged, "
                            f"{item['truncated_pages']} truncated page(s)")
    distinct = set(leaders.values())
    if stage == 1:
        verdict = "PRELIMINARY - no winner"
        guidance = ("Six pages cannot establish a winner. Use Stage 1 to check that both providers work "
                    "end to end, then run Stage 2 (all pages) and complete human review before deciding.")
        if len(distinct) > 1:
            guidance += " Different providers lead on different page types, so a hybrid route is a hypothesis to test."
    elif not leaders:
        verdict = "NO CLEAR DIFFERENCE"
        guidance = ("Neither provider leads clearly. Consider multiple models with disagreement-triggered review, "
                    "and decide on cost, latency and operational fit, after human review.")
    elif len(distinct) > 1:
        verdict = "HYBRID CANDIDATE"
        guidance = ("Leaders differ by page type (" + ", ".join(f"{k}: {v}" for k, v in sorted(leaders.items()))
                    + "). A hybrid route by page type is supported by the indicative evidence; confirm with "
                    "human review and verified Ground Truth before adopting it.")
    elif len(leaders) == len(categories) and not any(summary[pid]["pages_not_completed"] for pid in distinct):
        verdict = "SINGLE-MODEL CANDIDATE"
        guidance = (f"{next(iter(distinct))} leads on every page type. Confirm with human review and verified "
                    "Ground Truth; a hosted result does not prove the same for local weights.")
    else:
        verdict = "MIXED EVIDENCE"
        guidance = ("One provider leads on some page types and the others are tied or incomplete; "
                    "keep both under evaluation and rely on human review.")
    return {"verdict": verdict, "guidance": guidance, "evidence": evidence,
            "limits": [NO_GT_NOTE, "Hosted APIs report no model revision; local weights may behave differently.",
                       "Text-layer comparisons cannot see chart text and are distorted where the PDF text "
                       "layer is imperfect."]}


# ------------------------------------------------------------------ exports


CSV_FIELDS = [
    "item_id", "registry_id", "language", "page", "category", "provider_id", "status", "reused", "run_id",
    "latency_ms", "prompt_tokens", "completion_tokens", "cost_usd", "cost_source", "finish_reason", "truncated",
    "reported_model", "chars", "words", "text_layer_word_recall", "text_layer_word_precision",
    "numbers_recall", "numbers_missing", "numbers_unsupported", "percentages_missing", "dates_missing",
    "tables", "table_rows", "reading_order_tau", "repetition_flagged", "possibly_missing_lines", "error",
]


def to_csv(analysis: dict) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in analysis["rows"]:
        result = row.get("analysis") or {}
        value = result.get("values") or {}
        writer.writerow({key: row.get(key) for key in CSV_FIELDS} | ({
            "chars": result["chars"], "words": result["words"],
            "text_layer_word_recall": _fmt(result["text_layer_word_recall"]),
            "text_layer_word_precision": _fmt(result["text_layer_word_precision"]),
            "numbers_recall": _fmt(value["numbers"]["recall"]),
            "numbers_missing": len(value["numbers"]["missing"]),
            "numbers_unsupported": len(value["numbers"]["unsupported"]),
            "percentages_missing": len(value["percentages"]["missing"]),
            "dates_missing": len(value["dates"]["missing"]),
            "tables": result["tables"]["count"], "table_rows": result["tables"]["rows"],
            "reading_order_tau": _fmt(result["reading_order_tau_vs_text_layer"]),
            "repetition_flagged": result["repetition"]["flagged"],
            "possibly_missing_lines": result["possibly_missing_line_count"],
        } if result else {}))
    return buffer.getvalue()


def _fmt(value) -> str:
    return "" if value is None else f"{value:.4f}"


def to_json(analysis: dict) -> str:
    return json.dumps(analysis, ensure_ascii=False, indent=2, default=str)


def to_markdown(analysis: dict) -> str:
    lines = [
        f"# E2-D Stage {analysis['stage']} report: {analysis['definition_title']}",
        "",
        f"Batch `{analysis['batch_id']}` | plan SHA-256 `{analysis['plan_sha256']}` | status **{analysis['overall']}** "
        f"| generated {analysis['generated_at']}",
        "",
        "## Evidence classes",
        f"- **API**: {analysis['evidence_classes']['api']}",
        f"- **Reference**: {analysis['evidence_classes']['reference']}",
        f"- **Ground Truth**: {analysis['evidence_classes']['ground_truth']}",
        "",
        *[f"> {note}" for note in analysis["notes"]],
        "",
        "## Documents",
        "| Registry id | Language | Pages | SHA-256 | Provenance |",
        "|---|---|---|---|---|",
        *[f"| {d['registry_id']} | {d['language']} | {d['page_count']} | `{d['sha256']}` | {d['provenance']} |"
          for d in analysis["documents"]],
        "",
        "## Provider summary",
        "| Provider | With output | Reused | Not completed | Mean latency ms | Tokens in/out | Cost USD "
        "| Truncated | Repetition | Mean text-layer word recall | Mean reading-order tau |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pid, item in analysis["provider_summary"].items():
        lines.append(
            f"| {pid} | {item['pages_with_output']}/{item['pages_total']} | {item['pages_reused']} | "
            f"{item['pages_not_completed']} | {_num(item['mean_latency_ms'], 0)} | {item['prompt_tokens']}/"
            f"{item['completion_tokens']} | {item['cost_usd']:.4f} | {item['truncated_pages']} | "
            f"{item['repetition_flagged_pages']} | {_num(item['mean_text_layer_word_recall'])} | "
            f"{_num(item['mean_reading_order_tau'])} |")
    lines += ["", "### By page type (indicative)", "| Provider | Page type | Pages | Word recall | Number recall | Tables found |",
              "|---|---|---|---|---|---|"]
    for pid, item in analysis["provider_summary"].items():
        for category, values_ in item["by_category"].items():
            lines.append(f"| {pid} | {category} | {values_['pages']} | {_num(values_['mean_text_layer_word_recall'])} | "
                         f"{_num(values_['mean_number_recall'])} | {values_['tables_found']} |")
    lines += ["", "## Per page", "| Document | Page | Type | Provider | Status | Words | Word recall | Numbers missing/unsupported "
              "| Tables | Order tau | Latency ms | Cost USD |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in analysis["rows"]:
        result = row.get("analysis")
        if result:
            numbers = result["values"]["numbers"]
            lines.append(
                f"| {row['registry_id']} | {row['page']} | {row['category']} | {row['provider_id']} | {row['status']} | "
                f"{result['words']} | {_num(result['text_layer_word_recall'])} | {len(numbers['missing'])}/"
                f"{len(numbers['unsupported'])} | {result['tables']['count']} | "
                f"{_num(result['reading_order_tau_vs_text_layer'])} | {_num(row['latency_ms'], 0)} | "
                f"{_num(row['cost_usd'], 4)} |")
        else:
            lines.append(f"| {row['registry_id']} | {row['page']} | {row['category']} | {row['provider_id']} | "
                         f"{row['status']} | - | - | - | - | - | - | - |")
    lines += ["", "## Cross-provider agreement (no reference implied)", "| Document | Page | Word-sequence similarity | Numbers agreed |",
              "|---|---|---|---|"]
    for page in analysis["pages"]:
        cross = page.get("cross_provider") or {}
        if cross:
            lines.append(f"| {page['registry_id']} | {page['page']} | {cross['word_sequence_similarity']:.2f} | "
                         f"{cross['values_agreed']['numbers']} |")
    lines += ["", "## Errors and incomplete items"]
    lines += [f"- {row['item_id']}: **{row['status']}** - {row['detail'] or row['error'] or ''}" for row in analysis["errors"]] \
        or ["- none"]
    lines += ["", "## Possible missing content (text-layer lines without a similar output line)"]
    found = False
    for row in analysis["rows"]:
        result = row.get("analysis")
        if result and result["possibly_missing_lines"]:
            found = True
            lines.append(f"- {row['item_id']} ({result['possibly_missing_line_count']}):")
            lines += [f"  - {line}" for line in result["possibly_missing_lines"][:5]]
    if not found:
        lines.append("- none detected")
    rec = analysis["recommendation"]
    lines += ["", "## Recommendation", f"**{rec['verdict']}**: {rec['guidance']}", "", "Evidence:",
              *[f"- {item}" for item in rec["evidence"]], "", "Limits:", *[f"- {item}" for item in rec["limits"]],
              "", f"## Human review tasks ({len(analysis['review_tasks'])})"]
    lines += [f"- [{task['priority']}] {task['item_id']}: {task['task']}" for task in analysis["review_tasks"][:200]]
    return "\n".join(lines) + "\n"


def _num(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


__all__ = ["analyze_batch", "analyze_output", "recommendation", "to_csv", "to_json", "to_markdown"]
