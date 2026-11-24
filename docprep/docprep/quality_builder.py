"""Evidence-based quality inspection across Nass, Structure and Relationships."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from typing import Any

from .quality_config import QualityConfig
from .quality_inputs import QualityInputs, rehash_quality_inputs
from .quality_model import InventoryRecord, QualityFinding, QualityResult
from .quality_policy import (
    RAG_FLAG_BY_SUBTYPE,  # noqa: F401  (re-exported for callers that imported it from here)
    affects_content_reliability,
    finding_reason,
    rag_flag,
    reliability_dimension,
    unflagged_content_concern,
)
from .relation_text import NON_VALUE_DIMENSIONS, content_terms, normalize_text, numeric_tokens

CATEGORIES = {
    "A": "source_extraction",
    "B": "ocr_corruption",
    "C": "numeric_integrity",
    "D": "duplicate_leakage",
    "E": "structure_quality",
    "F": "relationship_quality",
    "G": "table_quality",
    "H": "figure_chart_quality",
    "I": "cross_page_continuity",
}
SEVERITY_WEIGHT = {"critical": 100, "high": 70, "medium": 40, "low": 20, "info": 5}
CONFIDENCE_WEIGHT = {"high": 15, "medium": 8, "low": 2}
TERMINAL_RE = re.compile(r"[.!?؟…]\s*$")
MALFORMED_ARITHMETIC_RE = re.compile(r"(?<!\w)[+\-]?[0-9٠-٩۰-۹][0-9٠-٩۰-۹,.٬٫\s]*\+\s*(?:=|$)")
TRUNCATED_DECIMAL_RE = re.compile(r"(?<!\w)[0-9٠-٩۰-۹]+[.٫,]\s*(?:$|[|])")
MOJIBAKE_RE = re.compile(r"(?:(?:Ã.|Â.|Ø.|Ù.|Ð.|Ñ.)){2,}")
REPEATED_TOKEN_RE = re.compile(r"(?i)(?<!\w)([A-Za-z\u0621-\u06ff]{2,})(?:\s+\1){2,}(?!\w)")
MIXED_TOKEN_RE = re.compile(r"(?=\S*[A-Za-z])(?=\S*[\u0621-\u06ff])\S+")
SUSPICIOUS_PUNCT_RE = re.compile(r"([!?؟،,;:])\1{3,}")
TECHNICAL_RE = re.compile(
    r"(?:https?://|www\.|\w+@\w+|\w+[\\/]\w+|\w+\.\w{1,5}$|[_{}<>])", re.IGNORECASE,
)
RELIABLE_GEOMETRY = {"precise", "approximate"}
EXPLICIT_FAILURE_CODES = {
    "extraction_failure", "processing_error", "ocr_failure", "job_failed", "page_processing_failed",
}
OPTIONAL_CHECK_TERMS = ("layout", "validation", "verification", "inspection", "structure_check")
CHART_CLASS_RE = re.compile(r"(?i)\b(?:chart|plot|graph|bar_chart|line_chart|pie_chart)\b|(?:مخطط|رسم\s+بياني)")
MAP_VISUAL_RE = re.compile(r"(?i)\b(?:map|province)\b|(?:خريط(?:ة|ه)|منطق(?:ة|ه)\s+جغرافية)")
CODE_VISUAL_RE = re.compile(
    r"(?i)(?:\b(?:class|def|dtype|randomforestclassifier|decisiontreeclassifier)\b|\.fit\s*\(|"
    r"\.predict\s*\(|#\s*model)"
)
WORD_TOKEN_RE = re.compile(r"[A-Za-z\u0621-\u06ff]{4,}")
LOOP_TOKEN_RE = re.compile(r"[0-9A-Za-z\u0621-\u064a\u0660-\u0669\u06f0-\u06f9%\u066a]+(?:[.,:][0-9\u0660-\u0669]+)*")
MEASURE_TERM_RE = re.compile(
    r"(?<![\w])(?:[وبلف]?(?:ال|لل)?)(?:عدد|اعداد|نسبه|معدل|متوسط|اجمالي|مجموع|count|number of|rate|average|mean|percentage|percent|"
    r"total|frequency)(?![\w])"
)
MIRRORED_BRACKET_RE = re.compile(r"\)(?P<inner>[A-Za-z0-9](?:[A-Za-z0-9.\-_/ ]{0,38}[A-Za-z0-9])?)\(")
LOOP_MIN_PHRASE_TOKENS = 3
LOOP_MAX_PHRASE_TOKENS = 12
LOOP_MIN_REPEATS = 3


def _stable_id(category: str, subtype: str, source_ids: list[str], evidence: dict[str, Any]) -> str:
    canonical = json.dumps([category, subtype, sorted(source_ids), evidence], ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"), default=str)
    return f"qf_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:18]}"


def _excerpt(text: str, start: int | None = None, end: int | None = None, radius: int = 80) -> str:
    if start is None or end is None:
        return text[:240]
    return text[max(0, start - radius):min(len(text), end + radius)]


def _nested_numeric_values(value: Any, path: str = "") -> list[tuple[str, float]]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            if key in {"ocr_confidence", "recognition_confidence", "text_confidence"} and isinstance(child, (int, float)):
                found.append((key_path, float(child)))
            else:
                found.extend(_nested_numeric_values(child, key_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_nested_numeric_values(child, f"{path}[{index}]"))
    return found


def _nested_status_records(value: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Return only explicitly failed/skipped records from saved metadata."""
    found = []
    if isinstance(value, dict):
        status = str(value.get("status") or "").casefold()
        has_error = bool(value.get("error") or value.get("exception"))
        if status in {"failed", "failure", "error", "skipped", "incomplete"} or has_error:
            found.append((path or "execution_manifest", value))
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.extend(_nested_status_records(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_nested_status_records(child, f"{path}[{index}]"))
    return found


def _edit_distance(left: str, right: str, maximum: int = 2) -> int:
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left_char != right_char)))
        if min(current) > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def _script(token: str) -> str:
    return "arabic" if re.search(r"[ء-ۿ]", token) else "latin"


def _word_records(text: str) -> list[dict[str, Any]]:
    records = []
    for match in WORD_TOKEN_RE.finditer(text):
        normalized = normalize_text(match.group(0))
        if normalized:
            records.append({
                "raw": match.group(0), "normalized": normalized,
                "start": match.start(), "end": match.end(),
            })
    return records


def _context_alignment(
    visual_tokens: list[dict[str, Any]], visual_index: int,
    support_tokens: list[dict[str, Any]], support_index: int,
) -> list[dict[str, Any]]:
    matches = []
    for relative in (-2, -1, 1, 2):
        visual_position = visual_index + relative
        support_position = support_index + relative
        if not (0 <= visual_position < len(visual_tokens) and 0 <= support_position < len(support_tokens)):
            continue
        visual_token = visual_tokens[visual_position]["normalized"]
        support_token = support_tokens[support_position]["normalized"]
        if visual_token == support_token or _morphological_variant(visual_token, support_token):
            matches.append({
                "relative_position": relative,
                "visual_token": visual_tokens[visual_position]["raw"],
                "support_token": support_tokens[support_position]["raw"],
                "comparison": "exact" if visual_token == support_token else "morphological_variant",
            })
    return matches


def _has_strong_local_context(matches: list[dict[str, Any]]) -> bool:
    return len(matches) >= 2 and any(abs(item["relative_position"]) == 1 for item in matches)


def _strip_one_suffix(token: str, suffixes: tuple[str, ...]) -> set[str]:
    return {token[:-len(suffix)] for suffix in suffixes if token.endswith(suffix) and len(token) - len(suffix) >= 3}


def _morphological_variant(left: str, right: str) -> bool:
    """Conservatively recognize common morphology that edit distance must not call OCR noise."""
    if left == right:
        return True
    if left.isascii() and right.isascii():
        suffixes = ("ing", "ies", "ied", "ed", "es", "s")
        left_stems = {left, *_strip_one_suffix(left, suffixes)}
        right_stems = {right, *_strip_one_suffix(right, suffixes)}
        return bool(left_stems & right_stems)

    def arabic_stems(token: str) -> set[str]:
        stems = {token}
        if token.startswith("و") and len(token) > 4:
            stems.add(token[1:])
        if token.startswith("ال") and len(token) > 5:
            stems.add(token[2:])
        suffixes = ("يات", "ات", "ون", "ين", "ان", "يه", "ها", "هم", "هن", "ت", "ه")
        stems.update(_strip_one_suffix(token, suffixes))
        for value in tuple(stems):
            if value.startswith("ال") and len(value) > 5:
                stems.add(value[2:])
            if value.startswith("و") and len(value) > 4:
                stems.add(value[1:])
        return stems

    left_stems, right_stems = arabic_stems(left), arabic_stems(right)
    if left_stems & right_stems:
        return True
    if any(
        longer.endswith("ي") and any(
            char == "ا" and longer[:index] + longer[index + 1:] == shorter
            for index, char in enumerate(longer)
        )
        for shorter in left_stems for longer in right_stems if len(longer) == len(shorter) + 1
    ) or any(
        longer.endswith("ي") and any(
            char == "ا" and longer[:index] + longer[index + 1:] == shorter
            for index, char in enumerate(longer)
        )
        for shorter in right_stems for longer in left_stems if len(longer) == len(shorter) + 1
    ):
        return True
    # Common Arabic singular/plural spelling pattern, e.g. final -ى versus -اوى after normalization.
    return any(
        len(longer) == len(shorter) + 1 and longer == f"{shorter[:-1]}ا{shorter[-1:]}"
        for shorter in left_stems for longer in right_stems
    ) or any(
        len(longer) == len(shorter) + 1 and longer == f"{shorter[:-1]}ا{shorter[-1:]}"
        for shorter in right_stems for longer in left_stems
    )


class QualityBuilder:
    def __init__(self, inputs: QualityInputs, config: QualityConfig):
        self.inputs, self.config = inputs, config
        self.elements = {item.element_id: item for item in inputs.run.elements}
        self.ordered = list(inputs.run.elements)
        self.assignments = list(inputs.assignments.get("assignments") or [])
        self.assignment_by_element: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for assignment in self.assignments:
            self.assignment_by_element[str(assignment.get("element_id"))].append(assignment)
        self.sections = {str(item.get("section_id")): item for item in inputs.section_tree.get("sections") or []}
        self.relationships = list(inputs.relationships.get("relationships") or [])
        self.relation_nodes = {str(item.get("node_id")): item for item in inputs.relationships.get("nodes") or []}
        self.visuals = {str(item.get("visual_id")): item for item in inputs.visuals.get("visuals") or []}
        self.findings: list[QualityFinding] = []
        self.finding_ids: set[str] = set()
        self.evidence_audit: list[dict[str, Any]] = []
        self.inventory = self._inventory()
        self.inventory_by_id = {item.source_id: item for item in self.inventory}
        self.nodes = self._nodes()

    def audit_check(
        self, check: str, status: str, element: Any, reason: str, **evidence: Any,
    ) -> None:
        """Record one explicit, source-preserving decision for a quality check."""
        self.evidence_audit.append({
            "component": "document_quality",
            "check": check,
            "status": status,
            "source_id": element.element_id,
            "page": element.page,
            "reason": reason,
            "evidence": evidence,
        })

    @staticmethod
    def _valid_bbox(value: Any) -> list[float] | None:
        if not isinstance(value, list) or len(value) != 4 or not all(isinstance(item, (int, float)) for item in value):
            return None
        bbox = [float(item) for item in value]
        return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None

    def _geometry_reliability(self, element: Any, assignments: list[dict[str, Any]]) -> str:
        paragraph_assignments = [item for item in assignments if item.get("role") == "paragraph"]
        for assignment in paragraph_assignments:
            bbox = self._valid_bbox(assignment.get("bbox_pt"))
            if bbox and (assignment.get("span_id") or assignment.get("unit_id") != element.element_id):
                return "approximate"
            if bbox:
                return "precise"
        if element.element_type == "page_text":
            return "page_level_only" if self._valid_bbox(element.bbox) else "unavailable"
        return "precise" if self._valid_bbox(element.bbox) else "unavailable"

    def _assignment_geometry(self, assignment: dict[str, Any]) -> tuple[str, list[float] | None]:
        element = self.elements[str(assignment.get("element_id"))]
        reliability = self._geometry_reliability(element, [assignment])
        bbox = self._valid_bbox(assignment.get("bbox_pt"))
        if bbox is None and assignment.get("unit_id") == element.element_id:
            bbox = self._valid_bbox(element.bbox)
        return reliability, bbox

    def _nodes(self) -> dict[str, dict[str, Any]]:
        nodes = dict(self.relation_nodes)
        for element in self.ordered:
            nodes.setdefault(element.element_id, {
                "node_id": element.element_id, "node_type": element.element_type, "page": element.page,
                "text": element.text or "", "source_id": element.element_id, "start": None, "end": None,
                "provenance": {"validation_state": element.validation_state},
            })
        for item in self.inputs.assignments.get("source_inventory") or []:
            for span in item.get("derived_spans") or []:
                span_id = str(span.get("span_id"))
                nodes.setdefault(span_id, {
                    "node_id": span_id, "node_type": "structure_derived_span", "page": item.get("page"),
                    "text": span.get("text"), "source_id": item.get("element_id"),
                    "start": span.get("start"), "end": span.get("end"), "provenance": span,
                })
        return nodes

    def _inventory(self) -> list[InventoryRecord]:
        relations_by_id: dict[str, list[str]] = defaultdict(list)
        for relation in self.relationships:
            for source_id in (relation.get("source_id"), relation.get("target_id")):
                if source_id in self.elements:
                    relations_by_id[str(source_id)].append(str(relation.get("rel_id")))
        output = []
        for element in self.ordered:
            assignments = self.assignment_by_element.get(element.element_id, [])
            assignment = next((item for item in assignments if item.get("unit_id") == element.element_id),
                              assignments[0] if assignments else None)
            skipped = {}
            if element.element_type not in {"figure", "table"}:
                skipped["figure_chart_quality"] = "not a figure or table"
            if element.element_type != "table":
                skipped["table_quality"] = "not a table"
            if not element.text:
                skipped["text_pattern_checks"] = "source has no extracted text"
                skipped["numeric_integrity"] = "source has no extracted text"
            geometry_reliability = self._geometry_reliability(element, assignments)
            if not any(item.get("role") == "paragraph" for item in assignments):
                skipped["cross_page_continuity"] = "source is not assigned as paragraph text"
            elif geometry_reliability not in RELIABLE_GEOMETRY:
                skipped["cross_page_continuity"] = (
                    f"insufficient paragraph geometry ({geometry_reliability}); page-level coordinates are not used"
                )
            output.append(InventoryRecord(
                element.element_id, element.page, element.element_type, element.validation_state, assignment,
                sorted(relations_by_id.get(element.element_id, [])), list(CATEGORIES.values()), skipped,
                geometry_reliability,
            ))
        return output

    def add_finding(self, category: str, subtype: str, severity: str, confidence: str,
                    source_ids: list[str], pages: list[int], evidence: dict[str, Any], explanation: str,
                    action: str, status: str = "open", *, offsets: list[dict[str, Any]] | None = None,
                    related_ids: list[str] | None = None, relationship_ids: list[str] | None = None,
                    limitations: list[str] | None = None, method: str = "heuristic") -> QualityFinding:
        source_ids = sorted(set(source_ids))
        pages = sorted({page for page in pages if isinstance(page, int)})
        finding_id = _stable_id(category, subtype, source_ids, evidence)
        if finding_id in self.finding_ids:
            return next(item for item in self.findings if item.finding_id == finding_id)
        priority = SEVERITY_WEIGHT[severity] + CONFIDENCE_WEIGHT[confidence]
        if category in {"numeric_integrity", "source_extraction"}:
            priority += 10
        if subtype in {"recorded_processing_failure", "numeric_conflict", "missing_extraction"}:
            priority += 15
        finding = QualityFinding(
            finding_id, category, subtype, severity, confidence, source_ids, pages, evidence, explanation, action,
            status, offsets or [], sorted(set(related_ids or [])), sorted(set(relationship_ids or [])),
            limitations or [], method, priority,
        )
        self.findings.append(finding)
        self.finding_ids.add(finding_id)
        for source_id in source_ids:
            if source_id in self.inventory_by_id:
                self.inventory_by_id[source_id].finding_ids.append(finding_id)
        return finding

    def _record_source_ids(self, record: dict[str, Any], page: int | None, *, page_wide: bool) -> list[str]:
        """Resolve the narrowest saved scope; page fallback is explicit, never accidental."""
        element_id = record.get("element_id")
        if element_id and str(element_id) in self.elements:
            return [str(element_id)]
        region_id = record.get("region_id")
        if region_id:
            region_keys = {str(region_id)}
            for page_record in (self.inputs.run.layout or {}).get("pages") or []:
                if not isinstance(page_record, dict):
                    continue
                selection = page_record.get("selection") if isinstance(page_record.get("selection"), dict) else {}
                for region in selection.get("regions") or []:
                    if isinstance(region, dict) and str(region.get("region_id")) == str(region_id):
                        region_keys.update(str(value) for value in (
                            region.get("region_id"), region.get("job_id"),
                        ) if value)
            matched = []
            for element in self.ordered:
                provenance = element.raw.get("provenance")
                if isinstance(provenance, dict) and str(provenance.get("job_id")) in region_keys:
                    matched.append(element.element_id)
            return matched
        if page_wide and page:
            return [item.element_id for item in self.ordered if item.page == page]
        return []

    def detect_source_quality(self) -> None:
        by_state: dict[str, list[Any]] = defaultdict(list)
        for element in self.ordered:
            if element.validation_state in {"needs_review", "pending_review"}:
                by_state[str(element.validation_state)].append(element)
            for path, score in _nested_numeric_values(element.raw):
                if score < self.config.low_ocr_confidence:
                    self.add_finding(
                        "source_extraction", "low_recorded_ocr_confidence", "medium", "high",
                        [element.element_id], [element.page], {"field_path": path, "recorded_value": score,
                         "threshold": self.config.low_ocr_confidence},
                        "The saved extraction metadata explicitly records low OCR confidence.",
                        "Inspect the original page beside the exact extracted text.", method="saved_metadata",
                    )
        for state, elements in by_state.items():
            self.add_finding(
                "source_extraction", f"nass_{state}", "medium", "high",
                [item.element_id for item in elements], [item.page for item in elements],
                {"validation_state": state, "count": len(elements)},
                f"Nass explicitly marked {len(elements)} element(s) as {state}.",
                "Review the marked source elements; do not assume the OCR is wrong without the page image.",
                "needs_review", method="nass_validation_state",
            )
        for path, record in _nested_status_records(self.inputs.run.execution_manifest or {}):
            page = record.get("page") if isinstance(record.get("page"), int) else None
            source_ids = self._record_source_ids(record, page, page_wide=not record.get("region_id"))
            status_value = str(record.get("status") or "").casefold()
            context = " ".join((path, str(record.get("code") or ""), str(record.get("kind") or ""))).casefold()
            optional = bool(record.get("optional")) or (
                any(term in context for term in OPTIONAL_CHECK_TERMS)
                and not any(term in context for term in ("extraction", "processing", "ocr", "page_job"))
            )
            explicit_failure = status_value in {"failed", "failure", "error"} or bool(
                record.get("error") or record.get("exception")
            )
            if explicit_failure and not optional:
                subtype, severity, status = "recorded_processing_failure", "high", "open"
                explanation = "The saved execution metadata explicitly records an actual processing failure."
            elif explicit_failure:
                subtype, severity, status = "optional_verification_failure", "medium", "unverified"
                explanation = "A saved optional verification or inspection step failed; extraction failure is not established."
            elif status_value == "skipped":
                subtype, severity, status = "recorded_processing_skip", "low", "unverified"
                explanation = "The saved execution metadata records a skipped operation; this is not a confirmed failure."
            else:
                subtype, severity, status = "unknown_processing_status", "low", "unverified"
                explanation = "The saved execution metadata records an incomplete or unknown processing status."
            evidence = {
                "field_path": path,
                "status": record.get("status"),
                "reason": record.get("reason"),
                "error": record.get("error") or record.get("exception"),
                "region_id": record.get("region_id"),
            }
            self.add_finding(
                "source_extraction", subtype, severity, "high", source_ids,
                [page] if page else [], evidence,
                explanation,
                "Inspect the saved execution record; do not infer missing content.", status,
                limitations=[] if source_ids else ["execution record has no directly identified source element"],
                method="saved_execution_manifest",
            )
        for flag in (self.inputs.run.validation_report or {}).get("flags") or []:
            if not isinstance(flag, dict):
                continue
            code = str(flag.get("code") or "unknown")
            message = str(flag.get("message") or "")
            page = flag.get("page") if isinstance(flag.get("page"), int) else None
            text = f"{code} {message}".casefold()
            code_key = code.casefold()
            if "reading_order" in text or "reading order" in text:
                subtype, severity, status = "ambiguous_reading_order", "medium", "needs_review"
            elif code_key == "layout_review" and any(term in text for term in ("check failed", "verification failed")):
                subtype, severity, status = "optional_layout_verification_failure", "medium", "unverified"
            elif code_key == "layout_review":
                subtype, severity, status = "nonfatal_layout_warning", "low", "unverified"
            elif any(term in text for term in ("missing_visual", "missing visual", "skipped_visual",
                                                "skipped visual", "image skipped", "dropped image")):
                subtype, severity, status = "missing_extraction", "high", "open"
            elif code_key == "incomplete_region":
                subtype, severity, status = "incomplete_region_coverage", "medium", "unverified"
            elif code_key in EXPLICIT_FAILURE_CODES:
                subtype, severity, status = "recorded_processing_failure", "high", "open"
            elif any(term in text for term in ("masked", "coverage", "unmatched")):
                subtype, severity, status = "layout_coverage_limitation", "low", "unverified"
            else:
                continue
            page_wide = subtype == "ambiguous_reading_order" or not (
                flag.get("region_id") or flag.get("element_id")
            )
            source_ids = self._record_source_ids(flag, page, page_wide=page_wide)
            self.add_finding(
                "source_extraction", subtype, severity, "high", source_ids, [page] if page else [],
                {"saved_flag": flag}, message or f"Saved validation flag {code}.",
                "Inspect the saved flag and affected page; preserve the source output.", status,
                limitations=[] if source_ids else ["flag has no directly identified source element"],
                method="saved_validation_report",
            )

    def detect_ocr_suspects(self) -> None:
        for element in self.ordered:
            text = element.text or ""
            if not text or TECHNICAL_RE.search(text):
                continue
            patterns = [
                (MOJIBAKE_RE, "mojibake_sequence", "medium", "high"),
                (REPEATED_TOKEN_RE, "repeated_token_run", "medium", "medium"),
                (SUSPICIOUS_PUNCT_RE, "repeated_punctuation", "low", "medium"),
                (TRUNCATED_DECIMAL_RE, "truncated_numeric_expression", "medium", "medium"),
            ]
            if element.element_type != "table":
                patterns.insert(0, (MALFORMED_ARITHMETIC_RE, "malformed_arithmetic_sequence", "medium", "high"))
            for pattern, subtype, severity, confidence in patterns:
                for match in pattern.finditer(text):
                    self.add_finding(
                        "ocr_corruption", subtype, severity, confidence, [element.element_id], [element.page],
                        {"verbatim": match.group(0), "offsets": [match.start(), match.end()]},
                        "The source contains a suspicious extraction pattern; the intended text is unknown.",
                        "Compare this exact span with the original page. Do not auto-correct it.", "unverified",
                        offsets=[{"source_id": element.element_id, "start": match.start(), "end": match.end(),
                                  "text": match.group(0)}], method="conservative_text_pattern",
                    )
            for match in MIXED_TOKEN_RE.finditer(text):
                token = match.group(0)
                if len(token) < 4 or TECHNICAL_RE.search(token):
                    continue
                self.add_finding(
                    "ocr_corruption", "unexpected_mixed_script_token", "low", "low",
                    [element.element_id], [element.page], {"verbatim": token,
                                                          "offsets": [match.start(), match.end()]},
                    "A single token mixes Arabic and Latin scripts; this may be valid technical text.",
                    "Review only if the token is not a name, identifier or technical expression.", "unverified",
                    offsets=[{"source_id": element.element_id, "start": match.start(), "end": match.end(),
                              "text": token}],
                    limitations=["mixed script is not proof of OCR corruption"],
                )
            if "\ufffd" in text or any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in text):
                positions = [index for index, char in enumerate(text)
                             if char == "\ufffd" or (unicodedata.category(char) == "Cc" and char not in "\n\r\t")]
                self.add_finding(
                    "ocr_corruption", "invalid_or_replacement_character", "high", "high",
                    [element.element_id], [element.page], {"positions": positions[:50], "count": len(positions)},
                    "The extracted text contains replacement or invalid control characters.",
                    "Inspect the source and extraction logs.", method="unicode_integrity",
                )

    def _upstream_ocr_attempts(self, element: Any) -> dict[str, Any] | None:
        """Saved OCR-service loop flags for an element's crop (read-only; endpoint fields are not copied)."""
        crop = element.raw.get("crop") if isinstance(element.raw.get("crop"), dict) else {}
        provenance = element.raw.get("provenance") if isinstance(element.raw.get("provenance"), dict) else {}
        job_ids = [str(value) for value in (
            provenance.get("job_id"), Path(str(crop.get("file"))).stem if crop.get("file") else None,
        ) if value]
        run_dir = Path(str(self.inputs.run.run_dir))
        for job_id in dict.fromkeys(job_ids):
            folder = run_dir / "raw" / job_id
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", job_id) or not folder.is_dir():
                continue
            results = sorted(folder.glob("attempt_*.result.json"))
            if not results:
                continue
            try:
                record = json.loads(results[-1].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            response = record.get("response") if isinstance(record.get("response"), dict) else {}
            final = response.get("final_flags") if isinstance(response.get("final_flags"), dict) else {}
            attempts = [item for item in response.get("attempts") or [] if isinstance(item, dict)]
            return {
                "record": f"raw/{job_id}/{results[-1].name}",
                "final_loop_stopped": final.get("loop_stopped"),
                "final_hit_limit": final.get("hit_limit"),
                "retried": response.get("retried"),
                "final_attempt": response.get("final_attempt"),
                "attempt_loop_stopped": [item.get("loop_stopped") for item in attempts],
            }
        return None

    @staticmethod
    def _repetition_runs(text: str) -> list[dict[str, Any]]:
        """Consecutive repeats (>= LOOP_MIN_REPEATS) of one phrase of >= LOOP_MIN_PHRASE_TOKENS tokens."""
        tokens = [(match.group(0), match.start(), match.end()) for match in LOOP_TOKEN_RE.finditer(text)]
        normalized = [normalize_text(token[0]) for token in tokens]
        runs = []
        index = 0
        while index < len(tokens):
            best = None
            for size in range(LOOP_MIN_PHRASE_TOKENS, LOOP_MAX_PHRASE_TOKENS + 1):
                if index + LOOP_MIN_REPEATS * size > len(tokens):
                    break
                phrase = normalized[index:index + size]
                if len(set(phrase)) < 2 or not any(re.search(r"[A-Za-zء-ي]", item) for item in phrase):
                    continue
                count = 1
                while normalized[index + count * size:index + (count + 1) * size] == phrase:
                    count += 1
                if count >= LOOP_MIN_REPEATS and (best is None or count * size > best[0] * best[1]):
                    best = (count, size)
            if best is None:
                index += 1
                continue
            count, size = best
            starts = [tokens[index + k * size][1] for k in range(count)]
            start, end = tokens[index][1], tokens[index + count * size - 1][2]
            runs.append({
                "start": start, "end": end, "phrase": text[start:tokens[index + size - 1][2]],
                "phrase_tokens": size, "repeat_count": count,
                "line_aligned": all(pos == 0 or text[pos - 1] in "\n\r" for pos in starts),
                "token_share": round(count * size / max(1, len(tokens)), 3),
            })
            index += count * size
        return runs

    def detect_ocr_repetition_loops(self) -> None:
        for element in self.ordered:
            text = element.text or ""
            if element.element_type == "table":
                self.inventory_by_id[element.element_id].skipped_checks.setdefault(
                    "ocr_repetition_loop", "tables legitimately repeat row and column labels")
                continue
            upstream = self._upstream_ocr_attempts(element) if element.element_type == "figure" else None
            upstream_loop = bool(upstream and upstream.get("final_loop_stopped") is True)
            # Line-aligned repeats (one label per line, as in legends or lists) need upstream loop evidence.
            runs = [run for run in self._repetition_runs(text) if upstream_loop or not run["line_aligned"]]
            if not runs and not upstream_loop:
                continue
            signals = []
            if runs:
                signals.append("consecutive_phrase_repetition")
            if upstream_loop:
                signals.append("upstream_final_loop_stopped")
            both = len(signals) == 2
            self.add_finding(
                "ocr_corruption", "ocr_repetition_loop", "high" if both else "medium",
                "high" if both else "medium", [element.element_id], [element.page],
                {"runs": runs, "signals": signals, "upstream_ocr_record": upstream,
                 "upstream_validation_state": element.validation_state,
                 "rule": {"min_phrase_tokens": LOOP_MIN_PHRASE_TOKENS, "min_consecutive_repeats": LOOP_MIN_REPEATS,
                          "normalization": "comparison only (case, diacritics, Arabic letter variants)"}},
                "The OCR output repeats the same phrase consecutively"
                + (" and the saved OCR record reports that generation stopped in a loop" if upstream_loop else "")
                + "; the repeated span is probably a decoding loop, not document content.",
                "Review the crop against the page; keep the extracted text unchanged and do not use the repeated "
                "span as evidence.", "needs_review",
                offsets=[{"source_id": element.element_id, "start": run["start"], "end": run["end"],
                          "text": text[run["start"]:run["end"]]} for run in runs],
                limitations=["the image is not re-read; which words are genuinely in the visual is unknown",
                             "the upstream validation_state is preserved as recorded"],
                method="+".join(signals),
            )

    def _pdf_page_chars(self, page: int) -> list[tuple[str, float, float, float, float]] | None:
        """(char, x0, y0, x1, y1) for one page of the saved source PDF; None when unavailable."""
        cache = self.__dict__.setdefault("_pdf_char_cache", {})
        if page in cache:
            return cache[page]
        chars = None
        pdf = Path(str(self.inputs.run.run_dir)) / "source.pdf"
        try:
            import fitz  # PyMuPDF, already used by the Structure stage for typography
            if pdf.is_file():
                with fitz.open(pdf) as document:
                    if 1 <= page <= document.page_count:
                        raw = document[page - 1].get_text("rawdict")
                        chars = [(item["c"], *item["bbox"]) for block in raw.get("blocks", [])
                                 for line in block.get("lines", []) for span in line.get("spans", [])
                                 for item in span.get("chars", [])]
        except Exception:  # noqa: BLE001 - geometry is optional evidence; absence is recorded, not raised
            chars = None
        cache[page] = chars
        return chars

    def _bracket_geometry(self, element: Any, inner: str) -> list[dict[str, Any]]:
        """Layout evidence that an extracted ')X(' is the visual left-to-right sequence of a logical '(X)'.

        True when the PDF draws '(' to the right of X and ')' to its left on the same line: in a right-to-left
        line the logical order then starts with '(' — the extracted order follows ascending x instead."""
        chars = self._pdf_page_chars(element.page)
        bbox = self._valid_bbox(element.bbox)
        found: list[dict[str, Any]] = []
        if not chars or not bbox:
            return found
        pad = 2.0
        local = [c for c in chars if c[1] >= bbox[0] - pad and c[3] <= bbox[2] + pad
                 and c[2] >= bbox[1] - pad and c[4] <= bbox[3] + pad]
        target = normalize_text(inner).replace(" ", "")
        for opening in (c for c in local if c[0] == "("):
            mid = (opening[2] + opening[4]) / 2
            height = max(1.0, opening[4] - opening[2])
            same_line = [c for c in local if abs((c[2] + c[4]) / 2 - mid) <= 0.4 * height]
            for closing in (c for c in same_line if c[0] == ")" and c[1] < opening[1]):
                between = sorted((c for c in same_line if closing[3] - 0.5 <= c[1] and c[3] <= opening[1] + 0.5
                                  and c[0] not in "()"), key=lambda c: c[1])
                if normalize_text("".join(c[0] for c in between)).replace(" ", "") != target:
                    continue
                arabic = re.compile(r"[ء-ۿ]")
                if any(arabic.match(c[0]) for c in same_line):
                    rtl_context = "arabic_on_same_pdf_line"
                else:
                    # A wrapped line of a right-aligned Arabic cell or paragraph: the line directly above, in the
                    # same column, is Arabic and ends at the same right edge.
                    right = max(c[3] for c in same_line)
                    above = [c for c in local if opening[2] - 1.5 * height <= c[2] < opening[2] - 0.2 * height
                             and c[3] <= right + 2.0 and c[1] >= closing[1] - 12 * height]
                    if not above or abs(max(c[3] for c in above) - right) > 2.0 or \
                            not any(arabic.match(c[0]) for c in above):
                        continue
                    rtl_context = "right_aligned_arabic_line_directly_above"
                found.append({"open_bracket_bbox": [round(v, 1) for v in opening[1:]],
                        "close_bracket_bbox": [round(v, 1) for v in closing[1:]],
                        "enclosed_glyphs": "".join(c[0] for c in between),
                        "rtl_context": rtl_context,
                        "observation": "the PDF places '(' right of the enclosed text and ')' left of it in a "
                                       "right-to-left context: logical order is '(…)', but the extracted text lists "
                                       "the brackets in ascending x"})
                break
        # Reading order of the matches (top to bottom), so the k-th text occurrence maps to the k-th pair.
        return sorted(found, key=lambda item: (round(item["open_bracket_bbox"][1]), -item["open_bracket_bbox"][0]))

    def detect_bidi_order_suspects(self) -> None:
        bracket_flagged: set[str] = set()
        for element in self.ordered:
            text = element.text or ""
            seen: Counter[str] = Counter()
            for match in MIRRORED_BRACKET_RE.finditer(text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                line = text[line_start:line_end if line_end >= 0 else len(text)]
                if not re.search(r"[ء-ۿ]", line):
                    continue
                inner = match.group("inner")
                occurrence = seen[inner]
                seen[inner] += 1
                matches = self._bracket_geometry(element, inner)
                geometry = matches[occurrence] if occurrence < len(matches) else None
                if geometry is None:
                    self.inventory_by_id[element.element_id].skipped_checks.setdefault(
                        "bidi_bracket_order", "mirrored-looking brackets without confirming PDF glyph geometry "
                                              "(language switching alone is not flagged)")
                    continue
                self.add_finding(
                    "ocr_corruption", "bidi_order_suspect", "medium", "high",
                    [element.element_id], [element.page],
                    {"suspected_span": match.group(0), "offsets": [match.start(), match.end()],
                     "line_text": line.strip(), "layout_evidence": geometry,
                     "signals": ["mirrored_bracket_pair_in_rtl_line", "pdf_glyph_geometry"]},
                    "Brackets around an embedded left-to-right term appear in visual (left-to-right) order inside a "
                    "right-to-left line; the logical order in the source is probably '(' first.",
                    "Review the span against the page; keep the extracted text unchanged (no silent reversal).",
                    "needs_review",
                    offsets=[{"source_id": element.element_id, "start": match.start(), "end": match.end(),
                              "text": match.group(0)}],
                    limitations=["only bracket order is assessed; the enclosed term itself is not reordered"],
                    method="mirrored_brackets+pdf_glyph_geometry",
                )
                bracket_flagged.add(element.element_id)
        for element in self.ordered:
            text = element.text or ""
            if not text:
                self.audit_check("bidi_order_suspect", "skipped", element, "source has no extracted text")
                continue
            if not re.search(r"[A-Za-z]", text) or not re.search(r"[\u0621-\u06ff]", text):
                self.audit_check(
                    "bidi_order_suspect", "checked", element,
                    "no mixed Arabic/Latin sequence requiring bidirectional-order inspection",
                )
                continue
            lines = text.splitlines(keepends=True)
            cursor = 0
            flagged: list[dict[str, Any]] = []
            for index, line in enumerate(lines):
                clean = line.strip()
                start = cursor + len(line) - len(line.lstrip())
                cursor += len(line)
                latin_only = bool(re.search(r"[A-Za-z]", clean)) and not re.search(r"[\u0621-\u06ff]", clean)
                previous = lines[index - 1].strip() if index else ""
                following = lines[index + 1].strip() if index + 1 < len(lines) else ""
                unfinished_previous = bool(re.search(r"[\u0621-\u06ff]", previous)) and not TERMINAL_RE.search(previous)
                connector_following = bool(re.match(r"(?:و|ف|ثم|في|على|من|إلى)\b", following))
                signals: list[str] = []
                suspected_start, suspected_end = start, start + len(clean)
                if latin_only and index and index + 1 < len(lines) and unfinished_previous and connector_following:
                    signals = ["standalone_ltr_fragment", "unfinished_arabic_before", "arabic_connector_after"]
                elif index and unfinished_previous:
                    embedded = re.search(r"(?<![A-Za-z])([A-Za-z][A-Za-z0-9._/-]*)\s+"
                                         r"((?:أن|إن|حيث|بينما|وقد|وأن|لكن|إلا)\b)", clean)
                    preceding_connector = re.search(r"(?:إلى|في|من|على|داخل|بين)\s*$", previous)
                    if embedded and preceding_connector and re.search(r"[\u0621-\u06ff]", clean[:embedded.start()]):
                        suspected_start = start + embedded.start(1)
                        suspected_end = start + embedded.end(2)
                        signals = [
                            "embedded_ltr_fragment", "unfinished_arabic_before",
                            "previous_line_ends_with_continuation_connector",
                            "arabic_clause_connector_immediately_after_ltr_fragment",
                        ]
                if not signals:
                    continue
                self.add_finding(
                    "ocr_corruption", "bidi_order_suspect", "medium", "medium",
                    [element.element_id], [element.page],
                    {"suspected_span": text[suspected_start:suspected_end],
                     "offsets": [suspected_start, suspected_end], "previous_line": previous,
                     "current_line": clean, "following_line": following, "signals": signals},
                    "An LTR fragment has strong local line-order signals that it interrupts the logical Arabic clause.",
                    "Review line order against the page; keep the extracted text unchanged.", "needs_review",
                    offsets=[{"source_id": element.element_id, "start": suspected_start,
                              "end": suspected_end, "text": text[suspected_start:suspected_end]}],
                    limitations=["mixed-language text is valid unless all recorded structural signals occur"],
                    method="conservative_line_sequence_bidi_signals",
                )
                flagged.append({"offsets": [suspected_start, suspected_end], "signals": signals})
            was_flagged = bool(flagged) or element.element_id in bracket_flagged
            self.audit_check(
                "bidi_order_suspect", "flagged" if was_flagged else "checked", element,
                "strong local bidirectional-order signals found" if was_flagged else
                "mixed-script content inspected; no complete conservative line-order signal set found",
                candidates=flagged, bracket_geometry_confirmed=element.element_id in bracket_flagged,
            )

    def _visual_vocabulary_references(self, element: Any) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        section_ids = {
            str(item.get("section_id")) for item in self.assignment_by_element.get(element.element_id, [])
            if item.get("section_id")
        }
        for section_id in sorted(section_ids):
            section = self.sections.get(section_id) or {}
            text = str(section.get("title") or "")
            if text:
                heading_ids = [str(value) for value in section.get("heading_element_ids") or [] if value]
                references.append({
                    "kind": "section_heading", "text": text,
                    "source_ids": heading_ids or [section_id],
                })
        visual = self.visuals.get(element.element_id) or {}
        caption = visual.get("caption") if isinstance(visual.get("caption"), dict) else {}
        if caption and caption.get("text"):
            references.append({
                "kind": "explicit_caption", "text": str(caption["text"]),
                "source_ids": [str(caption.get("source_id"))],
            })
        caption_source_ids = {
            str(relation.get("source_id")) for relation in self.relationships
            if relation.get("type") == "caption_of" and relation.get("target_id") == element.element_id
        }
        for source_id in sorted(caption_source_ids):
            source = self.elements.get(source_id)
            if source and source.text:
                references.append({"kind": "explicit_caption", "text": source.text, "source_ids": [source_id]})
        for key in ("verified_text", "alternative_text"):
            value = element.raw.get(key)
            if isinstance(value, str) and value.strip() and element.raw.get(f"{key}_verified") is True:
                references.append({
                    "kind": "verified_same_source_text", "text": value,
                    "source_ids": [element.element_id],
                })
        unique = {}
        for reference in references:
            key = reference["kind"], normalize_text(reference["text"]), tuple(reference["source_ids"])
            unique[key] = reference
        return list(unique.values())

    def detect_chart_and_vocabulary_quality(self) -> None:
        composite_nodes = {
            str(relation.get(key))
            for relation in self.relationships if relation.get("type") == "composite_part"
            for key in ("source_id", "target_id")
        }
        body_counts: Counter[str] = Counter()
        body_occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for element in self.ordered:
            if element.element_type in {"figure", "table"}:
                continue
            tokens = _word_records(element.text or "")
            for index, token in enumerate(tokens):
                body_counts[token["normalized"]] += 1
                body_occurrences[token["normalized"]].append({
                    "source_id": element.element_id, "page": element.page,
                    "order": element.order,
                    "section_ids": sorted({
                        str(item.get("section_id")) for item in self.assignment_by_element.get(
                            element.element_id, []
                        ) if item.get("section_id")
                    }),
                    "tokens": tokens, "index": index,
                })
        for element in self.ordered:
            if element.element_type == "table":
                self.audit_check(
                    "chart_without_values", "skipped", element,
                    "table content is evaluated by table-quality checks, not as a chart",
                )
                self.audit_check(
                    "ocr_vocabulary_noise", "skipped", element,
                    "table vocabulary is not treated as visual OCR vocabulary",
                )
                continue
            if element.element_type != "figure":
                continue
            if not element.text:
                reason = ("empty source is covered by a saved composite relationship; inspect its linked visual" if
                          element.element_id in composite_nodes or element.validation_state == "included_in_composite" else
                          "visual has no extracted text")
                self.audit_check("chart_without_values", "skipped", element, reason)
                self.audit_check("ocr_vocabulary_noise", "skipped", element, reason)
                continue
            if element.validation_state == "included_in_composite":
                self.audit_check(
                    "chart_without_values", "skipped", element,
                    "included composite part is assessed through its composite parent",
                )
                self.audit_check(
                    "ocr_vocabulary_noise", "skipped", element,
                    "included composite part is assessed through its composite parent",
                )
                continue
            metadata_text = json.dumps(element.raw, ensure_ascii=False, sort_keys=True, default=str)
            chart_classified = bool(CHART_CLASS_RE.search(" ".join((
                str(element.raw.get("label") or ""), str(element.raw.get("classification") or ""),
                str(element.raw.get("visual_type") or ""), str(element.raw.get("figure_type") or ""),
            ))))
            visual_record = self.visuals.get(element.element_id) or {}
            internal_title = visual_record.get("internal_title") or {}
            title_end = ((internal_title.get("offsets") or [None, None])[1]
                         if isinstance(internal_title, dict) else None)
            trailing_labels = element.text[title_end:].strip() if isinstance(title_end, int) else ""
            title_and_labels = bool(internal_title and content_terms(trailing_labels))
            usable_numbers = [token for token in numeric_tokens(element.text)
                              if token.dimension != "identifier"]
            excluded_kind = None
            if CODE_VISUAL_RE.search(element.text):
                excluded_kind = "code screenshot or code-like visual"
            elif MAP_VISUAL_RE.search(element.text) and not internal_title:
                excluded_kind = "map-like visual without independently verified chart title"
            chart_eligible = bool(chart_classified or title_and_labels) and excluded_kind is None
            if chart_eligible and not usable_numbers:
                self.add_finding(
                    "figure_chart_quality", "chart_without_values", "medium", "high",
                    [element.element_id], [element.page],
                    {"saved_chart_classification": {
                        key: element.raw.get(key) for key in ("label", "classification", "visual_type", "figure_type")
                        if element.raw.get(key) is not None
                    }, "numeric_token_count": 0, "content_terms": sorted(content_terms(element.text))[:50]},
                    "Independent chart evidence is present, but the extracted visual text has labels and no "
                    "usable numeric tokens.",
                    "Review whether the chart OCR is incomplete; do not infer any missing values.", "unverified",
                    limitations=["the source image is not inspected; the check does not claim which values exist"],
                    method="chart_evidence+numeric_token_inventory",
                )
                self.audit_check(
                    "chart_without_values", "flagged", element,
                    "eligible chart has title/label evidence but no usable numeric token",
                    saved_chart_classification=chart_classified,
                    internal_title_node_id=internal_title.get("node_id"),
                    trailing_label_terms=sorted(content_terms(trailing_labels)),
                )
            elif excluded_kind:
                self.audit_check("chart_without_values", "skipped", element, excluded_kind)
            else:
                self.audit_check(
                    "chart_without_values", "checked" if chart_eligible else "skipped", element,
                    "eligible chart contains usable numeric evidence" if chart_eligible else
                    "no independent chart classification or verified title-plus-label evidence",
                    numeric_token_count=len(usable_numbers),
                )
            if not chart_eligible and excluded_kind is None and not usable_numbers:
                self._probable_missing_numeric_evidence(element)
            ocr_derived = "ocr" in metadata_text.casefold() or element.validation_state in {
                "needs_review", "pending_review",
            }
            if not ocr_derived:
                self.audit_check(
                    "ocr_vocabulary_noise", "skipped", element,
                    "visual is not recorded as OCR-derived or pending OCR review",
                )
                continue
            visual_tokens = _word_records(element.text)
            if not visual_tokens:
                self.audit_check(
                    "ocr_vocabulary_noise", "checked", element,
                    "OCR-derived visual contains no eligible lexical tokens",
                )
                continue
            references = self._visual_vocabulary_references(element)
            reference_records = [{**reference, "tokens": _word_records(reference["text"])}
                                 for reference in references]
            candidate_vocabulary = set(body_counts)
            candidate_vocabulary.update(token["normalized"] for reference in reference_records
                                        for token in reference["tokens"])
            # Verification is per script: text in another script (or a translation) is never evidence.
            for script in sorted({_script(token["normalized"]) for token in visual_tokens}):
                script_tokens = [token for token in visual_tokens if _script(token["normalized"]) == script]
                if len(script_tokens) < self.config.vocabulary_min_visual_script_tokens:
                    continue
                body_same = sum(count for value, count in body_counts.items() if _script(value) == script)
                reference_same = sorted({source_id for reference in reference_records
                                         if any(_script(token["normalized"]) == script for token in reference["tokens"])
                                         for source_id in reference["source_ids"]})
                if body_same >= self.config.vocabulary_min_same_script_tokens or reference_same:
                    continue
                self.add_finding(
                    "ocr_corruption", "unverifiable_script_vocabulary", "info", "high",
                    [element.element_id], [element.page],
                    {"script": script, "visual_word_tokens": len(script_tokens),
                     "independent_same_script_body_tokens": body_same,
                     "minimum_same_script_body_tokens": self.config.vocabulary_min_same_script_tokens,
                     "same_script_reference_source_ids": reference_same,
                     "other_script_text_used": False},
                    f"The visual's {script} OCR words cannot be verified: the document has too little independent "
                    f"{script} text and no same-script heading, caption or verified text for this visual.",
                    "If the visual matters, review its words against the image; absence of a finding is not "
                    "evidence that the OCR is correct.", "unverified",
                    limitations=["no word is accused of being wrong", "rarity alone is never flagged",
                                 "text in another language is not treated as a reference"],
                    method="per_script_reference_inventory",
                )
            for visual_index, visual_token in enumerate(visual_tokens):
                raw_token = visual_token["raw"]
                token = visual_token["normalized"]
                audit_base = {
                    "token": raw_token, "normalized_token": token,
                    "offsets": [visual_token["start"], visual_token["end"]],
                }
                if not token:
                    self.audit_check("ocr_vocabulary_noise", "skipped", element,
                                     "token has no normalized lexical form", **audit_base)
                    continue
                if token in body_counts:
                    self.audit_check("ocr_vocabulary_noise", "checked", element,
                                     "token occurs in independent document text", **audit_base)
                    continue
                if raw_token.isupper():
                    self.audit_check("ocr_vocabulary_noise", "skipped", element,
                                     "all-uppercase token may be an acronym", **audit_base)
                    continue
                candidates: list[dict[str, Any]] = []
                gate_counts: Counter[str] = Counter()
                for candidate in candidate_vocabulary:
                    if candidate == token:
                        gate_counts["same_normalized_token"] += 1
                        continue
                    if _script(candidate) != _script(token):
                        gate_counts["different_script"] += 1
                        continue
                    if _morphological_variant(token, candidate):
                        gate_counts["plausible_morphological_variant"] += 1
                        continue
                    distance = _edit_distance(token, candidate, maximum=4)
                    if not 1 <= distance <= 4:
                        gate_counts["outside_edit_distance"] += 1
                        continue
                    supports = []
                    for reference in reference_records:
                        for support_index, support_token in enumerate(reference["tokens"]):
                            if support_token["normalized"] != candidate:
                                continue
                            context = _context_alignment(
                                visual_tokens, visual_index, reference["tokens"], support_index,
                            )
                            if _has_strong_local_context(context):
                                supports.append({
                                    "method": f"aligned_{reference['kind']}",
                                    "source_ids": reference["source_ids"],
                                    "context_matches": context,
                                    "support_text": reference["text"],
                                })
                    repeated_contexts = []
                    if not supports:
                        for occurrence in body_occurrences.get(candidate, []):
                            if abs(occurrence["page"] - element.page) > 1:
                                continue
                            context = _context_alignment(
                                visual_tokens, visual_index, occurrence["tokens"], occurrence["index"],
                            )
                            if _has_strong_local_context(context):
                                repeated_contexts.append({
                                    "source_id": occurrence["source_id"], "page": occurrence["page"],
                                    "order": occurrence["order"], "section_ids": occurrence["section_ids"],
                                    "context_matches": context,
                                })
                        repeated_source_ids = sorted({item["source_id"] for item in repeated_contexts})
                        if len(repeated_source_ids) >= 2:
                            supports.append({
                                "method": "repeated_matching_local_phrase",
                                "source_ids": repeated_source_ids,
                                "occurrences": repeated_contexts,
                            })
                        elif len(repeated_source_ids) == 1 and any(
                            abs(item["page"] - element.page) <= 1 for item in repeated_contexts
                        ):
                            supports.append({
                                "method": "neighboring_element_matching_local_phrase",
                                "source_ids": repeated_source_ids,
                                "occurrences": repeated_contexts,
                            })
                    if not supports and distance <= 2 and body_counts.get(candidate, 0) >= 2:
                        anchored = []
                        for occurrence in body_occurrences.get(candidate, []):
                            if abs(occurrence["page"] - element.page) > 1:
                                continue
                            context = _context_alignment(
                                visual_tokens, visual_index, occurrence["tokens"], occurrence["index"],
                            )
                            if any(abs(item["relative_position"]) == 1 for item in context):
                                anchored.append({
                                    "source_id": occurrence["source_id"], "page": occurrence["page"],
                                    "context_matches": context,
                                })
                        if anchored:
                            supports.append({
                                "method": "repeated_document_token_with_immediate_context_anchor",
                                "source_ids": sorted({item["source_id"] for item in anchored}),
                                "occurrences": anchored,
                            })
                    if distance >= 3 and not any(
                        sum(match.get("comparison") == "exact"
                            for match in item.get("context_matches") or []) >= 2 or any(
                            sum(match.get("comparison") == "exact"
                                for match in value.get("context_matches") or []) >= 2
                            for value in item.get("occurrences") or []
                        ) for item in supports
                    ):
                        gate_counts["large_distance_without_two_token_context"] += 1
                        continue
                    if distance >= 3 and SequenceMatcher(None, token, candidate, autojunk=False).ratio() < \
                            self.config.vocabulary_large_distance_min_similarity:
                        # A short valid word aligned with an unrelated word is not OCR damage.
                        gate_counts["large_distance_low_character_similarity"] += 1
                        continue
                    if raw_token[:1].isupper() and raw_token[1:].islower() and not any(
                        item["method"].startswith("aligned_") for item in supports
                    ):
                        gate_counts["possible_proper_name_without_reference_alignment"] += 1
                        continue
                    if not supports:
                        gate_counts["no_independent_local_context"] += 1
                        continue
                    reference_support = any(item["method"].startswith("aligned_") for item in supports)
                    candidates.append({
                        "candidate": candidate,
                        "distance": distance,
                        "supports": supports,
                        "priority": 2 if reference_support else 1,
                        "context_match_count": sum(
                            len(item.get("context_matches") or []) +
                            sum(len(value["context_matches"]) for value in item.get("occurrences") or [])
                            for item in supports
                        ),
                    })
                if not candidates:
                    self.audit_check(
                        "ocr_vocabulary_noise", "checked", element,
                        "no independently supported close vocabulary candidate",
                        **audit_base, candidate_gate_counts=dict(sorted(gate_counts.items())),
                    )
                    continue
                candidates.sort(key=lambda item: (
                    -item["priority"], item["distance"], -item["context_match_count"], item["candidate"],
                ))
                selected = candidates[0]
                if len(candidates) > 1 and (
                    candidates[1]["priority"], candidates[1]["distance"]
                ) == (selected["priority"], selected["distance"]):
                    self.audit_check(
                        "ocr_vocabulary_noise", "skipped", element,
                        "multiple equally supported vocabulary candidates; no correction is guessed",
                        **audit_base,
                        candidates=[item["candidate"] for item in candidates
                                    if (item["priority"], item["distance"]) ==
                                    (selected["priority"], selected["distance"])],
                    )
                    continue
                candidate = selected["candidate"]
                supporting_source_ids = sorted({
                    source_id for support in selected["supports"] for source_id in support["source_ids"]
                })
                self.add_finding(
                    "ocr_corruption", "ocr_vocabulary_noise", "medium", "high",
                    [element.element_id], [element.page],
                    {"original_token": raw_token, "normalized_token": token,
                     "candidate_body_token": candidate, "candidate_body_count": body_counts.get(candidate, 0),
                     "edit_distance": selected["distance"], "maximum_edit_distance": 4,
                     "large_distance_requires_two_token_context": True,
                     "supporting_source_ids": supporting_source_ids,
                     "support_methods": [item["method"] for item in selected["supports"]],
                     "contextual_evidence": selected["supports"],
                     "evidence_strength": "strong",
                     "candidate_selection": "unique_best_independently_aligned_candidate"},
                    "An OCR-derived visual token differs from a uniquely aligned token in independent local context.",
                    "Review the token against the source; do not replace it automatically.", "needs_review",
                    offsets=[{"source_id": element.element_id, "start": visual_token["start"],
                              "end": visual_token["end"], "text": raw_token}],
                    related_ids=supporting_source_ids,
                    limitations=["this remains a suspected OCR error; no source text is changed",
                                 "valid terminology can still require human review despite strong local alignment"],
                    method="ocr_visual_independent_local_alignment",
                )
                self.audit_check(
                    "ocr_vocabulary_noise", "flagged", element,
                    "unique independently supported close vocabulary candidate",
                    **audit_base, selected_candidate=candidate,
                    edit_distance=selected["distance"], supporting_source_ids=supporting_source_ids,
                )

    def _probable_missing_numeric_evidence(self, element: Any) -> None:
        """A visual whose OCR names a quantitative measure (an axis or series title) but contains no number.

        Only the saved OCR text is evidence: photos, maps, diagrams and decorative visuals without measure terms
        are never flagged, composite parts are judged through their parent, and no value is inferred."""
        if element.validation_state == "included_in_composite":
            return
        text = element.text or ""
        normalized = normalize_text(text)
        terms = sorted({match.group(0) for match in MEASURE_TERM_RE.finditer(normalized)})
        if not terms:
            return
        linked = []
        for relation in self.relationships:
            if relation.get("type") in {"describes", "caption_of"} and relation.get("target_id") == element.element_id:
                source = self.nodes.get(str(relation.get("source_id"))) or {}
                count = len([item for item in numeric_tokens(str(source.get("text") or ""))
                             if item.dimension != "identifier"])
                linked.append({"relationship_id": relation.get("rel_id"), "type": relation.get("type"),
                               "source_id": relation.get("source_id"), "numeric_token_count": count})
        self.add_finding(
            "figure_chart_quality", "probable_missing_numeric_evidence", "medium", "medium",
            [element.element_id], [element.page],
            {"measure_terms": terms, "numeric_token_count": 0,
             "numeric_evidence_by_role": {"axis_ticks": 0, "legend_values": 0, "data_labels": 0},
             "explanatory_text_numbers": linked,
             "saved_visual_label": element.raw.get("label"),
             "text_excerpt": _excerpt(text, 0, 0, 160)},
            "The visual's OCR names a quantitative measure but contains no numeric axis ticks, legend values or "
            "data labels; the extracted chart evidence is probably incomplete.",
            "Use the visual only for its labels; take values from linked explanatory text or the page image. "
            "Do not infer values.", "unverified",
            limitations=["the image is not inspected; whether values are drawn in the visual is not established",
                         "numbers in explanatory text are listed for context and are not attributed to the visual"],
            method="measure_term_without_numeric_tokens",
        )

    def detect_known_missing_quality(self) -> None:
        for relation in self.relationships:
            if relation.get("type") != "known_missing":
                continue
            target_id = str(relation.get("target_id"))
            target = self.relation_nodes.get(target_id) or {}
            evidence = relation.get("evidence") or {}
            self.add_finding(
                "source_extraction", "recorded_known_missing", "medium", "high", [target_id],
                [target.get("page") or evidence.get("page")],
                {"relationship_id": relation.get("rel_id"), "upstream_evidence": evidence,
                 "node_provenance": target.get("provenance")},
                "Upstream saved metadata explicitly records skipped, dropped, or unavailable visual content.",
                "Keep the placeholder visible to downstream consumers; do not reconstruct missing content.",
                "needs_review", relationship_ids=[str(relation.get("rel_id"))],
                method="component2_known_missing_record",
            )

    @staticmethod
    def _local_label(text: str, start: int) -> set[str]:
        return content_terms(text[max(0, start - 55):start])

    def detect_numeric_integrity(self) -> None:
        for relation in self.relationships:
            if relation.get("type") != "describes":
                continue
            source_id, target_id = str(relation.get("source_id")), str(relation.get("target_id"))
            source, target = self.nodes.get(source_id, {}), self.nodes.get(target_id, {})
            matches = (relation.get("evidence") or {}).get("numeric_matches") or []
            if matches:
                self.add_finding(
                    "numeric_integrity", "numeric_agreement", "info", "high", [source_id, target_id],
                    [source.get("page"), target.get("page")],
                    {"matches": matches, "count": len(matches), "relationship_id": relation.get("rel_id"),
                     "conclusive_count": sum(1 for item in matches
                                             if item.get("evidential_status", "supports") != "inconclusive"),
                     "inconclusive_count": sum(1 for item in matches
                                               if item.get("evidential_status") == "inconclusive")},
                    "An existing describes relationship contains compatible numeric agreement.",
                    "No correction is needed; retain this evidence for grounded review.", "not_applicable",
                    relationship_ids=[str(relation.get("rel_id"))], method="existing_relationship_evidence",
                )
            left, right = str(source.get("text") or ""), str(target.get("text") or "")
            if not left or not right:
                continue
            matched_pairs = {(str(item.get("source_raw")), str(item.get("target_raw"))) for item in matches}
            conflicts = []
            for left_token in numeric_tokens(left):
                if left_token.dimension in NON_VALUE_DIMENSIONS:
                    continue
                left_label = self._local_label(left, left_token.start)
                for right_token in numeric_tokens(right):
                    if right_token.dimension in NON_VALUE_DIMENSIONS or left_token.dimension != right_token.dimension:
                        continue
                    right_label = self._local_label(right, right_token.start)
                    shared = sorted(left_label & right_label)
                    if len(shared) < self.config.numeric_label_min_shared_terms:
                        continue
                    if (left_token.raw, right_token.raw) in matched_pairs or left_token.value == right_token.value:
                        continue
                    conflicts.append({
                        "source_raw": left_token.raw, "source_offsets": [left_token.start, left_token.end],
                        "target_raw": right_token.raw, "target_offsets": [right_token.start, right_token.end],
                        "dimension": left_token.dimension, "shared_metric_terms": shared,
                    })
            if conflicts:
                self.add_finding(
                    "numeric_integrity", "numeric_conflict", "high", "medium", [source_id, target_id],
                    [source.get("page"), target.get("page")],
                    {"conflicts": conflicts, "relationship_id": relation.get("rel_id")},
                    "Aligned metric labels and units contain different saved numeric values. This is a potential conflict,"
                    " not a correction.",
                    "Review the paragraph and visual together against the page image.", "needs_review",
                    relationship_ids=[str(relation.get("rel_id"))],
                    limitations=["OCR order can misalign labels and values; no intended value is inferred"],
                    method="aligned_metric_and_unit_comparison",
                )

    def detect_duplicates(self) -> None:
        linked_pairs = set()
        for relation in self.relationships:
            if relation.get("type") != "leaked_text":
                continue
            source_id, target_id = str(relation.get("source_id")), str(relation.get("target_id"))
            linked_pairs.add((source_id, target_id))
            self.add_finding(
                "duplicate_leakage", "recorded_leaked_text", "medium", str(relation.get("confidence") or "medium"),
                [source_id, target_id], [relation.get("source_page"), relation.get("target_page")],
                {"relationship_evidence": relation.get("evidence"), "relationship_id": relation.get("rel_id")},
                "Component 2 recorded text overlap between a source element and a visual crop.",
                "Inspect both elements; do not delete or deduplicate source text here.", "needs_review",
                relationship_ids=[str(relation.get("rel_id"))], method="existing_leaked_text_relationship",
            )
        text_items = [item for item in self.ordered if item.text and len(normalize_text(item.text)) >=
                      self.config.near_duplicate_min_chars]
        visuals = [item for item in text_items if item.element_type in {"figure", "table"}]
        ordinary = [item for item in text_items if item.element_type not in {"figure", "table"}]
        assignment_role = {element_id: (items[0].get("role") if items else None)
                           for element_id, items in self.assignment_by_element.items()}
        for visual in visuals:
            for source in ordinary:
                if source.page != visual.page or (source.element_id, visual.element_id) in linked_pairs:
                    continue
                if assignment_role.get(source.element_id) in {"header", "footer", "caption"}:
                    continue
                left, right = normalize_text(source.text or ""), normalize_text(visual.text or "")
                ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
                if ratio < self.config.near_duplicate_ratio:
                    continue
                self.add_finding(
                    "duplicate_leakage", "near_duplicate_visual_text", "medium", "medium",
                    [source.element_id, visual.element_id], [source.page],
                    {"normalized_similarity": round(ratio, 4), "source_excerpt": _excerpt(source.text or ""),
                     "visual_excerpt": _excerpt(visual.text or "")},
                    "A visual OCR crop and a separate source element are near-duplicates.",
                    "Review whether crop margins captured neighbouring text; preserve both originals.", "unverified",
                    limitations=["similarity may reflect intentional repetition"], method="same_page_near_duplicate",
                )

    def _heading_audit(self, candidate: dict[str, Any], source_ids: list[str]) -> dict[str, Any]:
        """Full, explicit evidence for a heading decision. A null confidence never stands alone: the decision
        status and the reason for the uncertainty are always stated."""
        thresholds = self.inputs.structure_manifest.get("detection_thresholds") or {}
        status = candidate.get("status")
        score = candidate.get("score")
        promote = thresholds.get("promote")
        if status == "promoted":
            decision = "promoted: score at or above the promotion threshold"
        elif status == "excluded":
            decision = "excluded by rule: " + "; ".join(candidate.get("exclusions") or [])
        else:
            decision = "rejected: score below the promotion threshold"
        uncertainty = []
        if candidate.get("ambiguous"):
            uncertainty.append("detection score within the ambiguity band around the promotion threshold"
                               + (f" ({score} vs {promote})" if promote is not None else ""))
        if candidate.get("confidence") == "low":
            uncertainty.append("promoted with low detection confidence")
        if candidate.get("level_confidence") == "low":
            uncertainty.append("low level confidence")
        if candidate.get("level_ambiguous"):
            uncertainty.append("level marked ambiguous")
        uncertainty.extend(candidate.get("level_conflicts") or [])
        return {
            "candidate_id": candidate.get("candidate_id"),
            "source_ids": source_ids,
            "unit_ids": candidate.get("unit_ids") or [],
            "text": candidate.get("text"),
            "page": candidate.get("page"),
            "decision_status": status,
            "decision": decision,
            "score": score,
            "thresholds": {key: thresholds.get(key) for key in ("promote", "medium", "high", "ambiguity_band")
                           if key in thresholds},
            "signals": candidate.get("signals") or [],
            "exclusions": candidate.get("exclusions") or [],
            "notes": candidate.get("notes") or [],
            "detection_confidence": candidate.get("confidence"),
            "detection_confidence_meaning": (
                "not applicable: the candidate was not promoted" if candidate.get("confidence") is None
                else "confidence of the promoted heading decision"),
            "level": candidate.get("level"),
            "level_confidence": candidate.get("level_confidence"),
            "level_confidence_meaning": (
                "not applicable: no level is assigned to a non-promoted candidate"
                if candidate.get("level_confidence") is None else "confidence of the assigned level"),
            "level_basis": candidate.get("level_basis") or [],
            "conflicts": candidate.get("conflicts") or candidate.get("level_conflicts"),
            "uncertainty_reasons": uncertainty,
            **self._heading_audit_explanation(candidate, thresholds),
        }

    def _heading_audit_explanation(self, candidate: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
        """Additive audit detail: supporting versus conflicting evidence, the confidence band the score falls in,
        signals that most same-style headings carry but this one lacks, and how the parent section was chosen."""
        signals = [item for item in candidate.get("signals") or [] if isinstance(item, dict)]
        supporting = [f"{item.get('name')} (+{item.get('contribution')}): {item.get('detail')}"
                      for item in signals if (item.get("contribution") or 0) > 0]
        conflicting = [f"{item.get('name')} ({item.get('contribution')}): {item.get('detail')}"
                       for item in signals if (item.get("contribution") or 0) < 0]
        conflicting += [str(item) for item in candidate.get("conflicts") or []
                        if not any(str(item).split(":", 1)[0] == entry.split(" (", 1)[0] for entry in conflicting)]
        conflicting += [f"level: {item}" for item in candidate.get("level_conflicts") or []]
        score, promote = candidate.get("score"), thresholds.get("promote")
        medium, high = thresholds.get("medium"), thresholds.get("high")
        band = "detection thresholds are not recorded in the structure manifest: confidence band not evaluated"
        if isinstance(score, (int, float)) and None not in (promote, medium, high):
            if score < promote:
                band = f"score {score} is below the promotion threshold {promote}"
            elif score < medium:
                band = f"score {score} is in the low-confidence band [{promote}, {medium})"
            elif score < high:
                band = f"score {score} is in the medium-confidence band [{medium}, {high})"
            else:
                band = f"score {score} is at or above the high-confidence threshold {high}"
        # Signals carried by most promoted headings of the same typographic style but absent here.
        style = candidate.get("style_key")
        peers = [item for item in self.inputs.section_tree.get("heading_candidates") or []
                 if item.get("candidate_id") != candidate.get("candidate_id") and item.get("status") == "promoted"
                 and style is not None and item.get("style_key") == style]
        own = {item.get("name") for item in signals}
        absent = []
        if len(peers) >= 2:
            counts: dict[str, int] = {}
            for peer in peers:
                for name in {item.get("name") for item in peer.get("signals") or [] if isinstance(item, dict)
                             and (item.get("contribution") or 0) > 0}:
                    counts[name] = counts.get(name, 0) + 1
            absent = [f"{name}: present on {count} of {len(peers)} same-style promoted headings, absent here"
                      for name, count in sorted(counts.items()) if name not in own and count * 2 > len(peers)]
        return {
            "supporting_signals": supporting,
            "conflicting_signals": conflicting,
            "confidence_explanation": band,
            "absent_peer_signals": absent,
            "placement": self._heading_placement(candidate),
        }

    def _heading_placement(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        sections = self.inputs.section_tree.get("sections") or []
        by_id = {item.get("section_id"): item for item in sections}
        section = next((item for item in sections
                        if item.get("heading_candidate_id") == candidate.get("candidate_id")), None)
        if section is None:
            return None
        parent = by_id.get(section.get("parent"))
        if parent is None:
            reason = "top of the section tree: no preceding heading at a lower level"
        else:
            reason = (f"nearest preceding section at a lower level: {parent.get('section_id')} "
                      f"(level {parent.get('level')}); the section tree nests a heading under the closest earlier "
                      f"heading whose level is smaller than its own")
        return {
            "section_id": section.get("section_id"),
            "assigned_level": section.get("level"),
            "raw_level": section.get("raw_level"),
            "parent_section_id": parent.get("section_id") if parent else None,
            "parent_title": parent.get("title") if parent else None,
            "parent_heading_candidate_id": parent.get("heading_candidate_id") if parent else None,
            "parent_selection": reason,
            "level_basis": candidate.get("level_basis") or [],
            "level_notes": section.get("level_notes") or [],
        }

    def detect_structure_quality(self) -> None:
        for candidate in self.inputs.section_tree.get("heading_candidates") or []:
            if candidate.get("confidence") == "low" or candidate.get("level_confidence") == "low" or \
                    candidate.get("ambiguous") or candidate.get("level_ambiguous"):
                source_ids = [str(value) for value in candidate.get("unit_ids") or []]
                source_ids = [self.nodes[value].get("source_id") or value for value in source_ids if value in self.nodes]
                self.add_finding(
                    "structure_quality", "uncertain_heading_or_level", "medium", "high", source_ids,
                    [candidate.get("page")], self._heading_audit(candidate, source_ids),
                    "The Structure Builder marked a heading decision or level as uncertain.",
                    "Review the heading and neighbouring section boundaries; do not move content automatically.",
                    "needs_review", method="structure_heading_diagnostics",
                )
        low_assignments = [item for item in self.assignments if item.get("confidence") == "low" or
                           item.get("review_recommended")]
        if low_assignments:
            source_ids = sorted({str(item.get("element_id")) for item in low_assignments})
            self.add_finding(
                "structure_quality", "uncertain_assignments", "medium", "high", source_ids,
                sorted({int(item["page"]) for item in low_assignments if isinstance(item.get("page"), int)}),
                {"assignment_count": len(low_assignments),
                 "unit_ids": [item.get("unit_id") for item in low_assignments],
                 "assignments": [{
                     "unit_id": item.get("unit_id"), "element_id": item.get("element_id"),
                     "section_id": item.get("section_id"), "role": item.get("role"),
                     "confidence": item.get("confidence"),
                     "reasons": list(item.get("conflicting") or []) or list(item.get("reasons") or []),
                 } for item in low_assignments],
                 "reliability_scope": "structural placement only; does not lower source text reliability"},
                "Structure assignments are low confidence or explicitly review-recommended.",
                "Review section membership and role without modifying the saved Structure output.", "needs_review",
                method="structure_assignment_confidence",
            )
        unassigned = [item for item in self.assignments if item.get("section_id") is None and
                      item.get("role") not in {"header", "footer", "page_number", "empty"}]
        if unassigned:
            self.add_finding(
                "structure_quality", "unassigned_content", "low", "high",
                sorted({str(item.get("element_id")) for item in unassigned}),
                sorted({int(item["page"]) for item in unassigned if isinstance(item.get("page"), int)}),
                {"count": len(unassigned), "roles": dict(Counter(str(item.get("role")) for item in unassigned))},
                "Content remains outside any inferred section; this is not automatically an error.",
                "Review whether the document legitimately has front matter or unsectioned content.", "unverified",
                method="structure_assignments",
            )
        for list_node in self.inputs.section_tree.get("lists") or []:
            possible = list_node.get("possible_continuations") or []
            if possible:
                ids = [str(item.get("element_id") or item.get("unit_id")) for item in list_node.get("items") or []]
                self.add_finding(
                    "structure_quality", "uncertain_list_continuation", "low", "high", ids,
                    list_node.get("pages") or [], {"list_id": list_node.get("list_id"), "possible": possible},
                    "The Structure Builder retained possible list continuations without linking them.",
                    "Inspect list flow and geometry.", "needs_review", method="saved_list_diagnostics",
                )
        for diagnostic in self.inputs.section_tree.get("layout_heading_diagnostics") or []:
            if diagnostic.get("association") in {"approximate", "none"}:
                source_ids = [str(diagnostic.get("element_id"))] if diagnostic.get("element_id") else []
                self.add_finding(
                    "structure_quality", "approximate_layout_heading", "low", "high", source_ids,
                    [diagnostic.get("page")], {"diagnostic": diagnostic},
                    "A saved layout heading region lacks a verified one-to-one text association.",
                    "Review the page layout; do not fabricate span coordinates.", "unverified",
                    limitations=["scanned OCR may not provide per-span coordinates"], method="saved_layout_diagnostic",
                )

    def detect_relationship_quality(self) -> None:
        validation = self.inputs.relationships.get("validation") or {}
        for item in validation.get("findings") or []:
            if item.get("severity") == "ok":
                continue
            source_ids = [str(value) for value in item.get("items") or [] if str(value) in self.nodes]
            pages = [self.nodes[value].get("page") for value in source_ids]
            evidence = item.get("evidence") or {}
            if isinstance(evidence, dict) and isinstance(evidence.get("page"), int):
                pages.append(evidence["page"])
            self.add_finding(
                "relationship_quality", str(item.get("code") or "relationship_limitation"), "medium", "high",
                source_ids, pages, {"component2_finding": item},
                str(item.get("message") or "Component 2 recorded an unresolved relationship issue."),
                "Inspect the competing or missing relationship evidence; do not create a link in this layer.",
                "needs_review",
                limitations=[] if source_ids else ["Component 2 did not identify a resolvable source node"],
                method="component2_validation",
            )
        for relation in self.relationships:
            if relation.get("type") == "co_located_in_section":
                continue
            if relation.get("confidence") not in {"medium", "low"} and not relation.get("review_recommended"):
                continue
            source_ids = [str(relation.get("source_id")), str(relation.get("target_id"))]
            pages = [relation.get("source_page"), relation.get("target_page")]
            self.add_finding(
                "relationship_quality", "uncertain_existing_relationship", "medium", "high", source_ids, pages,
                {"relationship_id": relation.get("rel_id"), "type": relation.get("type"),
                 "confidence": relation.get("confidence"), "conflicts": relation.get("conflicts"),
                 "limitations": relation.get("limitations")},
                "An existing relationship is medium/low confidence or review-recommended.",
                "Review its saved evidence; the Quality layer does not replace it.", "needs_review",
                relationship_ids=[str(relation.get("rel_id"))], method="component2_confidence",
            )
        linked_visuals = {str(relation.get("target_id")) for relation in self.relationships
                          if relation.get("type") in {"caption_of", "describes"}}
        unlinked = [visual_id for visual_id in self.visuals if visual_id not in linked_visuals]
        if unlinked:
            self.add_finding(
                "relationship_quality", "visual_without_caption_or_description", "info", "high", unlinked,
                [self.nodes[value].get("page") for value in unlinked if value in self.nodes],
                {"count": len(unlinked), "visual_ids": sorted(unlinked)},
                "These visuals have no reliable caption_of or describes relationship. This is not an extraction failure.",
                "Review only when downstream use requires a textual explanation.", "unverified",
                limitations=["a valid visual may legitimately have no separate caption or description"],
                method="relationship_inventory_coverage",
            )
        unresolved = [visual_id for visual_id, item in self.visuals.items()
                      if (item.get("composite") or {}).get("unresolved")]
        if unresolved:
            self.add_finding(
                "relationship_quality", "unresolved_composite_parent", "medium", "high", unresolved,
                [self.nodes[value].get("page") for value in unresolved], {"visual_ids": unresolved},
                "Composite parts lack a provenance- or geometry-supported parent.",
                "Identify the parent manually or leave the relationship unresolved.", "needs_review",
                method="component2_visual_inventory",
            )

    @staticmethod
    def _markdown_rows(text: str) -> list[list[str]]:
        rows = []
        for line in text.splitlines():
            clean = line.strip()
            if clean.startswith("|") and clean.endswith("|") and not re.fullmatch(r"[|:\-–—\s]+", clean):
                rows.append([value.strip() for value in clean.strip("|").split("|")])
        return rows

    def detect_table_quality(self) -> None:
        tables = [item for item in self.ordered if item.element_type == "table"]
        for table in tables:
            text, cells = table.text or "", table.raw.get("cells")
            rows = self._markdown_rows(text)
            if rows:
                widths = [len(row) for row in rows]
                if len(set(widths)) > 1:
                    self.add_finding(
                        "table_quality", "inconsistent_markdown_columns", "high", "high",
                        [table.element_id], [table.page], {"row_column_counts": widths},
                        "Markdown table rows contain inconsistent column counts.",
                        "Inspect row alignment and source extraction; do not insert cells automatically.",
                        method="markdown_structure",
                    )
            elif not isinstance(cells, list) or not cells:
                self.add_finding(
                    "table_quality", "flattened_unstructured_table", "medium", "high",
                    [table.element_id], [table.page], {"text_length": len(text), "cells_available": False},
                    "A table is represented as flat text without saved cell structure or Markdown rows.",
                    "Review row/column interpretation manually.", "unverified",
                    limitations=["flat OCR text does not prove that values are missing"], method="table_representation",
                )
            for match in MALFORMED_ARITHMETIC_RE.finditer(text):
                provenance = table.raw.get("provenance") if isinstance(table.raw.get("provenance"), dict) else {}
                explicit_source_origin = any(
                    str(value).casefold() in {"source_document", "original_pdf", "source_content"}
                    for value in (
                        table.raw.get("anomaly_origin"), table.raw.get("content_origin"),
                        provenance.get("anomaly_origin"), provenance.get("content_origin"),
                    )
                ) or bool(table.raw.get("source_content_verified") or provenance.get("source_content_verified"))
                subtype = "source_content_anomaly" if explicit_source_origin else "unverified_anomaly"
                self.add_finding(
                    "table_quality", subtype, "medium", "high" if explicit_source_origin else "medium",
                    [table.element_id], [table.page],
                    {"verbatim": match.group(0), "offsets": [match.start(), match.end()],
                     "origin_classification": "source_document" if explicit_source_origin else "unverified",
                     "source_provenance": provenance,
                     "explicit_origin_fields": {
                         key: table.raw.get(key) for key in
                         ("anomaly_origin", "content_origin", "source_content_verified")
                         if table.raw.get(key) is not None
                     }},
                    "The saved table contains an incomplete numeric expression; its origin is "
                    + ("explicitly recorded as source content." if explicit_source_origin else
                       "not established by available provenance."),
                    "Preserve the expression and review its source provenance; do not infer the missing value.",
                    "unverified" if not explicit_source_origin else "needs_review",
                    offsets=[{"source_id": table.element_id, "start": match.start(), "end": match.end(),
                              "text": match.group(0)}], method="malformed_numeric_pattern",
                    limitations=[] if explicit_source_origin else [
                        "digital extraction alone does not prove whether the anomaly originated in the PDF"
                    ],
                )
        for node in self.relation_nodes.values():
            if node.get("node_type") != "logical_table":
                continue
            provenance = node.get("provenance") or {}
            repeated = provenance.get("repeated_header_between_parts") or []
            if any(repeated):
                parts = [str(value) for value in provenance.get("ordered_parts") or []]
                self.add_finding(
                    "table_quality", "confirmed_repeated_continuation_header", "info", "high", parts,
                    [self.nodes[value].get("page") for value in parts if value in self.nodes],
                    {"logical_table_id": node.get("node_id"), "repeated_header_between_parts": repeated},
                    "A confirmed logical table repeats its header across page parts; this is intentional, not duplicate leakage.",
                    "Retain the header and logical-table provenance.", "not_applicable",
                    related_ids=[str(node.get("node_id"))], method="logical_table_provenance",
                )

    def detect_figure_quality(self) -> None:
        linked = {str(relation.get("target_id")) for relation in self.relationships
                  if relation.get("type") in {"caption_of", "describes"}}
        composite_parts = {str(relation.get("source_id")) for relation in self.relationships
                           if relation.get("type") == "composite_part"}
        composite_parents = {str(relation.get("target_id")) for relation in self.relationships
                             if relation.get("type") == "composite_part"}
        for visual_id in self.visuals:
            element = self.elements.get(visual_id)
            if not element or element.element_type != "figure":
                continue
            terms = content_terms(element.text or "")
            if element.validation_state == "included_in_composite" or (
                not element.text and visual_id in composite_parts | composite_parents
            ):
                self.inventory_by_id[visual_id].skipped_checks["sparse_visual_ocr"] = (
                    "empty visual is covered by an explicit saved composite relationship; "
                    "the source review state remains preserved"
                )
                continue
            if len(terms) < self.config.sparse_visual_terms:
                severity = "medium" if element.validation_state in {"needs_review", "pending_review"} else "low"
                self.add_finding(
                    "figure_chart_quality", "sparse_visual_ocr", severity, "medium", [visual_id], [element.page],
                    {"content_term_count": len(terms), "text_length": len(element.text or ""),
                     "validation_state": element.validation_state},
                    "The saved figure OCR is sparse; this may be legitimate or may reflect unreadable labels.",
                    "Inspect the crop and extraction status; absence of text is not proof of missing content.", "unverified",
                    limitations=["no image analysis or new OCR was performed"], method="saved_text_density",
                )
            if visual_id not in linked:
                self.inventory_by_id[visual_id].skipped_checks.setdefault(
                    "figure_numeric_consistency", "no reliable describes relationship identifies a comparison target"
                )

    def detect_cross_page(self) -> None:
        for node in self.inputs.section_tree.get("lists") or []:
            if node.get("crosses_pages") and node.get("possible_continuations"):
                ids = [str(item.get("element_id")) for item in node.get("items") or []]
                self.add_finding(
                    "cross_page_continuity", "uncertain_cross_page_list", "medium", "high", ids,
                    node.get("pages") or [], {"list_id": node.get("list_id"),
                                             "possible_continuations": node.get("possible_continuations")},
                    "A cross-page list has unresolved continuation evidence.",
                    "Inspect the end and start of the adjacent pages.", "needs_review",
                    method="structure_list_provenance",
                )
        for section in self.sections.values():
            if any("bottom of page" in str(flag) for flag in section.get("flags") or []):
                ids = [str(value) for value in section.get("heading_element_ids") or []]
                self.add_finding(
                    "cross_page_continuity", "heading_at_page_boundary", "low", "high", ids,
                    [section.get("page_start")], {"section_id": section.get("section_id"),
                                                  "flags": section.get("flags")},
                    "A heading appears at a page boundary and may separate from its following content.",
                    "Review the page break and next-page assignment.", "unverified",
                    method="structure_section_flags",
                )
        assignments = [item for item in self.assignments if item.get("role") == "paragraph"
                       and item.get("element_id") in self.elements]
        assignments.sort(key=lambda item: tuple(item.get("order") or [0, 0]))
        for left, right in pairwise(assignments):
            if right.get("page") != left.get("page", 0) + 1 or left.get("section_id") != right.get("section_id"):
                continue
            left_element = self.elements[str(left["element_id"])]
            right_element = self.elements[str(right["element_id"])]
            left_text = str(left.get("text") or "")
            right_text = str(right.get("text") or "")
            if not left_text or not right_text or TERMINAL_RE.search(left_text):
                continue
            left_reliability, left_bbox = self._assignment_geometry(left)
            right_reliability, right_bbox = self._assignment_geometry(right)
            if left_reliability not in RELIABLE_GEOMETRY or right_reliability not in RELIABLE_GEOMETRY:
                for element, reliability in ((left_element, left_reliability), (right_element, right_reliability)):
                    self.inventory_by_id[element.element_id].skipped_checks["cross_page_continuity"] = (
                        f"insufficient paragraph geometry ({reliability}); page-level coordinates are not used"
                    )
                continue
            if not left_bbox or not right_bbox or left_bbox[3] < self.config.cross_page_bottom_y or \
                    right_bbox[1] > self.config.cross_page_top_y:
                continue
            source_ids = sorted({left_element.element_id, right_element.element_id})
            related_ids = [str(item.get("unit_id")) for item in (left, right)
                           if item.get("unit_id") not in source_ids]
            self.add_finding(
                "cross_page_continuity", "possible_paragraph_continuation", "low", "medium",
                source_ids, [left_element.page, right_element.page],
                {"same_section": left.get("section_id"), "previous_bbox": left_bbox, "next_bbox": right_bbox,
                 "previous_geometry_reliability": left_reliability,
                 "next_geometry_reliability": right_reliability,
                 "previous_ends_terminal": False, "reading_order_adjacent": True},
                "A bottom-of-page paragraph without terminal punctuation is followed by a paragraph in the same section.",
                "Review whether the text continues across the page break; do not merge it automatically.", "unverified",
                related_ids=related_ids,
                limitations=["line and paragraph boundaries may be approximate"], method="multiple_cross_page_signals",
            )

    def _summary(self) -> dict[str, Any]:
        element_section = {}
        for element_id, items in self.assignment_by_element.items():
            sections = [str(item.get("section_id")) for item in items if item.get("section_id")]
            element_section[element_id] = Counter(sections).most_common(1)[0][0] if sections else "unassigned"
        by_page, by_section, by_type = defaultdict(list), defaultdict(list), defaultdict(list)
        for finding in self.findings:
            for page in finding.page_numbers:
                by_page[page].append(finding)
            for source_id in finding.source_ids:
                if source_id in self.elements:
                    by_section[element_section.get(source_id, "unassigned")].append(finding)
                    by_type[self.elements[source_id].element_type].append(finding)

        def scorecard(items: list[QualityFinding], source_ids: set[str] | None = None) -> dict[str, Any]:
            unique = {item.finding_id: item for item in items}.values()
            card = {
                "finding_count": len(unique),
                "by_category": dict(sorted(Counter(item.category for item in unique).items())),
                "by_severity": dict(sorted(Counter(item.severity for item in unique).items())),
                "by_confidence": dict(sorted(Counter(item.confidence for item in unique).items())),
            }
            if source_ids is not None:
                card["elements_inspected"] = len(source_ids)
                card["needs_review_elements"] = sum(
                    self.elements[source_id].validation_state in {"needs_review", "pending_review"}
                    for source_id in source_ids
                )
            return card

        relation_warnings = [item for item in (self.inputs.relationships.get("validation") or {}).get("findings") or []
                             if item.get("severity") != "ok"]
        page_sources: dict[int, set[str]] = defaultdict(set)
        section_sources: dict[str, set[str]] = defaultdict(set)
        type_sources: dict[str, set[str]] = defaultdict(set)
        for element in self.ordered:
            page_sources[element.page].add(element.element_id)
            section_sources[element_section.get(element.element_id, "unassigned")].add(element.element_id)
            type_sources[element.element_type].add(element.element_id)
        for section_id in self.sections:
            section_sources.setdefault(section_id, set())
        return {
            "document": {
                "source_run_id": self.inputs.run.run_id,
                "elements_inspected": len(self.ordered),
                "needs_review_elements": sum(item.validation_state in {"needs_review", "pending_review"}
                                             for item in self.ordered),
                **scorecard(self.findings),
                "unresolved_relationship_findings": len(relation_warnings),
                "confirmed_extraction_failures": sum(item.subtype == "recorded_processing_failure"
                                                     for item in self.findings),
                "accuracy_percentage": None,
                "accuracy_limitation": "No ground truth is used; absence of findings does not prove OCR correctness.",
            },
            "pages": {str(page): scorecard(by_page.get(page, []), source_ids)
                      for page, source_ids in sorted(page_sources.items())},
            "sections": {section: scorecard(by_section.get(section, []), source_ids)
                         for section, source_ids in sorted(section_sources.items())},
            "element_types": {kind: scorecard(by_type.get(kind, []), source_ids)
                              for kind, source_ids in sorted(type_sources.items())},
            "inspection_coverage": {
                "inventory_records": len(self.inventory),
                "categories_attempted_per_element": len(CATEGORIES),
                "skipped_checks": dict(sorted(Counter(reason for item in self.inventory
                                                       for reason in item.skipped_checks.values()).items())),
                "available_signals": {
                    "nass_validation_report": self.inputs.run.validation_report is not None,
                    "execution_manifest": self.inputs.run.execution_manifest is not None,
                    "saved_layout": self.inputs.run.layout is not None,
                    "structure_assignments": bool(self.assignments),
                    "relationship_inventory": bool(self.relation_nodes),
                },
            },
            "rag_evidence": {
                "reliability_states": dict(sorted(Counter(
                    item.evidence_reliability for item in self.inventory
                ).items())),
                "flags": dict(sorted(Counter(flag for item in self.inventory for flag in item.rag_flags).items())),
                "policy": "evidence annotation only; no retrieval filtering or automatic high rating",
            },
        }

    def apply_rag_evidence_signals(self) -> None:
        findings_by_source: dict[str, list[QualityFinding]] = defaultdict(list)
        for finding in self.findings:
            for source_id in finding.source_ids:
                findings_by_source[source_id].append(finding)
        relation_by_id = {str(item.get("rel_id")): item for item in self.relationships}
        for record in self.inventory:
            flags = []
            reasons = []
            actionable_findings = []
            unflagged: list[str] = []
            dimensions: dict[str, dict[str, Any]] = {
                name: {"finding_ids": [], "reasons": []}
                for name in ("source_text_reliability", "structural_placement", "relationship_confidence",
                             "evidence_completeness")
            }
            if record.validation_state in {"needs_review", "pending_review"}:
                flags.append("needs_review")
                reasons.append(f"upstream validation_state is {record.validation_state}")
                dimensions["source_text_reliability"]["reasons"].append(
                    f"upstream validation_state is {record.validation_state}")
            relevant = findings_by_source.get(record.source_id, [])
            for finding in relevant:
                dimension = dimensions[reliability_dimension(finding.category, finding.subtype)]
                dimension["finding_ids"].append(finding.finding_id)
                dimension["reasons"].append(finding_reason(finding.to_dict()))
                flag = rag_flag(finding.category, finding.subtype, finding.confidence, finding.evidence)
                actionable = affects_content_reliability(
                    finding.category, finding.subtype, finding.confidence, finding.evidence,
                )
                if actionable:
                    actionable_findings.append(finding)
                if unflagged_content_concern(finding.category, finding.subtype, finding.severity, finding.status,
                                             finding.confidence, finding.evidence):
                    unflagged.append(finding_reason(finding.to_dict()))
                if flag:
                    flags.append(flag)
                    reasons.append(finding_reason(finding.to_dict()))
                if actionable and finding.status in {"open", "needs_review"}:
                    flags.append("needs_review")
                    reasons.append("requires review: " + finding_reason(finding.to_dict()))
            record.rag_flags = sorted(set(flags))
            record.evidence_reliability_reasons = list(dict.fromkeys(reasons))
            if record.rag_flags:
                severe = any(finding.severity in {"critical", "high"} for finding in actionable_findings)
                record.evidence_reliability = "low" if severe or "known_missing" in record.rag_flags else "medium"
            else:
                record.evidence_reliability = "unassessed"
            record.reliability_dimensions = self._dimensions(record, dimensions, relation_by_id, unflagged)

    def _dimensions(self, record: InventoryRecord, dimensions: dict[str, dict[str, Any]],
                    relation_by_id: dict[str, dict[str, Any]], unflagged: list[str] | None = None) -> dict[str, Any]:
        """Per-source reliability dimensions. Only source text and completeness feed evidence_reliability."""
        source = dimensions["source_text_reliability"]
        source["state"] = (record.evidence_reliability if source["finding_ids"] or source["reasons"]
                           else "no_finding")
        assignments = self.assignment_by_element.get(record.source_id, [])
        placement = dimensions["structural_placement"]
        placement["assignments"] = [{
            "unit_id": item.get("unit_id"), "section_id": item.get("section_id"), "role": item.get("role"),
            "confidence": item.get("confidence"), "review_recommended": bool(item.get("review_recommended")),
            "conflicting": list(item.get("conflicting") or []),
        } for item in assignments]
        if not assignments:
            placement["state"] = "not_assigned"
        elif any(item.get("confidence") == "low" or item.get("review_recommended") for item in assignments)                 or placement["finding_ids"]:
            placement["state"] = "uncertain"
        else:
            placement["state"] = "assigned"
        relation = dimensions["relationship_confidence"]
        relation["relationship_confidences"] = {
            rel_id: (relation_by_id.get(rel_id) or {}).get("confidence") for rel_id in record.relationship_ids
        }
        relation["state"] = ("uncertain" if relation["finding_ids"] else
                             "no_relationship" if not record.relationship_ids else "no_finding")
        completeness = dimensions["evidence_completeness"]
        completeness["state"] = "concern_recorded" if completeness["finding_ids"] else "no_finding"
        if record.evidence_reliability == "low" or "known_missing" in record.rag_flags:
            readiness = "review_before_use"
        elif record.rag_flags:
            readiness = "use_with_flags"
        elif unflagged:
            # A source-text/completeness concern is recorded without a specific flag (e.g. a page-level layout
            # warning): never reported as no_recorded_concern.
            readiness = "content_concern_recorded"
        elif placement["state"] == "uncertain" or relation["state"] == "uncertain":
            readiness = "content_unflagged_placement_or_links_uncertain"
        else:
            readiness = "no_recorded_concern"
        return {**dimensions, "retrieval_readiness": {
            "state": readiness,
            **({"unflagged_content_concerns": list(dict.fromkeys(unflagged))} if unflagged else {}),
            "basis": "derived from source text and completeness flags; structural and relationship uncertainty "
                     "is reported separately and does not lower evidence reliability",
            "note": "no_recorded_concern is not a high-reliability rating",
        }}

    def build(self) -> QualityResult:
        self.detect_source_quality()
        self.detect_ocr_suspects()
        self.detect_ocr_repetition_loops()
        self.detect_bidi_order_suspects()
        self.detect_numeric_integrity()
        self.detect_duplicates()
        self.detect_structure_quality()
        self.detect_relationship_quality()
        self.detect_table_quality()
        self.detect_figure_quality()
        self.detect_chart_and_vocabulary_quality()
        self.detect_known_missing_quality()
        self.detect_cross_page()
        self.findings.sort(key=lambda item: (-item.priority_score, item.category, item.subtype, item.finding_id))
        for rank, finding in enumerate(self.findings, 1):
            finding.priority_rank = rank
        for record in self.inventory:
            record.finding_ids.sort()
        self.apply_rag_evidence_signals()
        self.evidence_audit.sort(key=lambda item: (
            int(item.get("page") or 0), str(item.get("source_id") or ""),
            str(item.get("check") or ""),
            json.dumps(item.get("evidence") or {}, ensure_ascii=False, sort_keys=True, default=str),
        ))
        return QualityResult(
            self.inputs.run.run_id, str(self.inputs.run.run_dir), str(self.inputs.structure_dir),
            str(self.inputs.relations_dir), self.findings, self.inventory, self._summary(), self.nodes,
            list(self.inputs.run.warnings), self.inputs.hashes, rehash_quality_inputs(self.inputs),
            self.evidence_audit,
        )


def build_quality(inputs: QualityInputs, config: QualityConfig | None = None) -> QualityResult:
    return QualityBuilder(inputs, config or QualityConfig()).build()

