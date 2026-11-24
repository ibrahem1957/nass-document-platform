"""Explainable heading detection. Every contribution to a score is recorded as a Signal."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

from .config import Config
from .layout import HEADING_LABELS
from .model import HeadingCandidate, Signal, Unit
from .semantic import SemanticEvidence, SemanticVerifier
from .text_patterns import list_intro_cues, words
from .typography import BodyProfile, color_distance

VISUAL_TYPES = ("figure", "table")


def style_key(unit: Unit, cfg: Config) -> tuple | None:
    """Typographic style class (size bucket, font bold, colour). Fake bold is deliberately not part of it."""
    t = unit.typography
    if t is None or not t.available or t.size is None:
        return None
    tol = cfg.typography.size_class_tolerance
    return (round(round(t.size / tol) * tol, 2), (t.bold_share or 0.0) >= cfg.typography.bold_share, t.color)


def size_ratio(unit: Unit, body: BodyProfile) -> float | None:
    t = unit.typography
    if t is None or not t.available or not t.size or not body.size:
        return None
    return t.size / body.size


def _bold_vs_body(unit: Unit, body: BodyProfile, cfg: Config) -> bool:
    t = unit.typography
    return bool(t and t.available and (t.bold_share or 0) >= cfg.typography.bold_share and (body.bold_share or 0) < 0.5)


def matches_body_style(unit: Unit, body: BodyProfile, cfg: Config) -> bool | None:
    """True when the unit has the predominant body-text style (size class, font weight, colour);
    None when typography is unavailable. Fake bold is ignored here (it is a separate, low-weight signal)."""
    t = unit.typography
    if t is None or not t.available or not t.size or not body.size:
        return None
    same_size = abs(t.size - body.size) <= cfg.typography.size_class_tolerance
    bold = (t.bold_share or 0) >= cfg.typography.bold_share
    body_bold = (body.bold_share or 0) >= 0.5
    same_color = color_distance(t.color, body.color) < cfg.typography.color_distance
    return same_size and bold == body_bold and same_color


def _layout_heading(unit: Unit) -> str | None:
    return HEADING_LABELS.get(unit.layout.label) if unit.layout is not None else None


def strong_visual(unit: Unit, body: BodyProfile, cfg: Config) -> bool:
    ratio = size_ratio(unit, body)
    return bool((ratio and ratio >= cfg.typography.size_ratio_medium) or _bold_vs_body(unit, body, cfg)
                or _layout_heading(unit))


def is_list_item(unit: Unit, body: BodyProfile, cfg: Config) -> bool:
    f = unit.features
    if f.get("list_marker"):
        return True
    return bool(f.get("numbering") and not strong_visual(unit, body, cfg) and f.get("terminal_punctuation"))


def in_pool(unit: Unit, body: BodyProfile, cfg: Config) -> bool:
    """Units that look heading-like on at least one signal. Others are never heading candidates."""
    if not unit.clean or unit.element_type in VISUAL_TYPES:
        return False
    f = unit.features
    p = cfg.patterns
    if unit.element_type == "heading" or strong_visual(unit, body, cfg):
        return True
    if f.get("numbering") or f.get("ends_colon"):
        return True
    if unit.span is not None and unit.span.role_hint == "heading":
        return True
    return (f.get("n_words", 0) <= p.short_words and not f.get("terminal_punctuation")
            and f.get("n_lines", 1) <= p.max_heading_lines)


def heading_word_threshold(units: list[Unit], body: BodyProfile, cfg: Config) -> tuple[int, str]:
    """Word count above which a colon-ended line reads like body text: derived from the document's anchor
    headings (Nass headings or visually prominent short lines), with a configured floor."""
    counts = [u.features.get("n_words", 0) for u in units
              if u.clean and u.element_type not in VISUAL_TYPES and not u.features.get("caption")
              and (u.element_type == "heading" or strong_visual(u, body, cfg))
              and u.features.get("n_words", 0) <= cfg.patterns.max_heading_words]
    floor = cfg.patterns.list_intro_min_words
    if not counts:
        return floor, f"no anchor headings; configured floor {floor} words"
    median = statistics.median(counts)
    derived = math.ceil(median * cfg.patterns.list_intro_heading_factor)
    return max(floor, derived), (f"median of {len(counts)} anchor headings = {median} words x "
                                 f"{cfg.patterns.list_intro_heading_factor}; floor {floor}")


def list_intro_evidence(unit: Unit, nxt: Unit | None, body: BodyProfile, cfg: Config,
                        min_words: int) -> dict[str, Any] | None:
    """A body-style, colon-ended sentence that leads straight into list items. A colon alone is never
    enough: it also needs sentence-like language cues or a body-like length, and no independent typographic,
    layout or numbering evidence of a heading."""
    f = unit.features
    if not f.get("ends_colon") or f.get("numbering") or f.get("list_marker"):
        return None
    if nxt is None or not is_list_item(nxt, body, cfg):
        return None
    if strong_visual(unit, body, cfg):
        return None
    style = matches_body_style(unit, body, cfg)
    if style is False:
        return None
    cues = list_intro_cues(unit.clean)
    if f.get("n_words", 0) >= min_words:
        cues.append(f"{f.get('n_words')} words, at least the derived body-like length of {min_words}")
    if not cues:
        return None
    return {"cues": cues, "first_list_unit": nxt.unit_id,
            "typography": "same as body text" if style else "typography unavailable (no contrary evidence)"}


def exclusions(unit: Unit, body: BodyProfile, cfg: Config) -> list[str]:
    f, p = unit.features, cfg.patterns
    reasons: list[str] = []
    if unit.furniture is not None:
        reasons.append(f"page furniture: {unit.furniture.kind} ({unit.furniture.reason})")
    if f.get("page_number"):
        reasons.append("page number")
    if f.get("caption"):
        reasons.append(f"figure/table caption ({f.get('caption_reason')})")
    elif unit.layout is not None and unit.layout.role == "caption" and not unit.layout.ambiguous:
        reasons.append(f"layout region labelled '{unit.layout.label}' (caption)")
    elif unit.span is not None and unit.span.role_hint == "caption":
        reasons.append("derived span recognised as a caption")
    if f.get("numeric_label"):
        reasons.append("short numeric label")
    if f.get("axis_like"):
        reasons.append("axis-like sequence of numbers")
    if f.get("fragment"):
        reasons.append("OCR fragment (too short or without letters)")
    if f.get("list_marker") and not f.get("ends_colon"):
        reasons.append(f"list item (marker '{f.get('list_marker')}' at the {f.get('marker_position')} of the line)")
    if f.get("n_words", 0) > p.max_heading_words or f.get("n_chars", 0) > p.max_heading_chars:
        reasons.append(f"too long for a heading ({f.get('n_words')} words)")
    if f.get("n_lines", 1) > p.max_heading_lines:
        reasons.append(f"spans {f.get('n_lines')} lines")
    if f.get("ends_colon") and f.get("n_words", 0) > p.colon_heading_max_words and not strong_visual(unit, body, cfg):
        reasons.append(f"ordinary sentence ending with a colon ({f.get('n_words')} words) that introduces the "
                       "following content")
    if f.get("list_intro"):
        reasons.append("list introduction: " + "; ".join(f["list_intro"]["cues"]) + " (followed by list item "
                       f"{f['list_intro']['first_list_unit']})")
    if f.get("aside") and f["aside"]["kind"] == "label":
        reasons.append(f"callout or side-box label (layout-derived, uncertain): {f['aside']['reason']}")
    if reasons and unit.element_type == "heading":
        reasons.append("this overrides the Nass element type 'heading'")
    return reasons


def _signals(unit: Unit, body: BodyProfile, cfg: Config, nxt: Unit | None, nxt_in_pool: bool,
             first_content_id: str | None) -> list[Signal]:
    w, tc, p, f = cfg.weights, cfg.typography, cfg.patterns, unit.features
    out: list[Signal] = []
    if unit.element_type == "heading":
        out.append(Signal("nass_heading_type", w.nass_heading_type, "Nass classified the element as 'heading'",
                          "source_type"))
    t = unit.typography
    ratio = size_ratio(unit, body)
    if ratio is not None:
        detail = f"font {t.size}pt = {ratio:.2f} x body {body.size}pt"
        if matches_body_style(unit, body, cfg):
            out.append(Signal("body_style", 0.0, "same size, weight and colour as the predominant body text: "
                                                 "no prominence or consistency bonus", "typography"))
        elif ratio >= tc.size_ratio_large:
            out.append(Signal("font_size_large", w.size_large, detail, "typography"))
        elif ratio >= tc.size_ratio_medium:
            out.append(Signal("font_size_larger", w.size_medium, detail, "typography"))
        elif ratio <= tc.size_ratio_small:
            out.append(Signal("font_size_smaller_than_body", w.size_smaller_than_body, detail, "typography"))
        else:
            out.append(Signal("font_size_like_body", 0.0, detail, "typography"))
        if _bold_vs_body(unit, body, cfg):
            out.append(Signal("bold", w.bold, f"{(t.bold_share or 0):.0%} of characters bold; body "
                                             f"{(body.bold_share or 0):.0%}", "typography"))
        if color_distance(t.color, body.color) >= tc.color_distance:
            out.append(Signal("distinct_color", w.distinct_color, f"colour {t.color} vs body {body.color}",
                              "typography"))
        if t.size_share is not None and t.size_share < tc.min_consistency:
            out.append(Signal("mixed_font_sizes", w.mixed_sizes, f"only {t.size_share:.0%} of characters at the "
                                                                  "dominant size", "typography"))
        if t.fake_bold_status == "fill_and_stroke" and (body.fake_bold_share or 0) < 0.5:
            out.append(Signal("fake_bold", w.fake_bold, f"{t.fake_bold_note} (fill-and-stroke rendering; "
                                                         "supporting evidence only)", "typography"))
        elif t.fake_bold_status == "unknown":
            out.append(Signal("fake_bold_unknown", 0.0, t.fake_bold_note, "typography"))
    elif unit.span is None and unit.route in ("DIGITAL", "MIXED"):
        reason = t.reason if t is not None else "typography not computed"
        out.append(Signal("typography_unavailable", 0.0, reason, "typography"))
    m = unit.layout
    if m is not None:
        factor = 1.0 if (m.score or 0.0) >= cfg.layout.min_score else 0.5
        if m.ambiguous:
            factor *= 0.5
        detail = (f"layout '{m.label}' (score {m.score}, containment {m.containment})"
                  + ("; overlapping regions with different roles" if m.ambiguous else ""))
        heading_label = HEADING_LABELS.get(m.label)
        if heading_label == "doc_title":
            out.append(Signal("layout_doc_title", w.layout_doc_title * factor, detail, "layout"))
        elif heading_label == "paragraph_title":
            out.append(Signal("layout_paragraph_title", w.layout_paragraph_title * factor, detail, "layout"))
        elif m.role == "body":
            out.append(Signal("layout_body_text", w.layout_body_text, detail, "layout"))
        elif m.role == "furniture" and (f.get("geometry") or {}).get("top_share", 1) < 0.2:
            out.append(Signal("layout_header_region_top", w.layout_header_top * factor,
                              detail + "; weak: also used for running headers", "layout"))
    if unit.span is not None:
        if unit.span.role_hint == "heading":
            out.append(Signal("span_heading_hint", w.span_heading_hint,
                              f"segmentation marked the derived span as heading-like ({unit.span.cut_reason})",
                              "pattern"))
        if "text:title_echoed_by_following_sentence" in unit.span.basis:
            out.append(Signal("title_echo", w.title_echo, "the same words open the following sentence", "pattern"))
        heading_hits = [b for b in unit.span.basis if b.startswith("layout:") and any(
            f"layout:{label} " in b for label in HEADING_LABELS)]
        if heading_hits and unit.span.role_hint == "heading":
            out.append(Signal("layout_heading_in_segment", w.layout_paragraph_title * 0.5,
                              "a heading region exists in this page-OCR segment (position inside the text unknown)",
                              "layout"))
    n = f.get("numbering")
    if n:
        if n["scheme"] in ("ordinal", "keyword"):
            out.append(Signal("ordinal_or_keyword", w.ordinal, f"starts with '{n['label']}'", "pattern"))
        else:
            out.append(Signal("numbering", w.numbering, f"{n['scheme']} '{n['label']}'", "pattern"))
            if n["scheme"] == "decimal_multi":
                out.append(Signal("multilevel_numbering", w.multilevel_numbering, f"depth {n['depth']}", "pattern"))
        nn = nxt.features.get("numbering") if nxt is not None else None
        if nn and nn["scheme"] == n["scheme"] and nn["depth"] == n["depth"]:
            out.append(Signal("numbered_list_sequence", w.numbered_list_sequence,
                              f"directly followed by the next {n['scheme']} item '{nn['label']}' with no content "
                              "between (list-like)", "context"))
    if f.get("ends_colon") and f.get("n_words", 0) <= p.colon_heading_max_words:
        out.append(Signal("short_colon_label", w.colon_short, f"{f.get('n_words')} words ending with a colon",
                          "pattern"))
    if f.get("terminal_punctuation"):
        out.append(Signal("terminal_punctuation", w.terminal_period, "ends like a sentence", "pattern"))
    else:
        out.append(Signal("no_terminal_punctuation", w.no_terminal_punctuation, "no sentence-final punctuation",
                          "pattern"))
    if f.get("n_words", 0) <= p.short_words and f.get("n_lines", 1) <= 1:
        out.append(Signal("short_standalone", w.short_standalone, f"{f.get('n_words')} words on one line",
                          "pattern"))
    if f.get("list_marker"):
        out.append(Signal("bullet_prefixed", w.bullet_prefixed, f"list marker '{f.get('list_marker')}' at the "
                                                                f"{f.get('marker_position')} of the line", "pattern"))
    g = f.get("geometry")
    if g:
        median = g.get("page_median_gap_pt")
        if g.get("gap_before_pt") and median and g["gap_before_pt"] > 1.5 * median:
            out.append(Signal("space_before", w.gap_before, f"{g['gap_before_pt']}pt before vs median {median}pt",
                              "position"))
        if g.get("alignment") == "center":
            out.append(Signal("centered", w.centered, "centred on the page", "position"))
    if first_content_id is not None and unit.unit_id == first_content_id:
        out.append(Signal("first_content_of_document", w.first_on_first_page, "first text of the document",
                          "position"))
    if nxt is not None:
        nf = nxt.features
        body_like = nxt.element_type not in VISUAL_TYPES and nxt.clean and (
            nf.get("n_words", 0) > max(p.short_words, 2 * f.get("n_words", 0)) or nf.get("terminal_punctuation")
            or nf.get("list_marker"))
        if body_like and not nxt_in_pool:
            out.append(Signal("followed_by_body", w.followed_by_body, "followed by body text or a list", "context"))
        if nxt_in_pool and style_key(nxt, cfg) is not None and style_key(nxt, cfg) == style_key(unit, cfg) and \
                not f.get("numbering"):
            if starts_numbered_sequence(nxt) and matches_body_style(unit, body, cfg) is False                     and not f.get("ends_colon") and not is_list_item(nxt, body, cfg):
                # A distinctive (non-body) unnumbered line directly followed by a same-style, non-list item
                # numbered "1" introduces a numbered heading run; it is not a wrapped line. Body-style lines and
                # colon-ended introductions keep the penalty (list-introduction protection).
                out.append(Signal("introduces_numbered_sequence", w.introduces_numbered_sequence,
                                  f"the next unit has the same distinctive style and starts a numbered sequence "
                                  f"('{nf['numbering']['label']}')", "context"))
            else:
                out.append(Signal("followed_by_same_style_candidate", w.followed_by_same_style_candidate,
                                  "the next unit has the same style (could be a list or a wrapped line)", "context"))
    return out


FIRST_SEQUENCE_LABELS = {"1", "a", "A", "I", "ا", "أ"}


def starts_numbered_sequence(unit: Unit) -> bool:
    n = unit.features.get("numbering")
    return bool(n and n.get("depth", 1) == 1 and str(n.get("label")) in FIRST_SEQUENCE_LABELS)


def _following_texts(index: int, content: list[Unit], heading_like: set[str], cfg: Config) -> list[str]:
    texts = []
    for unit in content[index + 1:]:
        if unit.unit_id in heading_like:
            break
        if unit.element_type in VISUAL_TYPES or not unit.clean:
            continue
        texts.append(unit.clean)
        if len(texts) >= cfg.semantic.lookahead_units:
            break
    return texts


def _wraps(a: Unit, b: Unit, cfg: Config) -> bool:
    """Two consecutive lines of one heading: same page, same style, tight vertical gap, overlapping horizontally,
    first line not finished, second line not a new numbered item."""
    if a.page != b.page or a.span is not None or b.span is not None or a.bbox is None or b.bbox is None:
        return False
    ka, kb = style_key(a, cfg), style_key(b, cfg)
    if ka is None or ka != kb:
        return False
    fa, fb = a.features, b.features
    if fa.get("ends_colon") or fa.get("terminal_punctuation") or fb.get("numbering") or fb.get("list_marker"):
        return False
    size = a.typography.size or 0.0
    gap = b.bbox[1] - a.bbox[3]
    if not (-0.2 * size <= gap <= 0.8 * size):
        return False
    overlap = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
    narrower = min(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0])
    if overlap < 0.5 * narrower:
        return False
    return fa.get("n_words", 0) + fb.get("n_words", 0) <= cfg.patterns.max_heading_words


def pattern_class(unit: Unit) -> str:
    f = unit.features
    n = f.get("numbering")
    if unit.layout is not None and unit.layout.label in HEADING_LABELS:
        return f"layout:{HEADING_LABELS[unit.layout.label]}"
    if n:
        return "decimal" if n["scheme"] == "decimal_multi" else n["scheme"]
    if f.get("list_marker") and f.get("ends_colon"):
        return "bullet_colon_label"
    if f.get("ends_colon"):
        return "colon_label"
    if unit.span is not None and "text:title_echoed_by_following_sentence" in unit.span.basis:
        return "title_echo"
    if unit.element_type == "heading":
        return "nass_heading"
    return "plain"


def _confidence(score: float, cfg: Config) -> str:
    t = cfg.thresholds
    return "high" if score >= t.high else "medium" if score >= t.medium else "low"


def detect_headings(units: list[Unit], body: BodyProfile, cfg: Config,
                    verifier: SemanticVerifier) -> list[HeadingCandidate]:
    content = [u for u in units if u.furniture is None]
    position = {u.unit_id: i for i, u in enumerate(content)}

    def following(unit: Unit) -> Unit | None:
        i = position.get(unit.unit_id)
        return content[i + 1] if i is not None and i + 1 < len(content) else None

    # Derived roles that depend on the neighbour: list introductions.
    min_words, threshold_basis = heading_word_threshold(units, body, cfg)
    for unit in units:
        if unit.furniture is None and unit.clean and unit.element_type not in VISUAL_TYPES:
            evidence = list_intro_evidence(unit, following(unit), body, cfg, min_words)
            if evidence is not None:
                evidence["threshold_basis"] = threshold_basis
                unit.features["list_intro"] = evidence

    first_content = next((u for u in content if u.clean and u.element_type not in VISUAL_TYPES), None)
    first_content_id = first_content.unit_id if first_content is not None else None
    pool = [u for u in units if in_pool(u, body, cfg)]
    pool_ids = {u.unit_id for u in pool}
    excluded: dict[str, list[str]] = {u.unit_id: exclusions(u, body, cfg) for u in pool}
    signals: dict[str, list[Signal]] = {}
    for unit in pool:
        nxt = following(unit)
        nxt_in_pool = nxt is not None and nxt.unit_id in pool_ids and not excluded.get(nxt.unit_id)
        signals[unit.unit_id] = _signals(unit, body, cfg, nxt, nxt_in_pool, first_content_id)

    def score(uid: str) -> float:
        return sum(s.contribution for s in signals[uid])

    live = [u for u in pool if not excluded[u.unit_id]]
    likely = {u.unit_id for u in live if score(u.unit_id) >= cfg.thresholds.promote}
    semantic: dict[str, dict[str, Any]] = {}
    for unit in live:
        if unit.unit_id not in position:
            continue
        texts = _following_texts(position[unit.unit_id], content, likely - {unit.unit_id}, cfg)
        nxt = following(unit)
        if not texts and nxt is not None and nxt.unit_id in likely:
            # The next structural unit is itself a heading (e.g. a child heading): judged later, by the tree.
            evidence = SemanticEvidence(verifier.name, 0.0, [(
                "next structural unit is a heading candidate; the check is deferred to the section tree "
                "(no penalty here)"
            )], [])
            semantic[unit.unit_id] = evidence.to_dict() | {"deferred": True}
            continue
        evidence = verifier.verify(unit.clean, texts)
        semantic[unit.unit_id] = evidence.to_dict()
        if evidence.verifier != "none":
            w = cfg.weights
            value = evidence.score * w.semantic_max if evidence.score >= 0 else -evidence.score * w.semantic_min
            detail = "; ".join(evidence.supporting + evidence.conflicting) or "no evidence either way"
            signals[unit.unit_id].append(Signal("semantic_relation", value, detail, "semantic"))
    # Document-wide consistency: the same distinctive style as at least two confident headings.
    # Body-style text never receives this bonus.
    def distinctive(u: Unit) -> bool:
        return style_key(u, cfg) is not None and not matches_body_style(u, body, cfg)

    strong = Counter(style_key(u, cfg) for u in live if distinctive(u) and score(u.unit_id) >= cfg.thresholds.medium)
    for unit in live:
        if not distinctive(unit):
            continue
        key = style_key(unit, cfg)
        others = strong[key] - (1 if score(unit.unit_id) >= cfg.thresholds.medium else 0)
        if others >= 2:
            signals[unit.unit_id].append(Signal("style_consistency", cfg.weights.style_consistency,
                                                f"same distinctive style as {others} other confident heading(s): "
                                                f"{key}", "style"))
    # Document-wide repeated distinctive style: a colour or weight that differs from the body and that the
    # document uses (almost) only for short, standalone, heading-like lines on several pages. This does not
    # need already-confident members, so a consistently styled heading class at body size is not lost.
    # Independent of style_consistency: a unit never receives both.
    tol = cfg.typography.size_class_tolerance
    pc = cfg.patterns

    def family(u: Unit) -> tuple | None:
        t = u.typography
        key = style_key(u, cfg)
        if key is None or body.size is None or matches_body_style(u, body, cfg) or t.size < body.size - tol:
            return None
        if color_distance(t.color, body.color) < cfg.typography.color_distance and not _bold_vs_body(u, body, cfg):
            return None
        return (key[1], key[2])

    def heading_like(u: Unit) -> bool:
        f = u.features
        return (not f.get("terminal_punctuation") and not f.get("list_marker") and not excluded.get(u.unit_id)
                and f.get("n_words", 0) <= pc.max_heading_words and f.get("n_lines", 1) <= pc.max_heading_lines)

    by_family: dict[tuple, list[Unit]] = {}
    for unit in content:
        if unit.clean and unit.element_type not in VISUAL_TYPES:
            fam = family(unit)
            if fam is not None:
                by_family.setdefault(fam, []).append(unit)
    for fam, members in by_family.items():
        heading_members = [u for u in members if u.unit_id in pool_ids and heading_like(u)]
        pages = sorted({u.page for u in heading_members})
        share = len(heading_members) / len(members)
        if len(heading_members) < pc.repeated_style_min_members or len(pages) < pc.repeated_style_min_pages \
                or share < pc.repeated_style_min_share:
            continue
        for unit in heading_members:
            # The stronger document-wide evidence supersedes style_consistency; the two are never added.
            replaced = [s for s in signals[unit.unit_id] if s.name == "style_consistency"]
            signals[unit.unit_id] = [s for s in signals[unit.unit_id] if s.name != "style_consistency"]
            signals[unit.unit_id].append(Signal(
                "repeated_heading_style", cfg.weights.repeated_heading_style,
                f"distinctive style (bold {fam[0]}, colour {fam[1]}) used by {len(heading_members)} short "
                f"standalone line(s) on {len(pages)} page(s); {share:.0%} of all {len(members)} line(s) in this "
                "style are heading-like" + ("; supersedes style_consistency" if replaced else ""), "style"))
    # Wrapped headings: consecutive live candidates that are one heading split over lines.
    groups: list[list[Unit]] = []
    for unit in sorted(live, key=lambda u: u.order):
        if groups and position.get(unit.unit_id) == position.get(groups[-1][-1].unit_id, -2) + 1 \
                and _wraps(groups[-1][-1], unit, cfg):
            groups[-1].append(unit)
        else:
            groups.append([unit])
    for unit in pool:
        if excluded[unit.unit_id]:
            groups.append([unit])
    groups.sort(key=lambda g: g[0].order)

    candidates: list[HeadingCandidate] = []
    for number, group in enumerate(groups, start=1):
        first = group[0]
        sigs = list(signals[first.unit_id])
        # Rounded so that a sum of recorded contributions that equals the threshold is not lost to float error.
        total = round(max(score(u.unit_id) for u in group), 6)
        if len(group) > 1:
            sigs.append(Signal("merged_wrapped_lines", 0.0,
                               f"{len(group)} consecutive lines with the same style, tight spacing and textual "
                               "continuity were merged", "context"))
        reasons = excluded[first.unit_id]
        conflicts = [f"{s.name}: {s.detail}" for s in sigs if s.contribution < 0]
        if first.element_type == "heading" and reasons:
            conflicts.append("Nass type 'heading' conflicts with the exclusion")
        status = "excluded" if reasons else ("promoted" if total >= cfg.thresholds.promote else "rejected")
        notes = []
        fake = sum(s.contribution for s in sigs if s.name == "fake_bold")
        if status == "promoted" and fake > 0 and total - fake < cfg.thresholds.promote:
            status = "rejected"
            notes.append("not promoted: the score passes the threshold only because of fake-bold rendering")
        confidence = _confidence(total, cfg) if status == "promoted" else None
        if status == "promoted" and first.span is not None and first.span.uncertain and confidence == "high":
            confidence = "medium"
            notes.append("capped at medium: derived span with uncertain segmentation")
        ambiguous = status != "excluded" and abs(total - cfg.thresholds.promote) < cfg.thresholds.ambiguity_band
        if status == "rejected" and total < cfg.thresholds.reject_floor:
            notes.append("weak candidate")
        if first.features.get("aside"):
            notes.append("callout/side-box classification is layout-derived and uncertain")
        review = (status == "promoted" and (confidence == "low" or ambiguous or bool(conflicts))) or \
                 (status == "rejected" and ambiguous) or bool(first.features.get("aside"))
        n = first.features.get("numbering")
        candidates.append(HeadingCandidate(
            candidate_id=f"c{number:04d}", unit_ids=[u.unit_id for u in group], page=first.page,
            text=" ".join(u.clean for u in group), parts=[u.text for u in group], signals=sigs, score=total,
            status=status, exclusions=reasons, confidence=confidence, ambiguous=ambiguous, review=review,
            conflicts=conflicts, semantic=semantic.get(group[-1].unit_id), style_key=style_key(first, cfg),
            pattern_class=pattern_class(first), numbering=n, notes=notes))
    return candidates


def word_count(text: str) -> int:
    return len(words(text))
