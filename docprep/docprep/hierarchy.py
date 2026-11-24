"""Heading levels from the document-wide style hierarchy (not from isolated elements).

Level 0 = document title (optional), 1 = major section, 2 = subsection, ... up to max_levels.
Level confidence is separate from detection confidence; ambiguity is recorded, never resolved by guessing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import pairwise

from .config import Config
from .model import HeadingCandidate, Unit
from .typography import BodyProfile, color_distance

PATTERN_RANK = ["layout:doc_title", "keyword", "ordinal", "roman", "layout:paragraph_title", "decimal",
                "paren_number", "latin_letter", "arabic_letter", "nass_heading", "colon_label",
                "bullet_colon_label", "title_echo", "plain"]
CONF_ORDER = {"low": 0, "medium": 1, "high": 2}


def _min_conf(a: str, b: str) -> str:
    return a if CONF_ORDER[a] <= CONF_ORDER[b] else b


def _typo_size(h: HeadingCandidate, units: dict[str, Unit]) -> float | None:
    sizes = []
    for uid in h.unit_ids:
        t = units[uid].typography
        if t is None or not t.available or t.size is None:
            return None
        sizes.append(t.size)
    return max(sizes) if sizes else None


def _depth(h: HeadingCandidate) -> int | None:
    return h.numbering["depth"] if h.numbering else None


def _document_title(promoted: list[HeadingCandidate], units: dict[str, Unit], ordered_content: list[str],
                    cfg: Config) -> HeadingCandidate | None:
    first_ids = ordered_content[:cfg.hierarchy.title_search_units]
    early = [h for h in promoted if h.unit_ids[0] in first_ids and h.page == min(h2.page for h2 in promoted)]
    if not early:
        return None
    sizes = {h.candidate_id: _typo_size(h, units) for h in promoted}
    tol = cfg.typography.size_class_tolerance
    for h in early:
        size = sizes[h.candidate_id]
        others = [s for cid, s in sizes.items() if cid != h.candidate_id and s is not None]
        if size is not None and others and all(size > s + tol for s in others):
            h.level_basis.append(f"document title: the largest heading font ({size}pt) among the first units")
            h.level_confidence = "high"
            return h
        same_class_on_page = [
            h2 for h2 in promoted if h2 is not h and h2.page == h.page and h2.style_key == h.style_key
        ]
        if (size is not None and others and h.numbering is None and h.unit_ids[0] == ordered_content[0]
                and all(size >= s - tol for s in others) and not same_class_on_page):
            # Tie: the title shares the largest size with later major headings. It still counts as the title
            # when it is the very first content, unnumbered, and the only heading of its class on its page.
            h.level_basis.append(
                f"document title: first content of the document, unnumbered, in the largest heading size "
                f"({size}pt, shared with later headings) and the only heading of its style on page {h.page}")
            h.level_confidence = "medium"
            return h
        if size is not None and not others and h.numbering is None:
            h.level_basis.append("document title: the only typographic heading, at the start of the document")
            h.level_confidence = "medium"
            return h
        if h.pattern_class == "layout:doc_title":
            h.level_basis.append("document title: layout region labelled doc_title at the start")
            h.level_confidence = "medium"
            return h
    if all(s is None for s in sizes.values()):
        h = early[0]
        if h.unit_ids[0] == ordered_content[0] and h.numbering is None and h.pattern_class in (
                "title_echo", "plain", "nass_heading", "layout:paragraph_title"):
            h.level_basis.append("document title: first heading-like text of the document (no typography available)")
            h.level_confidence = "low"
            h.level_ambiguous = True
            return h
    return None


def _plain_number(h: HeadingCandidate) -> int | None:
    n = h.numbering
    if (not n or n.get("depth") != 1 or n.get("scheme") not in ("decimal", "paren_number")
            or not str(n.get("label")).isdigit()):
        return None
    return int(n["label"])


def _numbered_runs(typed: list[HeadingCandidate]) -> list[tuple[HeadingCandidate | None, list[HeadingCandidate]]]:
    """Maximal runs of consecutive same-style headings numbered 1, 2, ... (at least two), with the heading
    directly before item 1 (the introducer, or None)."""
    runs = []
    index = 0
    while index < len(typed):
        if _plain_number(typed[index]) != 1:
            index += 1
            continue
        run = [typed[index]]
        for h in typed[index + 1:]:
            if (h.style_key != run[0].style_key or _plain_number(h) != len(run) + 1
                    or h.numbering.get("scheme") != run[0].numbering.get("scheme")):
                break
            run.append(h)
        if len(run) >= 2:
            runs.append((typed[index - 1] if index > 0 else None, run))
        index += len(run)
    return runs


def _run_indent(run: list[HeadingCandidate], units: dict[str, Unit]) -> float | None:
    unit = units.get(run[0].unit_ids[0])
    box = unit.bbox if unit is not None else None
    return float(box[0]) if box else None


def _align_parallel_numbered_runs(typed: list[HeadingCandidate], units: dict[str, Unit], cfg: Config) -> None:
    tol = cfg.hierarchy.numbered_run_indent_tolerance
    runs = [(intro, run, _run_indent(run, units)) for intro, run in _numbered_runs(typed)]
    levels = lambda run: {h.level for h in run}
    # Reference runs: uniform level, introduced directly by a heading of a different class one level above.
    references = [(intro, run, x) for intro, run, x in runs
                  if intro is not None and x is not None and len(levels(run)) == 1
                  and intro.style_key != run[0].style_key and intro.level == run[0].level - 1]
    for lead, run, x in runs:
        if (lead is None or x is None or lead.numbering is not None or lead.style_key != run[0].style_key
                or len(levels(run)) != 1 or run[0].level != lead.level + 1):
            continue
        matches = [(intro, ref) for intro, ref, rx in references
                   if ref[0].style_key == run[0].style_key
                   and ref[0].numbering.get("scheme") == run[0].numbering.get("scheme")
                   and abs(rx - x) <= tol and ref[0].level < run[0].level]
        if not matches or len({ref[0].level for _, ref in matches}) != 1:
            continue  # no reference, or the references disagree: leave the typographic placement
        target = matches[0][1][0].level
        refs = ", ".join(f"{ref[0].candidate_id}..{ref[-1].candidate_id} under {intro.candidate_id}"
                         for intro, ref in matches)
        old_lead = lead.level
        lead.level = target - 1
        lead.level_confidence = "low"
        lead.level_basis.append(
            f"parallel numbered runs: introduces {run[0].candidate_id}..{run[-1].candidate_id} "
            f"({len(run)} items, numbering restarts at 1, indent {x:.0f}pt), which matches the run(s) {refs} in "
            f"style, scheme and indent; those runs sit at level {target}, so this introducer is placed at level "
            f"{lead.level} (was {old_lead})")
        msg = (f"typography places {lead.candidate_id} in the same class as its numbered items (level {old_lead}); "
               f"parallel-run consistency places it at level {lead.level}")
        if msg not in lead.level_conflicts:
            lead.level_conflicts.append(msg)
        for h in run:
            old = h.level
            h.level = target
            h.level_confidence = _min_conf(h.level_confidence or "medium", "medium")
            h.level_basis.append(f"parallel numbered runs: same depth as the run(s) {refs}; level {old} -> {target}")


def assign_levels(candidates: list[HeadingCandidate], units: dict[str, Unit], body: BodyProfile,
                  ordered_content: list[str], cfg: Config) -> list[dict]:
    """Sets level fields on promoted candidates. Returns a description of the style classes found."""
    promoted = [h for h in candidates if h.status == "promoted"]
    classes_out: list[dict] = []
    if not promoted:
        return classes_out
    title = _document_title(promoted, units, ordered_content, cfg) if cfg.hierarchy.detect_document_title else None
    if title is not None:
        title.level, title.is_document_title = 0, True
        classes_out.append({"class": "document_title", "level": 0, "headings": [title.candidate_id]})
    rest = [h for h in promoted if h is not title]
    tol = cfg.typography.size_class_tolerance

    # 1. Typographic classes (size, weight, colour), most prominent first.
    typo = [(h, _typo_size(h, units)) for h in rest]
    typo_heads = [(h, s) for h, s in typo if s is not None]
    classes: list[dict] = []
    for h, size in sorted(typo_heads, key=lambda item: -item[1]):
        key = h.style_key
        bold = bool(key[1]) if isinstance(key, tuple) else False
        color = key[2] if isinstance(key, tuple) else None
        for cls in classes:
            if abs(cls["size"] - size) <= tol and cls["bold"] == bold and cls["color"] == color:
                cls["members"].append(h)
                break
        else:
            classes.append({"size": size, "bold": bold, "color": color, "members": [h]})
    body_color = body.color
    classes.sort(key=lambda c: (-c["size"], not c["bold"], c["color"] == body_color))
    # Size and weight define depth. A class that differs from an earlier class by colour only is a deeper level
    # only when the document interleaves them (A ... B ... A: B sits inside A's sections). When their members
    # never interleave, colour alone is no evidence of nesting and the classes share a level (recorded).
    order = {uid: i for i, uid in enumerate(ordered_content)}

    def positions(cls: dict) -> list[int]:
        return sorted(order.get(h.unit_ids[0], len(order)) for h in cls["members"])

    def interleaved(a: dict, b: dict) -> bool:
        pa, pb = positions(a), positions(b)
        return any(pa[0] < p < pa[-1] for p in pb) or any(pb[0] < p < pb[-1] for p in pa)

    next_rank = 0
    for position_index, cls in enumerate(classes):
        peer = next((other for other in classes[:position_index]
                     if abs(other["size"] - cls["size"]) <= tol and other["bold"] == cls["bold"]
                     and not interleaved(other, cls)), None)
        if peer is not None:
            cls["rank"], cls["colour_only_peer"] = peer["rank"], True
            peer["colour_only_peer"] = True
        else:
            next_rank += 1
            cls["rank"], cls["colour_only_peer"] = next_rank, False
    for index, cls in enumerate(classes):
        level = cls["rank"]
        neighbours = [abs(cls["size"] - other["size"]) for other in classes if other is not cls]
        gap = min(neighbours) if neighbours else None
        if len(classes) == 1:
            conf, why = "medium", "single typographic heading style: flat hierarchy"
        elif gap is not None and gap >= cfg.typography.distinct_size_gap:
            conf, why = "high", f"distinct font size ({cls['size']}pt, nearest other class {gap:.1f}pt away)"
        else:
            conf, why = "medium", "distinguished from another class by weight or colour only"
        body_like = (body.size is not None and abs(cls["size"] - body.size) <= tol
                     and cls["bold"] == ((body.bold_share or 0) >= 0.5)
                     and color_distance(cls["color"], body.color) < cfg.typography.color_distance)
        if body_like:
            independent = all(h.numbering or h.pattern_class.startswith("layout:") for h in cls["members"])
            conf = "medium" if independent else "low"
            why = ("same size, weight and colour as the body text: the level rests on "
                   + ("numbering or layout evidence" if independent else "weak evidence"))
        if cls["colour_only_peer"]:
            conf = _min_conf(conf, "medium")
            why += "; same size and weight as another heading class (differs by colour only): same level"
        depths = [d for d in (_depth(h) for h in cls["members"]) if d]
        min_depth = min(depths) if depths else 1
        for h in cls["members"]:
            extra = (_depth(h) - min_depth) if _depth(h) else 0
            h.level = level + extra
            h.level_confidence = conf
            h.level_basis.append(f"typographic class {index + 1} of {len(classes)} (level rank {level}): "
                                 f"{cls['size']}pt"
                                 f"{' bold' if cls['bold'] else ''}{' coloured ' + cls['color'] if cls['color'] != body_color and cls['color'] else ''}; {why}")
            if extra:
                h.level_basis.append(f"numbering depth {_depth(h)} below the class's depth {min_depth}")
                h.level_confidence = _min_conf(conf, "medium")
        classes_out.append({"class": f"typography_{index + 1}", "level": level, "size": cls["size"],
                            "bold": cls["bold"], "color": cls["color"], "confidence": conf,
                            "headings": [h.candidate_id for h in cls["members"]]})

    # 2. Numbering contradicting the typographic order is recorded as a conflict.
    numbered = [h for h, _ in typo_heads if h.numbering and h.level is not None]
    for a in numbered:
        for b in numbered:
            if a.numbering["scheme"] == b.numbering["scheme"] and a.level < b.level and _depth(a) > _depth(b):
                for h in (a, b):
                    msg = (f"numbering depth contradicts typography: {a.candidate_id} (depth {_depth(a)}) is more "
                           f"prominent than {b.candidate_id} (depth {_depth(b)})")
                    if msg not in h.level_conflicts:
                        h.level_conflicts.append(msg)
                    h.level_confidence = "low"

    # 2b. Top section level after a document title. When the first heading after the title belongs to a less
    # prominent class than later headings, the document's top sections start at that class: the more
    # prominent classes that appear only later are major sections at the same depth, not its parents.
    position = {uid: i for i, uid in enumerate(ordered_content)}

    def doc_order(h: HeadingCandidate) -> int:
        return position.get(h.unit_ids[0], len(position))

    typed = sorted((h for h, _ in typo_heads if h.level is not None), key=doc_order)
    if title is not None and typed and typed[0].level > 1:
        first = typed[0]
        shift = first.level - 1
        for h in typed:
            old = h.level
            h.level = max(1, old - shift)
            h.level_confidence = _min_conf(h.level_confidence or "medium", "medium")
            h.level_basis.append(
                f"top section level after the document title starts at the class of {first.candidate_id} "
                f"(typographic rank {first.level}); rank {old} placed at level {h.level}")

    # 2c. Numbering context inside one class: an unnumbered heading directly followed by a run of at least two
    # same-class headings numbered 1, 2, ... is the parent of that run (a numbered sub-sequence, not siblings).
    for index, lead in enumerate(typed):
        if lead.numbering is not None:
            continue
        run: list[HeadingCandidate] = []
        for h in typed[index + 1:]:
            n = h.numbering
            if (h.style_key != lead.style_key or h.level != lead.level or not n or n.get("depth") != 1
                    or n.get("scheme") not in ("decimal", "paren_number") or not str(n.get("label")).isdigit()):
                break
            if int(n["label"]) != len(run) + 1:
                break
            run.append(h)
        if len(run) < 2:
            continue
        for h in run:
            h.level = lead.level + 1
            h.level_confidence = _min_conf(h.level_confidence or "medium", "medium")
            h.level_basis.append(
                f"numbering context: item {h.numbering['label']} of a run of {len(run)} numbered headings directly "
                f"after the unnumbered same-style heading {lead.candidate_id}; placed one level below it")

    # 2d. Numbering continuity: item k directly continuing item k-1 of the same scheme, with the same weight and
    # colour and a size difference below a distinct class gap, is a sibling at the same level.
    numbered_typed = [h for h in typed if h.numbering and h.numbering.get("depth") == 1
                      and h.numbering.get("scheme") in ("decimal", "paren_number")
                      and str(h.numbering.get("label")).isdigit()]
    sizes = {id(h): s for h, s in typo_heads}
    for index, (prev, h) in enumerate(pairwise(numbered_typed), start=1):
        if (h.numbering["scheme"] != prev.numbering["scheme"]
                or int(h.numbering["label"]) != int(prev.numbering["label"]) + 1 or h.level == prev.level
                or not isinstance(h.style_key, tuple) or not isinstance(prev.style_key, tuple)
                or h.style_key[1:] != prev.style_key[1:]
                or abs(sizes[id(h)] - sizes[id(prev)]) > cfg.typography.distinct_size_gap):
            continue
        nxt = numbered_typed[index + 1] if index + 1 < len(numbered_typed) else None
        if (nxt is not None and nxt.numbering["scheme"] == h.numbering["scheme"]
                and int(nxt.numbering["label"]) == int(h.numbering["label"]) + 1 and nxt.level != prev.level):
            continue  # the following item does not confirm the sibling level: leave the typographic level
        old = h.level
        h.level = prev.level
        h.level_confidence = _min_conf(h.level_confidence or "medium", "medium")
        h.level_basis.append(
            f"numbering continuity: continues {prev.candidate_id} ({prev.numbering['label']} -> "
            f"{h.numbering['label']}) with the same weight and colour and a size difference of "
            f"{abs(sizes[id(h)] - sizes[id(prev)]):.1f}pt; level {old} -> {h.level}")

    # 2e. Parallel numbered runs. Two runs of the same typographic class and numbering scheme that each restart at
    # 1, start at the same indent and directly follow their introducing heading are parallel sequences: they sit
    # at the same depth. When one run is nested one level deeper only because its introducer shares the run's
    # class (2c), while the other run hangs directly under a more prominent heading, the deeper run and its
    # introducer are lifted to the reference depth. Typography alone cannot decide this case, so the introducer's
    # placement is recorded as a level conflict and its level confidence is capped at low.
    _align_parallel_numbered_runs(typed, units, cfg)

    # 3. Headings without typography: pattern classes.
    pattern_heads = [h for h, s in typo if s is None]
    if pattern_heads:
        typo_levels_by_pattern: dict[str, Counter] = defaultdict(Counter)
        for h, _ in typo_heads:
            typo_levels_by_pattern[h.pattern_class][h.level] += 1
        if typo_heads:
            top = min(h.level for h, _ in typo_heads)
            for h in pattern_heads:
                seen = typo_levels_by_pattern.get(h.pattern_class)
                if seen:
                    h.level = seen.most_common(1)[0][0]
                    h.level_basis.append(f"no typography; same text pattern ({h.pattern_class}) as typographic "
                                         f"headings at level {h.level}")
                else:
                    h.level = top
                    h.level_ambiguous = True
                    h.level_basis.append("no typography and no matching pattern: placed at the top section level "
                                         "(ambiguous)")
                h.level_confidence = "low"
        else:
            keys = sorted({(PATTERN_RANK.index(h.pattern_class) if h.pattern_class in PATTERN_RANK else len(PATTERN_RANK),
                            _depth(h) or 1) for h in pattern_heads})
            rank = {key: i + 1 for i, key in enumerate(keys)}
            multi = len(keys) > 1
            for h in pattern_heads:
                key = (PATTERN_RANK.index(h.pattern_class) if h.pattern_class in PATTERN_RANK else len(PATTERN_RANK),
                       _depth(h) or 1)
                h.level = rank[key]
                consistent_numbering = h.numbering is not None and h.numbering["scheme"] == "decimal_multi"
                h.level_confidence = "medium" if consistent_numbering else "low"
                h.level_basis.append(f"no typography: ordered by text pattern '{h.pattern_class}'"
                                     + (f", numbering depth {_depth(h)}" if _depth(h) else "")
                                     + ("" if multi else " (only one pattern: flat hierarchy)"))
                if not multi:
                    h.level_ambiguous = True
            classes_out.append({"class": "text_patterns", "levels": {f"{PATTERN_RANK[k[0]] if k[0] < len(PATTERN_RANK) else 'other'}/depth{k[1]}": v
                                                                    for k, v in rank.items()},
                                "confidence": "low"})
    # 4. Clamp.
    for h in rest:
        if h.level is not None and h.level > cfg.hierarchy.max_levels:
            h.level_basis.append(f"clamped from level {h.level} to the maximum {cfg.hierarchy.max_levels}")
            h.level = cfg.hierarchy.max_levels
            h.level_confidence = "low"
        if h.level is None:
            h.level, h.level_confidence, h.level_ambiguous = 1, "low", True
            h.level_basis.append("level could not be inferred; top section level used (ambiguous)")
        if h.level_ambiguous or h.level_conflicts:
            h.review = True
    if title is not None and title.level_ambiguous:
        title.review = True
    return classes_out
