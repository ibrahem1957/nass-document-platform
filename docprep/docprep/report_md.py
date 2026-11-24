"""structure_report.md"""

from __future__ import annotations

from collections import Counter

from .pipeline import StructureResult
from .tree import LEVEL_NAMES


def _clip(text: str, n: int = 120) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _cell(text: str) -> str:
    return _clip(text).replace("|", "\\|")


def render_markdown(result: StructureResult) -> str:
    doc = result.inputs.document.get("document") or {}
    cands = result.candidates
    counts = Counter(c.status for c in cands)
    sections = result.tree.sections
    cand_by_id = {c.candidate_id: c for c in cands}
    routes = Counter(str(p.get("route")) for p in result.inputs.pages.values())
    lines = [
        f"# Document structure: {doc.get('display_name') or result.inputs.run_id}", "",
        f"- Source run: `{result.inputs.run_id}` ({result.inputs.page_count} pages; routes {dict(routes)})",
        (f"- Elements: {len(result.inputs.elements)} original; {len(result.units)} assignable units "
         f"({sum(len(v) for v in result.spans.values())} derived spans from {len(result.spans)} element(s))"),
        f"- Body typography: {result.body.size or 'n/a'} pt, colour {result.body.color or 'n/a'} ({result.body.note})",
        "- Read-only inspection of saved outputs. Text is quoted verbatim; nothing was corrected.", "",
        "## Outline", "",
    ]
    for s in sections:
        c = cand_by_id[s.heading_candidate_id]
        raw = f", raw {s.raw_level}" if s.raw_level != s.level else ""
        lines.append(f"{'  ' * max(0, s.level)}- **{_clip(s.title, 100)}** — level {s.level}{raw} "
                     f"({LEVEL_NAMES.get(s.level, 'level')}), pages {s.page_start}–{s.page_end}, "
                     f"detection {c.confidence}, level {c.level_confidence}, `{s.section_id}`"
                     + (" ⚠ review" if c.review else ""))
    if not sections:
        lines.append("_No heading was promoted._")
    lines += ["", "## Heading candidates", "",
              (f"- Promoted: {counts.get('promoted', 0)}; rejected: {counts.get('rejected', 0)}; "
               f"excluded: {counts.get('excluded', 0)}"),
              (f"- Promoted from non-heading Nass types: "
               f"{sum(1 for c in cands if c.status == 'promoted' and not any(result_unit_type(result, u) == 'heading' for u in c.unit_ids))}"),
              (f"- Nass 'heading' elements not promoted: "
               f"{sum(1 for c in cands if c.status != 'promoted' and any(result_unit_type(result, u) == 'heading' for u in c.unit_ids))}"),
              "", "### Promoted", "", "| Candidate | Page | Text | Score | Detection | Level | Level conf. | Main signals |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in cands:
        if c.status != "promoted":
            continue
        main = ", ".join(f"{s.name} {s.contribution:+.2f}" for s in sorted(c.signals, key=lambda s: -abs(s.contribution))[:4])
        lines.append(f"| {c.candidate_id} | {c.page} | {_cell(c.text)} | {c.score:.2f} | {c.confidence} | {c.level} | "
                     f"{c.level_confidence} | {_cell(main)} |")
    lines += ["", "### Rejected and excluded", "", "| Candidate | Page | Text | Status | Score | Reasons |",
              "| --- | --- | --- | --- | --- | --- |"]
    for c in cands:
        if c.status == "promoted":
            continue
        why = "; ".join(c.exclusions) if c.exclusions else ("near threshold (ambiguous)" if c.ambiguous else
                                                             "score below the promotion threshold")
        lines.append(f"| {c.candidate_id} | {c.page} | {_cell(c.text)} | {c.status} | {c.score:.2f} | {_cell(why)} |")
    roles = Counter(a.role for a in result.tree.assignments)
    lines += ["", "## Derived roles", "",
              (f"- List introductions: {roles.get('list_intro', 0)}; list items: {roles.get('list_item', 0)}; "
               f"wrapped continuations linked: {roles.get('list_item_continuation', 0)}; possible continuations not "
               f"linked: {sum(len(n.possible_continuations) for n in result.tree.lists)}"),
              (f"- Callout/side-box labels: {roles.get('aside_label', 0)}; side-box text: "
               f"{roles.get('aside_text', 0)} (layout-derived, uncertain)"),
              (f"- Semantic warnings (content never moved): "
               f"{sum(1 for a in result.tree.assignments if any(c.startswith('semantic warning') for c in a.conflicting))}")]
    checks = Counter((a.semantic_check or {}).get("status") for a in result.tree.assignments if a.semantic_check)
    lines.append(f"- Semantic checks: flagged {checks.get('flagged', 0)}, passed {checks.get('passed', 0)}, "
                 f"skipped {checks.get('skipped', 0)} (a skipped check never lowers confidence)")
    diags = result.layout_diagnostics
    lines += ["", "## Layout heading regions (saved PP-DocLayout)", ""]
    if not diags:
        lines.append("_No doc_title / paragraph_title regions in plan/layout.json._")
    else:
        assoc = Counter(d.association for d in diags)
        lines.append(f"- {len(diags)} region(s): verified {assoc.get('verified', 0)}, approximate "
                     f"{assoc.get('approximate', 0)}, no reliable association {assoc.get('none', 0)}; became a "
                     f"section heading: {sum(d.became_heading for d in diags)}")
        lines += ["", ("| Page | Label | Score | Box (pt) | Association | Match conf. | Element / span | Text | "
                        "Heading? | Evidence or reason |"),
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for d in diags:
            why = "; ".join(d.evidence) if d.association != "none" else (d.reason_unmatched or "")
            lines.append(f"| {d.page} | {d.layout_label} | {d.layout_score if d.layout_score is not None else '—'} | "
                         f"{[round(v) for v in d.layout_bbox_pt]} | {d.association} | {d.match_confidence or '—'} | "
                         f"{d.span_id or d.element_id or '—'} | {_cell(d.extracted_text or '')} | "
                         f"{(d.section_id or 'yes') if d.became_heading else 'no'} | {_cell(why)} |")
    low = [a for a in result.tree.assignments if a.confidence == "low"]
    lines += ["", "## Low-confidence assignments", ""]
    lines += [f"- `{a.unit_id}` (p{a.page}, {a.role}) → `{a.section_id}`: {_clip('; '.join(a.conflicting or a.reasons), 200)}"
              for a in low] or ["_None._"]
    unassigned = [a for a in result.tree.assignments if a.section_id is None]
    lines += ["", "## Unassigned content", ""]
    lines += [f"- `{a.unit_id}` (p{a.page}, {a.role}): {_clip('; '.join(a.reasons), 160)}" for a in unassigned] \
        or ["_None._"]
    lines += ["", "## Typography and layout classes", ""]
    for level in result.levels:
        lines.append(f"- {level}")
    if not result.levels:
        lines.append("_No heading classes._")
    lines += ["", "## Warnings", ""] + ([f"- {w}" for w in result.warnings] or ["_None._"])
    lines += ["", "## Validation", ""]
    lines += [f"- **{f['severity']}** `{f['code']}`: {f['message']}" + (f" ({f['item_count']} item(s))" if f['item_count'] else "")
              for f in result.findings]
    return "\n".join(lines) + "\n"


def result_unit_type(result: StructureResult, unit_id: str) -> str:
    for unit in result.units:
        if unit.unit_id == unit_id:
            return unit.element_type
    return ""
