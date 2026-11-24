"""Local-only Markdown and HTML review reports for relationships."""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from typing import Any

from .relation_model import RelationResult
from .relation_validation import validation_summary


def _clip(text: str | None, limit: int = 240) -> str:
    clean = " ".join((text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def render_relations_markdown(result: RelationResult, findings: list[dict[str, Any]]) -> str:
    counts = Counter(relation.type for relation in result.relationships)
    confidence = Counter(relation.confidence for relation in result.relationships)
    lines = [
        "# Content Relationship Report", "",
        f"- Source run: `{result.run_id}`",
        f"- Original visuals: {len(result.visuals)}",
        f"- Relationships: {len(result.relationships)}",
        f"- Confidence: high {confidence['high']}, medium {confidence['medium']}, low {confidence['low']}",
        f"- Validation: {validation_summary(findings)}", "",
        "## Relationship counts", "",
    ]
    lines.extend(f"- `{kind}`: {count}" for kind, count in sorted(counts.items()))
    lines += ["", "## Visual inventory", ""]
    for visual in result.visuals:
        lines += [
            f"### `{visual['visual_id']}` — {visual['visual_type']} on page {visual['page']}", "",
            (f"- Section: `{visual.get('section_id') or 'unassigned'}`; Nass state: "
             f"`{visual.get('validation_state')}`"),
            f"- Caption: {_clip((visual.get('caption') or {}).get('text')) or '_none_' }",
            f"- Internal title: {_clip((visual.get('internal_title') or {}).get('text')) or '_none_' }",
            (f"- Descriptions: {len(visual.get('description_paragraphs') or [])}; numeric matches: "
             f"{len(visual.get('matched_numeric_evidence') or [])}"),
            (f"- Leakage overlaps: {len(visual.get('leaked_text_overlaps') or [])}; logical table: "
             f"`{visual.get('logical_table_id') or 'none'}`"),
            f"- Review flags: {', '.join(visual.get('review_flags') or []) or '_none_'}", "",
        ]
    lines += ["## Relationships", ""]
    for relation in result.relationships:
        suffix = " — **review**" if relation.review_recommended else ""
        lines.append(
            f"- `{relation.rel_id}` `{relation.type}`: `{relation.source_id}` → `{relation.target_id}` "
            f"({relation.confidence}, {relation.method}){suffix}"
        )
    lines += ["", "## Lists", ""]
    for node in result.lists.get("nodes") or []:
        lines.append(f"- `{node['list_id']}`: {len(node.get('items') or [])} item(s), pages {node.get('pages')}")
        for item in node.get("items") or []:
            lines.append(f"  - {'  ' * int(item.get('level') or 0)}`{item['node_id']}` {_clip(item.get('text'), 120)}")
    if not result.lists.get("nodes"):
        lines.append("_No saved lists._")
    lines += ["", "## Conflicts and uncertainty", ""]
    uncertain = [relation for relation in result.relationships if relation.review_recommended]
    lines.extend(
        f"- `{relation.rel_id}`: {_clip(str(relation.conflicts or relation.limitations), 300)}"
        for relation in uncertain
    )
    lines += ["", "## Validation", ""]
    lines.extend(
        f"- **{item.get('severity')}** `{item.get('code')}`: {item.get('message')}"
        for item in findings
    )
    return "\n".join(lines) + "\n"


CSS = """
:root{--bg:#f7f7f4;--panel:#fff;--ink:#20201e;--muted:#6a6964;--line:#deddd7;--accent:#315f89;
--ok:#267349;--warn:#9b6100;--bad:#ad2921}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,"Segoe UI",Tahoma,sans-serif}main{max-width:1400px;margin:auto;padding:22px 18px 70px}
h1{margin:0}h2{margin-top:30px;border-top:2px solid var(--line);padding-top:10px}.grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:10px}.card{background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:10px 12px}.muted{color:var(--muted)}.high,.ok{color:var(--ok)}.medium,.warning{color:var(--warn)}
.low,.error{color:var(--bad)}code,.text{background:#f0efeb;border-radius:4px;padding:2px 5px}.text{white-space:pre-wrap;
word-break:break-word}table{width:100%;border-collapse:collapse}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);
padding:5px}.tree{margin-left:18px;border-left:2px solid var(--line);padding-left:10px}.flag{display:inline-block;border-radius:10px;
background:#eeeae0;padding:1px 7px;margin:1px}details summary{cursor:pointer;color:var(--accent)}
"""


def render_relations_html(result: RelationResult, findings: list[dict[str, Any]]) -> str:
    esc = lambda value: html.escape("" if value is None else str(value))
    counts = Counter(relation.type for relation in result.relationships)
    confidence = Counter(relation.confidence for relation in result.relationships)
    relations_by_visual: dict[str, list[Any]] = defaultdict(list)
    for relation in result.relationships:
        if relation.target_id in {item["visual_id"] for item in result.visuals}:
            relations_by_visual[relation.target_id].append(relation)
        if relation.source_id in {item["visual_id"] for item in result.visuals}:
            relations_by_visual[relation.source_id].append(relation)
    out = [
        ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Content Relationship Review</title><style>{CSS}</style>"
         "</head><body><main>"),
        "<h1>Content Relationship Review</h1>",
        f"<p class='muted'>Run <code>{esc(result.run_id)}</code>. Local-only; every quoted source string is HTML escaped.</p>",
        (f"<div class='card'>Original visuals {len(result.visuals)} · relationships {len(result.relationships)} · "
         f"confidence: <span class=high>{confidence['high']} high</span>, "
         f"<span class=medium>{confidence['medium']} medium</span>, "
         f"<span class=low>{confidence['low']} low</span> · validation {esc(validation_summary(findings))}</div>"),
        "<h2>Relationship summary</h2><div class='card'>",
        "".join(f"<span class='flag'>{esc(kind)}: {count}</span>" for kind, count in sorted(counts.items())),
        "</div><h2>Visuals</h2><div class='grid'>",
    ]
    for visual in result.visuals:
        caption = visual.get("caption") or {}
        title = visual.get("internal_title") or {}
        related = relations_by_visual.get(visual["visual_id"], [])
        out += [
            "<article class='card'>",
            f"<h3><code>{esc(visual['visual_id'])}</code> — {esc(visual['visual_type'])}, page {visual['page']}</h3>",
            (f"<p class='muted'>section {esc(visual.get('section_id') or 'unassigned')} · Nass state "
             f"{esc(visual.get('validation_state'))}</p>"),
            f"<b>Caption</b><div class='text' dir='auto'>{esc(caption.get('text') or 'None')}</div>",
            f"<b>Inside title</b><div class='text' dir='auto'>{esc(title.get('text') or 'None')}</div>",
            (f"<p>Descriptions {len(visual.get('description_paragraphs') or [])}; compatible numeric matches "
             f"{len(visual.get('matched_numeric_evidence') or [])}; leakage overlaps "
             f"{len(visual.get('leaked_text_overlaps') or [])}</p>"),
            "<details><summary>Evidence and relationships</summary><ul>",
            "".join(
                f"<li><code>{esc(rel.type)}</code> {esc(rel.source_id)} → {esc(rel.target_id)} "
                f"<span class='{esc(rel.confidence)}'>{esc(rel.confidence)}</span>"
                f"{' <b class=warning>review</b>' if rel.review_recommended else ''}"
                f"<div class='muted'>{esc(rel.evidence)}</div></li>" for rel in related
            ),
            "</ul></details>",
            "".join(f"<span class='flag warning'>{esc(flag)}</span>" for flag in visual.get("review_flags") or []),
            "</article>",
        ]
    out += ["</div><h2>Nested and merged lists</h2>"]
    if result.lists.get("nodes"):
        for list_node in result.lists["nodes"]:
            out.append(f"<div class='card'><h3>{esc(list_node['list_id'])}</h3>")
            for item in list_node.get("items") or []:
                out.append(
                    f"<div class='tree' style='margin-left:{18 * int(item.get('level') or 0)}px' dir='auto'>"
                    f"<code>{esc(item['node_id'])}</code> {esc(item.get('marker'))} {esc(item.get('text'))}</div>"
                )
            out.append("</div>")
    else:
        out.append("<p class='muted'>No saved lists.</p>")
    if result.lists.get("derived_spans"):
        out.append("<div class='card'><h3>Derived merged-item spans</h3><ul>")
        out.extend(
            f"<li><code>{esc(item['span_id'])}</code> parent {esc(item['parent_element_id'])} "
            f"[{item['start']}:{item['end']}] <span dir='auto'>{esc(item['text'])}</span></li>"
            for item in result.lists["derived_spans"]
        )
        out.append("</ul></div>")
    uncertain = [relation for relation in result.relationships if relation.review_recommended]
    out += ["<h2>Conflicts and uncertain relationships</h2><div class='card'><ul>"]
    out.extend(
        f"<li><code>{esc(relation.rel_id)}</code> {esc(relation.type)}: {esc(relation.conflicts or relation.limitations)}</li>"
        for relation in uncertain
    )
    out += ["</ul></div><h2>Known missing visuals and metadata</h2><div class='card'><ul>"]
    out.extend(
        f"<li><code>{esc(relation.type)}</code> {esc(relation.source_id)} → {esc(relation.target_id)}: "
        f"{esc(relation.evidence)}</li>" for relation in result.relationships
        if relation.type in {"known_missing", "document_metadata"}
    )
    out += ["</ul></div><h2>Validation and Nass review states</h2><div class='card'><ul>"]
    out.extend(
        f"<li class='{esc(item.get('severity'))}'><b>{esc(item.get('severity'))}</b> "
        f"<code>{esc(item.get('code'))}</code>: {esc(item.get('message'))}</li>" for item in findings
    )
    needs_review = [node for node in result.nodes.values() if "source_needs_review" in node.review_flags]
    out.extend(
        f"<li class='warning'>Nass needs_review: <code>{esc(node.node_id)}</code>, page {esc(node.page)}</li>"
        for node in needs_review
    )
    out.append("</ul></div></main></body></html>")
    return "".join(out)

