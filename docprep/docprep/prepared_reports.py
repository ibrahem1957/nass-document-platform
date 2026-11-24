"""Self-contained local reports for the prepared document."""

from __future__ import annotations

import html
from collections import Counter

from .prepared_model import PreparedResult
from .prepared_validation import validation_summary


def render_prepared_markdown(result: PreparedResult) -> str:
    document = result.document
    relationships = Counter(str(item.get("type")) for item in document.get("relationships") or [])
    severity = Counter(str(item.get("severity")) for item in (document.get("quality") or {}).get("findings") or [])
    metadata = document.get("metadata") or {}
    lines = [
        "# Unified Prepared Document Report", "",
        f"- Document ID: `{document.get('document_id')}`",
        f"- Source run: `{document.get('source_run_id')}`",
        f"- Pages: {document.get('page_count')}",
        f"- Elements: {len(document.get('elements') or [])}",
        f"- Sections: {len(document.get('sections') or [])}",
        f"- Lists: {len(document.get('lists') or [])}",
        f"- Visuals: {len(document.get('visuals') or [])}",
        f"- Logical tables: {len(document.get('logical_tables') or [])}",
        f"- Validation: {validation_summary(result.validation)}", "",
        "> Source text is verbatim. This output contains no chunks, embeddings, corrections, retrieval, or inference.",
        "", "## Metadata", "",
        f"- Title: {metadata.get('title') or '_unavailable_'}",
        f"- Author: {metadata.get('author') or '_unavailable_'}",
        f"- Language: {metadata.get('language') or '_unavailable_'} ({metadata.get('language_status')})",
        f"- Source filename: {metadata.get('source_filename') or '_unavailable_'}", "",
        "## Section hierarchy", "",
    ]
    for section in document.get("sections") or []:
        indent = "  " * max(0, int(section.get("level") or 0))
        lines.append(
            f"{indent}- `{section.get('section_id')}` {section.get('title')} "
            f"(pages {section.get('page_start')}-{section.get('page_end')}, "
            f"confidence {section.get('confidence')}, review={section.get('review_recommended')})"
        )
    lines += ["", "## Lists", ""]
    for item in document.get("lists") or []:
        structure = item.get("structure") or {}
        lines.append(
            f"- `{item.get('list_id')}`: {len(item.get('item_ids') or [])} item(s), "
            f"pages {structure.get('pages') or []}, relationships {len(item.get('relationship_ids') or [])}"
        )
    lines += ["", "## Figures and tables", ""]
    for visual in document.get("visuals") or []:
        lines.append(
            f"- `{visual.get('visual_id')}` {visual.get('visual_type')} page {visual.get('page')}; "
            f"section `{visual.get('section_id')}`; validation `{visual.get('validation_state')}`; "
            f"relationships {len(visual.get('relationship_ids') or [])}; "
            f"quality findings {len(visual.get('quality_finding_ids') or [])}"
        )
    lines += ["", "## Multi-page logical tables", ""]
    for table in document.get("logical_tables") or []:
        lines.append(
            f"- `{table.get('logical_table_id')}`: parts {table.get('source_part_ids')}; "
            f"pages {table.get('pages')}; repeated headers {table.get('repeated_header_between_parts')}"
        )
    if not document.get("logical_tables"):
        lines.append("- None recorded.")
    lines += ["", "## Relationships", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(relationships.items()))
    lines += ["", "## Quality findings", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(severity.items()))
    lines += ["", "## Provenance coverage", ""]
    provenance = document.get("provenance") or {}
    lines += [
        f"- Input layers hashed: {', '.join(sorted((provenance.get('input_sha256') or {}).keys()))}",
        f"- Configuration SHA-256: `{provenance.get('processing_configuration_sha256')}`",
        f"- Policy: {provenance.get('content_policy')}", "", "## Validation", "",
    ]
    lines.extend(f"- **{item['severity']}** `{item['code']}`: {item['message']}" for item in result.validation)
    lines += ["", "## Remaining ambiguities", ""]
    review = [item for item in (document.get("quality") or {}).get("findings") or []
              if item.get("status") in {"needs_review", "unverified", "open"}]
    lines.append(f"- {len(review)} quality finding(s) remain open, unverified, or review-recommended.")
    lines.append("- Unknown metadata stays null; Component 4 does not infer author or language.")
    return "\n".join(lines) + "\n"


CSS = """
:root{--bg:#f5f5f1;--panel:#fff;--ink:#20211f;--muted:#666;--line:#d9d9d1;--accent:#285f85;--bad:#a1261d;--ok:#267349}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Segoe UI",Tahoma,sans-serif}
main{max-width:1450px;margin:auto;padding:22px 16px 70px}h1{margin:0}h2{border-top:2px solid var(--line);padding-top:10px;margin-top:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0}
.muted{color:var(--muted)}code{word-break:break-all}.error{color:var(--bad)}.ok{color:var(--ok)}table{width:100%;border-collapse:collapse}th,td{padding:6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}details summary{cursor:pointer;color:var(--accent)}
"""


def render_prepared_html(result: PreparedResult) -> str:
    e = lambda value: html.escape("" if value is None else str(value))
    document = result.document
    metadata = document.get("metadata") or {}
    relationship_counts = Counter(str(item.get("type")) for item in document.get("relationships") or [])
    severity = Counter(str(item.get("severity")) for item in (document.get("quality") or {}).get("findings") or [])
    out = [
        ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Prepared Document Report</title><style>{CSS}</style>"
         "</head><body><main>"),
        "<h1>Unified Prepared Document Report</h1>",
        f"<p class='muted'>Document <code>{e(document.get('document_id'))}</code>. Local, self-contained report.</p>",
        "<div class='grid'>",
    ]
    cards = {
        "Pages": document.get("page_count"), "Elements": len(document.get("elements") or []),
        "Sections": len(document.get("sections") or []), "Lists": len(document.get("lists") or []),
        "Visuals": len(document.get("visuals") or []),
        "Logical tables": len(document.get("logical_tables") or []),
    }
    out.extend(f"<div class='card'><b>{e(key)}</b><br>{e(value)}</div>" for key, value in cards.items())
    out += ["</div><h2>Metadata and unknowns</h2><div class='card'>",
            f"<p><b>Title:</b> <span dir='auto'>{e(metadata.get('title') or 'unavailable')}</span></p>",
            f"<p><b>Author:</b> {e(metadata.get('author') or 'unavailable')} — {e(metadata.get('author_status'))}</p>",
            (f"<p><b>Language:</b> {e(metadata.get('language') or 'unavailable')} — "
             f"{e(metadata.get('language_status'))}</p></div>"),
            "<h2>Section hierarchy</h2>"]
    for section in document.get("sections") or []:
        margin = max(0, int(section.get("level") or 0)) * 18
        out.append(
            f"<div class='card' style='margin-left:{margin}px'><b><code>{e(section.get('section_id'))}</code> "
            f"<span dir='auto'>{e(section.get('title'))}</span></b><br>pages {e(section.get('page_start'))}–"
            f"{e(section.get('page_end'))}; parent {e(section.get('parent'))}; confidence "
            f"{e(section.get('confidence'))}; review {e(section.get('review_recommended'))}</div>"
        )
    out += ["<h2>Lists and nesting</h2><div class='grid'>"]
    for item in document.get("lists") or []:
        out.append(
            f"<div class='card'><b><code>{e(item.get('list_id'))}</code></b><br>items "
            f"{len(item.get('item_ids') or [])}; relationships {len(item.get('relationship_ids') or [])}"
            f"<details><summary>Nested representation</summary><pre>{e(item.get('nested_items'))}</pre></details></div>"
        )
    out += [("</div><h2>Figure and table inventory</h2><table><tr><th>ID</th><th>Type</th><th>Page</th>"
             "<th>Section</th><th>State</th><th>Relationships</th><th>Quality</th></tr>")]
    for visual in document.get("visuals") or []:
        out.append(
            f"<tr><td><code>{e(visual.get('visual_id'))}</code></td><td>{e(visual.get('visual_type'))}</td>"
            f"<td>{e(visual.get('page'))}</td><td>{e(visual.get('section_id'))}</td>"
            f"<td>{e(visual.get('validation_state'))}</td><td>{len(visual.get('relationship_ids') or [])}</td>"
            f"<td>{len(visual.get('quality_finding_ids') or [])}</td></tr>"
        )
    out += ["</table><h2>Logical tables</h2>"]
    for table in document.get("logical_tables") or []:
        out.append(
            f"<div class='card'><code>{e(table.get('logical_table_id'))}</code>: parts "
            f"{e(table.get('source_part_ids'))}; pages {e(table.get('pages'))}; repeated header "
            f"{e(table.get('repeated_header_between_parts'))}</div>"
        )
    if not document.get("logical_tables"):
        out.append("<div class='card'>None recorded.</div>")
    out += ["<h2>Relationship and quality summaries</h2><div class='grid'>",
            f"<div class='card'><b>Relationships</b><pre>{e(dict(sorted(relationship_counts.items())))}</pre></div>",
            f"<div class='card'><b>Quality severity</b><pre>{e(dict(sorted(severity.items())))}</pre></div>",
            "</div><h2>Provenance</h2><div class='card'>",
            f"<p>Input layers: {e(sorted((document.get('provenance') or {}).get('input_sha256') or {}))}</p>",
            f"<p>Configuration SHA-256: <code>{e((document.get('provenance') or {}).get('processing_configuration_sha256'))}</code></p>",
            f"<p>{e((document.get('provenance') or {}).get('content_policy'))}</p></div>",
            "<h2>Validation and remaining ambiguities</h2><div class='card'><ul>"]
    out.extend(f"<li class='{e(item['severity'])}'><code>{e(item['code'])}</code>: {e(item['message'])}</li>"
               for item in result.validation)
    unresolved = sum(item.get("status") in {"open", "needs_review", "unverified"}
                     for item in (document.get("quality") or {}).get("findings") or [])
    out += [f"</ul><p>{unresolved} quality finding(s) remain open, unverified, or review-recommended.</p>",
            "<p>No unknown metadata was inferred and no source content was corrected.</p></div>",
            "</main></body></html>"]
    return "".join(out)
