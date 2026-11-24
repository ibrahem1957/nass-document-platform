"""Self-contained local reports for Component 3."""

from __future__ import annotations

import html
from collections import defaultdict
from typing import Any

from .quality_model import QualityResult
from .quality_validation import validation_summary


def _clip(value: Any, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_quality_markdown(result: QualityResult) -> str:
    doc = result.summary["document"]
    lines = [
        "# Document Quality & Validation Report", "",
        f"- Source run: `{result.run_id}`",
        f"- Elements inspected: {doc['elements_inspected']}",
        f"- Findings: {doc['finding_count']} — severity {doc['by_severity']}",
        f"- Nass review states: {doc['needs_review_elements']} element(s)",
        f"- Validation: {validation_summary(result.validation)}", "",
        "> This is an evidence-based review queue, not an OCR accuracy score. No text was corrected or removed.", "",
        "## Prioritized findings", "",
    ]
    for finding in result.findings:
        lines += [
            f"### {finding.priority_rank}. `{finding.finding_id}` — {finding.category}/{finding.subtype}", "",
            f"- Severity/confidence/status: **{finding.severity}** / {finding.confidence} / {finding.status}",
            (f"- Sources: {', '.join(f'`{value}`' for value in finding.source_ids) or '_none_'}; "
             f"pages {finding.page_numbers or '_not directly identified_'}"),
            f"- Explanation: {finding.explanation}",
            f"- Evidence: `{_clip(finding.evidence)}`",
            f"- Review action: {finding.suggested_review_action}",
        ]
        if finding.limitations:
            lines.append(f"- Limitations: {'; '.join(finding.limitations)}")
        lines.append("")
    lines += ["## Page scorecards", ""]
    for page, scorecard in result.summary.get("pages", {}).items():
        lines.append(f"- Page {page}: {scorecard}")
    lines += ["", "## Section scorecards", ""]
    for section, scorecard in result.summary.get("sections", {}).items():
        lines.append(f"- `{section}`: {scorecard}")
    lines += ["", "## Validation", ""]
    lines.extend(f"- **{item['severity']}** `{item['code']}`: {item['message']}" for item in result.validation)
    return "\n".join(lines) + "\n"


CSS = """
:root{--bg:#f6f6f2;--panel:#fff;--ink:#20201e;--muted:#696862;--line:#dddcd5;--accent:#2e5d87;
--critical:#7d0014;--high:#b3261e;--medium:#9b6100;--low:#536b28;--info:#316a83;--ok:#267349}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Segoe UI",Tahoma,sans-serif}
main{max-width:1500px;margin:auto;padding:20px 16px 70px}h1{margin:0}h2{margin-top:30px;border-top:2px solid var(--line);padding-top:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0}.grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px}.muted{color:var(--muted)}.critical{color:var(--critical)}
.high,.error{color:var(--high)}.medium{color:var(--medium)}.low{color:var(--low)}.info{color:var(--info)}.ok{color:var(--ok)}
.chip{display:inline-block;background:#eeece4;border-radius:10px;padding:1px 8px;margin:1px}.evidence{white-space:pre-wrap;word-break:break-word;
background:#f0efe9;border-radius:5px;padding:6px}table{width:100%;border-collapse:collapse}th,td{border-bottom:1px solid var(--line);
padding:5px;text-align:left;vertical-align:top}details summary{cursor:pointer;color:var(--accent)}a{color:var(--accent)}
"""


def render_quality_html(result: QualityResult) -> str:
    e = lambda value: html.escape("" if value is None else str(value))
    doc = result.summary["document"]
    by_category: dict[str, list[Any]] = defaultdict(list)
    for finding in result.findings:
        by_category[finding.category].append(finding)
    out = [
        ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Document Quality Review</title><style>{CSS}</style>"
         "</head><body><main>"),
        "<h1>Document Quality &amp; Validation Report</h1>",
        f"<p class='muted'>Run <code>{e(result.run_id)}</code>. Local-only; all source-derived text is escaped.</p>",
        (f"<div class='card'>Inspected {doc['elements_inspected']} elements · findings {doc['finding_count']} · "
         f"Nass review states {doc['needs_review_elements']} · validation {e(validation_summary(result.validation))}. "
         "No OCR accuracy percentage is reported.</div>"),
        "<h2>Findings by severity and category</h2><div class='card'>",
        "".join(f"<span class='chip {e(key)}'>{e(key)}: {value}</span>"
                for key, value in sorted(doc["by_severity"].items())),
        "".join(f"<span class='chip'>{e(key)}: {value}</span>"
                for key, value in sorted(doc["by_category"].items())),
        "</div><h2>Prioritized review queue</h2>",
    ]
    for finding in result.findings:
        excerpts = []
        for source_id in finding.source_ids:
            node = result.nodes.get(source_id, {})
            if node.get("text"):
                excerpts.append(f"<b>{e(source_id)}</b>: <span dir='auto'>{e(_clip(node['text'], 500))}</span>")
        out += [
            (f"<article class='card' id='{e(finding.finding_id)}'><h3>{finding.priority_rank}. "
             f"<code>{e(finding.finding_id)}</code> — {e(finding.category)}/{e(finding.subtype)}</h3>"),
            (f"<p><span class='chip {e(finding.severity)}'>{e(finding.severity)}</span> "
             f"<span class='chip'>{e(finding.confidence)}</span> <span class='chip'>{e(finding.status)}</span> "
             f"pages {e(finding.page_numbers)}</p>"),
            f"<p>{e(finding.explanation)}</p>",
            "".join(f"<div class='evidence'>{value}</div>" for value in excerpts),
            (f"<details><summary>Evidence and provenance</summary><div class='evidence'>{e(finding.evidence)}</div>"
             f"<p>Method: {e(finding.method)} · relationship IDs {e(finding.relationship_ids)} · related IDs "
             f"{e(finding.related_ids)}</p><p>Limitations: {e(finding.limitations)}</p></details>"),
            f"<p><b>Review:</b> {e(finding.suggested_review_action)}</p></article>",
        ]
    out += ["<h2>Page-by-page summary</h2><div class='grid'>"]
    for page, scorecard in sorted(result.summary.get("pages", {}).items(), key=lambda item: int(item[0])):
        out.append(f"<div class='card'><h3>Page {e(page)}</h3><div class='evidence'>{e(scorecard)}</div></div>")
    out.append("</div><h2>Section and element-type scorecards</h2><div class='grid'>")
    for name, scorecard in {**result.summary.get("sections", {}),
                            **{f"type:{key}": value for key, value in result.summary.get("element_types", {}).items()}}.items():
        out.append(f"<div class='card'><h3>{e(name)}</h3><div class='evidence'>{e(scorecard)}</div></div>")
    out.append("</div>")
    headings = [
        ("Source extraction and skipped items", "source_extraction"),
        ("Table quality", "table_quality"), ("Figure and chart quality", "figure_chart_quality"),
        ("Numeric integrity", "numeric_integrity"), ("OCR corruption suspects", "ocr_corruption"),
        ("Duplicate and leaked content", "duplicate_leakage"), ("Structure quality", "structure_quality"),
        ("Relationship quality", "relationship_quality"),
        ("Cross-page continuity", "cross_page_continuity"),
    ]
    for title, category in headings:
        items = by_category.get(category, [])
        out.append(f"<h2>{e(title)}</h2><div class='card'><ul>" +
                   "".join(f"<li><a href='#{e(item.finding_id)}'>{e(item.subtype)}</a> — {e(item.explanation)}</li>"
                           for item in items) + ("" if items else "<li>None recorded.</li>") + "</ul></div>")
    out += ["<h2>Known limitations and validation</h2><div class='card'><ul>"]
    out.extend(f"<li class='{e(item['severity'])}'><code>{e(item['code'])}</code>: {e(item['message'])}</li>"
               for item in result.validation)
    out.append("</ul><div class='evidence'>" + e(result.summary.get("inspection_coverage")) + "</div></div>")
    out.append("</main></body></html>")
    return "".join(out)

