"""structure_report.html: self-contained, local-only review page (no external scripts, fonts or images)."""

from __future__ import annotations

import html
from collections import Counter

from .pipeline import StructureResult
from .tree import LEVEL_NAMES

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--accent:#2f5d8a;--ok:#2e7d4f;
--warn:#a86500;--bad:#b3261e;--chip:#efeee9;--code:#f2f1ec;--promo:#e7f0fa}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--panel:#1f1f1d;--ink:#ecebe6;
--muted:#a3a29b;--line:#34332f;--accent:#8db8e0;--ok:#7cc79a;--warn:#e0a85a;--bad:#f08a80;--chip:#2a2a27;
--code:#262623;--promo:#1c2a38}}
:root[data-theme="dark"]{--bg:#161615;--panel:#1f1f1d;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;--accent:#8db8e0;
--ok:#7cc79a;--warn:#e0a85a;--bad:#f08a80;--chip:#2a2a27;--code:#262623;--promo:#1c2a38}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Segoe UI",Tahoma,sans-serif}
main{max-width:1400px;margin:0 auto;padding:20px 16px 80px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:18px;margin:30px 0 8px;padding-top:10px;border-top:2px solid var(--line)}h3{font-size:15px;margin:6px 0}
.muted{color:var(--muted)}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0}
.filters{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center}
.chip{display:inline-block;background:var(--chip);border-radius:10px;padding:0 8px;font-size:12px;margin:1px 2px}
.high{color:var(--ok)}.medium{color:var(--warn)}.low{color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}.txt{white-space:pre-wrap;word-break:break-word;background:var(--code);border-radius:4px;padding:4px 7px}
.sec{border-left:4px solid var(--accent)}.sec.review{border-left-color:var(--warn)}
.unit{border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin:5px 0;background:var(--panel)}
.unit.low{border-left:4px solid var(--bad)}.unit.promoted{background:var(--promo)}.unit.unassigned{border-left:4px solid var(--warn)}
.diff{background:var(--promo)}details>summary{cursor:pointer;color:var(--accent)}
.outline ul{margin:2px 0 2px 18px;padding:0}.outline li{margin:2px 0}
@media (max-width:700px){table{font-size:12px}}
"""

JS = """
function apply(){
  const conf=[...document.querySelectorAll('input[name=conf]:checked')].map(e=>e.value);
  const promo=document.getElementById('f-promoted').checked;
  const unas=document.getElementById('f-unassigned').checked;
  const page=document.getElementById('f-page').value;
  const route=document.getElementById('f-route').value;
  document.querySelectorAll('.item').forEach(el=>{
    const ok=conf.includes(el.dataset.conf||'none')&&(!promo||el.dataset.promoted==='1')
      &&(!unas||el.dataset.kind==='unassigned')&&(page==='all'||el.dataset.page===page)
      &&(route==='all'||el.dataset.route===route);
    el.hidden=!ok;});
}
document.addEventListener('change',e=>{if(e.target.closest('.filters'))apply();});
"""


def e(text) -> str:
    return html.escape("" if text is None else str(text))


def _conf(value: str | None) -> str:
    return f"<span class='{e(value or 'none')}'>{e(value or '—')}</span>"


def render_html(result: StructureResult) -> str:
    doc = result.inputs.document.get("document") or {}
    units = {u.unit_id: u for u in result.units}
    cands = {c.candidate_id: c for c in result.candidates}
    cand_by_unit = {uid: c for c in result.candidates for uid in c.unit_ids}
    assign = {a.unit_id: a for a in result.tree.assignments}
    sections = {s.section_id: s for s in result.tree.sections}
    routes = {n: str(p.get("route") or "UNKNOWN") for n, p in result.inputs.pages.items()}

    def item_attrs(kind: str, conf: str | None, promoted: bool, page: int, extra: str = "") -> str:
        return (f"class='item{(' ' + extra) if extra else ''}' data-kind='{kind}' data-conf='{e(conf or 'none')}' data-promoted='{1 if promoted else 0}' "
                f"data-page='{page}' data-route='{e(routes.get(page, 'UNKNOWN'))}'")

    def path(section_id: str | None) -> str:
        parts = []
        while section_id:
            s = sections[section_id]
            parts.append(s.section_id)
            section_id = s.parent
        return " › ".join(reversed(parts)) or "—"

    def signal_table(c) -> str:
        rows = "".join(f"<tr><td>{e(s.category)}</td><td>{e(s.name)}</td><td>{s.contribution:+.3f}</td>"
                       f"<td dir='auto'>{e(s.detail)}</td></tr>" for s in c.signals)
        return f"<table><tr><th>Category</th><th>Signal</th><th>Contribution</th><th>Detail</th></tr>{rows}</table>"

    def unit_block(uid: str) -> str:
        u, a = units[uid], assign.get(uid)
        conf = a.confidence if a else None
        cls = ["unit"] + (["low"] if conf == "low" else []) + (["unassigned"] if a and a.section_id is None else [])
        span = f" · span {e(u.span_id)} of {e(u.element_id)} [{u.span.start}:{u.span.end}] ({e(u.span.support)})" \
            if u.span else ""
        role = a.role if a else "?"
        diff = " · <b>role differs from Nass type</b>" if role not in (u.element_type, "paragraph") and not (
            role == "paragraph" and u.element_type in ("text", "page_text")) else ""
        reasons = "; ".join((a.reasons + a.supporting) if a else [])
        conflicts = "; ".join(a.conflicting) if a else ""
        return (f"<div {item_attrs('unassigned' if a and a.section_id is None else 'content', conf, False, u.page)}>"
                f"<div class='{' '.join(cls)}'><div class='muted'>p{u.page} · {e(u.element_id)}{span} · Nass "
                f"<code>{e(u.element_type)}</code> → role <code>{e(role)}</code>{diff} · assignment {_conf(conf)} · "
                f"Nass state <span class='chip'>{e(u.validation_state)}</span></div>"
                f"<div class='txt' dir='auto'>{e(u.text)}</div>"
                + (f"<div class='muted'>{e(reasons)}</div>" if reasons else "")
                + (f"<div class='low'>{e(conflicts)}</div>" if conflicts else "") + "</div></div>")

    out = [(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Document Structure Review</title><style>{CSS}</style>"
            f"</head><body><main>"),
           (f"<h1>Document Structure Review</h1><p class='muted'>{e(doc.get('display_name'))} · run "
            f"<code>{e(result.inputs.run_id)}</code> · {result.inputs.page_count} pages · local-only report; text is "
            f"quoted verbatim.</p>")]
    counts = Counter(c.status for c in result.candidates)
    errors = [f for f in result.findings if f["severity"] == "error"]
    out.append(f"<div class='card'>Headings promoted {counts.get('promoted', 0)}, rejected {counts.get('rejected', 0)}, "
               f"excluded {counts.get('excluded', 0)} · sections {len(result.tree.sections)} · lists "
               f"{len(result.tree.lists)} · unassigned {sum(1 for a in result.tree.assignments if a.section_id is None)}"
               f" · derived spans {sum(len(v) for v in result.spans.values())} · body font {e(result.body.size)} pt · "
               f"validation: {'<b class=low>' + str(len(errors)) + ' error(s)</b>' if errors else '<span class=high>no errors</span>'}</div>")
    pages = sorted(result.inputs.pages)
    out.append("<div class='card filters'><b>Filters</b>"
               + "".join(f"<label><input type='checkbox' name='conf' value='{v}' checked> {v}</label>"
                         for v in ("high", "medium", "low", "none"))
               + "<label><input type='checkbox' id='f-promoted'> promoted headings only</label>"
               + "<label><input type='checkbox' id='f-unassigned'> unassigned only</label>"
               + "<label>page <select id='f-page'><option value='all'>all</option>"
               + "".join(f"<option value='{p}'>{p}</option>" for p in pages) + "</select></label>"
               + "<label>page type <select id='f-route'><option value='all'>all</option>"
               + "".join(f"<option value='{r}'>{r.lower()}</option>" for r in sorted(set(routes.values())))
               + "</select></label></div>")

    # 1. Outline
    def outline_node(sid: str) -> str:
        s = sections[sid]
        c = cands[s.heading_candidate_id]
        kids = "".join(outline_node(k) for k in s.children)
        return (f"<li><a href='#{s.section_id}' dir='auto'>{e(s.title)}</a> <span class='muted'>· L{s.level} · p{s.page_start}"
                f"–{s.page_end} · detection {_conf(c.confidence)} · level {_conf(c.level_confidence)}"
                f"{' · ⚠ review' if c.review else ''}</span>{'<ul>' + kids + '</ul>' if kids else ''}</li>")
    roots = [s.section_id for s in result.tree.sections if s.parent is None]
    out.append("<h2>Outline</h2><div class='card outline'>" + (
        "<ul>" + "".join(outline_node(r) for r in roots) + "</ul>" if roots else "<p>No heading was promoted.</p>")
        + "</div>")

    # 2. Sections with their content
    out.append("<h2>Sections and assigned content</h2>")
    lists = {node.list_id: node for node in result.tree.lists}
    for s in result.tree.sections:
        c = cands[s.heading_candidate_id]
        head_unit = units[s.heading_unit_ids[0]]
        nass_types = sorted({units[u].element_type for u in s.heading_unit_ids})
        promoted_from = "" if nass_types == ["heading"] else f" · <b>promoted from Nass type {e(', '.join(nass_types))}</b>"
        out.append(f"<div {item_attrs('heading', c.confidence, True, head_unit.page)}><div id='{s.section_id}' "
                   f"class='card sec{' review' if c.review else ''}'><h3 dir='auto'>{e(s.title)}</h3>"
                   f"<div class='muted'>{s.section_id} · level {s.level} ({e(LEVEL_NAMES.get(s.level, 'level'))})"
                   f"{' · raw level ' + str(s.raw_level) if s.raw_level != s.level else ''} · "
                   f"parent {e(s.parent or '—')} · pages {s.page_start}–{s.page_end} · heading {e(', '.join(s.heading_unit_ids))}"
                   f"{promoted_from}</div><div>Detection {_conf(c.confidence)} (score {c.score:.2f}"
                   f"{', ambiguous' if c.ambiguous else ''}) · level {_conf(c.level_confidence)}"
                   f"{' (ambiguous)' if c.level_ambiguous else ''} · method heuristic</div>"
                   f"<details><summary>Evidence</summary>{signal_table(c)}"
                   + (f"<p class='low'>Conflicts: {e('; '.join(c.conflicts))}</p>" if c.conflicts else "")
                   + f"<p>Level basis: {e('; '.join(c.level_basis))}</p>"
                   + (f"<p class='low'>Level conflicts: {e('; '.join(c.level_conflicts))}</p>" if c.level_conflicts else "")
                   + (f"<p>Semantic check ({e(c.semantic.get('verifier'))}, score {e(c.semantic.get('score'))}): "
                      f"{e('; '.join(c.semantic.get('supporting', []) + c.semantic.get('conflicting', [])))} "
                      f"<span class='muted'>{e(c.semantic.get('limitations'))}</span></p>" if c.semantic else "")
                   + (f"<p>Flags: {e('; '.join(s.flags))}</p>" if s.flags else "")
                   + (f"<p>Level normalization: {e('; '.join(s.level_notes))}</p>" if s.level_notes else "")
                   + "</details></div></div>")
        members = [a.unit_id for a in result.tree.assignments if a.section_id == s.section_id and a.role != "heading"]
        shown_lists = set()
        for uid in members:
            a = assign[uid]
            if a.list_id and a.list_id not in shown_lists:
                shown_lists.add(a.list_id)
                out.append(f"<div class='muted' style='margin-left:12px'>list {e(a.list_id)} "
                           f"({len(lists[a.list_id].unit_ids)} items)</div>")
            out.append(f"<div style='margin-left:{12 + 12 * (1 if a.list_id else 0)}px'>{unit_block(uid)}</div>")

    # 3. Heading candidates
    out.append("<h2>Heading candidates</h2><div class='card'><table><tr><th>Candidate</th><th>Page</th><th>Text</th>"
               "<th>Nass type</th><th>Status</th><th>Score</th><th>Detection</th><th>Level</th><th>Reasons / main signals</th></tr>")
    for c in result.candidates:
        u = units[c.unit_ids[0]]
        why = "; ".join(c.exclusions) if c.exclusions else ", ".join(
            f"{s.name} {s.contribution:+.2f}" for s in sorted(c.signals, key=lambda s: -abs(s.contribution))[:5])
        out.append(f"<tr {item_attrs('candidate', c.confidence, c.status == 'promoted', c.page)}><td>{c.candidate_id}</td>"
                   f"<td>{c.page}</td><td dir='auto'>{e(c.text)}</td><td>{e(u.element_type)}</td>"
                   f"<td class='{'high' if c.status == 'promoted' else 'muted'}'>{c.status}{' (ambiguous)' if c.ambiguous else ''}</td>"
                   f"<td>{c.score:.2f}</td><td>{_conf(c.confidence)}</td><td>{e(c.level)}</td><td dir='auto'>{e(why)}</td></tr>")
    out.append("</table></div>")

    # 4. Unassigned
    out.append("<h2>Unassigned content</h2>")
    unassigned = [a.unit_id for a in result.tree.assignments if a.section_id is None]
    out.append("".join(unit_block(uid) for uid in unassigned) or "<p class='muted'>None.</p>")

    # 5. Derived spans of segmented elements
    out.append("<h2>Derived spans (segmented elements)</h2>")
    for element_id, span_list in result.spans.items():
        element = units[span_list[0].span_id].element
        rows = "".join(
            f"<tr {item_attrs('content', assign[s.span_id].confidence if s.span_id in assign else None, s.span_id in cand_by_unit and cand_by_unit[s.span_id].status == 'promoted', element.page)}>"
            f"<td>{e(s.span_id)}</td><td>{s.start}–{s.end}</td><td>{e(s.role_hint)}</td><td>{e(s.support)}</td>"
            f"<td>{e('; '.join(s.basis))}</td><td>{'yes' if s.uncertain else 'no'}</td>"
            f"<td>{e((s.geometry or {}).get('precision', 'none'))}</td><td>{e(path(assign[s.span_id].section_id) if s.span_id in assign else '—')}</td>"
            f"<td dir='auto' class='txt'>{e(s.text)}</td></tr>" for s in span_list)
        out.append(f"<div class='card'><b>{e(element_id)}</b> · page {element.page} · Nass {e(element.element_type)} · "
                   f"state {e(element.validation_state)}<div class='txt' dir='auto'>{e(element.text)}</div>"
                   f"<table><tr><th>Span</th><th>Offsets</th><th>Role hint</th><th>Support</th><th>Basis</th>"
                   f"<th>Uncertain</th><th>Geometry</th><th>Section</th><th>Text</th></tr>{rows}</table></div>")
    if not result.spans:
        out.append("<p class='muted'>No element was segmented.</p>")

    # 5b. Saved layout heading regions (scanned pages)
    out.append("<h2>Layout heading regions (saved PP-DocLayout)</h2>")
    if result.layout_diagnostics:
        rows = []
        for d in result.layout_diagnostics:
            conf = {"verified": "high", "approximate": "medium"}.get(d.association, "low")
            why = "; ".join(d.evidence) if d.association != "none" else (d.reason_unmatched or "")
            rows.append(f"<tr {item_attrs('candidate', d.match_confidence, d.became_heading, d.page)}>"
                        f"<td>{d.page}</td><td>{e(d.layout_label)}</td><td>{e(d.layout_score)}</td>"
                        f"<td>{e([round(v) for v in d.layout_bbox_pt])}</td>"
                        f"<td class='{conf}'>{e(d.association)}</td><td>{_conf(d.match_confidence)}</td>"
                        f"<td>{e(d.element_id or '—')}<br>{e(d.span_id or '')}</td>"
                        f"<td dir='auto' class='txt'>{e(d.extracted_text or '—')}</td>"
                        f"<td>{'<b class=high>' + e(d.section_id) + '</b>' if d.became_heading else 'no'}</td>"
                        f"<td dir='auto'>{e(why)}</td></tr>")
        out.append("<div class='card'><p class='muted'>verified = text and layout linked one to one; approximate = "
                   "matched by order/role inside a multi-block segment; none = no reliable link (never treated as a "
                   "heading). No coordinates are invented for OCR text.</p><table><tr><th>Page</th><th>Label</th>"
                   "<th>Score</th><th>Box (pt)</th><th>Association</th><th>Match</th><th>Element / span</th>"
                   "<th>Extracted text</th><th>Section heading?</th><th>Evidence or reason</th></tr>"
                   + "".join(rows) + "</table></div>")
    else:
        out.append("<p class='muted'>No doc_title / paragraph_title regions in plan/layout.json.</p>")

    # 6. Nass order vs inferred structure
    out.append("<h2>Nass order vs inferred structure</h2><div class='card'><table><tr><th>Page</th><th>Element</th>"
               "<th>Nass type</th><th>Reading order</th><th>Unit</th><th>Detected role</th><th>Section path</th>"
               "<th>Assignment</th></tr>")
    for u in result.units:
        a = assign.get(u.unit_id)
        role = a.role if a else "?"
        changed = (role == "heading" and u.element_type != "heading") or (u.element_type == "heading" and role != "heading")
        out.append(f"<tr {item_attrs('unassigned' if a and a.section_id is None else 'content', a.confidence if a else None, role == 'heading', u.page, 'diff' if changed else '')}"
                   f"><td>{u.page}</td><td>{e(u.element_id)}</td><td>{e(u.element_type)}</td>"
                   f"<td>{e(u.element.reading_order)}</td><td>{e(u.span_id or '—')}</td><td>{e(role)}"
                   f"{' ◆' if changed else ''}</td><td>{e(path(a.section_id) if a else '—')}</td><td>{_conf(a.confidence if a else None)}</td></tr>")
    out.append("</table><p class='muted'>◆ = detected role differs from the Nass element type.</p></div>")

    # 7. Warnings and validation
    out.append("<h2>Warnings and validation</h2><div class='card'><ul>"
               + "".join(f"<li>{e(w)}</li>" for w in result.warnings)
               + "".join(f"<li class='{'low' if f['severity'] == 'error' else 'high'}'>{e(f['severity'])} "
                         f"<code>{e(f['code'])}</code>: {e(f['message'])}"
                         + (f" — {e(', '.join(f['items'][:10]))}" if f['items'] else "") + "</li>"
                         for f in result.findings)
               + "</ul></div>")
    out.append(f"<script>{JS}</script></main></body></html>")
    return "".join(out)
