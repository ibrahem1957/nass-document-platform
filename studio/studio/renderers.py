from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class RenderError(ValueError):
    pass


def load_json(path: Path | str, expected_kind: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot load {Path(path).name}: {exc}") from exc
    if not isinstance(value, dict) or value.get("kind") != expected_kind:
        raise RenderError(f"{Path(path).name} has an incompatible kind")
    return value


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def nass_reading_view(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("kind") != "automatic_extraction_document":
        raise RenderError("incompatible Nass document")
    page_count = len(document.get("pages", []))
    elements = document.get("elements")
    if not isinstance(elements, list):
        raise RenderError("Nass document lacks elements")
    ordered = sorted(
        enumerate(elements),
        key=lambda pair: (pair[1].get("page", 0), pair[1].get("reading_order", 10**9), pair[0]),
    )
    pages = [{"page": page, "blocks": []} for page in range(1, page_count + 1)]
    seen: set[str] = set()
    omitted_composite_members = []
    for _position, element in ordered:
        element_id = element.get("id")
        if not isinstance(element_id, str) or element_id in seen:
            raise RenderError("Nass element IDs must be unique strings")
        seen.add(element_id)
        state = element.get("validation_state")
        if state == "included_in_composite":
            omitted_composite_members.append(element_id)
            continue
        page = element.get("page")
        if not isinstance(page, int) or not 1 <= page <= page_count:
            raise RenderError(f"Nass element {element_id} has an invalid page")
        pages[page - 1]["blocks"].append({
            "source_id": element_id,
            "type": element.get("type") or "unknown",
            "text": _text(element.get("content")),
            "validation_state": state,
            "requires_review": state not in (None, "ok"),
        })
    return {
        "kind": "nass_reading_view",
        "title": (document.get("document") or {}).get("display_name"),
        "page_count": page_count,
        "pages": pages,
        "source_element_count": len(elements),
        "displayed_element_count": sum(len(page["blocks"]) for page in pages),
        "omitted_composite_members": omitted_composite_members,
        "notice": "JSON is authoritative; this reading view suppresses only elements explicitly included in a composite.",
    }


def nass_markdown(document: dict[str, Any]) -> str:
    view = nass_reading_view(document)
    lines = ["# Nass — Extracted Content", "", (
        "> Faithful reading view derived from document.json. "
        "The JSON remains the authoritative artifact."
    ), ""]
    for page in view["pages"]:
        lines.extend((f"## Page {page['page']}", ""))
        for block in page["blocks"]:
            block_type = block["type"]
            if block_type in {"figure", "table"}:
                lines.extend((f"### Extracted {block_type}", ""))
            if block["requires_review"]:
                lines.extend((f"> Review required: {block['validation_state']}", ""))
            lines.extend((block["text"], ""))
        lines.extend(("---", ""))
    return "\n".join(lines).rstrip() + "\n"


def _list_block(record: dict[str, Any], elements: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    source_ids: set[str] = set()

    def convert(item: dict[str, Any]) -> dict[str, Any]:
        element_id = item.get("element_id")
        if isinstance(element_id, str):
            source_ids.add(element_id)
        text = _text(item.get("text"))
        if not text and element_id in elements:
            text = _text(elements[element_id].get("text"))
        return {
            "text": text, "page": item.get("page"), "marker": item.get("marker"),
            "requires_review": bool(item.get("uncertain")),
            "children": [convert(child) for child in item.get("children") or []],
        }

    nested = record.get("nested_items") or []
    return ({"kind": "list", "items": [convert(item) for item in nested],
             "pages": (record.get("structure") or {}).get("pages", []),
             "requires_review": bool(record.get("quality_finding_ids"))}, source_ids)


def prepared_reading_view(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("kind") != "docprep_prepared_document":
        raise RenderError("incompatible prepared document")
    elements_list = document.get("elements")
    sections_list = document.get("sections")
    if not isinstance(elements_list, list) or not isinstance(sections_list, list):
        raise RenderError("prepared document lacks elements or sections")
    elements = {item["element_id"]: item for item in elements_list}
    if len(elements) != len(elements_list):
        raise RenderError("prepared document contains duplicate element IDs")
    sections = {item["section_id"]: item for item in sections_list}
    order = {item["element_id"]: item.get("document_order", position)
             for position, item in enumerate(elements_list)}
    used: set[str] = set()

    logical_by_part: dict[str, dict[str, Any]] = {}
    logical_first: dict[str, dict[str, Any]] = {}
    for table in document.get("logical_tables", []):
        parts = [part for part in table.get("source_part_ids", []) if part in elements]
        if parts:
            first = min(parts, key=lambda item: order[item])
            logical_first[first] = table
            for part in parts:
                logical_by_part[part] = table

    lists_by_first: dict[str, tuple[dict[str, Any], set[str]]] = {}
    list_member_ids: set[str] = set()
    for record in document.get("lists", []):
        block, members = _list_block(record, elements)
        members &= elements.keys()
        if members:
            first = min(members, key=lambda item: order[item])
            lists_by_first[first] = (block, members)
            list_member_ids |= members

    visuals = {item.get("visual_id"): item for item in document.get("visuals", [])}

    def ordinary(element_id: str) -> dict[str, Any]:
        element = elements[element_id]
        visual = visuals.get(element_id)
        review = element.get("validation_state") not in (None, "ok") or bool(element.get("quality_finding_ids"))
        return {
            "kind": "visual" if visual else "content", "source_type": element.get("original_type"),
            "text": _text(element.get("text")), "page": element.get("page"),
            "requires_review": review, "visual_type": visual.get("visual_type") if visual else None,
            "relationship_confidence": "uncertain" if visual and visual.get("review_flags") else None,
        }

    def blocks_for(ids: Iterable[str], heading_ids: set[str]) -> list[dict[str, Any]]:
        blocks = []
        for element_id in ids:
            if element_id not in elements or element_id in used:
                continue
            if element_id in heading_ids:
                used.add(element_id)
                continue
            if element_id in logical_by_part:
                table = logical_by_part[element_id]
                if element_id not in logical_first:
                    continue
                parts = []
                for part_id in table["source_part_ids"]:
                    if part_id in elements:
                        part = elements[part_id]
                        parts.append({"text": _text(part.get("text")), "page": part.get("page"),
                                      "requires_review": bool(part.get("quality_finding_ids"))})
                        used.add(part_id)
                blocks.append({"kind": "logical_table", "parts": parts, "pages": table.get("pages", []),
                               "requires_review": bool(table.get("quality_finding_ids"))})
                continue
            if element_id in list_member_ids:
                if element_id not in lists_by_first:
                    continue
                block, members = lists_by_first[element_id]
                blocks.append(block)
                used.update(members)
                continue
            blocks.append(ordinary(element_id))
            used.add(element_id)
        return blocks

    visited_sections: set[str] = set()

    def build_section(section_id: str) -> dict[str, Any]:
        if section_id in visited_sections:
            raise RenderError("prepared section hierarchy contains a cycle")
        visited_sections.add(section_id)
        section = sections[section_id]
        block = {
            "kind": "section", "title": _text(section.get("title")), "level": section.get("level"),
            "pages": [section.get("page_start"), section.get("page_end")],
            "requires_review": bool(section.get("review_recommended") or section.get("quality_finding_ids")),
            "blocks": blocks_for(section.get("element_ids_in_reading_order", []),
                                 set(section.get("heading_element_ids", []))),
            "children": [],
        }
        for child_id in section.get("children", []):
            if child_id in sections:
                block["children"].append(build_section(child_id))
        return block

    roots = [item for item in sections_list if item.get("parent") not in sections]
    section_blocks = [build_section(item["section_id"]) for item in roots]
    for item in sections_list:
        if item["section_id"] not in visited_sections:
            section_blocks.append(build_section(item["section_id"]))
    remaining = [item["element_id"] for item in sorted(elements_list, key=lambda item: item.get("document_order", 0))
                 if item["element_id"] not in used]
    remaining_blocks = blocks_for(remaining, set())
    return {
        "kind": "prepared_reading_view", "title": document.get("metadata", {}).get("title"),
        "page_count": document.get("page_count"), "sections": section_blocks,
        "remaining_blocks": remaining_blocks, "source_element_count": len(elements_list),
        "represented_source_count": len(used),
        "notice": "Organization is derived only from prepared_document.json; uncertain evidence remains marked.",
    }


def _markdown_list(items: list[dict[str, Any]], depth: int, lines: list[str]) -> None:
    for item in items:
        marker = "  " * depth + "- "
        text = item["text"].replace("\n", "\n" + "  " * (depth + 1))
        lines.append(marker + text)
        _markdown_list(item.get("children", []), depth + 1, lines)


def prepared_markdown(document: dict[str, Any]) -> str:
    view = prepared_reading_view(document)
    lines = ["# DocPrep — Organized Document", "", (
        "> Readable organization derived from prepared_document.json. "
        "Original source strings are retained; the JSON remains authoritative."
    ), ""]

    def render_blocks(blocks: list[dict[str, Any]]) -> None:
        for block in blocks:
            if block["kind"] == "list":
                _markdown_list(block["items"], 0, lines)
                lines.append("")
            elif block["kind"] == "logical_table":
                lines.extend(("**Logical multi-page table**", ""))
                for part in block["parts"]:
                    lines.extend((f"*Source page {part['page']}*", "", part["text"], ""))
            else:
                if block["kind"] == "visual":
                    lines.extend((f"**Extracted {block.get('visual_type') or 'visual'}**", ""))
                if block.get("requires_review"):
                    lines.extend(("> Requires review", ""))
                lines.extend((block["text"], ""))

    def render_section(section: dict[str, Any]) -> None:
        level = max(2, min(6, int(section.get("level") or 0) + 2))
        lines.extend(("#" * level + " " + section["title"], ""))
        render_blocks(section["blocks"])
        for child in section["children"]:
            render_section(child)

    for section in view["sections"]:
        render_section(section)
    if view["remaining_blocks"]:
        lines.extend(("## Content without a confirmed section", ""))
        render_blocks(view["remaining_blocks"])
    return "\n".join(lines).rstrip() + "\n"
