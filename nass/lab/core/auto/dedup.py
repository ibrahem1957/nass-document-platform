"""Provenance-aware de-duplication. Nothing is corrected; every removed span is recorded.

Rule 1 - margin text in a crop's OCR (the proven 60 px margin can catch neighbouring text):
  a line of the crop's OCR is removed ONLY if it matches (normalized, Arabic-aware) a caption or text
  block that lies OUTSIDE the visual's detected box (the box before the margin was added) and whose
  text is known. Match = the whole normalized line equals a normalized line of that block or the whole
  block, or (lines of >= MIN_SPAN_TOKENS tokens) is a contiguous token run inside the block.
  On digital/mixed pages the block text is the native text layer (native text is preferred, so the
  OCR copy is the one removed). On scanned pages the text of individual blocks is not known (only the
  full page is transcribed), so no line is removed and that is recorded.
Rule 2 - figure text in full-page OCR:
  a span is removed from the full-page OCR only if it is >= MIN_SPAN_TOKENS consecutive tokens that
  match a crop transcription of the SAME page. Each crop token can account for one page occurrence
  only (repeated rows and distinct numbers are kept). A span inside a line whose remaining part has
  >= MIN_SPAN_TOKENS other tokens is ambiguous (body and figure text mixed): it is kept and recorded.
"""

from __future__ import annotations

import re
import unicodedata

MIN_SPAN_TOKENS = 3
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_PUNCT = re.compile(r"[^\w%‰٪$€£¥﷼.\-]", re.UNICODE)
_TRANSLATE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ـ": "",
                            "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7",
                            "٨": "8", "٩": "9", "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5",
                            "۶": "6", "۷": "7", "۸": "8", "۹": "9", "٫": ".", "٬": ",", "−": "-"})


def normalize_token(token: str) -> str:
    """Comparison form only (never written to the document)."""
    text = unicodedata.normalize("NFKC", token).translate(_TRANSLATE)
    text = _TASHKEEL.sub("", text).lower()
    text = _PUNCT.sub("", text).strip(".-")
    return text


def tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, normalized) of every token with a non-empty comparison form."""
    out = []
    for match in re.finditer(r"\S+", text or ""):
        norm = normalize_token(match.group())
        if norm:
            out.append((match.start(), match.end(), norm))
    return out


def normalize(text: str) -> str:
    return " ".join(token for _s, _e, token in tokens(text))


def _contains_run(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def remove_margin_lines(crop_text: str, outside_blocks: list[dict], *, min_tokens: int = MIN_SPAN_TOKENS,
                        reason: str = "margin text matching a block outside the visual's detected box") -> dict:
    """Rule 1. `outside_blocks`: [{"id", "text", "bbox_pt", "source"}] with KNOWN text."""
    prepared = []
    for block in outside_blocks:
        text = block.get("text") or ""
        if not text.strip():
            continue
        prepared.append((block, normalize(text).split(), {normalize(line) for line in text.splitlines() if normalize(line)}))
    kept_lines, removed = [], []
    for number, line in enumerate((crop_text or "").splitlines()):
        norm = normalize(line).split()
        match = None
        if norm and prepared:
            joined = " ".join(norm)
            for block, block_tokens, block_lines in prepared:
                if joined in block_lines or norm == block_tokens:
                    match = (block, "whole line")
                    break
                if len(norm) >= min_tokens and _contains_run(block_tokens, norm):
                    match = (block, f"contiguous run of {len(norm)} tokens")
                    break
        if match is None:
            kept_lines.append(line)
            continue
        block, how = match
        removed.append({"rule": "margin_line", "line_number": number + 1, "text": line, "reason": reason,
                        "match": how, "matched_block": block.get("id"), "matched_block_source": block.get("source"),
                        "matched_block_bbox_pt": block.get("bbox_pt")})
    return {"text": "\n".join(kept_lines), "removed": removed,
            "checked_blocks": [b.get("id") for b, _t, _l in prepared]}


def remove_figure_spans(page_text: str, crops: dict[str, str], *, min_tokens: int = MIN_SPAN_TOKENS) -> dict:
    """Rule 2. `crops`: {region_id: transcription} of the same page."""
    page = tokens(page_text)
    norms = [t[2] for t in page]
    crop_tokens = {rid: [t[2] for t in tokens(text)] for rid, text in crops.items() if (text or "").strip()}
    index: dict[tuple, list[tuple[str, int]]] = {}
    for rid, seq in crop_tokens.items():
        for i in range(len(seq) - min_tokens + 1):
            index.setdefault(tuple(seq[i:i + min_tokens]), []).append((rid, i))
    used: dict[str, set[int]] = {rid: set() for rid in crop_tokens}
    spans = []
    i = 0
    while i <= len(norms) - min_tokens:
        best = None
        for rid, start in index.get(tuple(norms[i:i + min_tokens]), []):
            if any(p in used[rid] for p in range(start, start + min_tokens)):
                continue
            seq, length = crop_tokens[rid], min_tokens
            while (i + length < len(norms) and start + length < len(seq) and norms[i + length] == seq[start + length]
                   and start + length not in used[rid]):
                length += 1
            if best is None or length > best[2]:
                best = (rid, start, length)
        if best is None:
            i += 1
            continue
        rid, start, length = best
        used[rid].update(range(start, start + length))
        spans.append({"page_tokens": (i, i + length), "region_id": rid, "crop_tokens": (start, start + length)})
        i += length
    # Ambiguity check per line: other tokens on the same line(s) besides the span
    line_of = [page_text.count("\n", 0, t[0]) for t in page]
    removed, ambiguous, cuts = [], [], []
    for span in spans:
        a, b = span["page_tokens"]
        lines = set(line_of[a:b])
        others = sum(1 for k, line in enumerate(line_of) if line in lines and not a <= k < b)
        start_char, end_char = page[a][0], page[b - 1][1]
        record = {"rule": "figure_span", "text": page_text[start_char:end_char], "tokens": b - a,
                  "matched_region": span["region_id"], "char_range": [start_char, end_char]}
        if others >= min_tokens:
            ambiguous.append(record | {"kept": True, "reason": f"line also has {others} other token(s): body and "
                                                                "figure text mixed, kept (ambiguous)"})
            continue
        removed.append(record | {"reason": f"{b - a} consecutive tokens match the transcription of "
                                           f"{span['region_id']} on the same page"})
        cuts.append((start_char, end_char))
    # Rebuild line by line: a line touched by a removal is dropped when nothing comparable remains on it
    # (only whitespace or punctuation such as table pipes); untouched lines, incl. blank ones, stay as they are.
    lines, offset = [], 0
    for line in page_text.split("\n"):
        start, end = offset, offset + len(line)
        offset = end + 1
        touched = [(max(a, start), min(b, end)) for a, b in cuts if a < end and b > start]
        if not touched:
            lines.append(line)
            continue
        kept, last = [], start
        for a, b in sorted(touched):
            kept.append(page_text[last:a])
            last = b
        kept.append(page_text[last:end])
        remainder = "".join(kept)
        if tokens(remainder):
            lines.append(remainder.rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return {"text": text, "removed": removed, "ambiguous_kept": ambiguous}


def outside_blocks_for(region: dict, blocks: list[dict]) -> list[dict]:
    """Known-text blocks lying outside the region's detected box (before margin) but inside its crop.
    `blocks`: [{"id", "text", "bbox_pt", "source"}]; region: box_pt_before_margin + crop_bbox_pt."""
    inner = region["box_pt_before_margin"]
    crop = region.get("crop_bbox_pt") or inner
    chosen = []
    for block in blocks:
        box = block["bbox_pt"]
        inter_crop = _intersection(box, crop)
        if inter_crop <= 0:
            continue
        inside_detected = _intersection(box, inner) / max(_area(box), 1e-6)
        if inside_detected < 0.5:  # the block lies (mostly) outside the detected visual
            chosen.append(block)
    return chosen


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


# ------------------------------------------------------------- Rule 3 (diagnostic only): caption leakage

LEAKAGE_SCHEMA = 1
LEAKAGE_MAX_DETECTED_OVERLAP = 0.10   # a block mostly inside the detected visual box belongs to the visual itself
CAPTION_LABEL = re.compile(
    r"(?i)(?<![\w])(?:figure|fig\.|table|chart|الشكل|شكل|الجدول|جدول|المخطط|مخطط)\s*[0-9٠-٩۰-۹]+\s*[:：.\-–—]?")
_TITLE_LABELS = frozenset({"figure_title", "figure_caption", "table_caption", "chart_caption"})
# Text-like layout blocks whose text can also be caught by the margin (a caption may be detected as plain text).
_MARGIN_TEXT_LABELS = frozenset({"text", "paragraph_title", "doc_title", "content", "reference", "footnote",
                                 "vision_footnote", "abstract"})


def _px_box(box) -> list[float] | None:
    return [float(v) for v in box] if isinstance(box, (list, tuple)) and len(box) == 4 else None


def crop_caption_labels(text: str | None) -> list[dict]:
    """Every caption label ('Figure 3:', 'الشكل 3:') in a crop transcription, with offsets. Not attributed to
    any layout block: the OCR may reorder text, so a label's position in the string is not its position on the
    page."""
    if not text:
        return []
    return [{"text": m.group(0).strip(), "offsets": [m.start(), m.start() + len(m.group(0).rstrip())]}
            for m in CAPTION_LABEL.finditer(text)]


def suspected_margin_leakage(region: dict, selection: dict, crop_text: str | None) -> list[dict]:
    """Diagnostic only (the text is never changed): saved layout title/text blocks that lie outside the visual's
    detected box but inside its margin-expanded crop, unless the block is this visual's own unambiguous attached
    title (which is meant to be in the crop).

    Geometry uses the saved detection pixels (block `bbox_px`, region `bbox_px`, crop `pixel_box`). The crop text
    is only searched for caption labels, which are listed unattributed: a geometric overlap does not show that
    the transcription contains a block, and no block on a scanned page has its own known text."""
    from lab.core.hybrid import layout_regions as lr

    crop = _px_box(region.get("pixel_box"))
    detected = _px_box(region.get("bbox_px"))
    if crop is None or detected is None:
        return []
    others = [r for r in selection.get("regions") or []
              if r is not region and r.get("region_id") != region.get("region_id") and r.get("bbox_pt_detected")]
    own_attached = {a.get("block_index") for a in region.get("title_attachments") or [] if isinstance(a, dict)}
    labels_in_text = crop_caption_labels(crop_text)
    out = []
    for block in (selection.get("others") or []) + (selection.get("attached_titles") or []):
        label = block.get("label")
        is_title = label in _TITLE_LABELS
        if not is_title and label not in _MARGIN_TEXT_LABELS:
            continue
        box = _px_box(block.get("bbox_px"))
        if box is None or _area(box) <= 0:
            continue
        inside_crop = _intersection(box, crop) / _area(box)
        if inside_crop <= 0:
            continue  # entirely outside the crop
        if _intersection(box, detected) / _area(box) > LEAKAGE_MAX_DETECTED_OVERLAP:
            continue  # inside the visual itself: an internal title, label or legend, not margin text
        association = block.get("title_association") if isinstance(block.get("title_association"), dict) else {}
        index = block.get("block_index")
        competitors = []
        block_pt = block.get("bbox_pt_detected") or block.get("bbox_pt")
        if is_title and block_pt:
            tops, bottoms = lr._title_relations({"bbox_pt": list(block_pt)}, [
                {"bbox_pt": list(r["bbox_pt_detected"]), "block_index": r.get("block_index")} for r in others])
            competitors = sorted({t[0].get("block_index") for t in tops + bottoms})
        status_text = str(block.get("title_status") or "")
        if not is_title:
            reason, status = "text_block_in_crop_margin", "not_a_title_block"
        elif index in own_attached:
            if not competitors and association.get("status") in (None, "attached"):
                continue  # this visual's own, unambiguous attached title: intended to be in the crop
            reason, status = "attached_title_with_competing_visual", "attached_but_contested"
        elif status_text == "ambiguous" or association.get("status") == "ambiguous":
            reason, status = "ambiguous_title_in_crop_margin", "ambiguous"
        elif status_text.startswith("attached to block"):
            reason, status = "other_visual_title_in_crop_margin", "attached_to_other_visual"
        else:
            reason, status = "unattached_title_in_crop_margin", association.get("status") or "unattached"
        side = "above" if (box[1] + box[3]) / 2 < (detected[1] + detected[3]) / 2 else "below"
        rows = max(0.0, min(box[3], crop[3]) - max(box[1], crop[1]))
        out.append({
            "schema": LEAKAGE_SCHEMA, "reason_code": reason, "region_id": region.get("region_id"),
            "block_index": index, "block_label": label, "block_bbox_px": [round(v, 2) for v in box],
            "visual_detected_bbox_px": [round(v, 2) for v in detected],
            "crop_pixel_box": [round(v, 2) for v in crop], "margin_px": region.get("margin_px"), "units": "px",
            "block_share_inside_crop": round(inside_crop, 3), "block_rows_inside_crop_px": round(rows, 2),
            "block_height_px": round(box[3] - box[1], 2), "outside_detected_visual": True, "side": side,
            "association_status": status, "competing_visual_blocks": competitors,
            "saved_title_decision": block.get("title_decision"),
            "crop_caption_labels": labels_in_text, "label_attribution": "unattributed",
            "evidence_level": "geometric_risk_with_caption_label_in_crop_text" if labels_in_text
            else "geometric_risk_only",
            "confirmed": False,
            "limitations": [
                "no block on a scanned page has its own OCR; the block's text is not known",
                "geometric overlap does not show that the crop transcription contains the block",
                "caption labels found in the crop text are not attributed: the OCR may reorder text",
                "full-page OCR is not used as spatial evidence",
            ],
        })
    return out
