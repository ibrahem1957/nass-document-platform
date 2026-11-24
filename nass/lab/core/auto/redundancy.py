"""Suppress review placeholders that are provably redundant with successfully processed regions.

A pending (not processed) region is suppressed from document.md and rag_chunks.jsonl ONLY when all hold:
- its status is `contained` or `container` (never `uncertain`, `small`, or anything else);
- the region relationship recorded by the layout rules names the covering region(s): a `contained`
  region's `contained_in` outer block, or a `container`'s `split into inner visuals` blocks;
- every covering region was processed with a completed, non-empty OCR result;
- geometry proves the coverage: >= COVERAGE of the pending region's area lies inside the covering
  regions' crops (sampled on a fixed grid, so overlapping crops are never counted twice);
- no other detected block (text, title, caption, other visual) lies mostly inside the pending region
  but outside the covering crops, i.e. no unique content is known to remain uncovered.
Suppressed regions stay in document.json (suppressed_redundant=true, covering ids, reason), keep their
review warning, and remain in the review queue with Process now. Anything not proven stays visible.
"""

from __future__ import annotations

REDUNDANCY_VERSION = "1"
COVERAGE = 0.98
GRID = 40                 # 40 x 40 sample points over the pending region
UNIQUE_INSIDE_SHARE = 0.5  # a block mostly (>= 50%) inside the pending region ...
COVERED_SHARE = 0.9        # ... is covered when >= 90% of it lies inside the covering crops


def _inside(point, box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def coverage(box: list[float], covers: list[list[float]]) -> float:
    """Share of `box` inside the union of `covers` (grid sampling at cell centres)."""
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0 or not covers:
        return 0.0
    hits = 0
    for i in range(GRID):
        for j in range(GRID):
            point = (x0 + (i + 0.5) * (x1 - x0) / GRID, y0 + (j + 0.5) * (y1 - y0) / GRID)
            hits += any(_inside(point, c) for c in covers)
    return hits / float(GRID * GRID)


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inter(a, b) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _relations(region: dict) -> list:
    """Block indices of the covering regions named by the layout containment decision."""
    for decision in region.get("decisions") or []:
        if decision.get("type") != "containment":
            continue
        if region.get("status") == "contained" and decision.get("contained_in") is not None:
            return [decision["contained_in"]]
        if region.get("status") == "container" and decision.get("decision") == "split into inner visuals":
            return list(decision.get("inner") or [])
    return []


def assess(region: dict, selection: dict, processed: dict[str, str]) -> dict:
    """Whether a pending region is redundant. `processed`: {region_id: non-empty OCR text}."""
    if region.get("status") not in {"contained", "container"}:
        return {"redundant": False, "reason": f"status {region.get('status')!r} is never suppressed"}
    related = _relations(region)
    if not related:
        return {"redundant": False, "reason": "no recorded containment relationship"}
    by_block = {r.get("block_index"): r for r in selection["regions"] if r.get("region_id")}
    covering = [by_block.get(index) for index in related]
    if any(c is None for c in covering):
        return {"redundant": False, "reason": f"covering block(s) {related} are not regions of this page"}
    missing = [c["region_id"] for c in covering if not (processed.get(c["region_id"]) or "").strip()]
    if missing:
        return {"redundant": False, "reason": f"covering region(s) {missing} have no completed OCR text"}
    box = region["box_pt_before_margin"]
    crops = [c.get("crop_bbox_pt") or c["box_pt_before_margin"] for c in covering]
    share = coverage(box, crops)
    if share < COVERAGE:
        return {"redundant": False, "reason": f"only {share:.0%} of the region lies inside the processed crops",
                "coverage": round(share, 3)}
    blocks = [b for b in (selection.get("others") or []) + (selection.get("attached_titles") or []) + selection["regions"]
              if b is not region and b.get("bbox_pt") and b.get("region_id") not in {c["region_id"] for c in covering}]
    unique = []
    for block in blocks:
        inside = _inter(block["bbox_pt"], box) / (_area(block["bbox_pt"]) or 1.0)
        if inside >= UNIQUE_INSIDE_SHARE and coverage(block["bbox_pt"], crops) < COVERED_SHARE:
            unique.append({"block_index": block.get("block_index"), "label": block.get("label")})
    if unique:
        return {"redundant": False, "reason": f"block(s) {unique} inside the region are not covered by the processed "
                                              "crops (possible unique content)", "coverage": round(share, 3)}
    ids = [c["region_id"] for c in covering]
    kind = "inside" if region["status"] == "contained" else "split into"
    return {"redundant": True, "covered_by": ids, "coverage": round(share, 3), "version": REDUNDANCY_VERSION,
            "reason": f"{region['status']} region {kind} {ids}: {share:.0%} of it lies inside their processed crops and "
                      "no other detected content remains outside them"}
