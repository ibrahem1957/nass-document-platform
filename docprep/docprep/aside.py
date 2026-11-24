"""Callouts and side boxes: short labels in a narrow text box beside other content (derived role aside_label).

Layout-derived and always marked uncertain. Pages that look like genuine multi-column layouts are skipped.
Linking a callout to the content it refers to is left to a later component.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from .config import AsideConfig, PatternConfig
from .model import Unit


@dataclass
class AsideInfo:
    kind: str            # label | text
    reason: str
    neighbour_id: str | None
    uncertain: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _width(b) -> float:
    return b[2] - b[0]


def _v_overlap(a, b) -> float:
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _h_gap(a, b) -> float:
    """Horizontal distance between two boxes (negative when they overlap horizontally)."""
    return max(a[0], b[0]) - min(a[2], b[2])


def body_column_width(units: list[Unit], cfg: AsideConfig) -> float | None:
    widths = [_width(u.bbox) for u in units if u.bbox and u.furniture is None and u.element_type not in ("figure", "table")
              and u.features.get("n_words", 0) >= cfg.body_min_words]
    return statistics.median(widths) if widths else None


def detect_asides(units: list[Unit], page_sizes: dict[int, tuple[float, float]], aside: AsideConfig,
                  patterns: PatternConfig) -> dict[str, AsideInfo]:
    found: dict[str, AsideInfo] = {}
    if not aside.enabled:
        return found
    body_width = body_column_width(units, aside)
    if not body_width:
        return found
    by_page: dict[int, list[Unit]] = {}
    for u in units:
        if u.bbox and u.furniture is None:
            by_page.setdefault(u.page, []).append(u)
    for page, items in by_page.items():
        size = page_sizes.get(page)
        if size is None:
            continue
        width = size[0]
        text_items = [u for u in items if u.element_type not in ("figure", "table") and u.clean]
        # Multi-column test against the page width (not the body column, which is itself narrow in a
        # two-column document): several column-width text blocks on both halves of the page.
        column_like = [u for u in text_items if _width(u.bbox) < 0.55 * width]
        left = sum(1 for u in column_like if (u.bbox[0] + u.bbox[2]) / 2 < width / 2)
        right = len(column_like) - left
        if left >= aside.multicolumn_min_units and right >= aside.multicolumn_min_units:
            continue  # genuine multi-column page: narrow blocks are ordinary columns
        narrow_text = [u for u in text_items if _width(u.bbox) < aside.max_width_share * body_width]
        for index, unit in enumerate(items):
            f = unit.features
            if unit not in narrow_text:
                continue
            label_like = f.get("ends_colon") or (f.get("n_words", 0) <= patterns.short_words
                                                 and not f.get("terminal_punctuation"))
            if not label_like or f.get("numbering"):
                continue
            neighbour = next((o for o in items if o is not unit
                              and _v_overlap(unit.bbox, o.bbox) >= aside.min_vertical_overlap
                              * max(1.0, unit.bbox[3] - unit.bbox[1])
                              and _h_gap(unit.bbox, o.bbox) >= aside.min_horizontal_gap), None)
            if neighbour is None:
                # The label may sit just above a narrow text column that runs beside a figure.
                below = [o for o in items[index + 1:index + 3] if o in narrow_text
                         and _h_gap(unit.bbox, o.bbox) < 0 and 0 <= o.bbox[1] - unit.bbox[3] <= 2 * (unit.bbox[3] - unit.bbox[1])]
                if below:
                    neighbour = next((o for o in items if o is not unit and o not in below
                                      and _v_overlap(below[0].bbox, o.bbox) > 0
                                      and _h_gap(below[0].bbox, o.bbox) >= aside.min_horizontal_gap), None)
            if neighbour is None:
                continue
            found[unit.unit_id] = AsideInfo("label", f"short label in a narrow box ({_width(unit.bbox):.0f}pt wide; "
                                                     f"body column {body_width:.0f}pt) beside {neighbour.unit_id}",
                                            neighbour.unit_id)
            # Following text in the same narrow column belongs to the side box.
            previous = unit
            for other in items[index + 1:]:
                if other not in narrow_text or _h_gap(previous.bbox, other.bbox) >= 0:
                    break
                if other.bbox[1] - previous.bbox[3] > 2 * max(1.0, previous.bbox[3] - previous.bbox[1]):
                    break
                found.setdefault(other.unit_id, AsideInfo("text", f"continues the side box of {unit.unit_id}",
                                                          neighbour.unit_id))
                previous = other
    return found
