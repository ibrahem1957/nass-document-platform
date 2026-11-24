"""DocPrep consumption of crop-margin leakage evidence (Nass dedup Rule 3) and its derivation for older runs.
All content is synthetic."""

from __future__ import annotations

from .helpers import el
from .test_relations import _build

TABLE_PX = [300, 100, 1400, 880]
FORM_PX = [300, 965, 1400, 1530]
TITLE_PX = [700, 901, 1000, 931]
SCALE = 1700 / 612  # px per pt on a 612 pt wide page rendered at 1700 px


def _pt(box):
    return [round(v / SCALE, 2) for v in box]


def _layout(title_attached_to_form: bool = True):
    title = {"block_index": 1, "label": "figure_title", "bbox_px": TITLE_PX, "bbox_pt": _pt(TITLE_PX),
             "title_status": "attached to block 4 (above)" if title_attached_to_form else "ambiguous"}
    form = {"region_id": "p1_R1", "job_id": "p001_R1", "block_index": 4, "label": "image", "bbox_px": FORM_PX,
            "bbox_pt": _pt(FORM_PX), "bbox_pt_without_title": _pt(FORM_PX),
            "title_attachments": [{"block_index": 1, "position": "above"}] if title_attached_to_form else []}
    table = {"region_id": "p1_R0", "job_id": "p001_R0", "block_index": 0, "label": "table", "bbox_px": TABLE_PX,
             "bbox_pt": _pt(TABLE_PX)}
    selection = {"regions": [table, form], "others": [] if title_attached_to_form else [title],
                 "attached_titles": [title] if title_attached_to_form else []}
    return {"pages": [{"page": 1, "boxes": [], "selection": selection}]}


def _elements(form_text: str, **form_extra):
    crop = {"pixel_box": [FORM_PX[0] - 60, FORM_PX[1] - 60, FORM_PX[2] + 60, FORM_PX[3] + 60], "margin_px": 60}
    return [
        el("page", 1, 0, "Maintenance Log Figure 13: Maintenance Log", type="page_text", source="qari_full_page",
           bbox=[0, 0, 612, 792]),
        el("table", 1, 1, "Maintenance Log bus date cost BUS-1042 4,850", type="table", bbox=_pt(TABLE_PX),
           provenance={"job_id": "p001_R0"}, source="qari_crop"),
        el("form", 1, 2, form_text, type="figure", bbox=_pt(FORM_PX), source="qari_crop", crop=crop,
           provenance={"job_id": "p001_R1", "title_attachments": [{"block_index": 1, "position": "above",
                                                                  "bbox_pt": _pt(TITLE_PX)}]}, **form_extra),
    ]


SPANS = {"page": [{"span_id": "page_span_001", "element_id": "page", "start": 0, "end": 15,
                   "text": "Maintenance Log", "role_hint": "paragraph", "uncertain": False, "geometry": None}]}
FORM_TEXT = "Maintenance Log Monthly subscription request Name Card number Send Cancel"


def test_contested_attachment_is_not_title_evidence_and_risk_is_reported_for_old_runs(tmp_path):
    elements = _elements(FORM_TEXT)
    result, _, _ = _build(tmp_path, elements, derived_spans=SPANS, layout=_layout(),
                          sections={item["id"]: "s1" for item in elements})
    form = next(item for item in result.visuals if item["visual_id"] == "form")
    assert form["internal_title"] is None  # leaked neighbouring caption words are not this visual's title
    risk = form["margin_leakage_risk"]
    assert risk["source"] == "derived_by_docprep_from_saved_layout" and risk["review_recommended"]
    record = risk["records"][0]
    assert record["reason_code"] == "attached_title_with_competing_visual"
    assert record["competing_visual_ids"] == ["table"] and record["confirmed"] is False
    assert record["evidence_level"] == "geometric_risk_only"  # no caption label in this crop text
    assert any(item["code"] == "possible_caption_margin_leakage" for item in result.findings)
    assert result.nodes["form"].text == FORM_TEXT
    assert any(item.get("reason", "").startswith("saved title attachment is contested")
               for item in result.evidence_audit)


def test_caption_label_in_crop_text_is_listed_unattributed(tmp_path):
    text = "Figure 13: Maintenance Log Monthly subscription request Name"
    elements = _elements(text)
    result, _, _ = _build(tmp_path, elements, derived_spans=SPANS, layout=_layout(),
                          sections={item["id"]: "s1" for item in elements})
    record = next(item for item in result.visuals if item["visual_id"] == "form")["margin_leakage_risk"]["records"][0]
    assert record["evidence_level"] == "geometric_risk_with_caption_label_in_crop_text"
    assert record["crop_caption_labels"][0]["text"] == "Figure 13:" and record["label_attribution"] == "unattributed"
    assert not [r for r in result.relationships if r.type == "caption_of" and r.target_id == "form"]
    assert result.nodes["form"].text == text


def test_nass_field_is_used_when_present(tmp_path):
    nass = [{"schema": 1, "reason_code": "ambiguous_title_in_crop_margin", "block_index": 1, "confirmed": False}]
    elements = _elements(FORM_TEXT, dedup={"suspected_margin_leakage": nass})
    result, _, _ = _build(tmp_path, elements, layout=_layout(title_attached_to_form=False),
                          sections={item["id"]: "s1" for item in elements})
    risk = next(item for item in result.visuals if item["visual_id"] == "form")["margin_leakage_risk"]
    assert risk["source"] == "nass_dedup" and risk["records"] == nass and risk["review_recommended"]


def test_run_without_layout_or_crop_metadata_reports_nothing(tmp_path):
    elements = [el("fig", 1, 0, "Figure 2: Revenue 12 Cost 8", type="figure", bbox=[100, 100, 300, 300])]
    result, _, _ = _build(tmp_path, elements, sections={"fig": "s1"})
    assert "margin_leakage_risk" not in result.visuals[0]
    assert not any(item["code"] == "possible_caption_margin_leakage" for item in result.findings)
