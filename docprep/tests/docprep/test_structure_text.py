"""End-to-end structure on synthetic runs without a source PDF (text, geometry and Nass types only)."""

from __future__ import annotations

from docprep.pipeline import assignments_json, build_structure, section_tree_json

from .helpers import el, make_config, write_run

BODY_A = "This report describes the operations of the fictional network in detail."


def _by_unit(result):
    return {a.unit_id: a for a in result.tree.assignments}


def _cand(result, unit_id):
    return next(c for c in result.candidates if unit_id in c.unit_ids)


def _section(result, title):
    return next(s for s in result.tree.sections if s.title == title)


def furniture_run(tmp_path):
    elements = [
        el("hdr1", 1, 0, "Northwind Report 2025", bbox=[72, 40, 300, 55]),
        el("b0", 1, 1, "Opening remarks that precede any heading in this synthetic document.", bbox=[72, 90, 540, 104]),
        el("h1", 1, 2, "Introduction", type="heading", bbox=[72, 120, 200, 136]),
        el("b1", 1, 3, BODY_A, bbox=[72, 140, 540, 154]),
        el("h2", 1, 4, "Network Changes", type="heading", bbox=[72, 700, 250, 716]),
        el("ftr1", 1, 5, "Page 1", bbox=[280, 760, 330, 772]),
        el("hdr2", 2, 0, "Northwind Report 2025", bbox=[72, 40, 300, 55]),
        el("b2", 2, 1, "Two new lines opened during the year and several stations were renovated.",
           bbox=[72, 90, 540, 104], state="needs_review"),
        el("b3", 2, 2, "Further details follow in the annex of this synthetic document.", bbox=[72, 110, 540, 124]),
        el("ftr2", 2, 3, "Page 2", bbox=[280, 760, 330, 772]),
        el("hdr3", 3, 0, "Northwind Report 2025", bbox=[72, 40, 300, 55]),
        el("b4", 3, 1, "The renovation work continued until the end of the year.", bbox=[72, 90, 540, 104]),
        el("ftr3", 3, 2, "Page 3", bbox=[280, 760, 330, 772]),
    ]
    return write_run(tmp_path, "run_furniture", elements, page_count=3)


def test_headers_footers_cross_page_and_text_before_first_heading(tmp_path):
    result = build_structure(furniture_run(tmp_path), make_config())
    a = _by_unit(result)
    assert [s.title for s in result.tree.sections] == ["Introduction", "Network Changes"]
    for uid in ("hdr1", "hdr2", "hdr3"):
        assert a[uid].role == "header" and a[uid].section_id is None
    for uid in ("ftr1", "ftr2", "ftr3"):
        assert a[uid].role == "page_number" and a[uid].section_id is None
    assert a["b0"].section_id is None and "precedes the first detected heading" in a["b0"].reasons
    changes = _section(result, "Network Changes")
    assert "heading at the bottom of page 1" in changes.flags
    assert a["b2"].section_id == changes.section_id
    assert any("bottom of page 1" in s for s in a["b2"].supporting)
    assert a["b4"].section_id == changes.section_id and any("continues from page 2" in s for s in a["b4"].supporting)
    assert (changes.page_start, changes.page_end) == (1, 3)
    assert not [f for f in result.findings if f["severity"] == "error"]


def test_needs_review_is_preserved(tmp_path):
    result = build_structure(furniture_run(tmp_path), make_config())
    b2 = _by_unit(result)["b2"]
    assert b2.validation_state == "needs_review"
    assert any("needs_review" in r for r in b2.reasons)
    inventory = {e["element_id"]: e for e in assignments_json(result)["source_inventory"]}
    assert inventory["b2"]["validation_state"] == "needs_review"


def test_missing_pdf_and_layout_are_reported(tmp_path):
    result = build_structure(furniture_run(tmp_path), make_config())
    assert any("source.pdf is missing" in w for w in result.warnings)
    assert any("plan/layout.json is missing" in w for w in result.warnings)
    assert any(s.name == "typography_unavailable" for s in _cand(result, "h1").signals)


def test_ambiguous_hierarchy_without_typography(tmp_path):
    result = build_structure(furniture_run(tmp_path), make_config())
    for s in result.tree.sections:
        c = _cand(result, s.heading_unit_ids[0])
        assert c.level == 1 and c.level_confidence == "low" and c.level_ambiguous and c.review


def test_captions_lists_and_incorrect_source_types(tmp_path):
    long_heading = ("This heading element is actually a long sentence that continues well beyond any reasonable "
                    "heading length for a document in every respect.")
    elements = [
        el("h", 1, 0, "Results", type="heading", bbox=[72, 80, 160, 96]),
        el("intro", 1, 1, "The following items summarise the results of the synthetic survey in this document, "
                          "including:", bbox=[72, 100, 540, 114]),
        el("li1", 1, 2, "• First finding of the survey.", bbox=[90, 118, 400, 130]),
        el("li2", 1, 3, "• Second finding of the survey.", bbox=[90, 134, 400, 146]),
        el("li3", 1, 4, "• Third finding of the survey.", bbox=[90, 150, 400, 162]),
        el("fig", 1, 5, "chart text 10 20 30", type="figure", bbox=[100, 170, 500, 400], source="qari_crop"),
        el("cap", 1, 6, "Figure 3: Monthly ridership by line", type="heading", bbox=[150, 404, 450, 418]),
        el("label", 1, 7, "• Data description:", bbox=[90, 430, 300, 442]),
        el("li4", 1, 8, "• Date of each trip.", bbox=[110, 446, 400, 458]),
        el("li5", 1, 9, "• Line and station.", bbox=[110, 462, 400, 474]),
        el("wrong", 1, 10, long_heading, type="heading", bbox=[72, 490, 540, 520]),
    ]
    result = build_structure(write_run(tmp_path, "run_lists", elements), make_config())
    a = _by_unit(result)
    results = _section(result, "Results")
    intro = _cand(result, "intro")
    assert intro.status == "excluded" and any("ordinary sentence ending with a colon" in r for r in intro.exclusions)
    cap = _cand(result, "cap")
    assert cap.status == "excluded" and any("caption" in r for r in cap.exclusions)
    assert any("overrides the Nass element type" in r for r in cap.exclusions)
    assert a["cap"].role == "caption" and a["cap"].section_id == results.section_id
    assert a["fig"].role == "figure" and a["fig"].section_id == results.section_id
    first_list = result.tree.lists[0]
    assert first_list.unit_ids == ["li1", "li2", "li3"] and first_list.section_id == results.section_id
    assert all(a[u].list_id == first_list.list_id for u in ("li1", "li2", "li3"))
    wrong = _cand(result, "wrong")
    assert wrong.status == "excluded" and any("too long" in r for r in wrong.exclusions)
    assert a["wrong"].role == "paragraph" and a["wrong"].section_id == results.section_id
    assert len(result.tree.sections) == 1


def test_arabic_rtl_numbered_headings(tmp_path):
    elements = [
        el("h1", 1, 0, "١) مؤشرات التحليل الزمني وخطورة الحوادث:", bbox=[300, 100, 540, 118]),
        el("p1", 1, 1, "تتناول هذه الفقرة توزيع الحوادث حسب ساعات اليوم وأيام الأسبوع في البيانات التجريبية.",
           bbox=[80, 125, 540, 140]),
        el("h2", 1, 2, "٢) مؤشرات الموقع:", bbox=[380, 160, 540, 178]),
        el("p2", 1, 3, "يعرض هذا القسم توزيع الحوادث حسب نوع الطريق في البيانات التجريبية.", bbox=[90, 185, 540, 200]),
    ]
    result = build_structure(write_run(tmp_path, "run_rtl", elements, directions={1: "rtl"}), make_config())
    titles = [s.title for s in result.tree.sections]
    assert titles == ["١) مؤشرات التحليل الزمني وخطورة الحوادث:", "٢) مؤشرات الموقع:"]  # verbatim
    a = _by_unit(result)
    assert a["p1"].section_id == result.tree.sections[0].section_id
    assert a["p2"].section_id == result.tree.sections[1].section_id
    unit = next(u for u in result.units if u.unit_id == "h1")
    assert unit.features["geometry"]["alignment"] == "right" and unit.features["geometry"]["start_aligned"]
    assert unit.features["numbering"]["label"] == "1"


def test_colon_label_alone_is_not_enough_but_numbering_helps(tmp_path):
    elements = [
        el("b0", 1, 0, BODY_A, bbox=[72, 80, 540, 94]),
        el("label", 1, 1, "Data description:", bbox=[72, 110, 200, 124]),
        el("b1", 1, 2, "Each record lists the date, the line, the station and the number of passengers counted.",
           bbox=[72, 128, 540, 142]),
        el("num", 1, 3, "3) Data cleaning:", bbox=[72, 160, 200, 174]),
        el("b2", 1, 4, "Missing values were removed and duplicate records were merged before any analysis.",
           bbox=[72, 178, 540, 192]),
    ]
    result = build_structure(write_run(tmp_path, "run_colon", elements), make_config())
    label = _cand(result, "label")
    assert label.status == "rejected" and label.ambiguous and label.review  # no rigid "short + colon" rule
    assert _cand(result, "num").status == "promoted"


def test_scanned_page_text_is_segmented_and_accounted_once(tmp_path):
    text = ("Network Performance Overview (synthetic) Network Performance Overview - synthetic data for testing. "
            "This chapter reviews volumes. Figure 1: Daily trips per line Figure 2: Waiting time")
    elements = [el("p1_e0", 1, 0, text, type="page_text", bbox=[0, 0, 612, 792], source="qari_full_page")]
    run = write_run(tmp_path, "run_scanned", elements, routes={1: "SCANNED"})
    result = build_structure(run, make_config())
    spans = result.spans["p1_e0"]
    assert len(spans) == 5 and "".join(s.text for s in spans) == text
    inventory = assignments_json(result)["source_inventory"]
    assert [e["element_id"] for e in inventory] == ["p1_e0"]
    assert inventory[0]["represented_by"] == [s.span_id for s in spans] and inventory[0]["text"] == text
    title = result.tree.sections[0]
    assert title.heading_unit_ids == ["p1_e0_span_001"] and title.level == 0
    a = _by_unit(result)
    assert all(a[s.span_id].section_id == title.section_id for s in spans[1:])
    assert a["p1_e0_span_004"].role == "caption"
    assert all(u.typography is None and u.bbox is None for u in result.units)  # no invented coordinates
    codes = {f["code"]: f["severity"] for f in result.findings}
    assert codes["source_inventory"] == "ok" and codes["span_coverage"] == "ok"


def test_scanned_visual_or_table_title_is_not_invented_as_a_document_section(tmp_path):
    text = (
        "Synthetic Operations Review. This page introduces a maintenance table and a dashboard. "
        "Weekly Route Performance Figure 4: Route Performance"
    )
    element = el("page", 1, 0, text, type="page_text", bbox=[0, 0, 612, 792], source="qari_full_page")
    layout = {"pages": [{"page": 1, "image_width": 1700, "image_height": 2200, "boxes": [
        {"block_index": 2, "label": "figure_title", "score": 0.95,
         "bbox_pt": [300, 220, 500, 245]},
    ]}]}
    run = write_run(tmp_path, "run_visual_title_not_section", [element], routes={1: "SCANNED"}, layout=layout)
    result = build_structure(run, make_config())
    assert len(result.tree.sections) <= 1
    if result.tree.sections:
        assert result.tree.sections[0].level == 0
    assert all(section.title != "Weekly Route Performance" for section in result.tree.sections)
    assert "".join(span.text for span in result.spans["page"]) == text


def test_derived_spans_of_one_element_go_to_different_places(tmp_path):
    text = "Intro sentence one. 2) Second topic: Details about the second topic follow here. More details."
    elements = [el("p1_e0", 1, 0, text, type="page_text", bbox=[0, 0, 612, 792], source="qari_full_page")]
    result = build_structure(write_run(tmp_path, "run_split", elements, routes={1: "SCANNED"}), make_config())
    a = _by_unit(result)
    assert a["p1_e0_span_001"].section_id is None
    section = result.tree.sections[0]
    assert section.heading_unit_ids == ["p1_e0_span_002"]
    assert a["p1_e0_span_003"].section_id == section.section_id
    assert a["p1_e0_span_003"].confidence in ("medium", "low")  # derived span: never above medium
    tree = section_tree_json(result)
    assert tree["unassigned"][0]["unit_id"] == "p1_e0_span_001"
    assert {f["code"]: f["severity"] for f in result.findings}["assignment_uniqueness"] == "ok"


def test_reading_order_flags_lower_assignment_confidence(tmp_path):
    elements = [el("h", 1, 0, "Overview", type="heading", bbox=[72, 80, 200, 96]),
                el("b", 1, 1, BODY_A, bbox=[72, 100, 540, 114])]
    validation = {"flags": [{"severity": "warning", "code": "layout_review", "page": 1,
                             "message": "ambiguous reading order: overlapping elements"}]}
    result = build_structure(write_run(tmp_path, "run_flags", elements, validation=validation), make_config())
    b = _by_unit(result)["b"]
    assert b.confidence == "low" and b.review_recommended and not b.deterministic
