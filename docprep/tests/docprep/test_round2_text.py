"""Round 2 regressions on synthetic runs without a PDF: list introductions, lists, continuations, child headings,
semantic warnings and callouts. Written for the user to run; not executed by the implementer."""

from __future__ import annotations

from docprep.pipeline import build_structure
from docprep.text_patterns import list_intro_cues, list_marker

from .helpers import el, make_config, write_run

LONG_EN = "This paragraph explains the topic of the section in enough words to read as ordinary body text."
LONG_AR = "يعتمد هذا التحليل على مجموعة بيانات تجريبية تحتوي على سجلات متعددة مرتبطة بعمليات الفحص والمعالجة."


def _a(result):
    return {a.unit_id: a for a in result.tree.assignments}


def _c(result, unit_id):
    return next(c for c in result.candidates if unit_id in c.unit_ids)


def test_language_cues_and_markers():
    assert list_intro_cues("وتتضمن البيانات عددًا من المتغيرات الرئيسية مثل:")
    assert list_intro_cues("لذلك يُوصى بـ:") and list_intro_cues("ويمكن دعم ذلك من خلال:")
    assert list_intro_cues("These records include the following fields:")
    assert list_intro_cues("وصف البيانات:") == [] and list_intro_cues("Data sources:") == []
    assert list_marker("Power BI -", "rtl") == ("-", "end")
    assert list_marker("Power BI -", "ltr") == (None, None)
    assert list_marker("• Item", "ltr") == ("•", "start")


def test_list_introduction_stays_in_the_enclosing_section(tmp_path):
    elements = [
        el("h", 1, 0, "البيانات:", type="heading", bbox=[440, 80, 540, 96]),
        el("p", 1, 1, LONG_AR, bbox=[72, 100, 540, 130]),
        el("intro", 1, 2, "وتتضمن السجلات عددًا من الحقول الرئيسية مثل:", bbox=[300, 140, 540, 154]),
        el("li1", 1, 3, "• تاريخ التسجيل.", bbox=[420, 158, 540, 170]),
        el("li2", 1, 4, "• اسم الفرع.", bbox=[440, 174, 540, 186]),
        el("after", 1, 5, LONG_AR, bbox=[72, 196, 540, 226]),
    ]
    result = build_structure(write_run(tmp_path, "run_intro", elements, directions={1: "rtl"}), make_config())
    a = _a(result)
    intro = _c(result, "intro")
    assert intro.status == "excluded" and any(r.startswith("list introduction") for r in intro.exclusions)
    section = result.tree.sections[0]
    assert len(result.tree.sections) == 1 and section.heading_unit_ids == ["h"]
    assert a["intro"].role == "list_intro" and a["intro"].section_id == section.section_id
    node = result.tree.lists[0]
    assert node.intro_unit_id == "intro" and node.unit_ids == ["li1", "li2"] and node.section_id == section.section_id
    assert a["after"].section_id == section.section_id


def test_colon_line_without_sentence_cues_is_not_a_list_introduction(tmp_path):
    elements = [
        el("h", 1, 0, "Inputs", type="heading", bbox=[72, 80, 200, 96]),
        el("p", 1, 1, LONG_EN, bbox=[72, 100, 540, 130]),
        el("lab", 1, 2, "Data sources:", bbox=[72, 140, 200, 154]),
        el("li1", 1, 3, "• Survey records.", bbox=[90, 158, 300, 170]),
        el("li2", 1, 4, "• Station logs.", bbox=[90, 174, 300, 186]),
    ]
    result = build_structure(write_run(tmp_path, "run_label", elements), make_config())
    lab = _c(result, "lab")
    assert "list_intro" not in next(u for u in result.units if u.unit_id == "lab").features
    assert not any(r.startswith("list introduction") for r in lab.exclusions)


def test_numbered_colon_heading_followed_by_a_list_stays_a_heading(tmp_path):
    elements = [
        el("p0", 1, 0, LONG_EN, bbox=[72, 80, 540, 110]),
        el("h", 1, 1, "1) Data description:", bbox=[72, 120, 240, 134]),
        el("li1", 1, 2, "• Date of each trip.", bbox=[90, 138, 300, 150]),
        el("li2", 1, 3, "• Line and station.", bbox=[90, 154, 300, 166]),
    ]
    result = build_structure(write_run(tmp_path, "run_num_list", elements), make_config())
    h = _c(result, "h")
    assert h.status == "promoted" and not h.exclusions
    assert _a(result)["li1"].section_id == result.tree.sections[0].section_id


def test_rtl_end_of_line_markers_form_a_list(tmp_path):
    elements = [
        el("h", 1, 0, "الأدوات المستخدمة:", type="heading", bbox=[400, 80, 540, 96]),
        el("t1", 1, 1, "Power BI -", bbox=[470, 100, 540, 112]),
        el("t2", 1, 2, "Python -", bbox=[480, 116, 540, 128]),
        el("t3", 1, 3, "Colab -", bbox=[485, 132, 540, 144]),
    ]
    result = build_structure(write_run(tmp_path, "run_rtl_list", elements, directions={1: "rtl"}), make_config())
    a = _a(result)
    assert [a[u].role for u in ("t1", "t2", "t3")] == ["list_item"] * 3
    node = result.tree.lists[0]
    assert [i.marker_position for i in node.items] == ["end"] * 3 and node.unit_ids == ["t1", "t2", "t3"]
    for uid in ("t1", "t2", "t3"):
        candidate = next((c for c in result.candidates if uid in c.unit_ids), None)
        assert candidate is None or candidate.status != "promoted"
    assert next(u for u in result.units if u.unit_id == "t1").text == "Power BI -"  # verbatim


def test_wrapped_bullet_continuation_is_linked(tmp_path):
    elements = [
        el("h", 1, 0, "معالجة البيانات:", type="heading", bbox=[420, 70, 540, 86]),
        el("li1", 1, 1, "• معالجة القيم المفقودة في بيانات المستشفى واستبدالها لتمييز الحوادث غير المصحوبة",
           bbox=[80, 100, 540, 114]),
        el("cont", 1, 2, "بإصابات.", bbox=[480, 116, 540, 130]),
        el("li2", 1, 3, "• تنظيف النصوص من الرموز الزائدة.", bbox=[300, 134, 540, 148]),
    ]
    result = build_structure(write_run(tmp_path, "run_wrap", elements, directions={1: "rtl"}), make_config())
    a = _a(result)
    assert a["cont"].role == "list_item_continuation" and a["cont"].list_id == a["li1"].list_id
    node = result.tree.lists[0]
    assert node.items[0].continuation_unit_ids == ["cont"] and node.unit_ids == ["li1", "cont", "li2"]
    assert next(u for u in result.units if u.unit_id == "cont").text == "بإصابات."


def test_unconfirmable_continuation_is_kept_separate_and_flagged(tmp_path):
    elements = [
        el("h", 1, 0, "Cleaning", type="heading", bbox=[72, 70, 200, 86]),
        el("li1", 1, 1, "• Replace missing hospital values to separate accidents without", bbox=[72, 740, 540, 754]),
        el("cont", 2, 0, "injuries.", bbox=[72, 80, 140, 94]),
        el("p", 2, 1, LONG_EN, bbox=[72, 100, 540, 130]),
    ]
    result = build_structure(write_run(tmp_path, "run_wrap_page", elements, page_count=2), make_config())
    a = _a(result)
    assert a["cont"].role == "paragraph" and a["cont"].role_uncertain and a["cont"].review_recommended
    assert any("cannot be confirmed" in c for c in a["cont"].conflicting)
    assert result.tree.lists[0].possible_continuations[0]["unit_id"] == "cont"


def test_list_extends_across_pages(tmp_path):
    elements = [
        el("h", 1, 0, "Steps", type="heading", bbox=[72, 70, 200, 86]),
        el("li1", 1, 1, "• Collect the records.", bbox=[90, 700, 300, 712]),
        el("li2", 1, 2, "• Clean the records.", bbox=[90, 716, 300, 728]),
        el("li3", 2, 0, "• Analyse the records.", bbox=[90, 80, 300, 92]),
    ]
    result = build_structure(write_run(tmp_path, "run_list_pages", elements, page_count=2), make_config())
    node = result.tree.lists[0]
    assert node.unit_ids == ["li1", "li2", "li3"] and node.to_dict()["crosses_pages"]
    assert any("list continues from page 1" in s for s in _a(result)["li3"].supporting)


def test_numbered_heading_versus_numbered_list_items(tmp_path):
    elements = [
        el("h", 1, 0, "Methodology", type="heading", bbox=[72, 70, 220, 86]),
        el("p", 1, 1, LONG_EN, bbox=[72, 90, 540, 120]),
        el("s2", 1, 2, "2. Clean the data and handle the missing values.", bbox=[90, 130, 420, 142]),
        el("s3", 1, 3, "3. Build derived variables for the analysis.", bbox=[90, 146, 420, 158]),
        el("s4", 1, 4, "4. Explore the data.", bbox=[90, 162, 300, 174]),
        el("n1", 1, 5, "1) Time indicators:", bbox=[72, 190, 220, 204]),
        el("q", 1, 6, LONG_EN, bbox=[72, 208, 540, 238]),
    ]
    result = build_structure(write_run(tmp_path, "run_numbered", elements), make_config())
    a = _a(result)
    for uid in ("s2", "s3"):
        c = _c(result, uid)
        assert c.status != "promoted" and any(s.name == "numbered_list_sequence" for s in c.signals)
    assert [a[u].role for u in ("s2", "s3", "s4")] == ["list_item"] * 3
    assert _c(result, "n1").status == "promoted"


def test_heading_followed_by_a_child_heading_is_not_penalised(tmp_path):
    elements = [
        el("h1", 1, 0, "1. Data:", type="heading", bbox=[72, 70, 200, 86]),
        el("h2", 1, 1, "1.1 Data description:", type="heading", bbox=[72, 92, 260, 106]),
        el("p", 1, 2, LONG_EN, bbox=[72, 110, 540, 140]),
    ]
    result = build_structure(write_run(tmp_path, "run_child", elements), make_config())
    parent = _c(result, "h1")
    assert parent.status == "promoted" and parent.semantic and parent.semantic.get("deferred")
    assert not any(c.startswith("semantic_relation") for c in parent.conflicts)
    sections = {s.heading_unit_ids[0]: s for s in result.tree.sections}
    assert sections["h2"].parent == sections["h1"].section_id
    assert "no direct content: its content is in child sections" in sections["h1"].flags


def test_semantic_warning_lowers_confidence_without_moving_content(tmp_path):
    elements = [
        el("h5", 1, 0, "5) Operational indicators:", bbox=[72, 70, 260, 86]),
        el("p1", 1, 1, "The branch handled most cases within the target time during the year.", bbox=[72, 90, 540, 120]),
        el("p2", 1, 2, "Inspector workload varies widely between inspectors and their weekly cases.",
           bbox=[72, 124, 540, 154]),
        el("h6", 1, 3, "6) Inspector workload and efficiency:", bbox=[72, 170, 320, 186]),
        el("p3", 1, 4, "The chart compares cases and processing time for every inspector.", bbox=[72, 190, 540, 220]),
    ]
    result = build_structure(write_run(tmp_path, "run_semantic", elements), make_config())
    a = _a(result)
    sections = {s.heading_unit_ids[0]: s for s in result.tree.sections}
    assert a["p2"].section_id == sections["h5"].section_id  # not moved
    assert any(c.startswith("semantic warning") for c in a["p2"].conflicting)
    assert a["p2"].review_recommended and not a["p2"].deterministic
    assert not any(c.startswith("semantic warning") for c in a["p1"].conflicting)


def test_callout_label_is_not_a_main_heading(tmp_path):
    body = [el(f"b{i}", 1, i + 1, LONG_EN, bbox=[72, 80 + 34 * i, 532, 110 + 34 * i]) for i in range(4)]
    elements = [el("h", 1, 0, "Network report", type="heading", bbox=[72, 50, 250, 66])] + body + [
        el("lab", 1, 5, "Outlier locations:", bbox=[72, 300, 280, 316]),
        el("txt", 1, 6, "The map shows several points outside the expected region.", bbox=[72, 320, 280, 400]),
        el("fig", 1, 7, "map", type="figure", bbox=[320, 300, 560, 450], source="qari_crop"),
    ]
    result = build_structure(write_run(tmp_path, "run_callout", elements), make_config())
    lab = _c(result, "lab")
    assert lab.status == "excluded" and any("callout" in r for r in lab.exclusions)
    a = _a(result)
    assert a["lab"].role == "aside_label" and a["lab"].role_uncertain
    assert a["txt"].role == "aside_text"


def test_multi_column_page_is_not_treated_as_callouts(tmp_path):
    left = [el(f"l{i}", 1, i, LONG_EN, bbox=[72, 120 + 60 * i, 290, 170 + 60 * i]) for i in range(3)]
    right = [el(f"r{i}", 1, 4 + i, LONG_EN, bbox=[320, 120 + 60 * i, 540, 170 + 60 * i]) for i in range(3)]
    elements = [el("h", 1, 0, "Overview", type="heading", bbox=[72, 70, 200, 86]),
                el("sum", 1, 3, "Summary:", bbox=[72, 300, 160, 314])] + left + right
    result = build_structure(write_run(tmp_path, "run_columns", elements), make_config())
    assert not any("aside" in u.features for u in result.units)
