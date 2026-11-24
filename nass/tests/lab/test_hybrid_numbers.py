"""Numeric unit validation (validation only; nothing is corrected or converted)."""

from __future__ import annotations

from lab.core.hybrid import numbers


def _values(text):
    return numbers.validate(text)["values"]


def test_percent_and_per_mille_are_never_interchangeable() -> None:
    percent, per_mille = _values("3.01% and 3.01‰")
    assert percent["unit"] == "%" and per_mille["unit"] == "‰" and percent["value"] == per_mille["value"] == "3.01"
    conflicts = numbers.unit_conflicts({"qari": numbers.validate("rate 3.01%"),
                                        "paddle": numbers.validate("rate 3.01‰")})
    assert conflicts == [{"value": "3.01", "units_by_output": {"qari": ["%"], "paddle": ["‰"]}}]
    assert numbers.unit_conflicts({"a": numbers.validate("3.01%"), "b": numbers.validate("3.01 %")}) == []


def test_separators_are_read_only_when_unambiguous() -> None:
    assert _values("٦,٤")[0]["value"] is None and "ambiguous" in _values("٦,٤")[0]["flags"][0]
    assert _values("١٬٢٨٤")[0]["value"] == "1284" and _values("٤٩٫٥٨٪")[0]["value"] == "49.58"
    assert _values("٤٩٫٥٨٪")[0]["unit"] == "%"
    assert _values("6,4")[0]["value"] is None and _values("1.234")[0]["value"] is None
    assert _values("1,284")[0]["value"] == "1284" and _values("1,284,330")[0]["value"] == "1284330"
    assert _values("12.50")[0]["value"] == "12.50" and _values("1,234.5")[0]["value"] == "1234.5"
    assert numbers.validate("٦,٤")["needs_review"]


def test_unicode_minus_is_read_as_minus_for_comparison_only() -> None:
    (value,) = _values("change −12.5%")
    assert value["value"] == "-12.5" and value["unit"] == "%" and value["minus_sign"] == "−"
    assert value["original"] == "−12.5%"  # the original text is kept


def test_units_and_formats() -> None:
    kinds = {v["original"]: (v["kind"], v["unit"]) for v in _values("At 10:45 AM on 2025-03-01: $1,284, 5.2M users, 12 min")}
    assert kinds["10:45 AM"] == ("time", None) and kinds["2025-03-01"] == ("date", None)
    assert kinds["$1,284"] == ("currency", "$") and kinds["5.2M"] == ("scale", "M")
    assert kinds["12 min"] == ("duration", "min")


def test_axis_like_sequences_are_candidates_not_data() -> None:
    result = numbers.validate("0 50 100 150 200\nQ1 37")
    assert result["axis_candidates"] == [["0", "50", "100", "150", "200"]]
    assert [v["axis_candidate"] for v in result["values"]] == [True, True, True, True, True, False, False]
    assert result["data_values"] == 2 and any("axis-like" in flag for flag in result["flags"])


def test_unit_mismatch_is_counted_separately_in_scores() -> None:
    mismatch = numbers.unit_mismatches("Share 42% of 1,284 cases", "Share 42‰ of 1,284 cases")
    assert mismatch["count"] == 1 and mismatch["items"][0]["value"] == "42"
    assert numbers.unit_mismatches("Share 42%", "Share 42%")["count"] == 0
