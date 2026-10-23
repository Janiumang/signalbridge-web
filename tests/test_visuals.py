"""Visual cards: escaping (XSS), edge cases, frame logic."""
from __future__ import annotations

import base64
import datetime
import re

import visuals

EVIL = '<script>alert(1)</script><img src=x onerror=alert(2)>'


def _ror(**kw):
    base = {
        "drug_name": "ibuprofen", "reaction": "Acute kidney injury",
        "filters_applied": {"min_age": 65, "max_age": None, "country": "US", "start_date": "2020-01-01", "end_date": None},
        "contingency_table": {"a_drug_and_reaction": 532, "b_drug_no_reaction": 9000, "c_reaction_no_drug": 60000, "d_neither": 3400000},
        "ror": 3.33, "ci_95_lower": 3.05, "ci_95_upper": 3.64, "signal": True,
    }
    base.update(kw)
    return base


def _decoded_svgs(html: str) -> str:
    return " ".join(base64.b64decode(m).decode() for m in re.findall(r"base64,([A-Za-z0-9+/=]+)", html))


def test_no_raw_html_from_data_in_any_card():
    cards = [
        visuals.ror_card(_ror(drug_name=EVIL, reaction=EVIL)),
        visuals.count_card({"drug_name": EVIL, "pivot": "reaction", "total_matching": 5,
                            "counts": [{"label": EVIL, "count": 5}]}),
        visuals.plan_rows([{"name": EVIL, "input": {"drug_name": EVIL, "reaction": EVIL}, "data": {}}]),
        visuals.ror_compare_card([_ror(drug_name=EVIL), _ror(drug_name="b", ror=1.2, signal=False)]),
    ]
    for html in cards:
        assert "<script" not in html and "<img src=x" not in html
        assert "<script" not in _decoded_svgs(html)


def test_ror_card_degenerate_and_zero():
    zero = visuals.ror_card(_ror(ror=0.0, ci_95_lower=None, ci_95_upper=None, signal=False))
    assert "NO SIGNAL" in zero and "undefined" in zero
    undefined = visuals.ror_card(_ror(ror=None, ci_95_lower=None, ci_95_upper=None, signal=None))
    assert "SIGNAL" not in undefined.replace("Signal criteria", "")
    assert visuals.ror_card({"error": "x"}) == ""


def test_compare_card_titles():
    both = visuals.ror_compare_card([_ror(), _ror(min_age=18, ror=5.28, signal=True)])
    assert "holds in both groups" in both
    mixed = visuals.ror_compare_card([_ror(), _ror(ror=0.9, ci_95_lower=0.7, ci_95_upper=1.1, signal=False)])
    assert "differs" in mixed
    assert visuals.ror_compare_card([_ror()]) == ""


def test_year_chart_marks_partial_current_year():
    y = datetime.date.today().year
    html = visuals.count_card({"drug_name": "d", "pivot": "year", "total_matching": 3,
                               "counts": [{"label": str(y - 1), "count": 1}, {"label": str(y), "count": 2}]})
    assert f"*{y} is a partial year" in html


def test_lawyer_alert_threshold():
    counts = [{"label": "Physician", "count": 90}, {"label": "Lawyer", "count": 10}]
    on = visuals.count_card({"drug_name": "d", "pivot": "reporter_qualification", "total_matching": 100, "counts": counts})
    assert "stimulated reporting" in on
    counts[1]["count"] = 1
    off = visuals.count_card({"drug_name": "d", "pivot": "reporter_qualification", "total_matching": 100, "counts": counts})
    assert "stimulated reporting" not in off


def test_frame_chips_and_consistent_title():
    f = {"drug_name": "ibuprofen", "reaction": "AKI", "country": "US"}
    calls = [{"name": "calculate_reporting_odds_ratio", "input": {**f, "min_age": 65}, "data": _ror()},
             {"name": "calculate_reporting_odds_ratio", "input": {**f, "min_age": 18, "max_age": 64}, "data": _ror()},
             {"name": "count_adverse_events", "input": {**f, "pivot": "year", "min_age": 65}, "data": {}}]
    html = visuals.plan_rows(calls)
    assert "3 tool calls, one consistent frame." in html
    assert "compare" in html and "country US" in html


def test_result_cards_order_and_fold():
    calls = [{"name": "count_adverse_events", "input": {}, "data": {"pivot": "year", "counts": [{"label": "2020", "count": 1}], "drug_name": "d", "total_matching": 1}},
             {"name": "calculate_reporting_odds_ratio", "input": {}, "data": _ror()},
             {"name": "calculate_reporting_odds_ratio", "input": {}, "data": _ror(min_age=18)}]
    cards = visuals.result_cards(calls)
    assert "ROR comparison" in cards[0] or "Stratified ROR" in cards[0]
    assert any("<details" in c for c in cards)


def test_sex_compare_labels_and_chips():
    def with_sex(sex, **kw):
        d = _ror(**kw)
        d["filters_applied"] = {**d["filters_applied"], "sex": sex}
        return d
    card = visuals.ror_compare_card([with_sex("female"), with_sex("male", ror=1.8, ci_95_lower=1.2, ci_95_upper=2.6)])
    assert "Female" in _decoded_svgs(card) and "Male" in _decoded_svgs(card)
    chips, compare, _ = visuals._frame([
        {"name": "calculate_reporting_odds_ratio", "input": {"drug_name": "x", "sex": "female"}},
        {"name": "calculate_reporting_odds_ratio", "input": {"drug_name": "x", "sex": "male"}},
    ])
    assert any("female vs male" in c for c in compare)
    assert visuals._PIVOT_TITLES["sex"] == "Reports by patient sex"
