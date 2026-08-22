import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "audit_temporal_gold.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_audit_temporal_gold", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_stated_interval_uses_end_of_first_date_range():
    text = "14 days after the weekend on April 20-21, beginning on May 5."

    assert _MODULE.stated_interval_days(text) == 14


def test_stated_interval_allows_gold_to_mention_later_date_first():
    text = "The follow-up on May 8 happened 15 days after submission on April 23."

    assert _MODULE.stated_interval_days(text) == 15


def test_audit_flags_inconsistent_gold_arithmetic():
    row = {
        "query_id": "17_temporal_reasoning_1",
        "query": "How many days passed?",
        "gold_answers": [
            "46 days passed between finishing casting on April 20 and the "
            "pilot episode being 75% complete by July 5."
        ],
        "answer": "76 days",
        "score": 0,
    }

    audit = _MODULE.audit_row(row)

    assert audit is not None
    assert audit["calendar_interval_days"] == 76
    assert audit["gold_claimed_days"] == 46
    assert audit["gold_arithmetic_matches"] is False
    assert audit["answer_quantity"] == 76


def test_audit_normalizes_weeks_to_days():
    row = {
        "query_id": "week",
        "gold_answers": [
            "Exactly 4 weeks passed between January 1 and January 29."
        ],
        "answer": "4 weeks",
    }

    audit = _MODULE.audit_row(row)

    assert audit is not None
    assert audit["gold_claimed_days"] == 28
    assert audit["calendar_interval_days"] == 28
    assert audit["gold_arithmetic_matches"] is True
