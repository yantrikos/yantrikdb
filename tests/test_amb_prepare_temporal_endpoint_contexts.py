import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "prepare_temporal_endpoint_contexts.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_prepare_temporal_endpoint_contexts", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_render_endpoint_context_groups_and_bounds_candidates():
    row = {
        "endpoints": ["first event", "second event"],
        "lane_hits": [
            [
                {"rid": "a", "text": "First A"},
                {"rid": "b", "text": "First B"},
                {"rid": "c", "text": "First C"},
            ],
            [
                {"rid": "b", "text": "Duplicate B"},
                {"rid": "d", "text": "Second D"},
            ],
        ],
    }

    context = _MODULE.render_endpoint_context(row, hits_per_endpoint=2)

    assert "## Temporal endpoint 1" in context
    assert "Query fragment: first event" in context
    assert "First A" in context
    assert "First B" in context
    assert "First C" not in context
    assert "Duplicate B" not in context
    assert "Second D" in context
