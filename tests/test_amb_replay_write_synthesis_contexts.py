from types import SimpleNamespace

import pytest

from benchmarks.amb.replay_write_synthesis_contexts import (
    configure_source_turn_rollups,
    select_queries,
)


class _Provider:
    @staticmethod
    def _extract_axis():
        return "unconfigured"

    @staticmethod
    def _build_semantic_threads(items):
        return items


def test_source_turn_rollup_configuration_disables_model_extraction():
    module = SimpleNamespace(
        _WRITE_SYNTH_AXES=("asked",),
        _WRITE_SYNTH_SOURCE_TURNS=False,
        _WRITE_SYNTH_THREADS=False,
        YantrikDBWriteTimeSynthesisMemoryProvider=_Provider,
    )

    provider = configure_source_turn_rollups(module, "semantic")

    assert module._WRITE_SYNTH_AXES == ()
    assert module._WRITE_SYNTH_SOURCE_TURNS is True
    assert module._WRITE_SYNTH_THREADS is True
    with pytest.raises(RuntimeError, match="attempted model extraction"):
        provider._extract_axis()


def test_global_rollup_mode_disables_semantic_handles():
    module = SimpleNamespace(
        _WRITE_SYNTH_AXES=("asked",),
        _WRITE_SYNTH_SOURCE_TURNS=False,
        _WRITE_SYNTH_THREADS=False,
        YantrikDBWriteTimeSynthesisMemoryProvider=_Provider,
    )

    provider = configure_source_turn_rollups(module, "global")

    assert provider._build_semantic_threads(["item"]) == []


def test_select_queries_uses_only_units_and_categories():
    queries = [
        SimpleNamespace(user_id="1", meta={"question_category": "summarization"}),
        SimpleNamespace(user_id="1", meta={"question_category": "event_ordering"}),
        SimpleNamespace(user_id="2", meta={"question_category": "summarization"}),
    ]
    dataset = SimpleNamespace(load_queries=lambda _split: queries)

    selected = select_queries(dataset, "100k", {"1"}, {"summarization"})

    assert selected == [queries[0]]
