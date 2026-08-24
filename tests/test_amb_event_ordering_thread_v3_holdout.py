import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "amb" / "verify_event_ordering_thread_v3_holdout.py"
SPEC = importlib.util.spec_from_file_location("thread_v3_holdout", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_gzip(path: Path, value) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle)


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    queries = [
        {
            "id": "2_event_ordering_0",
            "query": "fresh order question",
            "user_id": "2",
            "meta": {"question_category": "event_ordering"},
        },
        {
            "id": "2_summary_0",
            "query": "fresh summary question",
            "user_id": "2",
            "meta": {"question_category": "summarization"},
        },
        {
            "id": "1_event_ordering_0",
            "query": "development question",
            "user_id": "1",
            "meta": {"question_category": "event_ordering"},
        },
    ]
    documents = [
        {"id": "2_s0", "content": "sealed", "user_id": "2"},
        {"id": "1_s0", "content": "development", "user_id": "1"},
    ]
    burned = [
        {
            "id": "9_event_ordering_0",
            "query": "old order question",
            "user_id": "9",
            "meta": {"question_category": "event_ordering"},
        }
    ]
    query_path = tmp_path / "queries.json.gz"
    document_path = tmp_path / "documents.json.gz"
    burned_path = tmp_path / "burned.json.gz"
    _write_gzip(query_path, queries)
    _write_gzip(document_path, documents)
    _write_gzip(burned_path, burned)
    holdout_queries = queries[:2]
    holdout_event = queries[:1]
    holdout_documents = documents[:1]
    manifest = {
        "protocol": MODULE.PROTOCOL,
        "status": "active",
        "holdout_units": ["2"],
        "event_category": "event_ordering",
        "expected": {
            "query_rows": 2,
            "event_rows": 1,
            "document_rows": 1,
            "unit_count": 1,
            "exact_event_query_overlap_with_beam_100k": 0,
        },
        "source_sha256": {
            "queries.json.gz": _source_sha(query_path),
            "documents.json.gz": _source_sha(document_path),
        },
        "subset_sha256": {
            "all_queries": MODULE.canonical_sha256(holdout_queries),
            "event_queries": MODULE.canonical_sha256(holdout_event),
            "all_ordered_query_ids": MODULE.ordered_ids_sha256(holdout_queries),
            "event_ordered_query_ids": MODULE.ordered_ids_sha256(holdout_event),
            "documents": MODULE.canonical_sha256(holdout_documents),
        },
    }
    return manifest, query_path, document_path, burned_path


def test_verify_accepts_exact_sealed_holdout_without_emitting_content(tmp_path):
    manifest, queries, documents, burned = _fixture(tmp_path)

    report = MODULE.verify(manifest, queries, documents, burned)

    assert report["verified"] is True
    assert report["content_emitted"] is False
    assert report["counts"] == {
        "query_rows": 2,
        "event_rows": 1,
        "document_rows": 1,
        "unit_count": 1,
    }
    assert report["exact_event_query_overlap_with_beam_100k"] == 0
    assert "fresh order question" not in json.dumps(report)


def test_verify_rejects_source_drift_before_parsing(tmp_path):
    manifest, queries, documents, burned = _fixture(tmp_path)
    manifest["source_sha256"]["queries.json.gz"] = "0" * 64

    with pytest.raises(ValueError, match="queries.json.gz"):
        MODULE.verify(manifest, queries, documents, burned)


def test_verify_rejects_subset_order_drift(tmp_path):
    manifest, queries, documents, burned = _fixture(tmp_path)
    manifest["subset_sha256"]["all_ordered_query_ids"] = "0" * 64

    with pytest.raises(ValueError, match="all_ordered_query_ids"):
        MODULE.verify(manifest, queries, documents, burned)


def test_verify_rejects_overlap_with_burned_event_queries(tmp_path):
    manifest, queries, documents, burned = _fixture(tmp_path)
    burned_rows = MODULE.read_json(burned)
    burned_rows[0]["query"] = "fresh order question"
    _write_gzip(burned, burned_rows)

    with pytest.raises(ValueError, match="overlap mismatch"):
        MODULE.verify(manifest, queries, documents, burned)


def test_verify_rejects_void_seal(tmp_path):
    manifest, queries, documents, burned = _fixture(tmp_path)
    manifest["status"] = "void"

    with pytest.raises(ValueError, match="not active"):
        MODULE.verify(manifest, queries, documents, burned)
