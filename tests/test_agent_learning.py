"""Tests for provenance-safe post-interaction memory extraction."""

import asyncio
from types import SimpleNamespace

from yantrikdb.agent.learning import extract_and_learn


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    async def chat(self, messages, max_tokens):
        self.messages = messages
        return SimpleNamespace(content=self.payload)


class RecordingDB:
    def __init__(self):
        self.records = []
        self.edges = []

    def record(self, **kwargs):
        self.records.append(kwargs)
        return f"rid-{len(self.records)}"

    def relate(self, source, target, relationship):
        self.edges.append((source, target, relationship))


def test_extraction_records_user_grounded_provenance():
    llm = FakeLLM(
        """{
            "should_remember": true,
            "memory_text": "The user prefers weekly project updates.",
            "evidence_quote": "I prefer weekly project updates",
            "memory_type": "semantic",
            "importance": 0.8,
            "valence": 0.2,
            "domain": "work",
            "entities": [],
            "is_open_topic": false
        }"""
    )
    db = RecordingDB()

    asyncio.run(
        extract_and_learn(
            db,
            llm,
            "For this project, I prefer weekly project updates.",
            "I can also send a detailed report every day.",
        )
    )

    assert len(db.records) == 1
    record = db.records[0]
    assert record["source"] == "user"
    assert record["metadata"] == {
        "speaker_role": "user",
        "extracted_by": "companion",
        "evidence_quote": "I prefer weekly project updates",
        "provenance_verified": True,
        "provenance_method": "user_quote_v1",
    }
    assert "only admissible evidence" in llm.messages[1]["content"]


def test_extraction_rejects_assistant_only_claim():
    llm = FakeLLM(
        """{
            "should_remember": true,
            "memory_text": "The user wants daily reports.",
            "evidence_quote": "I can also send a detailed report every day.",
            "memory_type": "semantic",
            "importance": 0.8,
            "valence": 0.0,
            "domain": "work",
            "entities": [],
            "is_open_topic": false
        }"""
    )
    db = RecordingDB()

    asyncio.run(
        extract_and_learn(
            db,
            llm,
            "For this project, I prefer weekly project updates.",
            "I can also send a detailed report every day.",
        )
    )

    assert db.records == []
    assert db.edges == []


def test_extraction_accepts_quote_with_normalized_whitespace():
    llm = FakeLLM(
        """{
            "should_remember": true,
            "memory_text": "The user's launch target is April 22.",
            "evidence_quote": "launch target is April 22",
            "memory_type": "episodic",
            "importance": 0.7,
            "valence": 0.0,
            "domain": "work",
            "entities": [],
            "is_open_topic": false
        }"""
    )
    db = RecordingDB()

    asyncio.run(
        extract_and_learn(
            db,
            llm,
            "My launch target is\nApril 22, after the review.",
            "That gives us enough time.",
        )
    )

    assert len(db.records) == 1
