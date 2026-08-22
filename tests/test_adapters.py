"""Tests for agent framework adapters (LangChain, OpenAI, CrewAI)."""

import pytest
from yantrikdb import YantrikDB


class MockEmbedder:
    """Simple deterministic embedder for testing."""
    def encode(self, text):
        h = hash(text) & 0xFFFFFFFF
        return [(h >> i & 0xFF) / 255.0 for i in range(0, 32, 1)][:8]


@pytest.fixture
def db():
    return YantrikDB(":memory:", 8, MockEmbedder())


class TestLangChainAdapter:
    """Test LangChain chat memory adapter."""

    def test_save_and_load(self, db):
        from yantrikdb.adapters.langchain import YantrikDBChatMemory

        mem = YantrikDBChatMemory(db)
        mem.save_context(
            {"input": "What is Python?"},
            {"output": "Python is a programming language."},
        )

        result = mem.load_memory_variables({"input": "Tell me about Python"})
        assert mem.memory_key in result
        history = result[mem.memory_key]
        assert "Python" in history

    def test_save_preserves_turn_speaker_provenance(self, db):
        from yantrikdb.adapters.langchain import YantrikDBChatMemory

        mem = YantrikDBChatMemory(db)
        mem.save_context(
            {"input": "I prefer compact status reports."},
            {"output": "I will keep future reports compact."},
        )

        user_hits = db.recall(
            query="compact status reports",
            top_k=5,
            source="user",
            skip_reinforce=True,
        )
        assistant_hits = db.recall(
            query="future reports compact",
            top_k=5,
            source="assistant",
            skip_reinforce=True,
        )

        assert user_hits
        assert all(hit["source"] == "user" for hit in user_hits)
        assert user_hits[0]["metadata"]["speaker_role"] == "user"
        assert assistant_hits
        assert all(hit["source"] == "assistant" for hit in assistant_hits)
        assert assistant_hits[0]["metadata"]["speaker_role"] == "assistant"
        stats = db.stats()
        assert stats["provenance_verified_records"] == 2
        assert stats["unverified_user_source_records"] == 0
        assert stats["provenance_source_counts"] == {"user": 1, "assistant": 1}
        assert stats["provenance_method_counts"] == {"direct_turn_v1": 2}
        assert stats["unverified_source_counts"] == {}

    def test_memory_variables_property(self, db):
        from yantrikdb.adapters.langchain import YantrikDBChatMemory

        mem = YantrikDBChatMemory(db, memory_key="chat_history")
        assert mem.memory_variables == ["chat_history"]

    def test_empty_input_returns_empty(self, db):
        from yantrikdb.adapters.langchain import YantrikDBChatMemory

        mem = YantrikDBChatMemory(db)
        result = mem.load_memory_variables({"input": ""})
        assert result[mem.memory_key] == ""

    def test_clear_is_noop(self, db):
        from yantrikdb.adapters.langchain import YantrikDBChatMemory

        mem = YantrikDBChatMemory(db)
        mem.save_context({"input": "hi"}, {"output": "hello"})
        mem.clear()  # Should not raise
        # Memories should still be recallable after clear
        result = mem.load_memory_variables({"input": "hi"})
        assert result[mem.memory_key] != ""

    def test_multiple_turns(self, db):
        from yantrikdb.adapters.langchain import YantrikDBChatMemory

        mem = YantrikDBChatMemory(db, top_k=10)
        for i in range(5):
            mem.save_context(
                {"input": f"Turn {i} question about cats"},
                {"output": f"Turn {i} answer about cats"},
            )

        result = mem.load_memory_variables({"input": "cats"})
        history = result[mem.memory_key]
        assert len(history) > 0


class TestOpenAIAgentsAdapter:
    """Test OpenAI Agents SDK adapter."""

    def test_get_tools_structure(self):
        from yantrikdb.adapters.openai_agents import get_tools

        tools = get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 6

        names = {t["function"]["name"] for t in tools}
        assert names == {
            "memory_record",
            "memory_recall",
            "memory_forget",
            "entity_relate",
            "entity_edges",
            "memory_stats",
        }

        for tool in tools:
            assert tool["type"] == "function"
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_handle_record_and_recall(self, db):
        from yantrikdb.adapters.openai_agents import handle_tool_call

        result = handle_tool_call(db, "memory_record", {
            "text": "The sky is blue",
            "memory_type": "semantic",
            "importance": 0.8,
            "source": "user",
        })
        assert "rid" in result

        result = handle_tool_call(db, "memory_recall", {
            "query": "What color is the sky?",
            "top_k": 5,
            "source": "user",
        })
        assert "memories" in result
        assert len(result["memories"]) >= 1
        assert all(memory["source"] == "user" for memory in result["memories"])

    def test_record_defaults_agent_authored_memory_to_assistant(self, db):
        from yantrikdb.adapters.openai_agents import handle_tool_call

        handle_tool_call(db, "memory_record", {"text": "Try the compact layout."})
        result = handle_tool_call(db, "memory_recall", {
            "query": "compact layout",
            "source": "assistant",
        })

        assert len(result["memories"]) == 1
        assert result["memories"][0]["source"] == "assistant"
        assert result["memories"][0]["metadata"]["provenance_verified"] is False

    def test_handle_forget(self, db):
        from yantrikdb.adapters.openai_agents import handle_tool_call

        result = handle_tool_call(db, "memory_record", {"text": "secret info"})
        rid = result["rid"]

        result = handle_tool_call(db, "memory_forget", {"rid": rid})
        assert result["forgotten"] is True

    def test_handle_entity_relate_and_edges(self, db):
        from yantrikdb.adapters.openai_agents import handle_tool_call

        handle_tool_call(db, "entity_relate", {
            "source": "Alice",
            "target": "Bob",
            "relationship": "works_with",
        })

        result = handle_tool_call(db, "entity_edges", {"entity": "Alice"})
        assert "edges" in result
        assert any(e["dst"] == "Bob" for e in result["edges"])

    def test_handle_stats(self, db):
        from yantrikdb.adapters.openai_agents import handle_tool_call

        result = handle_tool_call(db, "memory_stats", {})
        assert "active_memories" in result

    def test_unknown_tool_raises(self, db):
        from yantrikdb.adapters.openai_agents import handle_tool_call

        with pytest.raises(ValueError, match="Unknown tool"):
            handle_tool_call(db, "nonexistent_tool", {})


class TestCrewAIAdapter:
    """Test CrewAI memory adapters."""

    def test_short_term_save_and_search(self, db):
        from yantrikdb.adapters.crewai import YantrikDBShortTermMemory

        mem = YantrikDBShortTermMemory(db)
        mem.save("The meeting is at 3pm", agent="scheduler")
        results = mem.search("When is the meeting?")
        assert len(results) >= 1
        assert "score" in results[0]
        assert "context" in results[0]
        assert "3pm" in results[0]["context"]
        stored = db.recall(
            query="meeting 3pm",
            top_k=5,
            source="assistant",
            skip_reinforce=True,
        )
        assert stored and stored[0]["metadata"]["speaker_role"] == "assistant"
        assert stored[0]["metadata"]["provenance_verified"] is True

    def test_long_term_save_and_search(self, db):
        from yantrikdb.adapters.crewai import YantrikDBLongTermMemory

        mem = YantrikDBLongTermMemory(db)
        mem.save("Python was created by Guido van Rossum")
        results = mem.search("Who created Python?")
        assert len(results) >= 1
        assert "Guido" in results[0]["context"]

    def test_entity_save_with_graph(self, db):
        from yantrikdb.adapters.crewai import YantrikDBEntityMemory

        mem = YantrikDBEntityMemory(db)
        mem.save(
            "Alice is the CTO of TechCorp",
            metadata={
                "entity": "Alice",
                "target_entity": "TechCorp",
                "relationship": "cto_of",
            },
        )

        # Verify graph edge was created
        edges = db.get_edges("Alice")
        assert any(e["dst"] == "TechCorp" for e in edges)

    def test_entity_search_with_expansion(self, db):
        from yantrikdb.adapters.crewai import YantrikDBEntityMemory

        mem = YantrikDBEntityMemory(db)
        mem.save(
            "Alice leads the engineering team",
            metadata={"entity": "Alice"},
        )

        results = mem.search("Who leads engineering?")
        assert len(results) >= 1

    def test_reset_is_noop(self, db):
        from yantrikdb.adapters.crewai import YantrikDBShortTermMemory, YantrikDBLongTermMemory, YantrikDBEntityMemory

        for cls in [YantrikDBShortTermMemory, YantrikDBLongTermMemory, YantrikDBEntityMemory]:
            mem = cls(db)
            mem.reset()  # Should not raise

    def test_search_limit(self, db):
        from yantrikdb.adapters.crewai import YantrikDBShortTermMemory

        mem = YantrikDBShortTermMemory(db, top_k=3)
        for i in range(10):
            mem.save(f"Event number {i} happened today")

        results = mem.search("What happened today?", limit=2)
        assert len(results) <= 2
