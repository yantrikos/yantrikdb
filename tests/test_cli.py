"""Tests for the YantrikDB CLI tool."""

import json
import math
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from yantrikdb import YantrikDB
from yantrikdb.cli import cli

DIM = 8


def _vec(seed: float) -> list[float]:
    raw = [(seed + i) * 0.1 for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def db_path(tmp_path):
    """Create a temp DB with some test data."""
    p = str(tmp_path / "test.db")
    db = YantrikDB(db_path=p, embedding_dim=DIM)
    db.record("hello world", embedding=_vec(1.0))
    db.record("important fact", memory_type="semantic", importance=0.9,
              embedding=_vec(2.0))
    db.relate("Alice", "Bob", rel_type="knows")
    db.close()
    return p


@pytest.fixture
def runner():
    return CliRunner()


class TestStats:
    def test_stats_output(self, runner, db_path):
        result = runner.invoke(cli, ["stats", "--db", db_path, "--dim", str(DIM)])
        assert result.exit_code == 0
        assert "Active memories:       2" in result.output

    def test_stats_shows_entities(self, runner, db_path):
        result = runner.invoke(cli, ["stats", "--db", db_path, "--dim", str(DIM)])
        assert "Entities:" in result.output
        assert "Edges:" in result.output


class TestInspect:
    def test_inspect_existing(self, runner, db_path):
        db = YantrikDB(db_path=db_path, embedding_dim=DIM)
        results = db.recall(query_embedding=_vec(1.0), top_k=1, skip_reinforce=True)
        rid = results[0]["rid"]
        db.close()

        result = runner.invoke(cli, ["inspect", "--db", db_path, "--dim", str(DIM), rid])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["rid"] == rid
        assert "embedding" not in data  # stripped from output

    def test_inspect_not_found(self, runner, db_path):
        result = runner.invoke(cli, ["inspect", "--db", db_path, "--dim", str(DIM),
                                     "nonexistent-rid"])
        assert result.exit_code == 1


class TestExportImport:
    def test_export_to_stdout(self, runner, db_path):
        result = runner.invoke(cli, ["export", "--db", db_path, "--dim", str(DIM)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["version"] == "yantrikdb-export-v1"
        assert len(data["memories"]) == 2
        assert len(data["edges"]) == 1

    def test_export_to_file(self, runner, db_path, tmp_path):
        out = str(tmp_path / "export.json")
        result = runner.invoke(cli, ["export", "--db", db_path, "--dim", str(DIM),
                                     "-o", out])
        assert result.exit_code == 0
        data = json.loads(Path(out).read_text())
        assert data["version"] == "yantrikdb-export-v1"
        assert len(data["memories"]) == 2

    def test_export_no_embeddings(self, runner, db_path):
        result = runner.invoke(cli, ["export", "--db", db_path, "--dim", str(DIM),
                                     "--no-embeddings"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for m in data["memories"]:
            assert "embedding" not in m

    def test_import_into_empty(self, runner, db_path, tmp_path):
        # Export first
        out = str(tmp_path / "export.json")
        runner.invoke(cli, ["export", "--db", db_path, "--dim", str(DIM), "-o", out])

        # Import into new DB
        new_db = str(tmp_path / "imported.db")
        result = runner.invoke(cli, ["import", "--db", new_db, "--dim", str(DIM), out])
        assert result.exit_code == 0
        assert "2 memories" in result.output

        # Verify
        verify = runner.invoke(cli, ["stats", "--db", new_db, "--dim", str(DIM)])
        assert "Active memories:       2" in verify.output

    def test_import_refuses_nonempty(self, runner, db_path, tmp_path):
        out = str(tmp_path / "export.json")
        runner.invoke(cli, ["export", "--db", db_path, "--dim", str(DIM), "-o", out])

        # Try importing back into the same (non-empty) DB
        result = runner.invoke(cli, ["import", "--db", db_path, "--dim", str(DIM), out])
        assert result.exit_code == 1
        assert "not empty" in result.output

    def test_import_merge(self, runner, db_path, tmp_path):
        out = str(tmp_path / "export.json")
        runner.invoke(cli, ["export", "--db", db_path, "--dim", str(DIM), "-o", out])

        # Merge into same DB (should skip duplicates)
        result = runner.invoke(cli, ["import", "--db", db_path, "--dim", str(DIM),
                                     "--merge", out])
        assert result.exit_code == 0

    def test_roundtrip_preserves_data(self, runner, db_path, tmp_path):
        """Export → import → export should produce identical data."""
        out1 = str(tmp_path / "export1.json")
        runner.invoke(cli, ["export", "--db", db_path, "--dim", str(DIM),
                            "--no-embeddings", "-o", out1])

        new_db = str(tmp_path / "roundtrip.db")
        runner.invoke(cli, ["import", "--db", new_db, "--dim", str(DIM), out1])

        out2 = str(tmp_path / "export2.json")
        runner.invoke(cli, ["export", "--db", new_db, "--dim", str(DIM),
                            "--no-embeddings", "-o", out2])

        data1 = json.loads(Path(out1).read_text())
        data2 = json.loads(Path(out2).read_text())
        assert len(data1["memories"]) == len(data2["memories"])
        assert len(data1["edges"]) == len(data2["edges"])
        assert len(data1["entities"]) == len(data2["entities"])


class TestMigrateAndImportAreDistinct:
    """The cross-system migrator registered itself as a second "import"
    and click silently REPLACED the export-restore command — every module
    still imported cleanly, only invocation broke. Pin both names."""

    def test_import_is_still_the_export_restore(self, runner):
        result = runner.invoke(cli, ["import", "--help"])
        assert result.exit_code == 0
        assert "JSON export" in result.output

    def test_migrate_is_the_cross_system_reader(self, runner):
        result = runner.invoke(cli, ["migrate", "--help"])
        assert result.exit_code == 0
        assert "mem0" in result.output


class TestThink:
    def test_think_empty_db(self, runner, tmp_path):
        p = str(tmp_path / "empty.db")
        result = runner.invoke(cli, ["think", "--db", p, "--dim", str(DIM)])
        assert result.exit_code == 0
        assert "Consolidations:" in result.output


class TestConflicts:
    def test_conflicts_empty(self, runner, db_path):
        result = runner.invoke(cli, ["conflicts", "--db", db_path, "--dim", str(DIM)])
        assert result.exit_code == 0
        assert "No conflicts." in result.output


class TestTriggers:
    def test_triggers_empty(self, runner, db_path):
        result = runner.invoke(cli, ["triggers", "--db", db_path, "--dim", str(DIM)])
        assert result.exit_code == 0
        assert "No pending triggers." in result.output
