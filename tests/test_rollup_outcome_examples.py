from yantrikdb import YantrikDB


def _vec(seed: float) -> list[float]:
    return [seed + index * 0.01 for index in range(8)]


def test_finalized_rollup_examples_are_available_to_python(tmp_path):
    db = YantrikDB(db_path=str(tmp_path / "rollup-examples.db"), embedding_dim=8)
    try:
        rollup = db.record("topic rollup", embedding=_vec(1.0), namespace="n")
        selected = db.record("selected child", embedding=_vec(2.0), namespace="n")
        corrected = db.record("corrected child", embedding=_vec(3.0), namespace="n")
        unselected = db.record("unselected child", embedding=_vec(4.0), namespace="n")

        impression = db.note_rollup_impression(
            rollup,
            "private query text",
            namespace="n",
            rank=1,
            score=0.75,
            impression_id="python-example",
        )
        db.note_rollup_expansion(impression, [selected, corrected, unselected])
        db.finalize_rollup_outcome(impression, [selected], [corrected])

        rows = db.rollup_outcome_examples(namespace="n")
        assert len(rows) == 3
        assert db.rollup_outcome_examples(namespace="other") == []
        assert all(row["export_schema_version"] == 1 for row in rows)
        assert all("private query text" not in str(row) for row in rows)
        assert rows[0]["selected"] is True
        assert rows[0]["corrected"] is False
        assert rows[1]["selected"] is True
        assert rows[1]["corrected"] is True
        assert rows[2]["selected"] is False
        assert rows[2]["corrected"] is False
        assert all(row["returned_child_count"] == 3 for row in rows)
    finally:
        db.close()


def test_membership_examples_include_explicit_omissions(tmp_path):
    db = YantrikDB(db_path=str(tmp_path / "membership-examples.db"), embedding_dim=8)
    try:
        rollup = db.record("topic rollup", embedding=_vec(1.0), namespace="n")
        selected = db.record("selected child", embedding=_vec(2.0), namespace="n")
        omitted = db.record("omitted positive", embedding=_vec(3.0), namespace="n")

        impression = db.note_rollup_impression_features(
            rollup,
            "List exactly two topic items",
            namespace="n",
            rank=1,
            score=0.75,
            requested_count=2,
            query_shape="list",
            impression_id="python-membership",
        )
        db.note_rollup_expansion_features(impression, [selected], [0.8])
        db.finalize_rollup_outcome(
            impression,
            [selected],
            omitted_child_rids=[omitted],
        )

        rows = db.rollup_membership_examples(namespace="n", limit_impressions=1)
        assert len(rows) == 2
        assert all(row["export_schema_version"] == 1 for row in rows)
        assert all("query_hash" not in row for row in rows)
        returned = next(row for row in rows if row["returned"])
        missing = next(row for row in rows if row["omitted_positive"])
        assert returned["child_score"] == 0.8
        assert returned["positive"] is True
        assert missing["child_rank"] is None
        assert missing["positive"] is True
        assert missing["omission_source"] == "caller_false_negative"
        report = db.rollup_membership_report(namespace="n")
        assert report["finalized_added_children"] == 1
        assert report["added_child_rate"] == 0.5
    finally:
        db.close()
