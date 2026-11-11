import json
import pytest
from core.data_loader import DataError, FieldMappingError, detect_fields, iter_records, load_json
from core.preprocessing import instruction_steps, make_recipe_id


def test_deterministic_id_generation():
    a = make_recipe_id("f.json", "key1", "T", ["a"], "x")
    assert a == make_recipe_id("f.json", "key1", "T", ["a"], "x")
    assert a != make_recipe_id("g.json", "key1", "T", ["a"], "x")
    assert a != make_recipe_id("f.json", "key2", "T", ["a"], "x")
    assert make_recipe_id("f.json", "k", "T", ["a"], "x", genuine_id="G1") == make_recipe_id("f.json", "other", "T2", ["b"], "y", genuine_id="G1")


def test_container_formats():
    assert iter_records([{"title": "a"}])[0][0][0] == 0
    recs, _ = iter_records({"h1": {"title": "a"}, "h2": {"title": "b"}})
    assert [k for k, _ in recs] == ["h1", "h2"]
    recs, _ = iter_records({"recipes": [{"title": "a"}]})
    assert recs[0][0] == "recipes[0]"
    with pytest.raises(DataError):
        iter_records({"a": 1, "b": 2})
    with pytest.raises(DataError):
        iter_records(5)


def test_field_detection_and_ambiguity():
    recs = [(i, {"title": "t", "ingredients": ["x"], "instructions": "a\nb"}) for i in range(5)]
    m = detect_fields(recs)
    assert (m["title"], m["ingredients"], m["instructions"], m["id"]) == ("title", "ingredients", "instructions", None)
    with pytest.raises(FieldMappingError):
        detect_fields([(i, {"title": "t", "name": "n", "ingredients": ["x"], "instructions": "a"}) for i in range(5)])
    with pytest.raises(FieldMappingError):
        detect_fields([(i, {"foo": "t", "ingredients": ["x"], "instructions": "a"}) for i in range(5)])


def test_missing_and_bad_json(tmp_path):
    with pytest.raises(DataError):
        load_json(tmp_path / "nope.json")
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(DataError):
        load_json(p)


def test_step_counting_rule():
    assert instruction_steps("a\nb\n\nc") == (["a", "b", "c"], "lines")
    assert instruction_steps("Do this. Then that.")[1] == "sentences"
    assert len(instruction_steps("Do this. Then that.")[0]) == 2
    assert instruction_steps(["x", "", "y"]) == (["x", "y"], "list_items")
    assert instruction_steps(None) == ([], "missing")
    assert instruction_steps("ADVERTISEMENT\n")[0] == []


def test_build_database_end_to_end(tmp_path):
    import sqlite3
    from scripts.build_database import build
    raw = tmp_path / "raw"
    raw.mkdir()
    for i, n in enumerate(["a.json", "b.json", "c.json"]):
        data = {f"k{i}{j}": {"title": f"T{i}{j}", "ingredients": ["1 cup rice", "ADVERTISEMENT"], "instructions": "one\ntwo"} for j in range(2)}
        data["bad"] = None
        data["notitle"] = {"title": "", "ingredients": ["x"], "instructions": "y"}
        (raw / n).write_text(json.dumps(data))
    rep1 = build(raw, tmp_path / "x.db", expected=["a.json", "b.json", "c.json"])
    ids1 = [r[0] for r in sqlite3.connect(tmp_path / "x.db").execute("select id from recipes order by id")]
    rep2 = build(raw, tmp_path / "y.db", expected=["a.json", "b.json", "c.json"])
    ids2 = [r[0] for r in sqlite3.connect(tmp_path / "y.db").execute("select id from recipes order by id")]
    assert ids1 == ids2 and len(ids1) == 6
    assert rep1["totals"]["skipped"] == 6
    assert rep1["files"]["a.json"]["skipped"] == {"record_not_an_object": 1, "missing_or_empty_title": 1}
    with pytest.raises(DataError):
        build(raw, tmp_path / "z.db", expected=["a.json", "missing.json"])
