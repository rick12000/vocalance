import pytest

from vocalance.app.services.storage.atomic_json import JsonReadError, read_json_dict, write_json_atomic


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "nested" / "data.json"
    payload = {"version": 1, "marks": {"a": {"x": 1, "y": 2}}}

    write_json_atomic(path, payload)

    assert read_json_dict(path) == payload


def test_overwrite_replaces_content_and_leaves_no_residue(tmp_path):
    path = tmp_path / "data.json"
    write_json_atomic(path, {"value": 1})
    write_json_atomic(path, {"value": 2})

    assert read_json_dict(path) == {"value": 2}
    assert list(path.parent.glob("*.tmp.*")) == []
    assert not (path.with_suffix(".backup")).exists()


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(JsonReadError):
        read_json_dict(tmp_path / "absent.json")


def test_read_invalid_json_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(JsonReadError):
        read_json_dict(path)
