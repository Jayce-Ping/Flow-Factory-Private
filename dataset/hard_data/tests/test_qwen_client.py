import pytest

from lib.qwen_client import _parse_json_object


def test_parse_json_accepts_fence_and_trailing_comment() -> None:
    assert _parse_json_object('```json\n{"ok": true}\n```', request_id="fenced") == {
        "ok": True
    }
    assert _parse_json_object('{"ok": true}\n// END', request_id="comment") == {
        "ok": True
    }


def test_parse_json_rejects_second_object_or_prose() -> None:
    with pytest.raises(ValueError, match="non-comment content"):
        _parse_json_object('{"ok": true}{"other": 1}', request_id="double")
    with pytest.raises(ValueError, match="non-comment content"):
        _parse_json_object('{"ok": true}\nDone.', request_id="prose")
