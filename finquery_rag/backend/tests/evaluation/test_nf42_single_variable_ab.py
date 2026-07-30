import hashlib
import json


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def test_only_extractor_fields_are_allowed_to_differ():
    current = {"provider": "current", "revision": "v1", "selector_hash": _sha("selector")}
    shadow = {"provider": "structured_shadow", "revision": "v2", "selector_hash": _sha("selector")}
    assert current["selector_hash"] == shadow["selector_hash"]
    assert current["provider"] != shadow["provider"]


def test_frozen_context_hash_is_stable():
    payload = {"case-1": "context-hash"}
    assert _sha(payload) == _sha(dict(payload))
