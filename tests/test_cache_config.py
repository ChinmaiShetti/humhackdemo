from ha.cache import KVCache
from ha.config import config_hash


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_cache_get_set_and_compute_once(tmp_path):
    cache = KVCache(tmp_path / "c.sqlite")
    key_obj = {"prompt": "नमस्ते", "model": "sha256:abc", "params": {"temperature": 0}}
    calls = []

    def compute():
        calls.append(1)
        return {"F": 5, "text": "kya karu"}

    assert cache.get_or_compute(key_obj, compute) == {"F": 5, "text": "kya karu"}
    assert cache.get_or_compute(key_obj, compute) == {"F": 5, "text": "kya karu"}
    assert len(calls) == 1
    assert KVCache.key_for(key_obj) in cache
    assert len(cache) == 1


def test_merge_keeps_existing_and_adds_new(tmp_path):
    local = KVCache(tmp_path / "local.sqlite")
    remote = KVCache(tmp_path / "remote.sqlite")
    local.set("k1", "local")
    remote.set("k1", "remote")
    remote.set("k2", "remote")
    remote.close()
    assert local.merge_from(tmp_path / "remote.sqlite") == 1
    assert local.get("k1") == "local"
    assert local.get("k2") == "remote"
