import json

import pytest

from ha.paths import get_paths
from ha.seal import SCOPES, SealedDataError, require_unsealed, sealed, unsealed_scopes


@sealed("heldout_A", what="authentic test set A")
def load_test_set(dataset: str) -> str:
    return f"A of {dataset}"


def test_everything_sealed_by_default():
    assert unsealed_scopes() == frozenset()
    for scope in SCOPES:
        with pytest.raises(SealedDataError):
            require_unsealed(scope, "x")


def test_sealed_loader_raises_and_writes_no_log():
    with pytest.raises(SealedDataError):
        load_test_set(dataset="cqa_android")
    assert not get_paths().unseal_log.exists()


def test_scoped_unseal_opens_only_that_scope(monkeypatch):
    monkeypatch.setenv("HA_UNSEAL", "cal_test")
    require_unsealed("cal_test", "cal-test split")
    with pytest.raises(SealedDataError):
        load_test_set(dataset="cqa_android")
    with pytest.raises(SealedDataError):
        require_unsealed("heldout_systems", "held-out system metrics")


def _signed_freeze():
    from ha import freeze

    get_paths().frozen.parent.mkdir(parents=True, exist_ok=True)
    get_paths().frozen.write_text(json.dumps({"analysis": {"git_sha": "abc"}}), encoding="utf-8")
    dec = get_paths().results / "decisions.md"
    dec.write_text("### D-001 · x\n", encoding="utf-8")
    freeze.sign("Test Lead", decisions=dec)


def test_heldout_A_needs_a_signed_unchanged_freeze(monkeypatch):
    monkeypatch.setenv("HA_UNSEAL", "heldout_A")
    with pytest.raises(SealedDataError, match="FROZEN.json"):
        load_test_set(dataset="cqa_android")                 # no record yet
    _signed_freeze()
    assert load_test_set(dataset="cqa_android") == "A of cqa_android"
    rec = json.loads(get_paths().frozen.read_text(encoding="utf-8"))
    rec["weights"] = {"M1": "tampered"}
    get_paths().frozen.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(SealedDataError, match="FROZEN.json"):
        load_test_set(dataset="cqa_android")                 # edited after signing


def test_full_unseal_logs_every_access(monkeypatch):
    monkeypatch.setenv("HA_UNSEAL", "1")
    _signed_freeze()
    assert load_test_set(dataset="hiaq") == "A of hiaq"
    require_unsealed("heldout_systems", "metrics")
    lines = get_paths().unseal_log.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    assert [e["scope"] for e in entries] == ["heldout_A", "heldout_systems"]
    assert entries[0]["dataset"] == "hiaq"
    assert all(e["timestamp"] and e["git_sha"] for e in entries)


def test_unknown_scope_is_an_error(monkeypatch):
    monkeypatch.setenv("HA_UNSEAL", "cal_test,everything")
    with pytest.raises(ValueError):
        unsealed_scopes()
    monkeypatch.setenv("HA_UNSEAL", "0")
    with pytest.raises(ValueError):
        require_unsealed("not_a_scope", "x")
