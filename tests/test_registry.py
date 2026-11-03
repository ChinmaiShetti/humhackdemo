import json

import pytest

from ha import registry
from ha.paths import get_paths


def _row(**over):
    row = {
        "run_id": registry.new_run_id("E0"),
        "exp": "E0",
        "dataset": "D1",
        "split_seed": None,
        "method": "S1",
        "system_set": "dev",
        "metric": "ndcg@10",
        "value": 0.236,
        "ci_low": None,
        "ci_high": None,
        "n_queries": 648,
    }
    row.update(over)
    return row


def test_round_trip_fills_auto_fields():
    n = registry.record([_row(), _row(value=0.3)], config={"k1": 0.9, "b": 0.4})
    assert n == 2
    df = registry.load()
    assert len(df) == 2
    assert set(registry.AUTO_FIELDS) <= set(df.columns)
    assert df["config_hash"].nunique() == 1
    assert df["frozen_hash"].isna().all()  # no FROZEN.json yet


def test_frozen_hash_recorded_when_frozen_exists():
    frozen = get_paths().frozen
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps({"judge": {"prompt_hash": "abc"}}), encoding="utf-8")
    registry.record([_row()], config={})
    assert registry.load()["frozen_hash"].iloc[0]


@pytest.mark.parametrize(
    "bad",
    [
        {"run_id": ""},
        {"value": float("nan")},
        {"value": None},
        {"value": True},
        {"metric": None},
        {"git_sha": "forged"},
    ],
)
def test_invalid_rows_rejected(bad):
    with pytest.raises(registry.RegistryError):
        registry.record([_row(**bad)], config={})
    assert not get_paths().registry.exists()


def test_missing_field_rejected():
    row = _row()
    del row["n_queries"]
    with pytest.raises(registry.RegistryError):
        registry.record([row], config={})


def test_batch_is_all_or_nothing():
    with pytest.raises(registry.RegistryError):
        registry.record([_row(), _row(run_id="")], config={})
    assert not get_paths().registry.exists()
