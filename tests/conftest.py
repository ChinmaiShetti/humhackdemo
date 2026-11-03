import pytest

from ha.paths import get_paths


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Point every writable location at a temp dir and start fully sealed."""
    for name in ("HA_DATA", "HA_CACHE", "HA_RESULTS", "HA_LOGS"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("HA_UNSEAL", "0")
    get_paths.cache_clear()
    yield tmp_path
    get_paths.cache_clear()
