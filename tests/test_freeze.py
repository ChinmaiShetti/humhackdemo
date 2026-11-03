import json

import pytest

from ha import freeze
from ha.paths import get_paths


def _write(rec):
    get_paths().frozen.parent.mkdir(parents=True, exist_ok=True)
    get_paths().frozen.write_text(json.dumps(rec), encoding="utf-8")


def test_sign_records_hash_and_appends_decision(tmp_path):
    _write({"analysis": {"git_sha": "abc123"}, "weights": {"M1": 1}, "signed_by": None})
    dec = tmp_path / "decisions.md"
    dec.write_bytes(b"### D-007 \xc2\xb7 a\r\n\r\n### D-012 \xc2\xb7 b\r\n")
    assert not freeze.is_signed()
    rec = freeze.sign("Chinmai Shetti", decisions=dec)
    assert freeze.is_signed() and rec["signed_by"] == "Chinmai Shetti"
    assert rec["record_sha256"] == freeze.record_sha256(rec)
    text = dec.read_bytes().decode("utf-8")
    assert "### D-013" in text and "Freeze signed" in text and "abc123" in text
    assert "\r\n" in text and "\n" not in text.replace("\r\n", "")      # CRLF kept, no bare LF
    with pytest.raises(freeze.FreezeError, match="already signed"):
        freeze.sign("Someone Else", decisions=dec)


def test_tampering_after_signing_breaks_the_signature(tmp_path):
    _write({"weights": {"M1": 1}})
    dec = tmp_path / "d.md"
    dec.write_text("", encoding="utf-8")
    freeze.sign("Lead", decisions=dec)
    rec = freeze.load()
    rec["weights"]["M1"] = 2
    _write(rec)
    assert not freeze.is_signed()


def test_sign_refuses_without_record_or_name(tmp_path):
    with pytest.raises(freeze.FreezeError, match="write FROZEN.json first"):
        freeze.sign("Lead", decisions=tmp_path / "d.md")
    _write({"x": 1})
    with pytest.raises(freeze.FreezeError, match="name"):
        freeze.sign("  ", decisions=tmp_path / "d.md")


def test_write_refuses_to_overwrite_a_signed_record(tmp_path):
    _write({"x": 1})
    dec = tmp_path / "d.md"
    dec.write_text("", encoding="utf-8")
    freeze.sign("Lead", decisions=dec)
    with pytest.raises(freeze.FreezeError, match="signed"):
        freeze.write()
