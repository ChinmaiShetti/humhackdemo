import pytest

from ha import prereg


def test_frozen_preregistration_body_unchanged():
    prereg.check_frozen()


def test_amendments_do_not_change_the_frozen_hash():
    text = prereg.PREREG_PATH.read_text(encoding="utf-8")
    amended = text + "\n### 2026-10-01 · example amendment\nSomething appended.\n"
    assert prereg.frozen_body_sha256(amended) == prereg.frozen_body_sha256(text)
    assert prereg.frozen_body_sha256(text.replace("\n", "\r\n")) == prereg.frozen_body_sha256(text)


def test_body_edit_is_detected():
    text = prereg.PREREG_PATH.read_text(encoding="utf-8")
    edited = text.replace("B = 1,000", "B = 2,000", 1)
    assert edited != text
    assert prereg.frozen_body_sha256(edited) != prereg.frozen_body_sha256(text)
    with pytest.raises(ValueError):
        prereg.frozen_body_sha256("no amendments heading here")
