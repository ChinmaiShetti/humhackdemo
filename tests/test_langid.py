"""LID/CMI tests on 30 hand-labelled examples (spec §6.5, §20.5).

PROVISIONAL LABELS: written by the agent strictly from the spec's class definitions (Hindi share
< 0.1 EN, > 0.8 ROM_HI, otherwise HINGLISH; DEV_HI / OTHER by script). The team verifies or
corrects them (human task H-09); a corrected label is a data change, not a code change.
Borderline items (Hindi share exactly on a threshold) are deliberately excluded.
"""

import pytest

from ha.data.langid import classify, cmi, tag_tokens

EXAMPLES = [
    # EN
    ("digilocker marksheet not showing what to do", "EN"),
    ("how to update mobile number in aadhaar", "EN"),
    ("upi payment failed but money deducted", "EN"),
    ("PF withdrawal status pending for 2 weeks", "EN"),
    ("can i file ITR after due date", "EN"),
    ("forgot my e-filing password", "EN"),
    ("how to link pan with aadhaar online", "EN"),
    ("is my pf account active", "EN"),
    # ROM_HI
    ("mera paisa kat gaya par transaction nahi hua", "ROM_HI"),
    ("aadhaar me naam kaise badle", "ROM_HI"),
    ("pf ka paisa kab tak aayega", "ROM_HI"),
    ("mujhe apna uan nahi pata chal raha", "ROM_HI"),
    ("kya main ghar baithe pan card bana sakta hu", "ROM_HI"),
    ("refund abhi tak kyu nahi aaya", "ROM_HI"),
    ("otp nahi aa raha hai", "ROM_HI"),
    # HINGLISH
    ("upi payment fail ho gaya par paise kat gaye", "HINGLISH"),
    ("mera account block ho gaya please help", "HINGLISH"),
    ("itr file karna hai but portal slow hai", "HINGLISH"),
    ("passbook download nahi ho raha please help karo", "HINGLISH"),
    ("bank account change karna hai epfo me", "HINGLISH"),
    ("income tax refund status kaise check kare", "HINGLISH"),
    ("digilocker me document nahi dikh raha", "HINGLISH"),
    ("e-kyc complete nahi ho raha kya karu", "HINGLISH"),
    ("mobile number link karne ka process kya hai", "HINGLISH"),
    ("pan card ko aadhaar se kaise link kare", "HINGLISH"),
    # DEV_HI
    ("आधार में मोबाइल नंबर कैसे अपडेट करें", "DEV_HI"),
    ("पीएफ का पैसा कब आएगा", "DEV_HI"),
    ("मेरा रिफंड अभी तक नहीं आया", "DEV_HI"),
    ("UPI से पैसे कट गए लेकिन payment fail", "DEV_HI"),
    # OTHER
    ("ಆಧಾರ್ ಅಪ್ಡೇಟ್ ಹೇಗೆ ಮಾಡುವುದು", "OTHER"),
]


def test_thirty_examples():
    assert len(EXAMPLES) == 30


@pytest.mark.parametrize(("text", "label"), EXAMPLES)
def test_language_class(text, label):
    assert classify(text)["label"] == label, classify(text)["tags"]


def test_named_entities_and_ambiguous_words():
    tags = dict(tag_tokens("mera UPI aur aadhaar to link hai"))
    assert tags["UPI"] == "ne" and tags["aadhaar"] == "ne"
    assert tags["to"] == "hi"  # ambiguous, Hindi context
    assert dict(tag_tokens("how to link aadhaar"))["to"] == "en"  # ambiguous, English context


def test_cmi_das_gamback():
    # upi(ne) payment(en) fail(en) ho gaya par paise kat gaye (6 hi): n=9, u=1 -> 100*(1-6/8)
    res = classify("upi payment fail ho gaya par paise kat gaye")
    assert res["n_hi"] == 6 and res["n_en"] == 2
    assert res["cmi"] == pytest.approx(25.0)
    assert cmi(["ne", "univ"]) == 0.0
    assert cmi(["en", "en"]) == 0.0


def test_sensitivity_shift_moves_thresholds():
    text = "digilocker me document nahi dikh raha"  # hi share 0.8 -> HINGLISH at default
    assert classify(text)["label"] == "HINGLISH"
    assert classify(text, shift=-0.05)["label"] == "ROM_HI"
