import pandas as pd

from ha.data.scenarios import LeakageChecker, build_sheet

TEXTS = {
    "c1": "To claim a refund reissue after a failed refund, validate your bank account and submit "
          "a reissue request under the services menu of the e-filing portal.",
    "c2": "Form 26AS shows tax credit statements including TDS deducted by your employer and "
          "advance tax paid during the financial year.",
    "c3": "Members can transfer their provident fund balance to a new employer online after the "
          "universal account number is activated and KYC is seeded.",
    "c4": "Pensioners must submit a life certificate every year, which can be generated through "
          "Jeevan Pramaan using biometric authentication.",
}


FILLER = ("Help for users: log in to your account online, check the status of your request, "
          "update details, and contact support if the new page shows an error or money is due.")


def _chunks():
    texts = dict(TEXTS)
    texts.update({f"f{i}": FILLER for i in range(16)})  # common words get a high document frequency
    rows = []
    for i, (cid, text) in enumerate(texts.items()):
        rows.append({"chunk_id": cid, "source": "incometax" if i < 2 else "epfo",
                     "url": f"https://x.gov.in/iec/foportal/help/t{i}/p", "heading_path": "",
                     "text": text, "n_words": 40, "kind": "section"})
    return pd.DataFrame(rows)


def test_rare_term_and_bm25_giveaway_are_rejected():
    checker = LeakageChecker(_chunks())
    leaky = checker.check("I need form 26AS to see advance tax credit statements", "c2")
    assert not leaky["pass"]
    natural = checker.check("You changed jobs and want your old savings with the new company",
                            "c3")
    assert natural["pass"] and not natural["warn"], natural
    echo = checker.check("How can members transfer provident fund balance to a new employer?",
                         "c3")
    assert echo["warn"] or not echo["pass"]


def test_sheet_allocation_and_authors():
    chunks = pd.concat([_chunks().iloc[:4]] * 30, ignore_index=True)
    chunks["chunk_id"] = [f"c{i}" for i in range(len(chunks))]
    sheet = build_sheet(chunks, n=20, seed=17)
    assert len(sheet) == 20 and sheet["chunk_id"].is_unique
    assert set(sheet["source"]) == {"incometax", "epfo"}
    assert (sheet["author"] == "A").sum() == 10
    assert sheet.equals(build_sheet(chunks, n=20, seed=17))


def test_professional_filter_does_not_catch_everyday_words():
    from ha.data.scenarios import PROFESSIONAL_TOPIC_RE
    for everyday in ["where is my refund", "here to help", "how to file itr-1 sahaj",
                     "confirm your bank account", "entrusted to the portal"]:
        assert not PROFESSIONAL_TOPIC_RE.search(everyday)
    for pro in ["help/all-topics/statutory-forms/popular-form/form29b-um", "help/company/x",
                "help/firm-llp/how-to-file-tax-returns",
                "form-15cb", "Register as CA"]:
        assert PROFESSIONAL_TOPIC_RE.search(pro.replace("Register as CA", "/ca/registration"))


def test_explicit_quota():
    chunks = pd.concat([_chunks().iloc[:4]] * 30, ignore_index=True)
    chunks["chunk_id"] = [f"c{i}" for i in range(len(chunks))]
    sheet = build_sheet(chunks, n=20, seed=17, quota={"incometax": 10, "epfo": 10}, author="team")
    assert sheet["source"].value_counts().to_dict() == {"incometax": 10, "epfo": 10}
    assert (sheet["author"] == "team").all()


def test_professional_filter_catches_eri_and_ca_headings():
    from ha.data.scenarios import PROFESSIONAL_TOPIC_RE
    for pro in ["Work with my ERI FAQ", "View Client Details (by ERIs) > User Manual",
                "Register for e-Filing (CA) User Manual",
                "Services Available > List of services accessible by Type of User"]:
        assert PROFESSIONAL_TOPIC_RE.search(pro), pro
    for everyday in ["Work with My CA User Manual", "series of steps", "ca-certificate"]:
        assert not PROFESSIONAL_TOPIC_RE.search(everyday), everyday
