from ha.data.hiaq_chunk import MAX_WORDS, OVERLAP, _windows, chunk_blocks, extract_blocks

HTML = """
<html><body><nav>Home | About</nav><main>
<h1>Help</h1><h2>Refunds</h2>
<p>Refunds are processed after your return is verified and assessed by the department. It can
take a few weeks, and the status is visible in your account dashboard under refund status.</p>
<h3>Common questions</h3>
<p>How do I check my refund status?</p>
<p>Log in and open the refund status page.</p>
<details><summary>What if my refund failed?</summary><p>Raise a refund reissue request after
validating your bank account.</p></details>
</main><footer>Copyright</footer></body></html>
"""


def test_blocks_and_chunks():
    blocks = extract_blocks(HTML)
    kinds = [(b.kind, b.text[:20]) for b in blocks]
    assert ("question", "What if my refund fa") in kinds       # <details> accordion
    assert ("question", "How do I check my re") in kinds       # "?" heuristic
    assert not any("Home" in b.text or "Copyright" in b.text for b in blocks)  # nav/footer gone
    chunks = chunk_blocks(blocks, boilerplate=set())
    qa = [c for c in chunks if c.kind == "qa"]
    sec = [c for c in chunks if c.kind == "section"]
    assert {c.extra["question"] for c in qa} == {"How do I check my refund status?",
                                                 "What if my refund failed?"}
    assert sec and sec[0].heading_path == ["Help", "Refunds"]
    status_qa = next(c for c in qa if c.extra["question"].startswith("How do I"))
    assert status_qa.heading_path == ["Help", "Refunds", "Common questions"]
    assert "refund status page" in status_qa.text


def test_boilerplate_is_dropped():
    blocks = extract_blocks(HTML)
    boiler = {b.text for b in blocks if b.kind == "text" and b.text.startswith("Refunds are")}
    chunks = chunk_blocks(blocks, boilerplate=boiler)
    assert not any(c.kind == "section" for c in chunks)


def test_windows_respect_max_and_overlap():
    words = [str(i) for i in range(800)]
    wins = _windows(words)
    assert all(len(w) <= MAX_WORDS for w in wins)
    assert wins[1][0] == str(MAX_WORDS - OVERLAP)
    assert wins[-1][-1] == "799"


def test_navigation_links_are_not_questions_or_content():  # noqa: D103
    html = ('<main><h2>Login help</h2><p><a href="/f">Forgot Password?</a></p>'
            '<li><a href="/r">Register Yourself</a></li>'
            '<p>Reset it from the login page using the OTP sent to your registered mobile.</p>'
            '</main>')
    blocks = extract_blocks(html)
    assert not any(b.kind == "question" for b in blocks)
    assert not any("Register Yourself" in b.text for b in blocks)
