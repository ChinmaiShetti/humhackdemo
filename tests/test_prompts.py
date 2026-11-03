import shutil

from ha import prompts


def test_templates_render_with_strict_variables():
    out = prompts.render("judge_v0_user", domain_description="d", passage="p",
                         neighbours=["a", "b"], query="q", lang_class="EN")
    assert "1. a" in out and "2. b" in out and 'TEST QUERY: "q"' in out
    iso = prompts.render("judge_v0_user", domain_description="d", passage="p",
                         neighbours=[], query="q", lang_class="EN")
    assert "REAL USER QUERIES" not in iso  # M1-iso: no neighbour block at all


def test_template_hash_ignores_line_endings(tmp_path, monkeypatch):
    src = prompts.PROMPT_DIR / "gen_std_v1.j2"
    lf_hash = prompts.template_hash("gen_std_v1")
    crlf_dir = tmp_path / "prompts"
    crlf_dir.mkdir()
    shutil.copy(src, crlf_dir / "gen_std_v1.j2")
    data = (crlf_dir / "gen_std_v1.j2").read_bytes().replace(b"\r\n", b"\n")
    (crlf_dir / "gen_std_v1.j2").write_bytes(data.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(prompts, "PROMPT_DIR", crlf_dir)
    assert prompts.template_hash("gen_std_v1") == lf_hash
