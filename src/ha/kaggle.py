"""Versioned code zips for Kaggle/Colab (plan I-18: no GitHub token ever goes to Kaggle)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ha.paths import REPO_ROOT
from ha.utils import git_sha


def pack(out_dir: Path | None = None) -> Path:
    """`git archive` of HEAD, so the zip contains exactly one commit. Refuses a dirty tree."""
    sha = git_sha()
    if sha == "nogit" or sha.endswith("-dirty"):
        raise RuntimeError(f"Commit your changes first (git_sha={sha}); zips must map to a commit.")
    out_dir = out_dir or REPO_ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"humanalign-rag-{sha[:10]}.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--prefix=humanalign-rag-{sha[:10]}/", "-o", str(out),
         "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )
    return out
