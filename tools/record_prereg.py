"""Record the frozen pre-registration and its configuration artefacts (D-030).

Hashes are computed from the git objects at the tag (LF-normalised by .gitattributes), so they
are independent of the working copy's line endings. Output: results/PREREG.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess

from ha.paths import REPO_ROOT
from ha.prereg import AMENDMENTS_HEADING, frozen_body_sha256
from ha.utils import utc_now_iso

TAG = "prereg-v1"
ARTEFACTS = [
    "docs/preregistration.md",
    "configs/models.yaml",
    "configs/generation.yaml",
    "configs/judge.yaml",
    "configs/datasets/fiqa.yaml",
    "configs/datasets/cqa_android.yaml",
    "src/ha/gen/prompts/gen_std_v1.j2",
    "src/ha/gen/prompts/gen_persona_v1.j2",
    "src/ha/gen/prompts/judge_v0_system.j2",
    "src/ha/gen/prompts/judge_v0_user.j2",
    "env/models.json",
    "requirements.txt",
]


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, check=True).stdout


def main() -> None:
    commit = _git("rev-list", "-n", "1", TAG).decode().strip()
    files = {}
    for path in ARTEFACTS:
        blob = _git("show", f"{TAG}:{path}")
        files[path] = hashlib.sha256(blob).hexdigest()
    prereg_text = _git("show", f"{TAG}:docs/preregistration.md").decode("utf-8")
    record = {
        "tag": TAG,
        "commit": commit,
        "frozen_on": "2026-09-27",
        "recorded_at": utc_now_iso(),
        "decisions": ["D-016", "D-027", "D-028", "D-029", "D-030"],
        "primary_test": "H2: pooled over D2+D4, tau_b(M1,A) > tau_b(B0,A) and > tau_b(B1,A); "
                        "one-sided paired bootstrap B=1000 seed 20260927; IUT alpha=0.05",
        "frozen_body_sha256": frozen_body_sha256(prereg_text),
        "frozen_body_scope": f"text above '{AMENDMENTS_HEADING}', LF-normalised",
        "artefact_sha256_at_tag": files,
        "models": {
            "generator_judge": {
                "tag": "qwen3:4b-instruct-2507-q4_K_M",
                "digest": "0edcdef34593eac1aa2be9c7d06c432d"
                          "cf81945adca5eca2f27662c18f168ba0",
            },
            "embeddings": {
                "tag": "bge-m3",
                "digest": "7907646426070047a77226ac3e684fbb"
                          "e8410524f7b4a74d02837e43f2146bab",
            },
            "reranker": {"id": "BAAI/bge-reranker-v2-m3",
                         "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"},
        },
    }
    out = REPO_ROOT / "results" / "PREREG.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
