"""Build the paper's figure and tables from the result files (no number is typed by hand).

Inputs:  results/eval/{dev-fiqa, heldout-cqa_android, exploratory-hiaq}.json  (frozen pipeline)
         results/eval/v2-{fiqa, cqa_android, hiaq}.json                     (D-068, OVL and NN)
         results/eval/explore_overlap.json                                  (overlap diagnosis)
Outputs: paper/figures/fig1_tau_by_dataset.{png,pdf}
         paper/tables/table1_tau.csv, table2_overlap.csv, table3_v2.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL, OUT = ROOT / "results" / "eval", ROOT / "paper"
MAIN = {"cqa_android": "heldout-cqa_android", "fiqa": "dev-fiqa", "hiaq": "exploratory-hiaq"}
LABEL = {"cqa_android": "D2 Android (primary)", "fiqa": "D1 FiQA (development)",
         "hiaq": "D4 HiAQ (exploratory)"}


def load(name: str) -> dict:
    return json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, f in MAIN.items():
        boot = load(f)["bootstrap"]
        for m, v in boot["tau_seed_mean"][ds].items():
            rows.append({"dataset": LABEL[ds], "method": m, "tau_b": v["tau_b"],
                         "ci_low": v["ci_low"], "ci_high": v["ci_high"], "source": f})
        for m, v in load(f"v2-{ds}")["bootstrap"]["tau_seed_mean"].items():
            if m in ("OVL", "NN"):
                rows.append({"dataset": LABEL[ds], "method": m, "tau_b": v["tau_b"],
                             "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                             "source": f"v2-{ds}"})
    t = pd.DataFrame(rows)
    t[t["method"].isin(["M1", "M1-iso", "B0", "B1"])].round(3).to_csv(
        OUT / "tables" / "table1_tau.csv", index=False)
    t3 = t.pivot(index="dataset", columns="method", values="tau_b")[
        ["B0", "B1", "M1", "OVL", "NN"]].round(3)
    t3.to_csv(OUT / "tables" / "table3_v2.csv")
    for tag, fname in (("R1", "table4_r1.csv"), ("R2", "table5_r2.csv")):
        rb = load(f"{tag}-replication")["bootstrap"]
        rows_t = []
        for ds, by in rb["tau_seed_mean"].items():
            row = {"dataset": ds, **{m: round(v["tau_b"], 3) for m, v in by.items()}}
            for comp in ("OVL-B0", "OVL-B1"):
                c = rb["comparisons"][comp][ds]
                row[f"delta_{comp}"] = round(c["delta"], 3)
                row[f"ci_{comp}"] = f"{c['ci_low']:.3f}..{c['ci_high']:.3f}"
            rows_t.append(row)
        pooled = {"dataset": f"pooled ({tag} test)"}
        for comp in ("OVL-B0", "OVL-B1"):
            c = rb["comparisons"][comp]["pooled"]
            pooled[f"delta_{comp}"] = round(c["delta"], 3)
            pooled[f"ci_{comp}"] = f"{c['ci_low']:.3f}..{c['ci_high']:.3f}"
            pooled[f"p_{comp}"] = round(c["p_one_sided"], 3)
        pd.DataFrame([*rows_t, pooled]).to_csv(OUT / "tables" / fname, index=False)
    ov = load("explore_overlap")["datasets"]
    t2 = pd.DataFrame({LABEL[d]: ov[d]["seed_mean"] for d in ("fiqa", "cqa_android", "hiaq")})
    t2.round(3).to_csv(OUT / "tables" / "table2_overlap.csv")

    style = {"M1": dict(marker="o", color="#2a6fdb", ms=9, label="M1 judge + examples"),
             "M1-iso": dict(marker="D", mfc="none", color="#9a9a8f", ms=7, label="M1-iso judge"),
             "B0": dict(marker="o", color="#b7b7ad", ms=8, label="B0 random"),
             "B1": dict(marker="o", mfc="none", color="#9a9a8f", ms=8, label="B1 stratified"),
             "OVL": dict(marker="s", color="#e8622c", ms=8, label="OVL overlap-matched*")}
    order = [LABEL[d] for d in MAIN]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for y, ds in enumerate(order):
        ax.axhline(y, color="#e6e6e6", lw=1, zorder=0)
        for m, st in style.items():
            v = t[(t["dataset"] == ds) & (t["method"] == m)]["tau_b"].iloc[0]
            ax.plot(v, y, ls="none", zorder=3, **{**st, "label": st["label"] if y == 0 else None})
            if m in ("M1", "OVL"):
                ax.annotate(f"{v:.2f}", (v, y), xytext=(0, 9 if m == "M1" else -15),
                            textcoords="offset points", ha="center", fontsize=8,
                            color=st["color"], fontweight="bold")
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(len(order) - 0.45, -0.55)          # inverted, with room for the value labels
    ax.set_xlim(0.35, 0.97)
    ax.set_xlabel("Kendall's τ_b with authentic A over C1–C7 (mean of 5 splits)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.22), ncol=5, frameon=False,
              fontsize=8, handletextpad=0.3, columnspacing=1.0)
    fig.text(0.01, 0.01, "*OVL is exploratory (§5.1).", fontsize=7, color="#666666")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"fig1_tau_by_dataset.{ext}", dpi=200, bbox_inches="tight")
    print(t3)


if __name__ == "__main__":
    main()
