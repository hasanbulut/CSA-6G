"""Shared plumbing for the experiment scripts."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Where generated artefacts land.  Defaults to the repository root, so
# ``make all`` behaves exactly as documented; set CSA6G_OUT to redirect a whole
# run into its own tree instead of overwriting the previous one:
#
#     CSA6G_OUT=version2 make all
#
# A relative value is resolved against the repository root.  Redirecting is the
# only way to keep two runs side by side -- every writer here overwrites by
# name, so a second run into the same tree silently replaces the first.
_out = os.environ.get("CSA6G_OUT", "").strip()
OUT = (ROOT / _out).resolve() if _out else ROOT

RESULTS = OUT / "results"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
for _d in (RESULTS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    # Embed TrueType outlines rather than Matplotlib's default Type-3 glyphs
    # so journal PDF preflight/graphics checks remain portable.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_json(obj, name: str) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def save_fig(fig, name: str) -> list[Path]:
    out = []
    for ext in ("png", "pdf"):
        p = FIGURES / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def quick_flag() -> bool:
    return "--quick" in sys.argv
