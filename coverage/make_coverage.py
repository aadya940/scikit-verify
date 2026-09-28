"""Regenerate doc/coverage.md from the battery outputs.

Run each script, save its stdout next to it as <name>.out, then run
this. The page is a tracking list of every function we tried, and what
happened, nothing else.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
LIBS = [
    ("NumPy dialect (every public callable)", "numpy_dialect.out"),
    ("NumPy (common-use battery)", "numpy_full.out"),
    ("SciPy", "scipy_full.out"),
    ("scikit-learn", "skl_full.out"),
    ("statsmodels", "sm_full.out"),
    ("cvxpy", "cvxpy_full.out"),
    ("Wild research code (random GitHub sample)", "wild_100.out"),
]

def parse(path):
    text = path.read_text()
    lifted = []
    m = re.search(r"^LIFT[^:]*: (.*)$", text, re.M)
    if m:
        lifted = [x.strip() for x in m.group(1).split(",") if x.strip()]
    ref_block = text[text.find("REFUSED"):text.find("DIED")] if "REFUSED" in text else ""
    died_block = text[text.find("DIED"):text.find("UNCALLABLE")] if "UNCALLABLE" in text else (text[text.find("DIED"):] if "DIED" in text else "")
    refused = re.findall(r"^\s{2,}(\S+)", ref_block, re.M)
    died = re.findall(r"^\s{2,}(\S+)", died_block, re.M)
    refused = [x.rstrip("|") for x in refused if x != "|"]
    died = [x.rstrip("|") for x in died if x != "|"]
    return lifted, refused, died

out = [
    "# Coverage",
    "",
    "Every function we have run through `to_sympy`, and what happened.",
    "**Works** means the traced value matched the library exactly on",
    "that call. **Refuses** means a one-sentence refusal instead of a",
    "result. The rules behind refusals are in",
    "[sharp bits](sharp-bits.md). Regenerate this page with the",
    "scripts in `coverage/`.",
    "",
]
for title, fname in LIBS:
    p = HERE / fname
    if not p.exists():
        continue
    lifted, refused, died = parse(p)
    out.append(f"## {title}")
    out.append("")
    out.append(f"**Works ({len(lifted)}):** " + ", ".join(f"`{x}`" for x in lifted))
    out.append("")
    if refused:
        out.append(f"**Refuses ({len(refused)}):** " + ", ".join(f"`{x}`" for x in refused))
        out.append("")
    if died:
        out.append(f"**Known walls ({len(died)}):** " + ", ".join(f"`{x}`" for x in died))
        out.append("")
(HERE.parent / "doc" / "coverage.md").write_text("\n".join(out) + "\n")
print("doc/coverage.md written")
