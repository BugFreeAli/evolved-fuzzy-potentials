"""Writes the data-driven parts of Section 5 so that no number in them is typed by hand:
    sections/ladder_rows.tex     rows of the per-seed table (nine methods)
    sections/pvalue_rows.tex     rows of the pairwise p-value table (nine methods, 36 tests)
    figures/tikz/fig_strip.tex   the per-seed strip chart (pgfplots, standalone source), nine methods;
                                 run figures/make_tikz_figures.py afterwards to turn it into figures/fig_strip.pdf
Usage (from the scripts folder):  python make_tables.py [DATA_DIR]      (default ../../data)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data as pdt

HERE = os.path.dirname(os.path.abspath(__file__))
D = pdt.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "data"))
SEC = os.path.join(HERE, "..", "sections")
hu = pdt.hu

# ---------------------------------------------------------------- per-seed table
FORMCOL = {"B0": "none", "B1": "PBRS, linear", "B2a": "additive, fuzzy", "B2b": "PBRS, fuzzy",
           "Ours-S1": "additive, evolved", "Ours-S2": "additive, evolved",
           "C1": "additive, bounded", "C2": "additive, linear", "C3": "PBRS, bounded"}
rows = []
for k in pdt.ORDER:
    v = D[k]; lo, hi = pdt.boot_ci(v)
    row = (f"{k:8s} & {FORMCOL[k]:19s} & {len(v)} & {', '.join(str(x) for x in v):31s} & {hu(np.mean(v))} & "
           f"{hu(np.std(v, ddof=1))} & {hu(np.median(v))} & [{hu(lo)}, {hu(hi)}] \\\\")
    if k == "C1": rows.append("\\midrule")
    rows.append(row)
rows[-1] = rows[-1].rstrip("\\").rstrip()           # the last row's line break is written after \input in the table
open(os.path.join(SEC, "ladder_rows.tex"), "w").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- p-value table (lower triangle)
cols, rws = pdt.ORDER[:-1], pdt.ORDER[1:]
lines = []
for j, r in enumerate(rws):
    cells = []
    for i, c in enumerate(cols):
        if i > j:                      # upper triangle left empty
            cells.append("")
        else:
            p = pdt.p_exact(D[r], D[c])
            t = hu(p, 3)
            cells.append(f"\\textbf{{{t}}}" if p < 0.05 else t)
    if r == "C1": lines.append("\\addlinespace[2pt]")
    lines.append(f"{r:8s} & " + " & ".join(cells) + " \\\\")
lines[-1] = lines[-1].rstrip("\\").rstrip()
open(os.path.join(SEC, "pvalue_rows.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- strip chart
COLOUR = {"B0": "refgrey", "B1": "refgrey", "B2a": "handblue", "B2b": "handblue", "Ours-S1": "evolorange",
          "Ours-S2": "evolorange", "C1": "ctrlgreen", "C2": "ctrlgreen", "C3": "ctrlgreen"}
def jitter(vals, step=0.05):
    """spread equal values sideways (symmetrically) so that every seed stays visible"""
    seen = {}
    for v in vals: seen[v] = seen.get(v, 0) + 1
    used, out = {}, []
    for v in vals:
        n = seen[v]; i = used.get(v, 0); used[v] = i + 1
        out.append((i - (n - 1) / 2) * step if n > 1 else 0.0)
    return out
L = ["\\begin{tikzpicture}", "\\begin{axis}[",
     "    width=0.96\\linewidth, height=6.2cm,",
     f"    xmin=0.4, xmax={len(D) + 0.6}, ymin=0, ymax=52,",
     f"    xtick={{{','.join(str(i + 1) for i in range(len(D)))}}},",
     f"    xticklabels={{{','.join(k for k in pdt.ORDER)}}},",
     "    ylabel={success rate (\\%)},", "    ymajorgrids=true, grid style={gray!25},",
     "    tick label style={font=\\footnotesize}, label style={font=\\small}]"]
for i, k in enumerate(pdt.ORDER, start=1):
    pts = " ".join(f"({i + dx:.2f},{v})" for dx, v in zip(jitter(D[k]), D[k]))
    L.append(f"  \\addplot[only marks, mark=*, mark size=1.7pt, {COLOUR[k]}, fill opacity=0.8] coordinates {{{pts}}};")
L += ["  \\addplot[only marks, mark=-, mark size=7pt, very thick, red!75!black, error bars/.cd, y dir=both, y explicit, error bar style={thick}]",
      "    table[row sep=crcr, x=x, y=y, y error plus=ep, y error minus=em] {", "      x y ep em\\\\"]
for i, k in enumerate(pdt.ORDER, start=1):
    m = float(np.mean(D[k])); lo, hi = pdt.boot_ci(D[k])
    L.append(f"      {i} {m:.2f} {hi - m:.2f} {m - lo:.2f}\\\\")
L += ["    };", "\\end{axis}", "\\end{tikzpicture}"]
head = ["\\documentclass[11pt,border=0pt]{standalone}",
        "% Standalone source of the per-seed strip chart (written by scripts/make_tables.py, do not edit by hand).",
        "\\input{preamble}", "\\begin{document}", "\\setlength{\\linewidth}{452.9679pt}"]
open(os.path.join(HERE, "..", "figures", "tikz", "fig_strip.tex"), "w").write("\n".join(head + L + ["\\end{document}"]) + "\n")
print("wrote ladder_rows.tex, pvalue_rows.tex, figures/tikz/fig_strip.tex")
