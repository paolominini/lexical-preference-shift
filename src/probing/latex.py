"""Formatting helpers for the thesis tables and figures: LaTeX numbers, p-values, booktabs tables, display labels.

Pure string functions, no I/O except `write_fragment`. Numbers are put in math mode so that minus signs render as minus signs.
"""
from __future__ import annotations

import math
from pathlib import Path

MODEL_LABELS = {"base": "Base", "sft": "SFT", "sft_dpo": "SFT-DPO", "sft_ppo": "SFT-PPO", "sft_kto": "SFT-KTO",
                "kto": "Direct KTO"}
_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def escape_tex(text: str) -> str:
    """Escape the LaTeX special characters of plain text."""
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(text))


def display_labels(cfg: dict) -> dict[str, str]:
    """Internal category name -> thesis display label (`config.yaml` `display_labels`)."""
    return dict(cfg["display_labels"])


def ordered_categories(cfg: dict) -> list[str]:
    """Internal category names sorted by display label (the row/axis order of every thesis table and figure)."""
    labels = display_labels(cfg)
    return sorted(labels, key=labels.get)


def fmt_num(x: float, nd: int = 2, signed: bool = False, math_mode: bool = True) -> str:
    """Round to `nd` decimals; `signed` adds '+' to positives; never prints '-0.00' (a value that rounds to zero is unsigned)."""
    r = round(float(x), nd)
    if r == 0:
        r = 0.0
    s = f"{r:+.{nd}f}" if signed and r != 0 else f"{r:.{nd}f}"
    return f"${s}$" if math_mode else s


def fmt_ci(lo: float, hi: float, nd: int = 2) -> str:
    """Confidence interval as '$[lo, hi]$'."""
    return f"$[{fmt_num(lo, nd, math_mode=False)}, {fmt_num(hi, nd, math_mode=False)}]$"


def fmt_range(lo: float, hi: float, nd: int = 2, signed: bool = False) -> str:
    """Range 'lo--hi' (en dash), both rounded to `nd` decimals; `signed` as in `fmt_num`."""
    return f"${fmt_num(lo, nd, signed, math_mode=False)}$--${fmt_num(hi, nd, signed, math_mode=False)}$"


def _sci_parts(p: float) -> tuple[str, int]:
    """Mantissa with one decimal and exponent of p, after rounding (9.96e-5 -> ('1.0', -4))."""
    mant, exp = f"{p:.1e}".split("e")
    return mant, int(exp)


def fmt_p(p: float) -> str:
    """p-value: >= 0.001 with two significant digits ('$0.26$', '$0.0015$'), 1 as '$1.00$', smaller as '$1.7 \\times 10^{-44}$'."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    p = float(p)
    if p >= 0.995:
        return "$1.00$"
    if p >= 0.001:
        return f"${p:.2g}$" if p < 0.1 else f"${p:.2f}$"
    mant, exp = _sci_parts(p)
    return f"${mant} \\times 10^{{{exp}}}$"


def fmt_p_bound(p: float) -> str:
    """Upper bound of a set of p-values, '$\\le 7.0 \\times 10^{-8}$' (used for ranges over models)."""
    return fmt_p(p).replace("$", "$\\le ", 1)


def tabular(colspec: str, header: list[str], rows: list[list[str]], midrules: tuple[int, ...] = (),
            header_lines: tuple[str, ...] = ()) -> str:
    """booktabs tabular. `midrules` = row indices before which a \\midrule is inserted; `header_lines` = raw LaTeX lines above
    `header` (e.g. a \\multicolumn group line and its \\cmidrule).

    Raises ValueError if a row does not have as many cells as `header`.
    """
    n = len(header)
    for i, r in enumerate(rows):
        if len(r) != n:
            raise ValueError(f"row {i} has {len(r)} cells, header has {n}")
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *header_lines]
    lines += [" & ".join(header) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append("\\midrule")
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_env(body: str, caption: str, label: str, note: str | None = None, size: str = "small",
              placement: str = "htbp", short: str | None = None) -> str:
    """Wrap a tabular in a floating table. `caption` is the short title (also the List of Tables entry); the optional `note`
    follows it inside the caption, `\\caption[title]{title. note}` (no separate note block under the table).
    `short`, if given, replaces the title in the List of Tables only (caption text that does not start with the short title)."""
    title = caption.rstrip(".")
    full = title + ("" if title.endswith("?") else ".") + (f" {note}" if note else "")
    lines = [f"\\begin{{table}}[{placement}]", "\\centering", f"\\{size}", f"\\caption[{{{short or title}}}]{{{full}}}",
             f"\\label{{{label}}}", body, "\\end{table}"]
    return "\n".join(lines)


def write_fragment(path: Path, text: str, sources: list[str], generator: str = "scripts/11_thesis_tables.py") -> None:
    """Write a .tex fragment with a provenance header (generator and source files); no date, so reruns are byte-identical."""
    head = [f"% Generated by {generator} - do not edit by hand; rerun the script instead.",
            "% Sources: " + ", ".join(sources)]
    path.write_text("\n".join(head) + "\n" + text + "\n")
