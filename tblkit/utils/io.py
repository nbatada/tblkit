# io.py — REPLACE THE WHOLE FILE WITH THIS
from __future__ import annotations
import sys, math
import pandas as pd
from typing import Optional

def pretty_print(df: pd.DataFrame, *, args=None, stream: str = "stdout") -> None:
    """
    ASCII table preview with MySQL-style borders (non-folding).
    Honors:
      - args.max_cols      : clip to first N columns (if provided)
      - args.max_col_width : truncate cells to width (default 40)
      - args.show_full     : disable truncation (show full cell text)
    NOTE: No row limiting; pipe through `head`/`tail` as desired.
    """
    # Column clipping
    max_cols = int(getattr(args, "max_cols", 0) or 0)
    df2 = df.iloc[:, :max_cols] if max_cols > 0 else df

    # Truncation policy
    max_col_width = None if getattr(args, "show_full", False) else int(getattr(args, "max_col_width", 40) or 40)

    # Prepare data
    headers = [str(c) for c in df2.columns]
    def cell(s):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            txt = ""
        else:
            txt = str(s)
        if max_col_width and len(txt) > max_col_width:
            return txt[: max(1, max_col_width - 1)] + "…"
        return txt

    # We print all rows (no head())—user controls via pipe
    rows = [[cell(v) for v in row] for row in df2.itertuples(index=False, name=None)]

    # Column widths (header-aware)
    widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))

    def hline(ch="-"):
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def render_row(vals):
        return "|" + "|".join(" " + v.ljust(w) + " " for v, w in zip(vals, widths)) + "|"

    out = sys.stdout if stream == "stdout" else sys.stderr
    try:
        out.write(hline() + "\n")
        out.write(render_row(headers) + "\n")
        out.write(hline() + "\n")
        for r in rows:
            out.write(render_row(r) + "\n")
        out.write(hline() + "\n")
    except BrokenPipeError:
        # Quiet when consumer (head/less) closes the pipe early
        return

def write_table(df: pd.DataFrame, path: Optional[str] = None, *, sep: str = "\t",
                index: bool = False, header: bool = True,
                encoding: str = "utf-8", na_rep: str = "") -> None:
    """Write a table to a file or stdout; remain quiet on BrokenPipe."""
    out = sys.stdout if path in (None, "-") else open(path, "w", encoding=encoding)
    close = (out is not sys.stdout)
    try:
        df.to_csv(out, sep=sep, index=index, header=header, na_rep=na_rep)
    except BrokenPipeError:
        return
    finally:
        if close:
            try:
                out.close()
            except Exception:
                pass
