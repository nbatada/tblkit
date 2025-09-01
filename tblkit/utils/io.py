from __future__ import annotations
import sys
import re
import pandas as pd
from typing import Optional

def _normalize_sep(sep: Optional[str]) -> str:
    """
    Map common tokens to actual separators; pass through others verbatim.
    Accepts: csv, tsv, tab, pipe, bar, space/spaces, '\t', ',', '|', regex.
    """
    if sep is None:
        return "\t"
    s = str(sep).strip()
    low = s.lower()
    if low in {"csv", "comma"}:
        return ","
    if low in {"tsv", "tab"}:
        return "\t"
    if s == r"\t":
        return "\t"
    if low in {"pipe", "bar"}:
        return "|"
    if low in {"space", "spaces", "whitespace"}:
        # robust to runs of spaces/tabs
        return r"\s+"
    return s  # literal or regex (multi-char ok)

def read_table(path: Optional[str],
               *,
               sep: str = "\t",
               header: Optional[int] = 0,
               encoding: str = "utf-8",
               na_values=None,
               on_bad_lines: str | None = "error") -> pd.DataFrame:
    """
    Read a delimited table from a path or stdin.
      - path None or "-" → read from stdin
      - sep: token or literal or regex (csv, tsv, tab, pipe, space, '\\t', ',', etc.)
      - header=0 includes header; header=None for headerless
      - on_bad_lines: 'error' | 'warn' | 'skip' (pandas >=1.3)
    """
    import io as _io
    from pandas.errors import EmptyDataError, ParserError

    use_sep = _normalize_sep(sep)
    if not use_sep:
        use_sep = ","
    # If multi-char and not obviously a regex, coerce to comma to avoid csv-engine errors.
    if len(use_sep) > 1 and not re.search(r"[\\\[\]\+\*\?\|\(\)]", use_sep):
        use_sep = ","
        
    need_python = (len(use_sep) > 1) or (use_sep.startswith("\\")
                    or bool(re.search(r"\\|[\[\]\+\*\?\|\(\)]", use_sep)))

    # Buffer stdin so we can detect empties/404/HTML and still parse once.
    if path in (None, "-"):
        raw = sys.stdin.buffer.read()
        try:
            text = raw.decode(encoding, errors="replace")
        except Exception:
            text = raw.decode("utf-8", errors="replace")

        sample = text.strip()
        if not sample:
            raise ValueError("No input data received on stdin.")
        if len(sample) <= 64 and "not found" in sample.lower():
            raise ValueError("Input looks like a 404/Not Found payload. Check the URL/path.")
        if "<html" in sample[:256].lower() or "<!doctype" in sample[:256].lower():
            raise ValueError("Input looks like HTML, not a delimited table. Use a raw file URL.")
        source = _io.StringIO(text)
        where = "stdin"
    else:
        source = path
        where = str(path)

    try:
        df = pd.read_csv(
            source,
            sep=use_sep,
            header=header,
            encoding=encoding,
            na_values=na_values,
            on_bad_lines=on_bad_lines,  # pandas >= 1.3
            engine="python" if need_python else None,
            compression="infer",
        )
        return df
    except TypeError as te:
        if "on_bad_lines" in str(te):
            df = pd.read_csv(
                source,
                sep=use_sep,
                header=header,
                encoding=encoding,
                na_values=na_values,
                engine="python" if need_python else None,
                compression="infer",
            )
            return df
        raise
    except (EmptyDataError, ParserError) as e:
        raise ValueError(f"Failed to read table from {where}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to read table from {where}: {e}") from e

def pretty_print(df: pd.DataFrame, *, args=None, stream: str = "stdout") -> None:
    """
    ASCII table preview with MySQL-style borders (non-folding).
    Honors:
      - args.max_cols      : clip to first N columns (if provided)
      - args.max_col_width : truncate cells to width (default 40)
      - args.show_full     : disable truncation (show full cells)
    NOTE: No row limiting; pipe through `head`/`tail` as desired.
    """
    import math

    max_cols = int(getattr(args, "max_cols", 0) or 0)
    df2 = df.iloc[:, :max_cols] if max_cols > 0 else df

    max_col_width = None if getattr(args, "show_full", False) else int(getattr(args, "max_col_width", 40) or 40)

    headers = [str(c) for c in df2.columns]

    def cell(s):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            txt = ""
        else:
            txt = str(s)
        if max_col_width and len(txt) > max_col_width:
            return txt[: max(1, max_col_width - 1)] + "…"
        return txt

    rows = [[cell(v) for v in row] for row in df2.itertuples(index=False, name=None)]

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
        return

def write_table(df: pd.DataFrame, path: Optional[str] = None, *,
                sep: str = "\t",
                index: bool = False,
                header: bool = True,
                encoding: str = "utf-8",
                na_rep: str = "") -> None:
    """Write a table to a file or stdout; remain quiet on BrokenPipe."""
    out = sys.stdout if path in (None, "-") else open(path, "w", encoding=encoding)
    close = (out is not sys.stdout)
    try:
        out_sep = _normalize_sep(sep)
        # pandas.to_csv requires a single-character delimiter (no regex).
        if not out_sep or len(out_sep) != 1:
            if out_sep == r"\s+":
                out_sep = " "
            else:
                out_sep = (str(out_sep)[:1] or ",")
        df.to_csv(out, sep=out_sep, index=index, header=header, na_rep=na_rep)
    except BrokenPipeError:
        return
    finally:
        if close:
            try:
                out.close()
            except Exception:
                pass
            
