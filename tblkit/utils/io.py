from __future__ import annotations
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import io as _io
import csv
import re
from typing import Optional
import sys

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
               sep: str | None = None,
               header: Optional[int] = 0,
               encoding: str = "utf-8",
               na_values=None,
               on_bad_lines: str | None = "error") -> pd.DataFrame:
    """
    Robust reader with delimiter auto-detect (when sep is None) and quote-respecting parse.
    Persists the effective input delimiter on df.attrs['tblkit_sep'] so writers can
    default to “same as input”. Warns on naive CSV→TSV via `tr` (thousands split).
    """
    import io as _io, re as _re, csv as _csv, sys as _sys
    from pandas.errors import ParserError, EmptyDataError

    def _norm(s):
        if s is None: return None
        t = str(s).lower()
        return {
            "csv": ",", ",": ",",
            "tsv": "\t", "tab": "\t", "\\t": "\t",
            "pipe": "|", "bar": "|", "|": "|",
            "space": r"\s+", "spaces": r"\s+", "whitespace": r"\s+",
            "auto": None, "guess": None,
        }.get(t, s)

    def _detect(text: str) -> str:
        # Prefer TAB if present; otherwise COMMA; then PIPE; else whitespace.
        sample = "\n".join([ln for ln in text.splitlines() if ln.strip()][:200])
        stripped = _re.sub(r'"[^"\n]*"', "", sample)  # ignore commas inside simple quotes
        counts = { "\t": sample.count("\t"), ",": stripped.count(","), "|": stripped.count("|") }
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else r"\s+"

    where = "stdin" if path in (None, "-") else str(path)
    raw = None
    if path in (None, "-"):
        raw = _io.TextIOWrapper(_sys.stdin.buffer, encoding=encoding).read()

    req = _norm(sep)
    use_sep = req if req is not None else _detect(raw or "")

    # Early warning for naive CSV→TSV (thousands split by `tr`)
    if raw is not None and use_sep == "\t":
        if _re.search(r"\b\d{1,3}\t\d{3}(?:\.\d+)?\b", raw):
            _sys.stderr.write(
                "tblkit: input looks like CSV converted with `tr`, which splits thousands like 1,521.64 -> 1\\t521.64.\n"
                "        Read CSV directly (e.g., `tblkit view --sep csv`) or use a CSV-aware converter.\n"
            )

    quoting = _csv.QUOTE_MINIMAL  # always honor quotes
    engine = "python" if use_sep == r"\s+" else None
    source = _io.StringIO(raw) if raw is not None else where

    def _read(try_sep: str) -> pd.DataFrame:
        return pd.read_csv(
            source,
            sep=try_sep,
            header=header,
            encoding=encoding,
            na_values=na_values,
            engine="python" if (engine or try_sep == r"\s+") else None,
            quoting=quoting,
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            on_bad_lines=on_bad_lines,
            compression="infer",
        )

    try:
        df = _read(use_sep)
    except ParserError:
        # Retry alternates if detection guessed poorly
        alts = [s for s in ("\t", ",", "|", r"\s+") if s != use_sep]
        last = None
        for a in alts:
            if isinstance(source, _io.StringIO): source.seek(0)
            try:
                df = _read(a); use_sep = a; break
            except ParserError as e:
                last = e
        else:
            raise last

    try: df.attrs["tblkit_sep"] = use_sep
    except Exception: pass
    return df


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
    from wcwidth import wcswidth

    max_cols = int(getattr(args, "max_cols", 0) or 0)
    df2 = df.iloc[:, :max_cols] if max_cols > 0 else df

    max_col_width = None if getattr(args, "show_full", False) else int(getattr(args, "max_col_width", 40) or 40)

    headers = [str(c) for c in df2.columns]

    def cell(s):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            txt = ""
        else:
            txt = str(s)
        # Truncate based on visual width
        if max_col_width and wcswidth(txt) > max_col_width:
            width = 0
            end_pos = 0
            for i, char in enumerate(txt):
                width += wcswidth(char)
                if width >= max_col_width:
                    end_pos = i
                    break
            return txt[:end_pos] + "…"
        return txt

    rows = [[cell(v) for v in row] for row in df2.itertuples(index=False, name=None)]

    widths = [wcswidth(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], wcswidth(v))

    def hline(ch="-"):
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def render_row(vals):
        cells = []
        for v, w in zip(vals, widths):
            padding = " " * (w - wcswidth(v))
            cells.append(" " + v + padding + " ")
        return "|" + "|".join(cells) + "|"

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
    
