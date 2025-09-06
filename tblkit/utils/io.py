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
               sep: str | None = "\t",
               header: Optional[int] = 0,
               encoding: str = "utf-8",
               na_values=None,
               on_bad_lines: str | None = "error") -> pd.DataFrame:
    """
    Robust reader: auto-detect delimiter for stdin/files; always respect quotes.
    Handles CSV/TSV/pipe/whitespace and retries sensible fallbacks on parser errors.
    """
    import io as _io, re as _re, csv as _csv
    from pandas.errors import EmptyDataError, ParserError

    def _norm(s):
        if s is None:
            return None
        low = str(s).lower()
        if low in {"auto", "guess"}:
            return None
        if low in {"csv", "comma", ","}:
            return ","
        if low in {"tsv", "tab", "\\t"}:
            return "\t"
        if low in {"pipe", "bar", "|"}:
            return "|"
        if low in {"space", "spaces", "whitespace"}:
            return r"\s+"
        return s

    def _detect(text):
        lines = [ln for ln in text.splitlines() if ln.strip()][:200]
        sample = "\n".join(lines)
        # Count candidates outside of obvious quoted regions (lightweight)
        def _score(ch):
            # remove simple quoted spans to avoid inflated counts
            stripped = _re.sub(r'"[^"\n]*"', "", sample)
            return stripped.count(ch)
        tab, com, pipe, sp = sample.count("\t"), _score(","), _score("|"), len(_re.findall(r"[ ]{1,}", sample))
        if tab > 0 and tab >= com and tab >= pipe:
            return "\t"
        if com > 0 and com >= pipe:
            return ","
        if pipe > 0:
            return "|"
        # Fallback: treat runs of whitespace as separator
        return r"\s+"

    # Read stdin (text) or use a file path; also keep a peek for detection
    where = "stdin" if path in (None, "-") else str(path)
    text = None
    if path in (None, "-"):
        text = _io.TextIOWrapper(__import__("sys").stdin.buffer, encoding=encoding).read()

    requested = _norm(sep)
    use_sep = requested if requested is not None else _detect(text if text is not None else open(where, "r", encoding=encoding).read(100_000))

    # Choose engine
    need_python = bool(use_sep) and (len(use_sep) > 1 or use_sep == r"\s+")
    quoting = _csv.QUOTE_MINIMAL  # respect quotes for ALL seps

    # Prepare source for pandas
    source = _io.StringIO(text) if text is not None else where

    def _read(try_sep):
        return pd.read_csv(
            source,
            sep=try_sep,
            header=header,
            encoding=encoding,
            na_values=na_values,
            engine="python" if (need_python or try_sep == r"\s+") else None,
            quoting=quoting,
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            on_bad_lines=on_bad_lines,
            compression="infer",
        )

    # First attempt
    try:
        return _read(use_sep)
    except ParserError:
        # Retry alternates if auto/guess likely went wrong
        alternates = []
        for cand in ("\t", ",", "|", r"\s+"):
            if cand != use_sep:
                alternates.append(cand)
        for cand in alternates:
            source.seek(0) if isinstance(source, _io.StringIO) else None
            try:
                return _read(cand)
            except ParserError:
                continue
        raise
    except (EmptyDataError, ParserError) as e:
        raise ValueError(f"Failed to read table from {where}: {e}") from e


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
    
