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
    Robust reader with auto-detect + diagnostics.
    On parse errors, reports line, expected vs seen fields, and a caret under the approximate offending column.
    Also handles empty stdin and naive CSV→TSV warnings.
    """
    import io as _io, re as _re, csv as _csv, sys as _sys
    from pandas.errors import ParserError, EmptyDataError

    def _norm(s):
        if s is None: return None
        t = str(s).lower()
        return {"csv": ",", ",": ",", "tsv": "\t", "tab": "\t", "\\t": "\t",
                "pipe": "|", "bar": "|", "|": "|", "space": r"\s+", "spaces": r"\s+",
                "whitespace": r"\s+", "auto": None, "guess": None}.get(t, s)

    def _detect(text: str) -> str:
        sample = "\n".join([ln for ln in text.splitlines() if ln.strip()][:200])
        stripped = _re.sub(r'"[^"\n]*"', "", sample)
        counts = {"\t": sample.count("\t"), ",": stripped.count(","), "|": stripped.count("|")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else r"\s+"

    where = "stdin" if path in (None, "-") else str(path)
    raw = None
    if path in (None, "-"):
        raw = _io.TextIOWrapper(_sys.stdin.buffer, encoding=encoding).read()
        if raw == "":
            raise ValueError("No input detected on stdin. Upstream produced no data (e.g., missing file) — pipe a table or use -i <file>.")

    req = _norm(sep)
    use_sep = req if req is not None else _detect(raw or "")

    # Warn for naive CSV→TSV via `tr` (thousands split)
    if raw is not None and use_sep == "\t":
        if _re.search(r"\b\d{1,3}\t\d{3}(?:\.\d+)?\b", raw):
            _sys.stderr.write(
                "tblkit: input looks like CSV converted with `tr`, which splits thousands like 1,521.64 -> 1\\t521.64.\n"
                "        Read CSV directly (e.g., --sep csv) or use a CSV-aware converter.\n"
            )

    quoting = _csv.QUOTE_MINIMAL
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

    def _diagnose_parse_error(e: Exception, active_sep: str) -> str:
        if raw is None:
            return f"{e}"
        lines = raw.splitlines()
        exp = None
        if header == 0 and lines:
            rdr = csv.reader([lines[0]], delimiter=(active_sep if len(active_sep) == 1 else ","), quotechar='"', escapechar="\\")
            exp = len(next(rdr, []))
        m = _re.search(r"[Ll]ine\s+(\d+)", str(e))
        line_no = int(m.group(1)) if m else None
        if line_no is None and exp is not None:
            for i, ln in enumerate(lines[1:], start=2):
                rdr = csv.reader([ln], delimiter=(active_sep if len(active_sep) == 1 else ","), quotechar='"', escapechar="\\")
                seen = len(next(rdr, []))
                if seen != exp:
                    line_no = i; break
        snippet, caret, seen = "", "", None
        if line_no is not None and 1 <= line_no <= len(lines):
            ln = lines[line_no - 1]; snippet = ln[:400]
            try:
                rdr = csv.reader([ln], delimiter=(active_sep if len(active_sep) == 1 else ","), quotechar='"', escapechar="\\")
                fields = next(rdr, []); seen = len(fields)
                bad_idx = exp if (exp is not None and seen > exp) else (seen if (exp is not None and seen < exp) else None)
                if bad_idx is not None:
                    cur = 0; delim = ("\t" if active_sep == "\t" else (active_sep if len(active_sep) == 1 else ","))
                    for _ in range(bad_idx):
                        pos = ln.find(delim, cur); 
                        if pos == -1: break
                        cur = pos + len(delim)
                    caret = " " * max(0, cur) + "^"
            except Exception:
                pass
        parts = []
        if line_no is not None: parts.append(f"line {line_no}")
        if exp is not None and seen is not None: parts.append(f"expected {exp} fields, saw {seen}")
        msg = f"Failed to read table from {where}: " + (", ".join(parts) if parts else str(e))
        if snippet:
            msg += f"\n  {snippet}\n  {caret if caret else ''}"
        msg += "\nHint: use --sep (csv/tsv/|/space), or --on-bad-lines skip|warn; check unclosed quotes."
        return msg

    try:
        df = _read(use_sep)
    except ParserError as e:
        for a in (s for s in ("\t", ",", "|", r"\s+") if s != use_sep):
            if isinstance(source, _io.StringIO): source.seek(0)
            try:
                df = _read(a); use_sep = a; break
            except ParserError:
                continue
        else:
            raise ValueError(_diagnose_parse_error(e, use_sep))
    except EmptyDataError:
        raise ValueError(f"Failed to read table from {where}: no data found.")
    except StopIteration as e:
        raise ValueError(_diagnose_parse_error(e, use_sep))

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
      - args.max_cols       : preview only first N columns
      - args.max_col_width  : truncate cells to this display width (default 40)
      - args.show_full      : disable truncation
    """
    import sys, re, warnings # <--- 1. Import warnings
    from wcwidth import wcswidth

    max_cols = int(getattr(args, "max_cols", 0) or 0)
    df2 = df.iloc[:, :max_cols] if max_cols > 0 else df

    max_col_width = None if getattr(args, "show_full", False) else int(getattr(args, "max_col_width", 40) or 40)

    # pick an ellipsis that the terminal can actually encode
    ell = "…"
    try:
        (_ := (sys.stdout if stream == "stdout" else sys.stderr).encoding)  # encoding may be None
        (ell.encode(_ or "utf-8"))
    except Exception:
        ell = "..."

    def _coerce(x) -> str:
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8", "replace")
            except Exception:
                return x.decode(errors="replace")
        s = str(x) if x is not None else ""
        s = s.replace("\r", "").replace("\n", "⏎")
        # strip control chars (except tab) to keep widths stable
        return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", s)

    headers = [ _coerce(c) for c in df2.columns ]
    
    # --- 2. Add a context manager to suppress the warning ---
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        rows = [ [ _coerce(v) for v in row ] for row in df2.astype(object).fillna("").to_numpy().tolist() ]
    # ---------------------------------------------------------

    def clip(s: str, wmax: int | None) -> str:
        if wmax is None:
            return s
        w = wcswidth(s)
        if w <= wmax:
            return s
        keep = wmax - wcswidth(ell)
        if keep <= 0:
            return ell if wmax >= wcswidth(ell) else "." * min(3, max(0, wmax))
        out = ""
        for ch in s:
            if wcswidth(out + ch) > keep:
                break
            out += ch
        return out + ell

    disp_headers = [clip(h, max_col_width) for h in headers]
    disp_rows = [[clip(v, max_col_width) for v in r] for r in rows]

    # column widths by display width
    cols = list(zip(*([disp_headers] + disp_rows))) if disp_headers else []
    widths = [max(wcswidth(x) for x in col) for col in cols] if cols else []

    def hline() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def render_row(vals) -> str:
        cells = []
        for v, w in zip(vals, widths):
            pad = " " * (w - wcswidth(v))
            cells.append(" " + v + pad + " ")
        return "|" + "|".join(cells) + "|"

    out = sys.stdout if stream == "stdout" else sys.stderr
    try:
        if widths:
            out.write(hline() + "\n")
            out.write(render_row(disp_headers) + "\n")
            out.write(hline() + "\n")
            for r in disp_rows:
                out.write(render_row(r) + "\n")
            out.write(hline() + "\n")
        else:
            out.write("(empty table)\n")
    except BrokenPipeError:
        return
    
