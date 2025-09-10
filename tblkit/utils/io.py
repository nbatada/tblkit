from __future__ import annotations
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import io as _io
import csv
import re
from typing import Optional
import sys

# /mnt/data/io.py — REPLACE the whole function
def _normalize_sep(sep: Optional[str]) -> str:
    """
    Map common tokens to actual separators; pass through others verbatim.
    Accepts: csv, tsv, tab, pipe, bar, space/spaces, '\t', ',', '|', regex.
    """
    if sep is None:
        return "\t"
    s = str(sep).strip()
    low = s.lower()
    if low in {"csv", "comma"}: return ","
    if low in {"tsv", "tab"}:   return "\t"
    if s == r"\t":              return "\t"
    if low in {"pipe", "bar"}:  return "|"
    if low in {"space","spaces","whitespace"}: return r"\s+"
    if low in {"auto", "guess"}: return "\t"  # default to TSV
    return s

# /mnt/data/io.py — REPLACE ENTIRE FUNCTION
def read_table(path: Optional[str],
               *,
               sep: str | None = None,
               header: Optional[int] = 0,
               encoding: str = "utf-8",
               na_values=None,
               on_bad_lines: str | None = "error",
               nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Robust reader with optional partial read (nrows) and clearer diagnostics.
    Default separator is TSV; 'auto'/'guess' triggers lightweight detection.
    """
    import io as _io, re as _re, csv as _csv, sys as _sys
    from pandas.errors import ParserError, EmptyDataError

    def _norm(s):
        if s is None: return None
        t = str(s).strip().lower()
        return {
            "tsv": "\t", "tab": "\t", "\\t": "\t",
            "csv": ",", ",": ",",
            "pipe": "|", "bar": "|", "|": "|",
            "space": r"\s+", "spaces": r"\s+", "whitespace": r"\s+",
            "auto": None, "guess": None
        }.get(t, s)

    def _detect(text: str) -> str:
        sample = "\n".join([ln for ln in text.splitlines() if ln.strip()][:200])
        stripped = _re.sub(r'"[^"\n]*"', "", sample)
        counts = {"\t": sample.count("\t"), ",": stripped.count(","), "|": stripped.count("|")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else "\t"

    where = "stdin" if path in (None, "-") else str(path)

    # Fast path: explicit non-auto sep on a real file
    req_sep = _norm(sep)
    if path not in (None, "-") and req_sep not in (None, r"\s+"):
        return pd.read_csv(
            where, sep=req_sep, header=header, encoding=encoding,
            na_values=na_values, on_bad_lines=on_bad_lines, nrows=nrows
        )

    # Otherwise, read a text buffer (stdin or for auto-detect)
    if path in (None, "-"):
        raw = _io.TextIOWrapper(_sys.stdin.buffer, encoding=encoding).read()
        if raw == "":
            raise ValueError("No input detected on stdin. Pipe a table or use -i <file>.")
    else:
        with open(where, "r", encoding=encoding, errors="replace") as fh:
            raw = fh.read()

    use_sep = _detect(raw) if req_sep is None else req_sep
    engine = "python" if use_sep == r"\s+" else None
    source = _io.StringIO(raw)

    try:
        return pd.read_csv(
            source, sep=use_sep, header=header, encoding=encoding,
            na_values=na_values, on_bad_lines=on_bad_lines, nrows=nrows,
            engine=engine
        )
    except EmptyDataError:
        raise ValueError(f"Failed to read table from {where}: empty input.")
    except ParserError as e:
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
            out_sep = " " if out_sep == r"\s+" else "\t"
        df.to_csv(out, sep=out_sep, index=index, header=header, na_rep=na_rep)
    except BrokenPipeError:
        return
    finally:
        if close:
            try: out.close()
            except Exception: pass

def pretty_print(df: pd.DataFrame, *, args=None, stream: str = "stdout") -> None:
    """
    ASCII table preview with MySQL-style borders (non-folding).
    Honors:
      - args.max_cols       : preview only first N columns
      - args.max_col_width  : truncate cells to this display width (default 40)
      - args.show_full      : disable truncation
    """
    import sys, re, warnings, os
    from wcwidth import wcswidth
    import pandas as pd

    out_stream = sys.stdout if stream == "stdout" else sys.stderr
    supports_color = out_stream.isatty() and os.environ.get("NO_COLOR") is None

    C_RESET = "\033[0m" if supports_color else ""
    C_BLUE = "\033[94m" if supports_color else ""
    C_RED = "\033[91m" if supports_color else ""
    C_WHITE = "\033[97m" if supports_color else ""

    max_cols = int(getattr(args, "max_cols", 0) or 0)
    df2 = df.iloc[:, :max_cols] if max_cols > 0 else df
    
    # --- START: New Smart Numeric Detection Logic ---
    NUM_LIKE_RE = re.compile(r"^\s*[\$]?[-+]?((?:\d{1,3}(?:,\d{3})*)|\d+)(?:\.\d+)?%?\s*$")

    def is_numeric_like(series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series.dtype):
            return True
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            # Check a sample of non-NA values
            sample = series.dropna().head(20)
            if len(sample) == 0:
                return False
            return all(isinstance(v, str) and NUM_LIKE_RE.match(v) for v in sample)
        return False

    display_as_numeric = [is_numeric_like(df2[col]) for col in df2.columns]
    # --- END: New Smart Numeric Detection Logic ---

    max_col_width = None if getattr(args, "show_full", False) else int(getattr(args, "max_col_width", 40) or 40)

    ell = "…"
    try:
        (ell.encode((out_stream.encoding or "utf-8")))
    except Exception:
        ell = "..."

    def _coerce(x) -> str:
        if isinstance(x, bytes):
            try: return x.decode("utf-8", "replace")
            except Exception: return x.decode(errors="replace")
        s = str(x) if x is not None else ""
        s = s.replace("\r", "").replace("\n", "⏎")
        return re.sub(r"[\x00-\x08\x0b-\x1f-\x7f]", "", s)

    def clip(s: str, wmax: int | None) -> str:
        if wmax is None: return s
        w = wcswidth(s)
        if w <= wmax: return s
        keep = wmax - wcswidth(ell)
        if keep <= 0: return ell if wmax >= wcswidth(ell) else "." * min(3, max(0, wmax))
        out = ""
        for ch in s:
            if wcswidth(out + ch) > keep: break
            out += ch
        return out + ell

    # --- START: New Decimal Padding Logic ---
    max_decimal_places = {}
    for i, col_name in enumerate(df2.columns):
        if display_as_numeric[i]:
            max_dp = 0
            for val in df2[col_name].dropna():
                s_val = str(val).strip()
                if '.' in s_val:
                    max_dp = max(max_dp, len(s_val.split('.')[-1].rstrip('%')))
            max_decimal_places[col_name] = max_dp
            
    def format_numeric_string(val, col_idx):
        col_name = df2.columns[col_idx]
        s = str(val).strip()
        
        has_dollar = s.startswith('$')
        has_percent = s.endswith('%')
        
        # Clean the string to its numeric core for formatting
        num_part = s.replace('$', '').replace(',', '').rstrip('%')
        
        try:
            num = float(num_part)
            max_dp = max_decimal_places.get(col_name, 0)
            formatted_num = f"{num:.{max_dp}f}"
            
            # Re-apply prefixes/suffixes
            if has_dollar: formatted_num = f"${formatted_num}"
            if has_percent: formatted_num = f"{formatted_num}%"
            return formatted_num
        except (ValueError, TypeError):
            return s # Not a number after all, return as is

    uncolored_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        for row_idx, row in enumerate(df2.to_numpy().tolist()):
            formatted_row = []
            for col_idx, v in enumerate(row):
                if pd.isna(v):
                    formatted_row.append("<NA>")
                elif display_as_numeric[col_idx]:
                    formatted_row.append(format_numeric_string(v, col_idx))
                else:
                    formatted_row.append(_coerce(v))
            uncolored_rows.append(formatted_row)
    # --- END: New Decimal Padding Logic ---

    headers = [_coerce(c) for c in df2.columns]
    disp_headers = [clip(h, max_col_width) for h in headers]
    disp_rows_uncolored = [[clip(v, max_col_width) for v in r] for r in uncolored_rows]

    cols_uncolored = list(zip(*([disp_headers] + disp_rows_uncolored))) if disp_headers else []
    widths = [max(wcswidth(x) for x in col) for col in cols_uncolored] if cols_uncolored else []

    disp_rows_colored = []
    for i, row in enumerate(df2.to_numpy().tolist()):
        colored_row = []
        for j, cell in enumerate(row):
            uncolored_clipped_cell = disp_rows_uncolored[i][j]
            color = C_WHITE
            if pd.isna(cell):
                color = C_RED
            elif display_as_numeric[j]:
                color = C_BLUE
            colored_row.append(f"{color}{uncolored_clipped_cell}{C_RESET}")
        disp_rows_colored.append(colored_row)

    def hline() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def render_row(vals, uncolored_vals, is_header=False) -> str:
        cells = []
        for i, w in enumerate(widths):
            v_colored = vals[i]
            v_uncolored = uncolored_vals[i]
            pad = " " * (w - wcswidth(v_uncolored))
            
            if not is_header and display_as_numeric[i]:
                cells.append(" " + pad + v_colored + " ")  # Right-align
            else:
                cells.append(" " + v_colored + pad + " ")  # Left-align
        return "|" + "|".join(cells) + "|"

    try:
        if widths:
            out_stream.write(hline() + "\n")
            out_stream.write(render_row(disp_headers, disp_headers, is_header=True) + "\n")
            out_stream.write(hline() + "\n")
            for i in range(len(disp_rows_colored)):
                out_stream.write(render_row(disp_rows_colored[i], disp_rows_uncolored[i], is_header=False) + "\n")
            out_stream.write(hline() + "\n")
        else:
            out_stream.write("(empty table)\n")
    except BrokenPipeError:
        return
    
    
def pretty_print_old(df: pd.DataFrame, *, args=None, stream: str = "stdout") -> None:
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
    
