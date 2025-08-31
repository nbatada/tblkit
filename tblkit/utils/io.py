from __future__ import annotations
import sys
import pandas as pd
from typing import Optional

def pretty_print(df: pd.DataFrame, max_rows: int = 10, max_cols: int = 20) -> None:
    """Print a small preview to stderr; useful in debugging and help flows."""
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", max_cols):
        print(df.head(max_rows), file=sys.stderr)

def read_table(path: Optional[str] = None, *, sep: str = "\t",
               header: bool | int = "infer", dtype: Optional[dict] = None,
               encoding: str = "utf-8", na_values: Optional[list[str]] = None,
               on_bad_lines: str = "error") -> pd.DataFrame:
    """Read a table from a file or stdin (if path is None or "-")."""
    src = sys.stdin if path in (None, "-") else path
    if header is True:
        header = 0
    if header is False:
        header = None
    return pd.read_csv(
        src,
        sep=sep,
        header=header,
        dtype=dtype,
        encoding=encoding,
        na_values=na_values,
        on_bad_lines=on_bad_lines,
    )

def write_table(df: pd.DataFrame, path: Optional[str] = None, *, sep: str = "\t",
                index: bool = False, header: bool = True,
                encoding: str = "utf-8", na_rep: str = "") -> None:
    """Write a table to a file or stdout (if path is None or "-")."""
    out = sys.stdout if path in (None, "-") else open(path, "w", encoding=encoding)
    close = (out is not sys.stdout)
    try:
        df.to_csv(out, sep=sep, index=index, header=header, na_rep=na_rep)
    finally:
        if close:
            out.close()
            
