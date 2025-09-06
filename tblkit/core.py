from __future__ import annotations
import sys
import argparse
import importlib
import traceback
import pandas as pd
import numpy as np
import re
from typing import Tuple
import os
import signal
from .utils import UtilsAPI
from .utils import io as UIO
from .utils import parsing as UP
from .utils import logging as ULOG
from .utils import columns as UCOL
from .utils import formatters as UFMT

__VERSION__=0.2

#-- Header Handlers --
def _handle_header_add(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Adds a generated header to a file read without one."""
    if df is None: raise ValueError("header add expects piped data.")
    
    if is_header_present and not args.force:
        # Idempotent: do nothing if header is already present
        return df
        
    n_cols = len(df.columns)
    new_header = [f"{args.prefix}{i}" for i in range(args.start, n_cols + args.start)]
    df.columns = new_header
    return df



def _handle_header_view(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame | None:
    """Creates a DataFrame with a vertical, indexed list of headers and pretty-prints it."""
    if df is None: raise ValueError("header view expects piped data")
    if not is_header_present:
        report_df = pd.DataFrame({"message": ["(no header to display)"]})
        UIO.pretty_print(report_df, args=argparse.Namespace())
        return None

    header = df.columns.tolist()
    if df.empty:
        first_row_values = ['(no data rows)'] * len(header)
    else:
        first_row_values = [str(item) if pd.notna(item) else "" for item in df.iloc[0].tolist()]

    report_df = pd.DataFrame({
        '#': range(1, len(header) + 1),
        'header': header,
        'sample_data_row_1': first_row_values
    })
    
    # Always pretty-print this command's output
    dummy_args = argparse.Namespace(max_cols=0, show_full=True, max_col_width=None)
    UIO.pretty_print(report_df, args=dummy_args, stream='stdout')
    return None

def _handle_header_rename(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    if df is None: raise ValueError("header rename expects piped data")
    
    rename_map = {}
    if args.map:
        pairs = [p.strip() for p in args.map.split(',')]
        for p in pairs:
            if ":" in p:
                old, new = p.split(":", 1)
                rename_map[old.strip()] = new.strip()
    elif args.from_file:
        with open(args.from_file, 'r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    old, new = parts
                    rename_map[old.strip()] = new.strip()
    
    # Resolve old names to actual column names before renaming
    resolved_map = {UCOL.parse_single_col(k, df.columns): v for k, v in rename_map.items()}
    return df.rename(columns=resolved_map)

def _handle_header_prefix_num(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Prefix columns with incrementing integers: 1_, 2_, ... or a custom format."""
    if df is None:
        raise ValueError("header prefix-num expects piped data")
    fmt = getattr(args, "fmt", None) or "{i}_"
    start = int(getattr(args, "start", 1) or 1)
    new_cols = [f"{fmt}".format(i=i) + str(c) for i, c in enumerate(df.columns, start=start)]
    return df.rename(columns=dict(zip(df.columns, new_cols)))

def _handle_header_add_prefix(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Add a fixed prefix to selected columns (rich selection)."""
    if df is None:
        raise ValueError("header add-prefix expects piped data")
    sel = UCOL.parse_multi_cols(getattr(args, "columns", "") or ",".join(df.columns), df.columns)
    pre = str(getattr(args, "prefix", "") or "")
    mapping = {c: (pre + c) for c in sel}
    return df.rename(columns=mapping)

def _handle_header_add_suffix(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Add a fixed suffix to selected columns (rich selection)."""
    if df is None:
        raise ValueError("header add-suffix expects piped data")
    sel = UCOL.parse_multi_cols(getattr(args, "columns", "") or ",".join(df.columns), df.columns)
    suf = str(getattr(args, "suffix", "") or "")
    mapping = {c: (c + suf) for c in sel}
    return df.rename(columns=mapping)

#-- Column Handlers --
def _handle_col_select(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    if df is None: raise ValueError("col select expects piped data")
    
    cols = UCOL.parse_multi_cols(args.columns, df.columns)
    
    # If --type is specified, filter the selected columns by dtype
    if args.type:
        selected_by_type = []
        dtype_map = {
            'string': ['object', 'string'],
            'numeric': ['number'],
            'integer': ['integer']
        }
        for col_name in cols:
            col_dtype = df[col_name].dtype.name
            is_match = any(pd.api.types.is_dtype_equal(col_dtype, t) for t in dtype_map.get(args.type, [])) or \
                       (args.type == 'numeric' and pd.api.types.is_numeric_dtype(df[col_name])) or \
                       (args.type == 'integer' and pd.api.types.is_integer_dtype(df[col_name]))
            
            if is_match:
                selected_by_type.append(col_name)
        cols = selected_by_type
        
    if args.invert:
        cols_to_drop = set(cols)
        final_cols = [c for c in df.columns if c not in cols_to_drop]
        return df[final_cols].copy()

    return df[cols].copy()

def _handle_col_drop(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None:
        raise ValueError("col drop expects piped data")
    drop_sel = UCOL.parse_multi_cols(args.columns, df.columns)
    keep_always = set(UCOL.parse_multi_cols(getattr(args, "keep_columns", "") or "", df.columns))
    if getattr(args, "invert", False):
        # Keep only listed columns plus forced keep set
        keep = list(dict.fromkeys(list(drop_sel) + list(keep_always)))
        return df.loc[:, keep]
    # Standard drop, but never drop forced keep columns
    target = [c for c in drop_sel if c not in keep_always]
    return df.drop(columns=target)

def _handle_col_rename(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    return _handle_header_rename(df, args, is_header_present=is_header_present)

def _handle_col_cast(df, args, column_names=None, **kwargs):
    """
    Cast columns to a target dtype.
    Defaults: int→ignore (keep original on failure), float→coerce, bool→coerce, datetime→coerce, others→raise.
    A user-provided --errors overrides these defaults.
    """
    from tblkit.utils.logging import get_logger
    from tblkit.utils.columns import resolve_columns_advanced
    logger = get_logger(__name__)

    cols_spec = getattr(args, "columns", None)
    cols = column_names or (resolve_columns_advanced(df, [cols_spec]) if cols_spec else None)
    if not cols:
        raise ValueError("No columns selected for casting. Use --columns.")

    to = (getattr(args, "to", None) or getattr(args, "dtype", None) or "").lower()
    if not to:
        raise ValueError("Target dtype is required. Use --to or --dtype.")

    fmt = getattr(args, "format", None)
    user_errors = getattr(args, "errors", None)
    out = df.copy()

    def default_errors_for(dtype_name: str) -> str:
        if dtype_name in {"int","int64","int32"}:      return "ignore"
        if dtype_name in {"float","float64","float32"}: return "coerce"
        if dtype_name in {"bool","boolean"}:            return "coerce"
        if dtype_name in {"datetime","datetime64"}:     return "coerce"
        return "raise"

    for c in cols:
        if c not in out.columns:
            raise KeyError(f"Unknown column: {c}")

        errors = (user_errors or default_errors_for(to)).lower()
        if errors not in {"raise","coerce","ignore"}:
            raise ValueError("--errors must be one of ['coerce','ignore','raise']")

        try:
            if to in {"int","int64","int32"}:
                s_num = pd.to_numeric(out[c], errors="coerce")
                if errors == "ignore":
                    new_vals = []
                    for orig, nval in zip(out[c], s_num):
                        if pd.isna(nval):
                            new_vals.append(orig)
                        else:
                            new_vals.append(int(nval) if float(nval).is_integer() else orig)
                    out[c] = pd.Series(new_vals, index=out.index)
                elif errors == "coerce":
                    nval = s_num.where(s_num.apply(lambda x: pd.isna(x) or float(x).is_integer()))
                    out[c] = nval.astype("Int64")
                else:  # raise
                    nval = pd.to_numeric(out[c], errors="raise")
                    if not all(float(x).is_integer() for x in nval):
                        raise ValueError("Non-integer values cannot be cast to Int64 under errors='raise'.")
                    out[c] = pd.Series([int(x) for x in nval], index=out.index, dtype="Int64")

            elif to in {"float","float64","float32"}:
                s = pd.to_numeric(out[c], errors=("coerce" if errors != "raise" else "raise"))
                out[c] = s.astype("float32" if to == "float32" else "float64")

            elif to in {"bool","boolean"}:
                def to_bool(v):
                    if pd.isna(v): return pd.NA
                    t = str(v).strip().lower()
                    if t in {"1","true","t","yes","y"}:  return True
                    if t in {"0","false","f","no","n"}:  return False
                    return pd.NA if errors != "raise" else (_ for _ in ()).throw(ValueError(f"Bad boolean: {v}"))
                out[c] = pd.Series([to_bool(v) for v in out[c]], index=out.index).astype("boolean")

            elif to in {"str","string"}:
                out[c] = out[c].astype("string")

            elif to in {"cat","category"}:
                out[c] = out[c].astype("category")

            elif to in {"datetime","datetime64"}:
                out[c] = pd.to_datetime(out[c], errors=("coerce" if errors != "raise" else "raise"), format=fmt)

            else:
                out[c] = out[c].astype(to)

        except Exception as e:
            if errors == "ignore":
                logger.debug(f"Ignored casting error for {c} to {to}: {e}")
            elif errors == "coerce":
                logger.debug(f"Coerced failures when casting {c} to {to}: {e}")
            else:
                raise
    return out

def _handle_col_fillna(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("col fillna expects piped data")
    out = df.copy()
    cols = UCOL.parse_multi_cols(args.columns, out.columns)
    try:
        fill_val = pd.to_numeric(args.value)
    except (ValueError, TypeError):
        fill_val = args.value
    out[cols] = out[cols].fillna(fill_val)
    return out

def _handle_col_split(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None:
        raise ValueError("col split expects piped data")
    out = df.copy()
    col = UCOL.parse_single_col(args.columns, df.columns)

    split_data = out[col].astype(str).str.split(
        args.pattern, n=args.maxsplit, expand=True, regex=not args.fixed
    )

    if args.names:
        new_names = [name.strip() for name in args.names.split(',')]
        if len(new_names) != split_data.shape[1]:
            raise ValueError(f"Number of names in --names must match resulting columns ({split_data.shape[1]})")
    else:
        new_names = [f"{col}_{i+1}" for i in range(split_data.shape[1])]

    split_data.columns = new_names
    out = pd.concat([out, split_data], axis=1)

    if getattr(args, "inplace", False):
        out = out.drop(columns=[col])
    return out

def _handle_col_add(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("col add expects piped data")
    out = df.copy()
    pos_col = UCOL.parse_single_col(args.columns, df.columns)
    pos_idx = df.columns.get_loc(pos_col)
    value = args.value if args.value else ''
    new_header = args.new_header
    out.insert(pos_idx, new_header, value)
    return out

def _handle_col_join(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("col join expects piped data")
    cols_to_join = UCOL.parse_multi_cols(args.columns, df.columns)
    joined_series = df[cols_to_join].astype(str).agg(args.delimiter.join, axis=1)
    
    min_idx = min(df.columns.get_loc(c) for c in cols_to_join)
    out = df.drop(columns=cols_to_join) if not args.keep else df.copy()
    out.insert(min_idx, args.output, joined_series)
    return out

def _handle_col_eval(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("col eval expects piped data")
    out = df.copy()
    out[args.output] = df.eval(args.expr)
    return out

#-- Row Handlers --
def _handle_row_grep(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Filters rows by matching against a list of words from an argument or file."""
    if df is None: raise ValueError("row grep expects piped data")

    words = []
    if args.words:
        words = [w.strip() for w in args.words.split(',')]
    elif args.word_file:
        with open(args.word_file, 'r', encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
    
    if not words:
        return df

    # Build regex pattern based on flags
    flags = re.IGNORECASE if args.ignore_case else 0
    pattern = "|".join(words) if args.regex else "|".join(re.escape(word) for word in words)
    
    # Determine which columns to search in
    search_df = df[UCOL.parse_multi_cols(args.columns, df.columns)] if args.columns else df

    # Create a boolean mask for rows that contain any of the words
    mask = search_df.apply(
        lambda col: col.astype(str).str.contains(pattern, na=False, regex=True, flags=flags)
    ).any(axis=1)

    return df[~mask] if args.invert else df[mask]

def _handle_row_head(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("row head expects piped data")
    return df.head(args.n).copy()

def _handle_row_tail(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("row tail expects piped data")
    return df.tail(args.n).copy()

def _handle_row_filter(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """
    Filter rows using a pandas expression. Example: --expr "a > 3 and b == 'x'".
    If --invert is set, returns the complement.
    """
    from tblkit.utils.logging import get_logger
    logger = get_logger(__name__)

    expr = getattr(args, "expr", None)
    invert = bool(getattr(args, "invert", False))
    if not expr or not str(expr).strip():
        raise ValueError("Row filter requires --expr <query>.")

    # Try numexpr first (fast), then pure python engine.
    try:
        kept = df.query(expr, engine="numexpr")
    except Exception as e1:
        logger.debug(f"numexpr failed on '{expr}': {e1}; falling back to python engine")
        try:
            kept = df.query(expr, engine="python")
        except Exception as e2:
            raise ValueError(f"Invalid filter expression: {expr}") from e2

    if invert:
        # Complement by index
        return df.loc[~df.index.isin(kept.index)]
    return kept

def _handle_row_sample(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("row sample expects piped data")
    n = args.n if args.n is not None else None
    frac = args.f if args.f is not None else None
    return df.sample(n=n, frac=frac, replace=args.with_replacement, random_state=args.seed)

def _handle_row_unique(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("row unique expects piped data")
    subset = UCOL.parse_multi_cols(args.columns, df.columns) if args.columns else None
    if args.invert:
        return df[df.duplicated(subset=subset, keep=False)].copy()
    return df.drop_duplicates(subset=subset, keep='first').copy()

def _handle_row_shuffle(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("row shuffle expects piped data")
    return df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

#-- Sort Handlers --
def _handle_sort_row(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("row sort expects piped data")
    sort_cols = UCOL.parse_multi_cols(args.by, df.columns)
    is_ascending = not args.descending

    # Date-aware path (takes precedence)
    if getattr(args, "date", False):
        fmt = getattr(args, "date_format", None)
        def key_dt(s: pd.Series):
            if pd.api.types.is_datetime64_any_dtype(s):
                return s
            return pd.to_datetime(s, errors="coerce",
                                  format=fmt if fmt else None,
                                  infer_datetime_format=False if fmt else True)
        return df.sort_values(by=sort_cols, ascending=is_ascending, key=key_dt)

    # Numeric-aware path (coerce text to numbers for ordering only; data unchanged)
    if getattr(args, "numeric", False):
        def key_num(s: pd.Series):
            if pd.api.types.is_numeric_dtype(s):
                return s
            ss = s.astype("string")
            # parentheses negatives → -value
            ss = ss.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            # drop leading currency symbols
            ss = ss.str.replace(r"^[\$\€\£]\s*", "", regex=True)
            # remove thousands separators (commas/underscores/spaces)
            ss = ss.str.replace(r"[,_\s]", "", regex=True)
            # strip trailing percent sign (treat -45.17% as -45.17 for ordering)
            ss = ss.str.replace(r"%$", "", regex=True)
            return pd.to_numeric(ss, errors="coerce")
        return df.sort_values(by=sort_cols, ascending=is_ascending, key=key_num)

    # Natural sort (strings) if requested
    try:
        from natsort import natsort_keygen
        key_func = natsort_keygen() if args.natural else None
        return df.sort_values(by=sort_cols, ascending=is_ascending, key=key_func)
    except ImportError:
        if args.natural:
            raise ImportError("Natural sort requires `natsort`. Please run `pip install natsort`.")
        return df.sort_values(by=sort_cols, ascending=is_ascending)
    
    

def _handle_sort_header(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("header sort expects piped data")
    if args.natural:
        try:
            from natsort import natsorted
            return df[natsorted(df.columns)]
        except ImportError:
            raise ImportError("Natural sort requires `natsort`. Please run `pip install natsort`.")
    return df[sorted(df.columns)]



#-- Table Handlers --

def _handle_tbl_aggregate(df, args, column_names=None, **kwargs):
    """
    Group-aggregate.
    Accepts:
      - --group/--by <cols>
      - EITHER --ops "a:sum,b:mean" OR --columns <cols> with --op <fn> OR --funcs "sum,mean"
    """
    from tblkit.utils.logging import get_logger
    from tblkit.utils.columns import resolve_columns_advanced
    logger = get_logger(__name__)

    by_spec = getattr(args, "by", None) or getattr(args, "group", None)
    if not by_spec:
        raise ValueError("Aggregation requires --by/--group <columns>.")
    by_cols = resolve_columns_advanced(df, [by_spec])

    ops_text = getattr(args, "ops", None)
    cols_text = getattr(args, "columns", None)
    op = getattr(args, "op", None)
    funcs_text = getattr(args, "funcs", None)

    if ops_text:
        agg_spec = {}
        for pair in str(ops_text).split(","):
            if not pair.strip():
                continue
            name, _, func = pair.partition(":")
            if not name or not func:
                raise ValueError(f"Bad --ops item: {pair!r}")
            agg_spec[name.strip()] = func.strip()
    else:
        if not cols_text:
            raise ValueError("Aggregation needs --columns when --ops is not used.")
        cols = resolve_columns_advanced(df, [cols_text]) if isinstance(cols_text, str) else cols_text
        if funcs_text:
            funcs = [f.strip() for f in str(funcs_text).split(",") if f.strip()]
            if not funcs:
                raise ValueError("Empty --funcs.")
            agg_spec = {c: funcs for c in cols}
        elif op:
            agg_spec = {c: op for c in cols}
        else:
            raise ValueError("Use --ops OR --op+--columns OR --funcs+--columns.")

    out = df.groupby(by=by_cols, dropna=False, sort=False).agg(agg_spec).reset_index()
    return out



def _handle_tbl_melt(df, args, column_names=None, **kwargs):
    """
    Melt wide → long.
    Requires: --id/--id_vars and --values/--value_vars
    Optional: --var_name, --value_name
    """
    from tblkit.utils.logging import get_logger
    from tblkit.utils.columns import resolve_columns_advanced
    logger = get_logger(__name__)

    id_spec = getattr(args, "id", None) or getattr(args, "id_vars", None)
    val_spec = getattr(args, "values", None) or getattr(args, "value_vars", None)
    if not id_spec or not val_spec:
        raise ValueError("Melt requires --id/--id_vars and --values/--value_vars.")

    id_vars = resolve_columns_advanced(df, [id_spec])
    value_vars = resolve_columns_advanced(df, [val_spec])

    var_name = getattr(args, "var_name", None) or getattr(args, "var", None) or "variable"
    value_name = getattr(args, "value_name", None) or getattr(args, "val", None) or "value"

    out = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    return out


def _handle_tbl_clean(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """
    Clean header and, by default, only non-numeric, non-date string cells:
      - Header: squeeze spaces→'_' (unless --keep-spaces), lowercase (unless --keep-case),
        strip non-ASCII (unless --keep-ascii), strip punctuation (unless --keep-punct),
        dedupe with sep (default '_').
      - Values: operate ONLY on object/string cells that are NOT numeric-like and NOT date-like.
        Numeric-like strings keep minus/decimals (and optional %); we normalize thousand separators
        by removing commas/underscores/spaces in the integer part. Dates are left unchanged.
        Punctuation in values is KEPT by default; use --strip-punct-values to remove.
    """
    if df is None:
        raise ValueError("tbl clean expects piped data.")

    sep = (args.dedupe if getattr(args, "dedupe", None) else "_") or "_"
    do_lower = not getattr(args, "keep_case", False)
    do_spaces = not getattr(args, "keep_spaces", False)
    do_ascii = not getattr(args, "keep_ascii", False)
    do_punct_header = not getattr(args, "keep_punct", False)
    do_punct_values = bool(getattr(args, "strip_punct_values", False))

    import re

    def _cleanup_header(s: str) -> str:
        new_s = str(s).strip()
        if do_lower:
            new_s = new_s.lower()
        if do_spaces:
            new_s = re.sub(r"\s+", " ", new_s).strip().replace(" ", "_")
        if do_ascii:
            new_s = new_s.encode("ascii", "ignore").decode("ascii")
        if do_punct_header:
            new_s = re.sub(r"[^\w\s-]", "", new_s).strip()
        return new_s

    # numeric-like: optional currency, sign, grouped thousands, optional decimals, optional %
    NUM_RE = re.compile(r"""^\s*            # start
                            [\$€£]?         # optional currency
                            [-+]?           # sign
                            (?:
                                \d{1,3}(?:[,_\s]\d{3})+  # 1,234 or 1_234 or 1 234
                                |\d+                      # or plain digits
                            )
                            (?:\.\d+)?      # optional decimals
                            %?              # optional percent
                            \s*$            # end
                         """, re.VERBOSE)

    DATE_RES = [
        re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$"),
        re.compile(r"^\d{2}/\d{2}/\d{4}$"),
        re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2,4}$"),
        re.compile(r"^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}$"),
    ]

    def _is_date_like(x: str) -> bool:
        s = str(x).strip()
        if not s:
            return False
        return any(r.match(s) for r in DATE_RES)

    def _normalize_numeric_like(s: str) -> str:
        t = s.strip()
        t = re.sub(r"^[\$€£]\s*", "", t)          # drop leading currency
        t = t.replace(",", "").replace("_", "").replace(" ", "")  # drop grouping
        return t  # keep '.' and '%'

    # 1) Header
    original = list(df.columns)
    new_cols = [_cleanup_header(c) for c in original]
    seen = {}
    for i, c in enumerate(new_cols):
        if c not in seen:
            seen[c] = 1
            continue
        k = seen[c] + 1
        while f"{c}{sep}{k}" in seen:
            k += 1
        new_cols[i] = f"{c}{sep}{k}"
        seen[new_cols[i]] = 1
    df = df.rename(columns=dict(zip(original, new_cols)))

    # 2) Values
    if not getattr(args, "header_only", False):
        exclude = set(UCOL.parse_multi_cols(getattr(args, "exclude", "") or "", df.columns))
        for col in df.columns:
            if col in exclude:
                continue
            # Skip numeric & datetimelike columns outright
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                continue

            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                def _cell(v):
                    if pd.isna(v):
                        return v
                    s = str(v)
                    if _is_date_like(s):
                        return s
                    if NUM_RE.match(s):
                        return _normalize_numeric_like(s)
                    # General text: lowercase/space/ascii like headers, but punctuation only if requested
                    if do_lower:
                        s = s.lower()
                    if do_spaces:
                        s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
                    if do_ascii:
                        s = s.encode("ascii", "ignore").decode("ascii")
                    if do_punct_values:
                        s = re.sub(r"[^\w\s-]", "", s).strip()
                    return s

                df[col] = df[col].apply(_cell)
    return df

def _handle_tbl_collapse(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Groups rows and collapses column values into delimited strings."""
    if df is None:
        raise ValueError("tbl collapse expects piped data")

    group_cols = UCOL.parse_multi_cols(args.group_by, df.columns)
    agg_cols = [c for c in df.columns if c not in group_cols]

    if not agg_cols:
        return df[group_cols].drop_duplicates().copy()

    if args.keep_all:
        agg_func = lambda x: args.delimiter.join(x.dropna().astype(str))
    else:
        agg_func = lambda x: args.delimiter.join(x.dropna().unique().astype(str))

    agg_dict = {col: agg_func for col in agg_cols}
    collapsed_df = df.groupby(group_cols, as_index=False, sort=False).agg(agg_dict)

    # Preserve original column order
    return collapsed_df[group_cols + agg_cols]


def _handle_tbl_pivot(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """
    Pivot long → wide with aggregation (default agg is 'first').
    Requires: --index <col>, --columns <col>, --value <col>
    Optional: --agg <func> (e.g., sum/mean/max)
    """
    from tblkit.utils.logging import get_logger
    from tblkit.utils.columns import resolve_columns_advanced
    logger = get_logger(__name__)

    idx_spec = getattr(args, "index", None)
    col_spec = getattr(args, "columns", None)
    val_spec = getattr(args, "value", None) or getattr(args, "values", None)
    agg = getattr(args, "agg", None) or "first"

    if not (idx_spec and col_spec and val_spec):
        raise ValueError("Pivot requires --index, --columns, and --value.")

    index = resolve_columns_advanced(df, [idx_spec])
    columns = resolve_columns_advanced(df, [col_spec])
    values = resolve_columns_advanced(df, [val_spec])

    if len(index) != 1 or len(columns) != 1 or len(values) != 1:
        raise ValueError("Pivot expects single columns for index/columns/value.")

    pv = pd.pivot_table(df, index=index[0], columns=columns[0], values=values[0], aggfunc=agg).reset_index()
    pv.columns.name = None
    return pv

def _handle_tbl_concat(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """
    Concatenate piped data and/or many files, with optional path-derived metadata columns.

    Supports:
      - positional FILES
      - --filelist FILE   (list of files, one per line; ignores blank lines and lines starting with '#')
      - --extract-from-path REGEX with named capture groups (?P<name>...)
      - --ancestor-cols-to-include COL1,COL2,... (rightmost name = immediate parent dir)
    """
    from pathlib import Path
    import os

    # Gather candidate files
    files: list[str] = []
    if getattr(args, "files", None):
        files.extend([f for f in args.files if f is not None])

    filelist = getattr(args, "filelist", None)
    if filelist:
        with open(filelist, "r", encoding=getattr(args, "encoding", "utf-8")) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                files.append(s)

    # Read each table
    sep = getattr(args, "sep", "\t")
    enc = getattr(args, "encoding", "utf-8")
    header = 0 if is_header_present else None

    all_dfs: list[pd.DataFrame] = []
    if df is not None:
        all_dfs.append(df)

    # Pre-parse path options
    path_regex = None
    if getattr(args, "extract_from_path", None):
        try:
            path_regex = re.compile(args.extract_from_path)
            if not path_regex.groupindex:
                raise ValueError("--extract-from-path regex must contain at least one named group, e.g., (?P<name>...).")
        except re.error as e:
            raise ValueError(f"Invalid regex for --extract-from-path: {e}")

    ancestor_cols: list[str] | None = None
    if getattr(args, "ancestor_cols_to_include", None):
        ancestor_cols = [c.strip() for c in args.ancestor_cols_to_include.split(",") if c.strip()]
        if not ancestor_cols:
            ancestor_cols = None

    for file_path in files:
        d = UIO.read_table(file_path, sep=sep, header=header, encoding=enc)
        # Optionally derive columns from the path
        extras: dict[str, object] = {}
        if path_regex is not None:
            m = path_regex.search(str(file_path))
            if m:
                extras.update({k: v for k, v in m.groupdict().items()})
            else:
                # If no match, still add named columns with NA to keep schema stable
                for k in path_regex.groupindex.keys():
                    extras.setdefault(k, pd.NA)
        elif ancestor_cols:
            p = Path(file_path).parent
            parts = list(p.parts)
            need = len(ancestor_cols)
            # take last N parts for the provided column names
            take = parts[-need:] if need <= len(parts) else [None] * (need - len(parts)) + parts
            # Map: rightmost provided name -> immediate parent
            for col_name, part in zip(ancestor_cols, take, strict=False):
                extras[col_name] = None if part is None else os.path.basename(part)

        if extras:
            for k, v in extras.items():
                # Ensure new columns filled with the same value for all rows in this file
                d[k] = v

        # Skip fully empty or all-NA frames
        if not d.empty and not d.isna().all().all():
            all_dfs.append(d)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True, sort=False)


def _handle_tbl_path2tbl(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """
    Build a table from filesystem paths.

    Inputs:
      --files "a,b,c" or --filelist file.txt or piped stdin (one path per line).
    Outputs:
      Always includes: path, name, stem, ext
      Optionally adds ancestry via:
        --anc 1..3      # gp1,gp2,gp3 (1=parent)
        --anc 1,3,5     # specific levels
        --anc-names a,b,c  # must match count in --anc
    """
    from pathlib import Path
    out_rows, paths = [], []

    if getattr(args, "files", None):
        paths += [p.strip() for p in str(args.files).split(",") if p.strip()]
    if getattr(args, "filelist", None):
        with open(args.filelist, "r", encoding=getattr(args, "encoding", "utf-8")) as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    paths.append(s)
    if not sys.stdin.isatty():
        for line in sys.stdin:
            s = line.strip()
            if s and not s.startswith("#"):
                paths.append(s)

    # dedupe preserve order
    seen = set(); uniq_paths = []
    for p in paths:
        if p not in seen:
            uniq_paths.append(p); seen.add(p)

    def parse_anc_spec(spec: str) -> list[int]:
        spec = spec.strip()
        if ".." in spec:
            a, b = spec.split("..", 1)
            a, b = int(a), int(b)
            if a <= 0 or b <= 0:
                raise ValueError("--anc indices must be positive (1=parent)")
            return list(range(a, b+1)) if a <= b else list(range(a, b-1, -1))
        out = []
        for tok in spec.split(","):
            tok = tok.strip()
            if tok:
                k = int(tok)
                if k <= 0: raise ValueError("--anc indices must be positive (1=parent)")
                out.append(k)
        return out

    anc_idxs: list[int] = []
    anc_names: list[str] | None = None
    if getattr(args, "anc", None):
        anc_idxs = parse_anc_spec(args.anc)
        if getattr(args, "anc_names", None):
            anc_names = [s.strip() for s in str(args.anc_names).split(",")]
            if len(anc_names) != len(anc_idxs):
                raise ValueError("--anc-names count must match --anc")
        else:
            anc_names = [f"gp{k}" for k in anc_idxs]

    missing_fill = getattr(args, "missing", None)
    if missing_fill is None:
        missing_fill = pd.NA

    for p in uniq_paths:
        pp = Path(p)
        row = {
            "path": p,
            "name": pp.name,
            "stem": pp.stem,
            "ext": pp.suffix[1:] if pp.suffix.startswith(".") else pp.suffix
        }
        if anc_idxs:
            parts = list(pp.parent.parts)  # e.g., ["/","u","v","x","y","z"] on POSIX
            for idx, col in zip(anc_idxs, anc_names, strict=False):
                row[col] = parts[-idx] if len(parts) >= idx else missing_fill
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows)

    # ensure unique column names (warn if changed)
    seen = {}
    new_cols = []
    for c in df_out.columns:
        if c not in seen:
            seen[c] = 0; new_cols.append(c)
        else:
            seen[c] += 1
            newc = f"{c}_{seen[c]}"
            sys.stderr.write(f"Warning: duplicate column '{c}' renamed to '{newc}'\n")
            new_cols.append(newc)
    df_out.columns = new_cols
    return df_out


def _handle_tbl_transpose(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("tbl transpose expects piped data")
    if is_header_present:
        # Set first column as index for new header, transpose the rest
        first_col = df.columns[0]
        transposed = df.set_index(first_col).T
        transposed.columns.name = None
        return transposed.reset_index()
    return df.T

#-- View Handlers --
# In core.py

def _handle_view_frequency(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame | None:
    """Calculates value frequencies and returns them in a wide DataFrame."""
    if df is None: raise ValueError("view frequency expects piped data")

    if args.columns:
        cols_to_analyze = UCOL.parse_multi_cols(args.columns, df.columns)
    elif args.all_columns:
        cols_to_analyze = df.columns.tolist()
    else:
        cols_to_analyze = df.select_dtypes(include=['object', 'string']).columns.tolist()

    if not cols_to_analyze:
        ULOG.get_logger("tblkit.core").info("No columns to analyze. Use -c or --all-columns.")
        return None

    top_n = args.n
    all_freqs = {}
    for col in cols_to_analyze:
        total_non_na = df[col].notna().sum()
        if total_non_na == 0:
            all_freqs[col] = ["(no values)"] + [""] * (top_n - 1)
            continue
        
        counts = df[col].value_counts().head(top_n)
        formatted_list = [f"{val} ({count}, {(count / total_non_na) * 100:.1f}%)" for val, count in counts.items()]
        
        if len(formatted_list) < top_n:
            formatted_list.extend([""] * (top_n - len(formatted_list)))
            
        all_freqs[col] = formatted_list

    report_df = pd.DataFrame(all_freqs)
    report_df.index = pd.RangeIndex(start=1, stop=len(report_df) + 1, name="rank")
    return report_df.reset_index()





def _handle_tbl_join(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Relational join between two tables on key columns, with optional fuzzy matching.

    Fuzzy behavior:
      - Only supported when: --fuzzy, single key, and --how left.
      - Normalization (--key-norm) is applied before exact attempt. Unmatched left keys are then
        matched to the closest right key by SequenceMatcher if score >= --threshold.
      - Optional --report writes left_key,right_key,score,method for transparency.
      - Optional --require-coverage raises if any left key remains unmatched after fuzzy.
    """
    import re, difflib

    if not args.left or not args.right:
        raise ValueError("Both --left and --right file paths are required for tbl join.")

    header_arg = 0 if not args.no_header else None
    df_left  = UIO.read_table(args.left,  sep=args.sep, header=header_arg)
    df_right = UIO.read_table(args.right, sep=args.sep, header=header_arg)

    keys = [k.strip() for k in args.keys.split(',')]
    for key in keys:
        if key not in df_left.columns:  raise ValueError(f"Key '{key}' not found in left table: {args.left}")
        if key not in df_right.columns: raise ValueError(f"Key '{key}' not found in right table: {args.right}")

    # Fast path: no fuzzy
    if not getattr(args, "fuzzy", False):
        merged = df_left.merge(df_right, on=keys, how=args.how, suffixes=(args.lsuffix, args.rsuffix))
        left_cols  = [c for c in df_left.columns  if c not in keys]
        right_cols = [c for c in df_right.columns if c not in keys]
        final_right_cols = []
        for c in right_cols:
            suffixed_c = f"{c}{args.rsuffix}"
            if suffixed_c in merged.columns:
                final_right_cols.append(suffixed_c)
            elif c in merged.columns:
                final_right_cols.append(c)
        final_order = keys + left_cols + final_right_cols
        return merged[final_order]

    # Fuzzy path constraints
    if len(keys) != 1:
        raise ValueError("Fuzzy join currently supports a single key column. Use --keys with one column.")
    if str(getattr(args, "how", "left")).lower() != "left":
        raise ValueError("Fuzzy join is currently implemented for --how left only.")

    key = keys[0]
    norms = []
    for item in (getattr(args, "key_norm", None) or []):
        norms.extend([t.strip() for t in str(item).split(",") if t.strip()])

    def apply_norms(s: pd.Series) -> pd.Series:
        out = s.astype("string").fillna("")
        for t in norms:
            if t == "lower":
                out = out.str.lower()
            elif t == "upper":
                out = out.str.upper()
            elif t == "trim":
                out = out.str.strip()
            elif t.startswith("strip_suffix:"):
                pat = t.split(":", 1)[1]
                out = out.str.replace(rf"{pat}$", "", regex=True)
            elif t.startswith("strip_prefix:"):
                pat = t.split(":", 1)[1]
                out = out.str.replace(rf"^{pat}", "", regex=True)
            elif t.startswith("strip:"):
                chars = t.split(":", 1)[1]
                out = out.str.replace(f"[{re.escape(chars)}]", "", regex=True)
            elif t == "rm_leading_zeros":
                def _rlz(x: str) -> str:
                    if not x: return x
                    y = x.lstrip("0")
                    return y if y != "" else "0"
                out = out.map(_rlz)
        return out

    left_norm  = apply_norms(df_left[key]) if norms else df_left[key].astype("string").fillna("")
    right_norm = apply_norms(df_right[key]) if norms else df_right[key].astype("string").fillna("")

    # Step 1: exact on normalized key
    L = df_left.copy()
    R = df_right.copy()
    L["__key_norm__"] = left_norm
    R["__key_norm__"] = right_norm

    exact = L.merge(R, on="__key_norm__", how="left", suffixes=(args.lsuffix, args.rsuffix), indicator=True)

    # Identify unmatched left rows
    if "_merge" in exact.columns:
        unmatched_mask = exact["_merge"] == "left_only"
        exact = exact.drop(columns=["_merge"])
    else:
        right_cols = [c for c in df_right.columns if c != key]
        cols_after = set(exact.columns)
        merged_right_cols = [c for c in cols_after if c not in L.columns or c in right_cols]
        unmatched_mask = exact[merged_right_cols].isna().all(axis=1)

    # Early out if nothing unmatched
    if not unmatched_mask.any():
        left_cols  = [c for c in df_left.columns  if c != key]
        right_cols = [c for c in df_right.columns if c != key]
        final_right_cols = []
        for c in right_cols:
            suffixed_c = f"{c}{args.rsuffix}"
            if suffixed_c in exact.columns:
                final_right_cols.append(suffixed_c)
            elif c in exact.columns:
                final_right_cols.append(c)
        final_order = [key] + left_cols + final_right_cols
        return exact[final_order]

    # Step 2: fuzzy match unmatched left keys against unique right norms
    threshold = float(getattr(args, "threshold", 0.9) or 0.9)
    left_unmatched = L.loc[unmatched_mask, [key, "__key_norm__"]].copy()

    right_unique = R[["__key_norm__", key]].drop_duplicates("__key_norm__")
    r_map = dict(zip(right_unique["__key_norm__"], right_unique[key]))

    chosen = []
    for ln, orig_left in zip(left_unmatched["__key_norm__"], left_unmatched[key]):
        best_norm = None
        best_score = 0.0
        for cand in r_map.keys():
            score = difflib.SequenceMatcher(None, str(ln), str(cand)).ratio()
            if score > best_score:
                best_score = score
                best_norm = cand
        if best_norm is not None and best_score >= threshold:
            chosen.append((orig_left, ln, r_map[best_norm], best_norm, best_score, "fuzzy"))
        else:
            chosen.append((orig_left, ln, None, None, 0.0, "none"))

    if getattr(args, "report", None):
        rep = pd.DataFrame(chosen, columns=[f"left_{key}", "left_norm", f"right_{key}", "right_norm", "score", "method"])
        try:
            rep.to_csv(args.report, index=False)
        except Exception as e:
            raise ValueError(f"Failed to write report to {args.report}: {e}") from e

    if getattr(args, "require_coverage", False):
        if any(m == "none" for *_, m in chosen):
            missing = [lk for (lk, *_rest, m) in chosen if m == "none"]
            raise ValueError(f"Fuzzy join: {len(missing)} left keys unmatched (e.g., {missing[:5]})")

    matched = [t for t in chosen if t[-1] != "none"]
    if matched:
        match_df = pd.DataFrame(matched, columns=[f"left_{key}", "left_norm", f"right_{key}", "right_norm", "score", "method"])
        L2 = L.loc[unmatched_mask, :].copy()
        L2 = L2.merge(match_df[["left_norm","right_norm"]].rename(columns={"left_norm":"__key_norm__","right_norm":"__match_norm__"}),
                      on="__key_norm__", how="left")
        R2 = R.copy()
        fuzzy_merged = L2.merge(R2, left_on="__match_norm__", right_on="__key_norm__", how="left", suffixes=(args.lsuffix, args.rsuffix))
        exact.loc[unmatched_mask, :] = fuzzy_merged[exact.columns].values

    left_cols  = [c for c in df_left.columns  if c != key]
    right_cols = [c for c in df_right.columns if c != key]
    final_right_cols = []
    for c in right_cols:
        suffixed_c = f"{c}{args.rsuffix}"
        if suffixed_c in exact.columns:
            final_right_cols.append(suffixed_c)
        elif c in exact.columns:
            final_right_cols.append(c)
    final_order = [key] + left_cols + final_right_cols

    final = exact
    for helper in ["__key_norm__", "__match_norm__"]:
        if helper in final.columns:
            final = final.drop(columns=[helper])
    return final[final_order]




#--  Column Handlers --

def _handle_col_replace(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Replaces values in selected columns."""
    if df is None:
        raise ValueError("col replace expects piped data.")
    out = df.copy()
    cols = UCOL.parse_multi_cols(args.columns, out.columns)

    import csv as _csv
    def _split_csv_aware(s: str) -> list[str]:
        return next(_csv.reader([s], skipinitialspace=True))

    keys_raw = [v.strip() for v in _split_csv_aware(args.vals_from)]
    vals_raw = [v.strip() for v in _split_csv_aware(args.vals_to)]
    if len(keys_raw) != len(vals_raw):
        raise ValueError("--from and --to must have the same number of comma-separated values.")

    for col in cols:
        s = out[col]
        if getattr(args, "na_only", False):
            for k, v in zip(keys_raw, vals_raw):
                if k.lower() in ("na", "nan", ""):
                    out[col] = s.fillna(v)
            continue
        if pd.api.types.is_numeric_dtype(s):
            k_num = pd.to_numeric(keys_raw, errors="coerce")
            v_num = pd.to_numeric(vals_raw, errors="coerce")
            repl = {k: v for k, v in zip(k_num.tolist(), v_num.tolist()) if pd.notna(k)}
            out[col] = s.replace(repl)
        else:
            repl = dict(zip(keys_raw, vals_raw))
            out[col] = s.astype("string").replace(repl, regex=getattr(args, "regex", False))
    return out



def _handle_col_strip(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """
    Trim/squeeze whitespace; optional substring strip from left/right; fixed-count strip; regex strip (side-aware).
    Defaults: --squeeze is True (collapse internal whitespace to single space).
    Sides for non-whitespace ops are chosen via --lstrip / --rstrip (both = both ends).
    """
    if df is None:
        raise ValueError("col strip expects piped data.")
    out = df.copy()
    cols = UCOL.parse_multi_cols(args.columns, out.columns)

    do_squeeze = not getattr(args, "no_squeeze", False)
    pat = getattr(args, "pattern", None)
    use_regex = bool(getattr(args, "regex", False))
    # Side flags for pattern/count; both True => both ends
    do_left = bool(getattr(args, "lstrip", False))
    do_right = bool(getattr(args, "rstrip", False))
    both_sides = (do_left and do_right) or (not do_left and not do_right)  # default to both when unspecified

    for col in cols:
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            s = out[col].astype("string")
            # 1) whitespace trim (always)
            s = s.str.strip()
            if do_squeeze:
                s = s.str.replace(r"\s+", " ", regex=True)

            # 2) fixed substring strip (literal), if provided
            if getattr(args, "lstrip_substr", None):
                sub = str(args.lstrip_substr)
                s = s.map(lambda v: v[len(sub):] if v is not None and v.startswith(sub) else v)
            if getattr(args, "rstrip_substr", None):
                sub = str(args.rstrip_substr)
                s = s.map(lambda v: v[:-len(sub)] if v is not None and v.endswith(sub) else v)

            # 3) regex or literal pattern strip (side-aware)
            if pat:
                if use_regex:
                    left_re = rf"^({pat})"
                    right_re = rf"({pat})$"
                    if both_sides or do_left:
                        s = s.str.replace(left_re, "", regex=True)
                    if both_sides or do_right:
                        s = s.str.replace(right_re, "", regex=True)
                else:
                    sub = str(pat)
                    def drop_left(v):
                        return v[len(sub):] if v is not None and v.startswith(sub) else v
                    def drop_right(v):
                        return v[:-len(sub)] if v is not None and v.endswith(sub) else v
                    if both_sides or do_left:
                        s = s.map(drop_left)
                    if both_sides or do_right:
                        s = s.map(drop_right)

            # 4) fixed-count strip with direction
            n = int(getattr(args, "strip_num_characters", 0) or 0)
            if n > 0:
                if both_sides:
                    # symmetric trim on both ends
                    s = s.str.slice(n).str.slice(stop=-n)
                elif do_left:
                    s = s.str.slice(n)
                elif do_right:
                    s = s.str.slice(stop=-n)

            out[col] = s
    return out

#==
def _handle_col_affix_add(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Add a fixed prefix/suffix to values in selected columns."""
    if df is None:
        raise ValueError("col affix-add expects piped data.")
    out = df.copy()
    cols = UCOL.parse_multi_cols(getattr(args, "columns", None), out.columns) if getattr(args, "columns", None) else list(out.columns)
    mode = (getattr(args, "mode", None) or "").lower()
    if mode not in {"prefix", "suffix"}:
        raise ValueError("--mode must be 'prefix' or 'suffix'")
    text = getattr(args, "text", None)
    if text is None:
        raise ValueError("--text is required")
    for c in cols:
        if pd.api.types.is_string_dtype(out[c]) or pd.api.types.is_object_dtype(out[c]):
            s = out[c].astype("string")
            out[c] = (text + s) if mode == "prefix" else (s + text)
    return out

def _handle_col_affix_rem(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Remove a prefix/suffix by fixed/regex pattern or by character count from the chosen side."""
    if df is None:
        raise ValueError("col affix-rem expects piped data.")
    out = df.copy()
    cols = UCOL.parse_multi_cols(getattr(args, "columns", None), out.columns) if getattr(args, "columns", None) else list(out.columns)
    mode = (getattr(args, "mode", None) or "").lower()
    if mode not in {"prefix", "suffix"}:
        raise ValueError("--mode must be 'prefix' or 'suffix'")
    pat = getattr(args, "pattern", None)
    is_regex = bool(getattr(args, "regex", False))
    count = int(getattr(args, "count", 0) or 0)

    if not pat and count <= 0:
        raise ValueError("Provide --pattern or --count N")

    for c in cols:
        if pd.api.types.is_string_dtype(out[c]) or pd.api.types.is_object_dtype(out[c]):
            s = out[c].astype("string")
            # Pattern removal (applies first, then count if provided)
            if pat:
                if mode == "prefix":
                    s = s.str.replace(rf"^({pat})", "", regex=True) if is_regex else s.map(lambda v: v[len(pat):] if v is not None and v.startswith(pat) else v)
                else:
                    s = s.str.replace(rf"({pat})$", "", regex=True) if is_regex else s.map(lambda v: v[:-len(pat)] if v is not None and len(pat)>0 and v.endswith(pat) else v)
            # Fixed count removal
            if count > 0:
                s = s.str.slice(count) if mode == "prefix" else s.str.slice(stop=-count)
            out[c] = s
    return out
#==

def _handle_col_move(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Reorders columns by moving a selection before or after a target."""
    if df is None: raise ValueError("col move expects piped data.")
    
    cols_to_move = UCOL.parse_multi_cols(args.columns, df.columns)
    target_col = UCOL.parse_single_col(args.before or args.after, df.columns)
    
    original_cols = df.columns.to_list()
    # Remove columns to move from their original positions
    remaining_cols = [c for c in original_cols if c not in cols_to_move]
    
    if target_col not in remaining_cols:
        raise ValueError(f"Target column '{target_col}' cannot be one of the columns being moved.")
        
    target_idx = remaining_cols.index(target_col)
    
    if args.before:
        new_order = remaining_cols[:target_idx] + cols_to_move + remaining_cols[target_idx:]
    else: # --after
        new_order = remaining_cols[:target_idx+1] + cols_to_move + remaining_cols[target_idx+1:]
        
    return df[new_order]

def _handle_col_extract(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Extracts regex capture groups from a column into new columns."""
    if df is None: raise ValueError("col extract expects piped data.")
    out = df.copy()
    source_col = UCOL.parse_single_col(args.columns, out.columns)
    
    extracted_df = out[source_col].str.extract(args.regex)
    
    # Insert new columns after the source column
    source_idx = out.columns.get_loc(source_col)
    for i, col_name in enumerate(extracted_df.columns):
        out.insert(source_idx + 1 + i, col_name, extracted_df[col_name])

    if args.drop_source:
        out = out.drop(columns=[source_col])
        
    return out

#-- [TIER 1] Column Handlers --

def _handle_col_clean(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Normalizes string values in selected columns."""
    if df is None: raise ValueError("col clean expects piped data.")
    out = df.copy()
    cols = UCOL.parse_multi_cols(args.columns, out.columns)
    
    for col in cols:
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            s = out[col].astype(str)
            if args.lower: s = s.str.lower()
            if args.upper: s = s.str.upper()
            if args.unicode_nfkc:
                import unicodedata
                s = s.apply(lambda x: unicodedata.normalize('NFKC', x) if pd.notna(x) else x)
            out[col] = s
    return out

#-- Row Handlers --

def _handle_row_drop(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Drop rows by 1-based indices; with -v/--invert keep only those."""
    if df is None:
        raise ValueError("row drop expects piped data.")
    indices = set()
    for part in args.indices.split(','):
        s = part.strip()
        if '-' in s:
            a, b = s.split('-', 1)
            a = int(a)
            b = int(b) if b else len(df)
            indices.update(range(a, b + 1))
        else:
            indices.add(int(s))
    zero = sorted(i - 1 for i in indices if 1 <= i <= len(df))
    if getattr(args, "invert", False):
        return df.iloc[zero, :].copy()
    return df.drop(index=zero)

def _handle_row_add(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Adds a new row with specified values."""
    if df is None: raise ValueError("row add expects piped data.")
    
    values = [v.strip() for v in args.values.split(',')]
    if len(values) != len(df.columns):
        raise ValueError(f"Number of values ({len(values)}) must match number of columns ({len(df.columns)}).")
    
    new_row = pd.DataFrame([values], columns=df.columns)
    
    if args.at is None: # Add to end
        return pd.concat([df, new_row], ignore_index=True)
    else:
        pos = args.at - 1 # 1-based to 0-based
        if not (0 <= pos <= len(df)):
            raise ValueError(f"Position --at must be between 1 and {len(df) + 1}.")
        
        df_top = df.iloc[:pos]
        df_bottom = df.iloc[pos:]
        return pd.concat([df_top, new_row, df_bottom], ignore_index=True)

def _handle_view(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Returns a DataFrame slice for viewing (no row limiting)."""
    if df is None:
        raise ValueError("view expects piped data.")
    out = df
    # Rich column selection (optional)
    if getattr(args, "columns", None):
        sel = UCOL.parse_multi_cols(args.columns, out.columns)
        out = out.loc[:, sel]
    # Keep max-cols if provided
    if getattr(args, "max_cols", None):
        out = out.iloc[:, : args.max_cols]
    return out

def _attach_view_group(subparsers: argparse._SubParsersAction, *, parents=None) -> None:
    """Attaches the simplified 'view' command (pretty-print only)."""
    p_view = subparsers.add_parser(
        "view",
        help="Pretty-print a table (ASCII, non-folding).",
        description="This command formats and displays a table with clear ASCII borders.",
        formatter_class=UFMT.CommandGroupHelpFormatter, parents=parents,
    )
    # Options: max-cols retained; max-rows removed; add truncation & rich column selection
    p_view.add_argument("--max-cols", type=int, help="Limit to first N columns.")
    p_view.add_argument("--max-col-width", type=int, default=40,
                        help="Truncate each column to this width (default: 40).")
    p_view.add_argument("--show-full", action="store_true",
                        help="Do not truncate wide fields (disables --max-col-width).")
    p_view.add_argument("-c", "--columns", help="Rich column selection (name/glob/pos/range/regex).")
    p_view.set_defaults(handler=_handle_view)
    
    
def _attach_tbl_group(subparsers: argparse._SubParsersAction, *, parents=None) -> None:
    """Attaches the 'tbl' command group and its actions."""
    p_tbl = subparsers.add_parser("tbl", help="Whole-table operations",
                                  description="This group contains commands for whole-table operations.",
                                  formatter_class=UFMT.ActionFirstHelpFormatter,
                                  parents=parents)
    tsub = p_tbl.add_subparsers(dest="action", title="Action", metavar="Action",
                                required=True, parser_class=UFMT.ActionParser)

    # clean
    t_clean = tsub.add_parser("clean", help="Clean headers and string values throughout the table.", parents=parents)
    t_clean.add_argument("--case", choices=["lower", "upper"], help="Convert case.")
    t_clean.add_argument("--spaces", help="Character to replace whitespace with.")
    t_clean.add_argument("--ascii", action="store_true", help="Remove non-ASCII characters.")
    t_clean.add_argument("--dedupe", help="Character for de-duplicating header names.")
    t_clean.add_argument("--header-only", action="store_true", help="Only clean the header, not cell values.")
    t_clean.add_argument("--exclude", help="Comma-separated columns to exclude from value cleaning.")
    t_clean.add_argument("--keep-spaces", action="store_true", help="Keep spaces as-is (no squeeze/underscore).")
    t_clean.add_argument("--keep-punct", action="store_true", help="Keep punctuation.")
    t_clean.add_argument("--keep-case", action="store_true", help="Do not lowercase.")
    t_clean.add_argument("--keep-ascii", action="store_true", help="Do not strip non-ASCII.")
    t_clean.add_argument("--strip-punct-values", action="store_true",
                         help="Also strip punctuation from string values (default: keep).")
    t_clean.set_defaults(handler=_handle_tbl_clean)

    # frequency
    t_freq = tsub.add_parser("frequency", help="Show top N values per column.", parents=parents)
    t_freq.add_argument("-c", "--columns", help="Columns to analyze (default: all string columns).")
    t_freq.add_argument("-n", type=int, default=5, help="Number of top values to show.")
    t_freq.add_argument("--all-columns", action="store_true", help="Analyze all columns, not just string ones.")
    t_freq.set_defaults(handler=_handle_view_frequency)

    # join
    t_join = tsub.add_parser("join", help="Relational join between two tables.", parents=parents)
    t_join.add_argument("--left", required=True, help="Path to the left table.")
    t_join.add_argument("--right", required=True, help="Path to the right table.")
    t_join.add_argument("--keys", required=True, help="Comma-separated join key(s).")
    t_join.add_argument("--how", choices=["left","right","inner","outer"], default="left")
    t_join.add_argument("--keep-left", help="Comma-separated columns to keep from left (default: all).")
    t_join.add_argument("--keep-right", help="Comma-separated columns to keep from right (default: all).")
    t_join.add_argument("--suffixes", default="_x,_y", help="Suffixes for overlapping right/left columns.")
    t_join.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching fallback.")
    t_join.add_argument("--key-norm", dest="key_norm", action="append",
                        help="Normalization(s) to apply to key(s) before matching.")
    t_join.add_argument("--fuzzy-threshold", type=float, default=0.9,
                        help="Fuzzy match similarity threshold in [0,1] (default: 0.9).")
    t_join.add_argument("--require-coverage", action="store_true",
                        help="Error if any left key remains unmatched after exact+fuzzy.")
    t_join.add_argument("--report", help="Write CSV report of matches (left_key,right_key,score,method).")
    t_join.set_defaults(handler=_handle_tbl_join, standalone=True)

    # sort (already inherited)
    t_sort = tsub.add_parser("sort", help="Sort rows by column values (alias for 'sort rows').", parents=parents)
    t_sort.add_argument("--by", required=True, help="Comma-separated columns to sort by.")
    t_sort.add_argument("--descending", action="store_true")
    t_sort.add_argument("--natural", action="store_true", help="Use natural sort order.")
    t_sort.add_argument("--date", action="store_true", help="Parse the sort keys as dates.")
    t_sort.add_argument("--date-format", help="Optional strptime format for dates.")
    t_sort.set_defaults(handler=_handle_sort_row)

    # pivot
    t_pivot = tsub.add_parser("pivot", help="Pivot wider.", parents=parents)
    t_pivot.add_argument("--index", required=True)
    t_pivot.add_argument("--columns", required=True)
    t_pivot.add_argument("--values", required=True)
    t_pivot.set_defaults(handler=_handle_tbl_pivot)

    # concat
    t_concat = tsub.add_parser("concat", help="Concatenate tables vertically.", parents=parents)
    t_concat.add_argument("--files", required=False, help="Comma-separated paths, or use --filelist.")
    t_concat.add_argument("--filelist", help="Path to a file listing table paths.")
    t_concat.add_argument("--fill-missing", action="store_true", help="Union columns, filling missing with NA.")
    t_concat.set_defaults(handler=_handle_tbl_concat)

    t_path = tsub.add_parser("path2tbl", help="Build a table from paths.", parents=parents)
    t_path.add_argument("--files", help="Comma-separated paths, or use --filelist, or pipe paths on stdin.")
    t_path.add_argument("--filelist", help="File with one path per line (supports comments with #).")
    t_path.add_argument("--anc", help="Ancestor spec from the right, e.g., '1..3' or '1,3,5' (1=parent).")
    t_path.add_argument("--anc-names", help="Comma-separated column names matching --anc (e.g., 'subproj,proj,org').")
    t_path.add_argument("--missing", help="Fill value for missing ancestry (default: NA).")
    t_path.set_defaults(handler=_handle_tbl_path2tbl, standalone=True)

    
    # aggregate
    t_agg = tsub.add_parser("aggregate", help="Group and aggregate numeric columns.", parents=parents)
    t_agg.add_argument("--group-by", required=True, help="Comma-separated group columns.")
    t_agg.add_argument("--columns", help="Columns to aggregate (default: all numeric).")
    t_agg.add_argument("--funcs", required=True, help="Comma-separated aggregation functions.")
    t_agg.set_defaults(handler=_handle_tbl_aggregate)

    # collapse
    t_collapse = tsub.add_parser("collapse", help="Group rows and collapse column values into delimited strings.", parents=parents)
    t_collapse.add_argument("-g", "--group-by", required=True, help="Column(s) to group by.")
    t_collapse.add_argument("-d", "--delimiter", default=",", help="Delimiter for joining values.")
    t_collapse.add_argument("--keep-all", action="store_true", help="Join all values, not just unique ones.")
    t_collapse.set_defaults(handler=_handle_tbl_collapse)

    # melt
    t_melt = tsub.add_parser("melt", help="Melt table to long format.", parents=parents)
    t_melt.add_argument("--id-vars", required=True)
    t_melt.add_argument("--value-vars")
    t_melt.add_argument("--var-name", default="variable")
    t_melt.add_argument("--value-name", default="value")
    t_melt.set_defaults(handler=_handle_tbl_melt)

    # transpose
    t_transpose = tsub.add_parser("transpose", help="Transpose the table.", parents=parents)
    t_transpose.set_defaults(handler=_handle_tbl_transpose)
    
    
def _attach_sort_group(subparsers: argparse._SubParsersAction, *, parents=None) -> None:
    """Attaches the 'sort' command group and its actions."""
    p_sort = subparsers.add_parser("sort", help="Sort rows or columns",
                                   description="This group contains commands for sorting table rows or columns.",
                                   formatter_class=UFMT.ActionFirstHelpFormatter,
                                   parents=parents)

    sosub = p_sort.add_subparsers(dest="action", title="Action", metavar="Action", required=True, parser_class=UFMT.ActionParser)
    
    so_rows = sosub.add_parser("rows", help="Sort rows by column values", parents=parents)
    so_rows.add_argument("--by", required=True, help="Comma-separated columns to sort by.")
    so_rows.add_argument("--descending", action="store_true")
    so_rows.add_argument("--natural", action="store_true", help="Use natural sort order.")
    so_rows.add_argument("--date", action="store_true", help="Parse the sort keys as dates.")
    so_rows.add_argument("--date-format", help="Optional strptime format for dates.")
    so_rows.add_argument("--numeric", action="store_true",
                     help="Coerce sort keys to numeric for ordering (data unchanged).")
    
    so_rows.set_defaults(handler=_handle_sort_row)
    
    so_cols = sosub.add_parser("cols", help="Sort columns by their names")
    so_cols.add_argument("--natural", action="store_true", help="Use natural sort order.")
    so_cols.set_defaults(handler=_handle_sort_header)
    
    
# Example for _attach_row_group (apply this change to all similar functions)
def _attach_row_group(subparsers: argparse._SubParsersAction, *, parents=None) -> None:
    """Attaches the 'row' command group and its actions."""
    p_row = subparsers.add_parser("row", help="Row operations",
                                  description="This group contains commands that operate on table rows.",
                                  formatter_class=UFMT.ActionFirstHelpFormatter, # <--- THIS IS THE CHANGE
                                  parents=parents)
    # ... rest of the function remains the same
    rsub = p_row.add_subparsers(dest="action", title="Action", metavar="Action", required=True, parser_class=UFMT.ActionParser)
    
    r_subset = rsub.add_parser("subset", help="Select a subset of rows using a query expression")
    r_subset.add_argument("expr", help="Pandas query expression (e.g., 'col_a > 5')")
    r_subset.add_argument("--invert", action="store_true", help="Invert the filter condition.")
    r_subset.set_defaults(handler=_handle_row_filter)
    
    r_grep = rsub.add_parser("grep", help="Filter rows by a list of words or phrases.")
    grep_group = r_grep.add_mutually_exclusive_group(required=True)
    grep_group.add_argument("--words", help="Comma-separated list of words to search for.")
    grep_group.add_argument("--word-file", help="File containing one word/phrase per line to search for.")
    r_grep.add_argument("-c", "--columns", help="Specific columns to search in (default: all).")
    r_grep.add_argument("--invert", action="store_true", help="Select rows that DO NOT match.")
    r_grep.add_argument("--ignore-case", action="store_true", help="Perform case-insensitive matching.")
    r_grep.add_argument("--regex", action="store_true", help="Treat words as regular expressions.")
    r_grep.set_defaults(handler=_handle_row_grep)
    
    r_head = rsub.add_parser("head", help="Select first N rows")
    r_head.add_argument("-n", type=int, default=10, help="Number of rows")
    r_head.set_defaults(handler=_handle_row_head)
    
    r_tail = rsub.add_parser("tail", help="Select last N rows")
    r_tail.add_argument("-n", type=int, default=10, help="Number of rows")
    r_tail.set_defaults(handler=_handle_row_tail)
    
    r_sample = rsub.add_parser("sample", help="Randomly sample rows")
    sg = r_sample.add_mutually_exclusive_group(required=True)
    sg.add_argument("-n", type=int, help="Number of rows to sample.")
    sg.add_argument("-f", type=float, help="Fraction of rows to sample.")
    r_sample.add_argument("--with-replacement", action="store_true")
    r_sample.set_defaults(handler=_handle_row_sample)
    
    r_unique = rsub.add_parser("unique", help="Filter unique or duplicate rows")
    r_unique.add_argument("-c", "--columns", help="Columns to consider for uniqueness (default: all).")
    r_unique.add_argument("--invert", action="store_true", help="Keep only duplicate rows.")
    r_unique.set_defaults(handler=_handle_row_unique)
    
    r_shuffle = rsub.add_parser("shuffle", help="Randomly shuffle all rows.")
    r_shuffle.set_defaults(handler=_handle_row_shuffle)
    
    r_drop = rsub.add_parser("drop", help="Drop rows by 1-based index.")
    r_drop.add_argument("--indices", required=True, help="Comma-separated indices or ranges (e.g., '1,3,10-12').")
    r_drop.add_argument("-v", "--invert", action="store_true", help="Keep only the specified indices (inverse drop).")

    r_drop.set_defaults(handler=_handle_row_drop)
    
    r_add = rsub.add_parser("add", help="Add a row with specified values.")
    r_add.add_argument("--values", required=True, help="Comma-separated values for the new row.")
    r_add.add_argument("--at", type=int, help="1-based position to insert the row (default: append).")
    r_add.set_defaults(handler=_handle_row_add)
    
    
    
def _attach_col_group(subparsers: argparse._SubParsersAction, *, parents=None) -> None:
    """Attaches the 'col' command group and its actions."""
    p_col = subparsers.add_parser("col", help="Column operations",
                                  description="This group contains commands that operate on table columns.",
                                  formatter_class=UFMT.ActionFirstHelpFormatter, parents=parents)
    csub = p_col.add_subparsers(dest="action", title="Action", metavar="Action", required=True, parser_class=UFMT.ActionParser)
    
    c_subset = csub.add_parser("subset", help="Select a subset of columns by name/glob/position/regex")
    c_subset.add_argument("-c", "--columns", required=True, help="Columns to keep (e.g., 'id,val*,2-5,re:tmp_')")
    c_subset.add_argument("--type", choices=['string', 'numeric', 'integer'], help="Select columns of a specific type.")
    c_subset.add_argument("--invert", action="store_true", help="Select all columns EXCEPT these.")
    c_subset.set_defaults(handler=_handle_col_select)
    
    c_clean = csub.add_parser("clean", help="Normalize string values in selected columns.")
    c_clean.add_argument("-c", "--columns", required=True)
    case_group = c_clean.add_mutually_exclusive_group()
    case_group.add_argument("--lower", action="store_true", help="Convert to lowercase.")
    case_group.add_argument("--upper", action="store_true", help="Convert to uppercase.")
    c_clean.add_argument("--spaces", help="Replace whitespace with this character.")
    c_clean.add_argument("--ascii", action="store_true", help="Strip non-ASCII.")
    c_clean.add_argument("--unicode-nfkc", action="store_true", help="Apply NFKC Unicode normalization.")
    c_clean.set_defaults(handler=_handle_col_clean)
    
    c_drop = csub.add_parser("drop", help="Drop columns by name/glob/position/regex")
    c_drop.add_argument("-c", "--columns", required=True, help="Columns to drop (rich selector).")
    c_drop.add_argument("-v", "--invert", action="store_true", help="Keep only these columns (inverse drop).")
    c_drop.add_argument("--keep-columns", help="Columns to always retain even if --invert is used.")
    c_drop.set_defaults(handler=_handle_col_drop)
    
    c_rename = csub.add_parser("rename", help="Rename column(s) via map string")
    c_rename.add_argument("--map", required=True, help="Map of 'old1:new1,old2:new2'")
    c_rename.set_defaults(handler=_handle_col_rename)
    
    c_replace = csub.add_parser("replace", help="Value replacement in selected columns.")
    c_replace.add_argument("-c", "--columns", required=True)
    c_replace.add_argument("--from", dest="vals_from", required=True, help="Comma-separated values to replace.")
    c_replace.add_argument("--to", dest="vals_to", required=True, help="Comma-separated replacement values.")
    c_replace.add_argument("--regex", action="store_true", help="Treat replacement keys as regex.")
    c_replace.add_argument("--na-only", action="store_true", help="Only replace missing (NA) values.")
    c_replace.set_defaults(handler=_handle_col_replace)
    
    c_strip = csub.add_parser("strip", help="Trim/squeeze whitespace; optional substring/fixed-count strip.")
    c_strip.add_argument("-c", "--columns", required=True)
    c_strip.add_argument("--no-squeeze", action="store_true", help="Do not collapse internal whitespace.")
    c_strip.add_argument("--lstrip-substr", help="Substring to strip from the left, if present.")
    c_strip.add_argument("--rstrip-substr", help="Substring to strip from the right, if present.")
    c_strip.add_argument("--strip-num-characters", type=int, help="Fixed number of characters to strip.")
    c_strip.add_argument("--lstrip", action="store_true", help="When using --strip-num-characters, strip from left.")
    c_strip.add_argument("--rstrip", action="store_true", help="When using --strip-num-characters, strip from right.")
    c_strip.add_argument("--pattern", help="Pattern to strip from chosen side(s); use --regex for regex; otherwise literal.")
    c_strip.add_argument("--regex", action="store_true", help="Interpret --pattern as a regular expression.")
    c_strip.set_defaults(handler=_handle_col_strip)
    
    c_move = csub.add_parser("move", help="Reorder columns by moving a selection.")
    c_move.add_argument("-c", "--columns", required=True, help="Column(s) to move.")
    move_group = c_move.add_mutually_exclusive_group(required=True)
    move_group.add_argument("--before", help="Target column to move selection before.")
    move_group.add_argument("--after", help="Target column to move selection after.")
    c_move.set_defaults(handler=_handle_col_move)
    
    c_extract = csub.add_parser("extract", help="Extract regex groups into new columns.")
    c_extract.add_argument("-c", "--columns", required=True, help="Source column.")
    c_extract.add_argument("--regex", required=True, help="Regex with named capture groups, e.g., '^(?P<name>...)'")
    c_extract.add_argument("--drop-source", action="store_true", help="Drop the original source column.")
    c_extract.set_defaults(handler=_handle_col_extract)

    c_split = csub.add_parser("split", help="Split a column by pattern into multiple columns")
    c_split.add_argument("-c", "--columns", required=True, help="Single column to split.")
    c_split.add_argument("--pattern", required=True, help="Split pattern (regex by default).")
    c_split.add_argument("--fixed", action="store_true", help="Treat pattern as a literal substring, not regex.")
    c_split.add_argument("--maxsplit", type=int, default=-1, help="Maximum number of splits (-1 for all).")
    c_split.add_argument("-n", "--names", help="Comma-separated names for new columns.")
    c_split.add_argument("--inplace", action="store_true", help="Drop the source column after split.")
    c_split.set_defaults(handler=_handle_col_split)
    
    c_add = csub.add_parser("add", help="Add a new column")
    c_add.add_argument("-c", "--columns", required=True, help="Column to position new column next to.")
    c_add.add_argument("--new-header", required=True, help="Name for the new column.")
    c_add.add_argument("-v", "--value", help="Static value for the new column.")
    c_add.set_defaults(handler=_handle_col_add)
    
    c_join = csub.add_parser("join", help="Join values from multiple columns into a new column.")
    c_join.add_argument("-c", "--columns", required=True)
    c_join.add_argument("-d", "--delimiter", default="", help="Delimiter between values.")
    c_join.add_argument("-o", "--output", required=True, help="Name for the new output column.")
    c_join.add_argument("--keep", action="store_true", help="Keep the original columns.")
    c_join.set_defaults(handler=_handle_col_join)
    
    #c_aff_add = csub.add_parser("affix-add", help="Add a fixed prefix/suffix to values in selected columns.")
    #c_aff_add = csub.add_parser("affix-add", help=argparse.SUPPRESS)
    #c_aff_add.add_argument("-c", "--columns", help="Column selection (default: all).")
    #c_aff_add.add_argument("--mode", required=True, choices=["prefix","suffix"], help="Which side to add the text.")
    #c_aff_add.add_argument("--text", required=True, help="Text to add.")
    #c_aff_add.set_defaults(handler=_handle_col_affix_add)

    #c_aff_rem = csub.add_parser("affix-rem", help="Remove a prefix/suffix by pattern or by character count.")
    #c_aff_rem = csub.add_parser("affix-rem", help=argparse.SUPPRESS)
    #c_aff_rem.add_argument("-c", "--columns", help="Column selection (default: all).")
    #c_aff_rem.add_argument("--mode", required=True, choices=["prefix","suffix"], help="Which side to remove from.")
    #c_aff_rem.add_argument("--pattern", help="Fixed string or regex to remove (use --regex for regex).")
    #c_aff_rem.add_argument("--regex", action="store_true", help="Interpret --pattern as a regular expression.")
    #c_aff_rem.add_argument("--count", type=int, help="Remove N characters from the chosen side.")
    #c_aff_rem.set_defaults(handler=_handle_col_affix_rem)

    c_paste = csub.add_parser("paste", help="Add fixed text as a prefix/suffix to values in selected columns.")
    c_paste.add_argument("-c", "--columns", help="Column selection (default: all).")
    c_paste.add_argument("--mode", required=True, choices=["prefix","suffix"], help="Which side to add the text.")
    c_paste.add_argument("--text", required=True, help="Text to add.")
    c_paste.set_defaults(handler=_handle_col_affix_add)
    
    
#----header group
def _attach_header_group(subparsers: argparse._SubParsersAction, *, parents=None) -> None:
    """Attaches the 'header' command group and its actions."""
    p_header = subparsers.add_parser("header", help="Header operations",
                                     description="This group contains commands for manipulating table headers.",
                                     formatter_class=UFMT.ActionFirstHelpFormatter, parents=parents)
    hsub = p_header.add_subparsers(dest="action", title="Action", metavar="Action", required=True, parser_class=UFMT.ActionParser)
    
    h_view = hsub.add_parser("view", help="View header column names")
    h_view.set_defaults(handler=_handle_header_view)
    
    h_rename = hsub.add_parser("rename", help="Rename headers via map string or file")
    map_group = h_rename.add_mutually_exclusive_group(required=True)
    map_group.add_argument("--map", help="Comma-separated map of old:new names.")
    map_group.add_argument("--from-file", help="Two-column file (old_name\\tnew_name) with renames.")
    h_rename.set_defaults(handler=_handle_header_rename)

    h_add = hsub.add_parser("add", help="Add a generated header to a headerless file.")
    h_add.add_argument("--prefix", default="col_", help="Prefix for generated column names.")
    h_add.add_argument("--start", type=int, default=1, help="Starting number for generated column names.")
    h_add.add_argument("--force", action="store_true", help="Add header even if one appears to exist.")
    h_add.set_defaults(handler=_handle_header_add)

    h_clean = hsub.add_parser(
        "clean",
        help="Normalize all column names (deprecated; use: tbl clean)",
        parents=parents
    )    
    h_clean.add_argument("--case", choices=["lower", "upper"], help="Convert case.")
    h_clean.add_argument("--spaces", help="Character to replace whitespace with.")
    h_clean.add_argument("--ascii", action="store_true", help="Remove non-ASCII characters.")
    h_clean.add_argument("--dedupe", help="Character to use as a separator for de-duplicating names.")
    h_clean.set_defaults(handler=_handle_tbl_clean)
    
    h_prefix_num = hsub.add_parser("prefix-num", help="Prefix headers with 1_, 2_, ... (or custom fmt).")
    h_prefix_num.add_argument("--fmt", default="{i}_", help="Format string with {i} (default: '{i}_').")
    h_prefix_num.add_argument("--start", type=int, default=1, help="Starting integer (default: 1).")
    h_prefix_num.set_defaults(handler=_handle_header_prefix_num)
    
    h_add_prefix = hsub.add_parser("add-prefix", help="Add a fixed prefix to columns.")
    h_add_prefix.add_argument("--prefix", required=True, help="Prefix text.")
    h_add_prefix.add_argument("-c", "--columns", help="Rich selection (default: all).")
    h_add_prefix.set_defaults(handler=_handle_header_add_prefix)
    
    h_add_suffix = hsub.add_parser("add-suffix", help="Add a fixed suffix to columns.")
    h_add_suffix.add_argument("--suffix", required=True, help="Suffix text.")
    h_add_suffix.add_argument("-c", "--columns", help="Rich selection (default: all).")
    h_add_suffix.set_defaults(handler=_handle_header_add_suffix)

    
def _attach_stable_groups(subparsers: argparse._SubParsersAction, *, parents=None) -> None:    
    """Attaches all stable command groups to the main parser."""
    #_attach_header_group(subparsers)
    _attach_header_group(subparsers, parents=parents)    
    #_attach_col_group(subparsers)
    _attach_col_group(subparsers, parents=parents)

    #_attach_row_group(subparsers)
    _attach_row_group(subparsers, parents=parents)

    #_attach_sort_group(subparsers)
    _attach_sort_group(subparsers, parents=parents)
    
    #_attach_tbl_group(subparsers)
    _attach_tbl_group(subparsers, parents=parents)    
    #_attach_view_group(subparsers)
    _attach_view_group(subparsers, parents=parents)

def safe_register(mod, subparsers, utils_api, logger):
    try:
        mod.register(subparsers, utils_api)
        logger.info("Loaded plugin: %s", getattr(mod, "__name__", mod))
        return True
    except Exception as e:
        logger.warning("Plugin failed to load: %s (%s)", getattr(mod, "__name__", mod), e)
        return False


def _add_global_io_flags(p):
    """
    Global I/O flags available to all subcommands.
    """
    p.add_argument("--sep", default="\t",
                   help="Input delimiter for reading tables (default: \\t).")
    p.add_argument("--output-sep", dest="output_sep", default="\t",
                   help="Output delimiter when printing tables (default: \\t).")
    p.add_argument("--encoding", default="utf-8",
                   help="File encoding for reading inputs (default: utf-8).")
    

def build_parser() -> argparse.ArgumentParser:
    ap = UFMT.CustomArgumentParser(
        prog="tblkit",
        description="Modular tabular toolkit",
        formatter_class=UFMT.MainHelpFormatter,
        add_help=False
    )
    # Global options
    UP.add_common_io_args(ap)
    common_parent = argparse.ArgumentParser(add_help=False)
    UP.add_common_io_args(common_parent)

    g = ap.add_argument_group("Global Options")
    g.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    g.add_argument("--plugins", action=UFMT.PluginsAction, help="List loaded plugins and exit.")
    g.add_argument("--version", action="version", version=__VERSION__)
    subs = ap.add_subparsers(dest="group", metavar="group", required=True,
                             parser_class=UFMT.CustomArgumentParser)

    #subs = ap.add_subparsers(dest="group", metavar="group", required=True)
    #_attach_stable_groups(subs)
    _attach_stable_groups(subs, parents=[common_parent])
    # Load plugins so their commands are available
    utils_api = UtilsAPI()
    logger = ULOG.get_logger("tblkit.core")
    _load_plugins(subs, utils_api, logger)

    
    return ap


# In core.py (add or restore this function)

def _load_plugins(subparsers, utils_api, logger) -> Tuple[list, list]:
    loaded, failed = [], []
    # We only have the compare plugin for now
    for name in ("tblkit.plugins.compare",):
        try:
            mod = importlib.import_module(name)
            ok = safe_register(mod, subparsers, utils_api, logger)
            (loaded if ok else failed).append(name)
        except ImportError:
            logger.debug("Plugin not found: %s. Skipping.", name)
        except Exception as e:
            logger.warning("Failed to import plugin %s: %s", name, e)
            failed.append(name)
    return loaded, failed


def run_handler(df, args, is_header_present, logger):
    handler = getattr(args, "handler", None)
    if handler is None:
        raise ValueError("No command selected. Use --help.")
    
    res = handler(df, args, is_header_present=is_header_present)

    if isinstance(res, pd.DataFrame):
        if getattr(args, "pretty", False):
            # Preview to stderr instead of writing a file/STDOUT.
            UIO.pretty_print(res, args=args, stream='stdout')
        else:
            out_sep = getattr(args, "output_sep", None) or getattr(args, "sep", "\t")
            UIO.write_table(
                res,
                path=getattr(args, "out_file", None),
                sep=out_sep,
                header=is_header_present,
                na_rep=getattr(args, "na_rep", ""),
                encoding=getattr(args, "encoding", "utf-8"),
            )
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    
    parser = build_parser()

    if not argv:
        parser.error("command group is required")
        return 2 # error() already exits, but this is for clarity
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass

    # Handle group-level help (e.g., 'tblkit col')
    if len(argv) == 1 and not argv[0].startswith('-') and hasattr(parser, '_subparsers'):
        sp_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        if sp_actions and argv[0] in sp_actions[0].choices:
            # Custom error message for missing action
            use_color = sys.stderr.isatty() and (os.getenv("NO_COLOR") is None)
            red = "\033[31m" if use_color else ""
            reset = "\033[0m" if use_color else ""
            
            sys.stderr.write("-------------------------------\n")
            sys.stderr.write(f"{red}Error: Must provide an Action for the '{argv[0]}' group.{reset}\n")
            sys.stderr.write("-------------------------------\n\n")

            gp = sp_actions[0].choices[argv[0]]
            groups = getattr(gp, "_action_groups", None)
            if isinstance(groups, list):
                original_groups = list(groups)
                try:
                    gp._action_groups = [
                        g for g in groups
                        if (getattr(g, "title", "") or "").strip().lower() != "i/o"
                    ]
                    gp.print_help(sys.stderr)
                finally:
                    gp._action_groups = original_groups
            else:
                gp.print_help(sys.stderr)
            return 2
        
    args0, _ = parser.parse_known_args(argv)
    ULOG.configure(quiet=getattr(args0,"quiet", False),
                   debug=getattr(args0,"debug", False),
                   log_file=getattr(args0,"log_file", None))
    logger = ULOG.get_logger("tblkit.core")

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2

    df = None
    is_header_present = not getattr(args, "no_header", False)
    
    if not getattr(args, "standalone", False) and not sys.stdin.isatty():
        header_arg = 0 if is_header_present else None
        df = UIO.read_table(
            None,
            sep=args.sep,
            header=header_arg,
            encoding=args.encoding,
            na_values=args.na_values,
            on_bad_lines=args.on_bad_lines
        )

    try:
        return run_handler(df, args, is_header_present, logger)

    except (ValueError, KeyError, ImportError) as e:
        logger.error(str(e))
        if getattr(args, "debug", False): traceback.print_exc()
        return 2
    except (FileNotFoundError, IOError) as e:
        logger.error(str(e))
        if getattr(args, "debug", False): traceback.print_exc()
        return 3
    except BrokenPipeError:
        try:
            try: sys.stdout.close()
            except Exception: pass
            try: sys.stderr.close()
            except Exception: pass
        finally:
            return 0    
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        if getattr(args, "debug", False): traceback.print_exc()
        return 4
    
    

if __name__ == "__main__":
    raise SystemExit(main())
