from __future__ import annotations
import sys
import argparse
import importlib
import traceback
import pandas as pd
import numpy as np
import re
from typing import Tuple

from .utils import UtilsAPI
from .utils import io as UIO
from .utils import parsing as UP
from .utils import logging as ULOG
from .utils import columns as UCOL
from .utils import formatters as UFMT

#region Handlers (Migrated from tblkit v1)

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



def _handle_header_view(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Creates a DataFrame with a vertical, indexed list of headers."""
    if df is None: raise ValueError("header view expects piped data")
    if not is_header_present:
        return pd.DataFrame({"message": ["(no header to display)"]})

    header = df.columns.tolist()
    if df.empty:
        first_row_values = ['(no data rows)'] * len(header)
    else:
        first_row_values = [str(item) if pd.notna(item) else "" for item in df.iloc[0].tolist()]

    return pd.DataFrame({
        '#': range(1, len(header) + 1),
        'header': header,
        'sample_data_row_1': first_row_values
    })

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
    if df is None: raise ValueError("col drop expects piped data")
    cols_to_drop = UCOL.parse_multi_cols(args.columns, df.columns)
    return df.drop(columns=cols_to_drop)

def _handle_col_rename(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    return _handle_header_rename(df, args, is_header_present=is_header_present)

# def _handle_col_clean(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
#     """Cleans string values in specified columns to be machine-readable."""
#     if df is None: raise ValueError("col clean expects piped data")

#     # Basic cleanup function migrated from tblkit v1
#     def _cleanup_string(s: Any) -> Any:
#         if s is None or (isinstance(s, float) and pd.isna(s)): return s
#         s = str(s).strip().lower().replace(' ', '_').replace('-', '_')
#         s = re.sub(r'[\\./()]+', '_', s)
#         s = re.sub(r'[^0-9A-Za-z_]+', '', s)
#         s = re.sub(r'_+', '_', s).strip('_')
#         return s

#     out = df.copy()
#     cols_to_process = UCOL.parse_multi_cols(args.columns, df.columns) if args.columns else df.columns.tolist()
    
#     if args.exclude:
#         cols_to_exclude = set(UCOL.parse_multi_cols(args.exclude, df.columns))
#         cols_to_process = [c for c in cols_to_process if c not in cols_to_exclude]

#     for col in cols_to_process:
#         if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
#             out[col] = out[col].apply(_cleanup_string)
            
#     return out

def _handle_col_cast(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("col cast expects piped data")
    out = df.copy()
    cols = UCOL.parse_multi_cols(args.columns, out.columns)
    for col in cols:
        if args.to in ("int", "integer"):
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.where(s.notna(), out[col])
        elif args.to in ("float", "numeric"):
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.where(s.notna(), out[col])
        else:
            out[col] = out[col].astype(args.to)
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
    if df is None: raise ValueError("col split expects piped data")
    out = df.copy()
    col = UCOL.parse_single_col(args.columns, df.columns)
    
    split_data = out[col].astype(str).str.split(
        args.delimiter, n=args.maxsplit, expand=True, regex=not args.fixed
    )

    if args.into:
        new_names = [name.strip() for name in args.into.split(',')]
        if len(new_names) != split_data.shape[1]:
            raise ValueError(f"Number of names in --into must match resulting columns ({split_data.shape[1]})")
    else:
        new_names = [f"{col}_{i+1}" for i in range(split_data.shape[1])]

    split_data.columns = new_names
    out = pd.concat([out, split_data], axis=1)

    if not args.keep:
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

def _handle_row_filter(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("row filter expects piped data")
    expr_to_run = f"not ({args.expr})" if args.invert else args.expr
    return df.query(expr_to_run).copy()

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
def _handle_tbl_clean(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Cleans header and optionally string values in the entire table."""
    if df is None: raise ValueError("tbl clean expects piped data.")

    def _cleanup_string(s: str) -> str:
        new_s = str(s).strip()
        if args.case == 'lower': new_s = new_s.lower()
        elif args.case == 'upper': new_s = new_s.upper()
        if args.spaces is not None: new_s = re.sub(r'\s+', args.spaces, new_s)
        if args.ascii: new_s = new_s.encode('ascii', 'ignore').decode('ascii')
        new_s = re.sub(r'[^\w\s-]', '', new_s).strip()
        return new_s

    # 1. Clean Header
    original_cols = df.columns.to_list()
    new_cols = [_cleanup_string(c) for c in original_cols]
    
    if args.dedupe is not None:
        counts = {}
        final_cols = []
        for c in new_cols:
            if c in counts:
                counts[c] += 1
                final_cols.append(f"{c}{args.dedupe}{counts[c]}")
            else:
                counts[c] = 0
                final_cols.append(c)
        new_cols = final_cols
    
    df.columns = new_cols

    # 2. Clean Values (if requested)
    if not args.header_only:
        cols_to_exclude = set(UCOL.parse_multi_cols(args.exclude, df.columns)) if args.exclude else set()
        
        for col in df.columns:
            if col in cols_to_exclude:
                continue
            # Clean columns that are object/string type
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].apply(lambda x: _cleanup_string(x) if pd.notna(x) else x)

    return df

def _handle_tbl_squash(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Groups rows and squashes column values into delimited strings."""
    if df is None: raise ValueError("tbl squash expects piped data")
    
    group_cols = UCOL.parse_multi_cols(args.group_by, df.columns)
    agg_cols = [c for c in df.columns if c not in group_cols]

    if not agg_cols:
        return df[group_cols].drop_duplicates().copy()

    if args.keep_all:
        agg_func = lambda x: args.delimiter.join(x.dropna().astype(str))
    else:
        agg_func = lambda x: args.delimiter.join(x.dropna().unique().astype(str))
        
    agg_dict = {col: agg_func for col in agg_cols}
    
    squashed_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # Preserve original column order
    return squashed_df[group_cols + agg_cols]

def _handle_tbl_pivot(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("tbl pivot expects piped data")
    index_cols = UCOL.parse_multi_cols(args.index, df.columns)
    pivot_cols = UCOL.parse_single_col(args.columns, df.columns)
    value_cols = UCOL.parse_single_col(args.values, df.columns)
    pivoted = df.pivot_table(index=index_cols, columns=pivot_cols, values=value_cols, aggfunc=args.agg).reset_index()
    pivoted.columns.name = None
    return pivoted

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


def _handle_tbl_aggregate(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("tbl aggregate expects piped data")
    group_cols = UCOL.parse_multi_cols(args.group, df.columns)
    if args.columns:
        agg_cols = UCOL.parse_multi_cols(args.columns, df.columns)
    else:
        agg_cols = df.select_dtypes(include=np.number).columns.tolist()

    funcs = [f.strip() for f in args.funcs.split(',')]
    agg_dict = {col: funcs for col in agg_cols}

    return df.groupby(group_cols).agg(agg_dict).reset_index()

def _handle_tbl_melt(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool):
    if df is None: raise ValueError("tbl melt expects piped data")
    id_vars = UCOL.parse_multi_cols(args.id_vars, df.columns)
    value_vars = UCOL.parse_multi_cols(args.value_vars, df.columns) if args.value_vars else None
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=args.var_name, value_name=args.value_name)

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




def _handle_view_tree(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> None:
    """Prints a nicely aligned and colored ASCII tree of the command structure."""
    import os
    parser = build_parser()
    tree_items = []

    # Define colors
    use_color = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
    cyan, orange, reset = ("", "", "")
    if use_color:
        cyan = "\033[96m"    # Cyan for groups
        orange = "\033[33m"  # Yellow/Orange for actions
        reset = "\033[0m"

    # First pass: Collect all groups and actions
    subparsers_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if not subparsers_actions:
        return

    top_level_parsers = subparsers_actions[0].choices
    group_help_map = {a.dest: a.help or '' for a in subparsers_actions[0]._choices_actions}
    groups = sorted(top_level_parsers.items())
    
    for i, (name, group_parser) in enumerate(groups):
        is_last_group = (i == len(groups) - 1)
        tree_items.append({
            'prefix': "└── " if is_last_group else "├── ",
            'name': name, 'help': group_help_map.get(name, ''), 'level': 0
        })
        
        action_parsers_actions = [a for a in group_parser._actions if isinstance(a, argparse._SubParsersAction)]
        if not action_parsers_actions: continue
            
        action_help_map = {a.dest: a.help or '' for a in action_parsers_actions[0]._choices_actions}
        actions = sorted(action_parsers_actions[0].choices.items())
        
        for j, (action_name, _) in enumerate(actions):
            is_last_action = (j == len(actions) - 1)
            child_prefix = "    " if is_last_group else "│   "
            action_tree_prefix = "└── " if is_last_action else "├── "
            tree_items.append({
                'prefix': child_prefix + action_tree_prefix,
                'name': action_name, 'help': action_help_map.get(action_name, ''), 'level': 1
            })

    # Second pass: Calculate alignment and format output
    if not tree_items:
        print("tblkit")
        return

    # Calculate width based on the visible characters, not the raw string length with color codes
    max_width = max(len(f"{item['prefix']}{item['name']}") for item in tree_items)
    
    output = ["tblkit"]
    for item in tree_items:
        color = orange if item['level'] == 0 else cyan
        colored_name = f"{color}{item['name']}{reset}"
        
        # Manually construct the line to handle color codes correctly for padding
        prefix_and_name = f"{item['prefix']}{item['name']}"
        padding = " " * (max_width - len(prefix_and_name))
        
        # Reconstruct the colored version for printing
        full_first_col = f"{item['prefix']}{colored_name}"
        
        output.append(f"{full_first_col}{padding}  ({item['help']})")

    print("\n".join(output))
    return None

# In core.py, within the #region Handlers

#-- [TIER 0] Table Handlers --

def _handle_tbl_join(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Performs a relational join between two tables on key columns."""
    if not args.left or not args.right:
        raise ValueError("Both --left and --right file paths are required for tbl join.")
    
    header_arg = 0 if not args.no_header else None
    
    df_left = UIO.read_table(args.left, sep=args.sep, header=header_arg)
    df_right = UIO.read_table(args.right, sep=args.sep, header=header_arg)
    
    keys = [k.strip() for k in args.keys.split(',')]
    
    # Validate keys are in both dataframes
    for key in keys:
        if key not in df_left.columns: raise ValueError(f"Key '{key}' not found in left table: {args.left}")
        if key not in df_right.columns: raise ValueError(f"Key '{key}' not found in right table: {args.right}")

    merged = pd.merge(
        df_left, df_right, how=args.how, on=keys,
        suffixes=(args.lsuffix, args.rsuffix)
    )
    
    # Enforce deterministic column order
    left_cols = [c for c in df_left.columns if c not in keys]
    right_cols = [c for c in df_right.columns if c not in keys]
    
    # Account for suffixes in right columns
    final_right_cols = []
    for c in right_cols:
        suffixed_c = f"{c}{args.rsuffix}"
        if suffixed_c in merged.columns:
            final_right_cols.append(suffixed_c)
        elif c in merged.columns: # No suffix applied if no overlap
             final_right_cols.append(c)

    final_order = keys + left_cols + final_right_cols
    return merged[final_order]






#--  Column Handlers --

def _handle_col_replace(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Replaces values in selected columns."""
    if df is None: raise ValueError("col replace expects piped data.")
    out = df.copy()
    cols = UCOL.parse_multi_cols(args.columns, out.columns)
    
    vals_from = [v.strip() for v in args.from_val.split(',')]
    vals_to = [v.strip() for v in args.to_val.split(',')]
    
    if len(vals_from) != len(vals_to):
        raise ValueError("--from and --to must have the same number of comma-separated values.")

    for col in cols:
        if args.na_only:
            for v_from, v_to in zip(vals_from, vals_to):
                if v_from.lower() in ('na', 'nan', ''): # Special handling for replacing NAs
                    out[col] = out[col].fillna(v_to)
        else:
            replace_map = dict(zip(vals_from, vals_to))
            out[col] = out[col].replace(replace_map, regex=args.regex)
    return out

def _handle_col_strip(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Trims whitespace from values in selected columns."""
    if df is None: raise ValueError("col strip expects piped data.")
    out = df.copy()
    cols = UCOL.parse_multi_cols(args.columns, out.columns)
    
    for col in cols:
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].str.strip()
            if args.collapse:
                out[col] = out[col].str.replace(r'\s+', ' ', regex=True)
    return out

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

#-- [TIER 1] Row Handlers --

def _handle_row_drop(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Drops rows by their 1-based indices."""
    if df is None: raise ValueError("row drop expects piped data.")
    
    indices_to_drop = set()
    for part in args.indices.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            start = int(start)
            end = int(end) if end else len(df)
            indices_to_drop.update(range(start, end + 1))
        else:
            indices_to_drop.add(int(part))
            
    # Convert 1-based to 0-based indices for pandas
    zero_based_indices = [i - 1 for i in indices_to_drop if 1 <= i <= len(df)]
    return df.drop(index=zero_based_indices)

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

#-- [TIER 1] View Handlers --
def _handle_view(df: pd.DataFrame | None, args: argparse.Namespace, *, is_header_present: bool) -> pd.DataFrame:
    """Applies slicing and returns a DataFrame for viewing."""
    if df is None: raise ValueError("view expects piped data.")
    
    if args.max_rows: df = df.head(args.max_rows)
    if args.max_cols: df = df.iloc[:, :args.max_cols]
    return df

def _attach_view_group(subparsers: argparse._SubParsersAction) -> None:
    """Attaches the 'view' command group and its actions."""
    p_view = subparsers.add_parser("view", help="Data inspection and summarization", formatter_class=UFMT.CommandGroupHelpFormatter)
    vsub = p_view.add_subparsers(dest="action", title="Action", metavar="Action", required=True, parser_class=UFMT.ActionParser)
    
    v_table = vsub.add_parser("table", help="Pretty-print a table for display.")
    v_table.add_argument("--max-rows", type=int, help="Limit output to the first N rows.")
    v_table.add_argument("--max-cols", type=int, help="Limit output to the first N columns.")
    v_table.set_defaults(handler=_handle_view)
    
    v_freq = vsub.add_parser("frequency", help="Show top N values per column.")
    v_freq.add_argument("-c", "--columns", help="Columns to analyze (default: all string columns).")
    v_freq.add_argument("-n", type=int, default=5, help="Number of top values to show.")
    v_freq.add_argument("--all-columns", action="store_true", help="Analyze all columns, not just string ones.")
    v_freq.set_defaults(handler=_handle_view_frequency)

    v_tree = vsub.add_parser("tree", help="Show the command structure as an ASCII tree.")
    v_tree.set_defaults(handler=_handle_view_tree)
    
def _attach_tbl_group(subparsers: argparse._SubParsersAction) -> None:
    """Attaches the 'tbl' command group and its actions."""
    p_tbl = subparsers.add_parser("tbl", help="Whole-table operations", formatter_class=UFMT.CommandGroupHelpFormatter)
    tsub = p_tbl.add_subparsers(dest="action", title="Action", metavar="Action", required=True, parser_class=UFMT.ActionParser)
    
    t_clean = tsub.add_parser("clean", help="Clean headers and string values throughout the table.")
    t_clean.add_argument("--case", choices=["lower", "upper"], help="Convert case.")
    t_clean.add_argument("--spaces", help="Character to replace whitespace with.")
    t_clean.add_argument("--ascii", action="store_true", help="Remove non-ASCII characters.")
    t_clean.add_argument("--dedupe", help="Character for de-duplicating header names.")
    t_clean.add_argument("--header-only", action="store_true", help="Only clean the header, not cell values.")
    t_clean.add_argument("--exclude", help="Comma-separated columns to exclude from value cleaning.")
    t_clean.set_defaults(handler=_handle_tbl_clean)
    
    t_join = tsub.add_parser("join", help="Relational join between two tables.")
    t_join.add_argument("--left", required=True, help="Path to the left table.")
    t_join.add_argument("--right", required=True, help="Path to the right table.")
    t_join.add_argument("--keys", required=True, help="Comma-separated key column(s).")
    t_join.add_argument("--how", default="inner", choices=["left", "right", "outer", "inner"])
    t_join.add_argument("--lsuffix", default="_x", help="Suffix for overlapping columns from left table.")
    t_join.add_argument("--rsuffix", default="_y", help="Suffix for overlapping columns from right table.")
    t_join.set_defaults(handler=_handle_tbl_join, standalone=True)
    
    t_sort = tsub.add_parser("sort", help="Sort rows by column values (alias for 'sort rows').")
    t_sort.add_argument("--by", required=True, help="Comma-separated columns to sort by.")
    t_sort.add_argument("--descending", action="store_true")
    t_sort.add_argument("--natural", action="store_true", help="Use natural sort order.")
    t_sort.set_defaults(handler=_handle_sort_row)
    
    t_pivot = tsub.add_parser("pivot", help="Pivot a table from long to wide format")
    t_pivot.add_argument("--index", required=True, help="Columns to use as new index.")
    t_pivot.add_argument("--columns", required=True, help="Column to pivot into new columns.")
    t_pivot.add_argument("--values", required=True, help="Column to use for new values.")
    t_pivot.add_argument("--agg", default="first", help="Aggregation function for duplicates.")
    t_pivot.set_defaults(handler=_handle_tbl_pivot)
    
    t_concat = tsub.add_parser("concat", help="Concatenate piped table with other files")
    t_concat.add_argument("files", nargs='*', help="Files to concatenate (if not piped).")
    t_concat.add_argument("--filelist", metavar="FILE", help="File containing a list of input files (one per line).")
    concat_path = t_concat.add_mutually_exclusive_group()
    concat_path.add_argument(
        "--ancestor-cols-to-include",
        dest="ancestor_cols_to_include",
        help="Comma-separated names for columns to create from parent directories (rightmost = immediate parent).",
    )
    concat_path.add_argument(
        "--extract-from-path",
        dest="extract_from_path",
        help="Regex with NAMED capture groups applied to each file path, e.g. '(?P<proj>[^/]+)/(?P<sample>[^/]+)/[^/]+$'.",
    )
    t_concat.set_defaults(handler=_handle_tbl_concat)

    
    t_agg = tsub.add_parser("aggregate", help="Group and aggregate data")
    t_agg.add_argument("-g", "--group", required=True, help="Grouping column(s).")
    t_agg.add_argument("-c", "--columns", help="Columns to aggregate (default: all numeric).")
    t_agg.add_argument("--funcs", required=True, help="Comma-separated aggregation functions.")
    t_agg.set_defaults(handler=_handle_tbl_aggregate)
    
    t_squash = tsub.add_parser("squash", help="Group rows and squash column values into delimited strings.")
    t_squash.add_argument("-g", "--group-by", required=True, help="Column(s) to group by.")
    t_squash.add_argument("-d", "--delimiter", default=",", help="Delimiter for joining values.")
    t_squash.add_argument("--keep-all", action="store_true", help="Join all values, not just unique ones.")
    t_squash.set_defaults(handler=_handle_tbl_squash)
    
    t_melt = tsub.add_parser("melt", help="Melt table to long format.")
    t_melt.add_argument("--id-vars", required=True)
    t_melt.add_argument("--value-vars")
    t_melt.add_argument("--var-name", default="variable")
    t_melt.add_argument("--value-name", default="value")
    t_melt.set_defaults(handler=_handle_tbl_melt)
    
    t_transpose = tsub.add_parser("transpose", help="Transpose the table.")
    t_transpose.set_defaults(handler=_handle_tbl_transpose)
    
def _attach_sort_group(subparsers: argparse._SubParsersAction) -> None:
    """Attaches the 'sort' command group and its actions."""
    p_sort = subparsers.add_parser("sort", help="Sort rows or columns", formatter_class=UFMT.CommandGroupHelpFormatter)
    sosub = p_sort.add_subparsers(dest="action", title="Action", metavar="Action", required=True, parser_class=UFMT.ActionParser)
    
    so_rows = sosub.add_parser("rows", help="Sort rows by column values")
    so_rows.add_argument("--by", required=True, help="Comma-separated columns to sort by.")
    so_rows.add_argument("--descending", action="store_true")
    so_rows.add_argument("--natural", action="store_true", help="Use natural sort order.")
    so_rows.set_defaults(handler=_handle_sort_row)
    
    so_cols = sosub.add_parser("cols", help="Sort columns by their names")
    so_cols.add_argument("--natural", action="store_true", help="Use natural sort order.")
    so_cols.set_defaults(handler=_handle_sort_header)
    
def _attach_row_group(subparsers: argparse._SubParsersAction) -> None:
    """Attaches the 'row' command group and its actions."""
    p_row = subparsers.add_parser("row", help="Row operations", formatter_class=UFMT.CommandGroupHelpFormatter)
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
    r_drop.set_defaults(handler=_handle_row_drop)
    
    r_add = rsub.add_parser("add", help="Add a row with specified values.")
    r_add.add_argument("--values", required=True, help="Comma-separated values for the new row.")
    r_add.add_argument("--at", type=int, help="1-based position to insert the row (default: append).")
    r_add.set_defaults(handler=_handle_row_add)
    
def _attach_col_group(subparsers: argparse._SubParsersAction) -> None:
    """Attaches the 'col' command group and its actions."""
    p_col = subparsers.add_parser("col", help="Column operations", formatter_class=UFMT.CommandGroupHelpFormatter)
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
    c_clean.add_argument("--unicode-nfkc", action="store_true", help="Apply NFKC Unicode normalization.")
    c_clean.set_defaults(handler=_handle_col_clean)
    
    c_drop = csub.add_parser("drop", help="Drop columns by name/glob/position/regex")
    c_drop.add_argument("-c", "--columns", required=True, help="Columns to drop.")
    c_drop.set_defaults(handler=_handle_col_drop)
    
    c_rename = csub.add_parser("rename", help="Rename column(s) via map string")
    c_rename.add_argument("--map", required=True, help="Map of 'old1:new1,old2:new2'")
    c_rename.set_defaults(handler=_handle_col_rename)
    
    c_cast = csub.add_parser("cast", help="Cast columns to a new type")
    c_cast.add_argument("-c", "--columns", required=True)
    c_cast.add_argument("--to", required=True, help="Target data type (e.g., int, float, str)")
    c_cast.set_defaults(handler=_handle_col_cast)
    
    c_fillna = csub.add_parser("fillna", help="Fill missing values in columns")
    c_fillna.add_argument("-c", "--columns", required=True)
    c_fillna.add_argument("-v", "--value", required=True, help="Value to fill with.")
    c_fillna.set_defaults(handler=_handle_col_fillna)
    
    c_split = csub.add_parser("split", help="Split a column into multiple new columns")
    c_split.add_argument("-c", "--columns", required=True, help="Single column to split.")
    c_split.add_argument("-d", "--delimiter", required=True, help="Delimiter to split on.")
    c_split.add_argument("--maxsplit", type=int, default=-1, help="Maximum number of splits.")
    c_split.add_argument("--fixed", action="store_true", help="Treat delimiter as literal string.")
    c_split.add_argument("--into", help="Comma-separated names for new columns.")
    c_split.add_argument("--keep", action="store_true", help="Keep original column.")
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
    
    c_eval = csub.add_parser("eval", help="Create a column by evaluating an expression.")
    c_eval.add_argument("--expr", required=True, help="A pandas-compatible expression string.")
    c_eval.add_argument("-o", "--output", required=True, help="Name for the new column.")
    c_eval.set_defaults(handler=_handle_col_eval)
    
    c_replace = csub.add_parser("replace", help="Replace values in selected columns.")
    c_replace.add_argument("-c", "--columns", required=True)
    c_replace.add_argument("--from", dest="from_val", required=True, help="Comma-separated values to replace.")
    c_replace.add_argument("--to", dest="to_val", required=True, help="Comma-separated replacement values.")
    c_replace.add_argument("--regex", action="store_true", help="Treat --from values as regex patterns.")
    c_replace.add_argument("--na-only", action="store_true", help="Only replace missing (NA) values.")
    c_replace.set_defaults(handler=_handle_col_replace)
    
    c_strip = csub.add_parser("strip", help="Trim whitespace from string values.")
    c_strip.add_argument("-c", "--columns", required=True)
    c_strip.add_argument("--collapse", action="store_true", help="Collapse internal whitespace to a single space.")
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

def _attach_header_group(subparsers: argparse._SubParsersAction) -> None:
    """Attaches the 'header' command group and its actions."""
    p_header = subparsers.add_parser("header", help="Header operations", formatter_class=UFMT.CommandGroupHelpFormatter)
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

    h_clean = hsub.add_parser("clean", help="Normalize all column names (deprecated; use: tbl clean)")
    h_clean.add_argument("--case", choices=["lower", "upper"], help="Convert case.")
    h_clean.add_argument("--spaces", help="Character to replace whitespace with.")
    h_clean.add_argument("--ascii", action="store_true", help="Remove non-ASCII characters.")
    h_clean.add_argument("--dedupe", help="Character to use as a separator for de-duplicating names.")
    h_clean.set_defaults(handler=_handle_tbl_clean)
    

def _attach_stable_groups(subparsers: argparse._SubParsersAction) -> None:
    """Attaches all stable command groups to the main parser."""
    _attach_header_group(subparsers)
    _attach_col_group(subparsers)
    _attach_row_group(subparsers)
    _attach_sort_group(subparsers)
    _attach_tbl_group(subparsers)
    _attach_view_group(subparsers)

def safe_register(mod, subparsers, utils_api, logger):
    try:
        mod.register(subparsers, utils_api)
        logger.info("Loaded plugin: %s", getattr(mod, "__name__", mod))
        return True
    except Exception as e:
        logger.warning("Plugin failed to load: %s (%s)", getattr(mod, "__name__", mod), e)
        return False

# In core.py (replace the entire build_parser function)

def build_parser() -> argparse.ArgumentParser:
    ap = UFMT.CustomArgumentParser(
        prog="tblkit",
        description="Modular tabular toolkit",
        formatter_class=UFMT.MainHelpFormatter,
        add_help=False
    )
    # Global options
    UP.add_common_io_args(ap)
    g = ap.add_argument_group("Global Options")
    g.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    g.add_argument("--plugins", action=UFMT.PluginsAction, help="List loaded plugins and exit.")
    g.add_argument("--version", action="version", version="tblkit 2.0.0")

    subs = ap.add_subparsers(dest="group", metavar="group", required=True)
    _attach_stable_groups(subs)
    
    # --- ADD THIS LOGIC ---
    # Load plugins so their commands are available
    utils_api = UtilsAPI()
    logger = ULOG.get_logger("tblkit.core")
    _load_plugins(subs, utils_api, logger)
    # ----------------------
    
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
            UIO.pretty_print(res)
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
        parser.print_help()
        return 0
    
    # Handle group-level help (e.g., 'tblkit col')
    if len(argv) == 1 and not argv[0].startswith('-') and hasattr(parser, '_subparsers'):
        sp_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        if sp_actions and argv[0] in sp_actions[0].choices:
            sp_actions[0].choices[argv[0]].print_help()
            return 0

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
        # Pass the global seed to any handler that needs it
        # (e.g., sample, shuffle). `args` now contains the global seed.
        return run_handler(df, args, is_header_present, logger)
    except (ValueError, KeyError, ImportError) as e:
        logger.error(str(e))
        if getattr(args, "debug", False): traceback.print_exc()
        return 2
    except (FileNotFoundError, IOError) as e:
        logger.error(str(e))
        if getattr(args, "debug", False): traceback.print_exc()
        return 3
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        if getattr(args, "debug", False): traceback.print_exc()
        return 4
    
    

if __name__ == "__main__":
    raise SystemExit(main())
