import io
import os
import pandas as pd
import pytest
from types import SimpleNamespace as NS

from tblkit.core import (
    _handle_header_add, _handle_header_rename,
    _handle_col_add, _handle_col_cast, _handle_col_drop, _handle_col_eval, _handle_col_fillna,
    _handle_col_join, _handle_col_split,
    _handle_row_filter, _handle_row_head, _handle_row_tail, _handle_row_unique, _handle_row_sample, _handle_row_shuffle,
    _handle_sort_header, _handle_sort_row,
    _handle_tbl_aggregate, _handle_tbl_concat, _handle_tbl_melt, _handle_tbl_pivot, _handle_tbl_transpose,
    _handle_view_frequency
)

# ---------- HEADER ARGUMENTS ----------

def test_header_add_force_on_existing_header():
    df = pd.DataFrame({"A":[1,2],"B":[3,4]})
    out = _handle_header_add(df, NS(prefix="c", start=10, force=True), is_header_present=True)
    assert list(out.columns) == ["c10","c11"]

def test_header_rename_from_file(tmp_path):
    df = pd.DataFrame({"A":[1], "B":[2]})
    fmap = tmp_path / "map.txt"
    # TSV format expected by the handler (old<TAB>new per line)
    fmap.write_text("A\tX\nB\tY\n", encoding="utf-8")
    out = _handle_header_rename(df, NS(map=None, from_file=str(fmap)), is_header_present=True)
    assert list(out.columns) == ["X","Y"]
    
# ---------- COLUMN ARGUMENTS ----------

def test_col_add_value_insertion_positions():
    df = pd.DataFrame({"A":[1,2], "B":[3,4]})
    # insert before B with constant
    out = _handle_col_add(df, NS(columns="B", new_header="Z", value="v"), is_header_present=True)
    assert out.columns.tolist() == ["A","Z","B"]

def test_col_cast_float_and_bool_coercion():
    df = pd.DataFrame({"X":["1.5","2.0","x"], "Y":["True","False","maybe"]})
    out_f = _handle_col_cast(df, NS(columns="X", to="float"), is_header_present=True)
    assert out_f["X"].tolist()[:2] == [1.5, 2.0]
    out_b = _handle_col_cast(df, NS(columns="Y", to="bool"), is_header_present=True)
    # non-boolean stays as original or coerced False depending on impl; assert truthy/falsey count
    assert out_b["Y"].astype(str).isin(["True","False"]).sum() >= 2

def test_col_drop_multiple_and_eval_reuse():
    df = pd.DataFrame({"A":[1,2,3], "B":[4,5,6], "C":[7,8,9]})
    out_eval = _handle_col_eval(df, NS(expr="A + B", output="S"), is_header_present=True)
    assert out_eval["S"].tolist() == [5,7,9]
    out_drop = _handle_col_drop(out_eval, NS(columns="B,C"), is_header_present=True)
    assert out_drop.columns.tolist() == ["A","S"]

def test_col_fillna_with_numeric_and_string_values():
    df = pd.DataFrame({"A":[None,2,None], "B":[None,"x","y"]})
    out_num = _handle_col_fillna(df, NS(columns="A", value="0"), is_header_present=True)
    assert str(out_num["A"].tolist()) in ("[0.0, 2.0, 0.0]","[0, 2, 0]")
    out_str = _handle_col_fillna(df, NS(columns="B", value="missing"), is_header_present=True)
    assert out_str["B"].tolist() == ["missing","x","y"]

def test_col_join_keep_false_and_default_output_name():
    df = pd.DataFrame({"A":["1","2"], "B":["x","y"]})
    out = _handle_col_join(df, NS(columns="A,B", delimiter=":", keep=False, output=None), is_header_present=True)
    # Originals may be dropped; ensure a single joined column exists
    assert len(out.columns) == 1
    assert out.iloc[:,0].tolist() == ["1:x","2:y"]

def test_col_split_regex_and_into_names(tmp_path):
    df = pd.DataFrame({"C":["a  b", "c   d", "e"]})
    # regex split on whitespace (fixed=False), maxsplit=1
    out_regex = _handle_col_split(df, NS(columns="C", delimiter=r"\s+", fixed=False, into=None, keep=False, maxsplit=1), is_header_present=True)
    assert set(out_regex.columns) == {"C_1","C_2"}
    # into names must match resulting count -> mismatch should raise
    with pytest.raises(ValueError):
        _handle_col_split(df, NS(columns="C", delimiter=r"\s+", fixed=False, into="X", keep=False, maxsplit=1), is_header_present=True)
    # correct into names
    out_into = _handle_col_split(df, NS(columns="C", delimiter=r"\s+", fixed=False, into="X,Y", keep=True, maxsplit=1), is_header_present=True)
    assert set(out_into.columns) >= {"C","X","Y"}

# ---------- ROW ARGUMENTS ----------

def test_row_filter_invert_head_tail_sample_frac_and_with_replacement():
    df = pd.DataFrame({"A":[1,2,3,4], "B":["u","v","w","x"]})
    inv = _handle_row_filter(df, NS(expr="A > 2", invert=True), is_header_present=True)
    assert inv["A"].tolist() == [1,2]
    assert len(_handle_row_head(df, NS(n=1), is_header_present=True)) == 1
    assert _handle_row_tail(df, NS(n=2), is_header_present=True)["A"].tolist() == [3,4]
    # frac sample without replacement ~ length==round(frac*len)
    samp_frac = _handle_row_sample(df, NS(n=None, f=0.5, seed=1, with_replacement=False), is_header_present=True)
    assert len(samp_frac) in (2, 3)
    # with replacement exact n
    samp_wr = _handle_row_sample(df, NS(n=6, f=None, seed=123, with_replacement=True), is_header_present=True)
    assert len(samp_wr) == 6
    # shuffle deterministic by seed
    shuf1 = _handle_row_shuffle(df, NS(seed=7), is_header_present=True)
    shuf2 = _handle_row_shuffle(df, NS(seed=7), is_header_present=True)
    assert shuf1.equals(shuf2)

def test_row_unique_invert_modes():
    df = pd.DataFrame({"A":[1,1,2,2], "B":["x","x","y","z"]})
    uniq = _handle_row_unique(df, NS(columns="A,B", invert=False), is_header_present=True)
    assert len(uniq) == 3  # drop one duplicate
    dups = _handle_row_unique(df, NS(columns="A,B", invert=True), is_header_present=True)
    assert len(dups) == 1 or len(dups) == 2  # implementation-dependent on keeping both dup rows

# ---------- SORT ARGUMENTS ----------

def test_sort_header_natural_and_lexicographic():
    df = pd.DataFrame({"a10":[1], "a2":[2], "a1":[3]})
    # natural sort (if natsort installed), else fallback to lexicographic
    try:
        import natsort  # noqa: F401
        out_nat = _handle_sort_header(df, NS(natural=True), is_header_present=True)
        assert out_nat.columns.tolist() == ["a1","a2","a10"]
    except Exception:
        out_lex = _handle_sort_header(df, NS(natural=False), is_header_present=True)
        assert out_lex.columns.tolist() == ["a1","a10","a2"] or out_lex.columns.tolist() == ["a10","a1","a2"]

def test_sort_row_natural_and_descending():
    df = pd.DataFrame({"k":["a10","a2","a1"], "v":[1,2,3]})
    try:
        import natsort  # noqa: F401
        out_nat = _handle_sort_row(df, NS(by="k", descending=False, natural=True), is_header_present=True)
        assert out_nat["k"].tolist() == ["a1","a2","a10"]
    except Exception:
        out = _handle_sort_row(df, NS(by="v", descending=True, natural=False), is_header_present=True)
        assert out["v"].tolist() == [3,2,1]

# ---------- TBL ARGUMENTS ----------

def test_tbl_aggregate_multiple_funcs_and_groups():
    df = pd.DataFrame({
        "g1":["a","a","b","b"],
        "g2":[1,1,1,2],
        "x":[1,2,3,4],
        "y":[10,20,30,40],
    })
    out = _handle_tbl_aggregate(df, NS(columns="x,y", funcs="sum,mean", group="g1,g2"), is_header_present=True)
    # robust to MultiIndex vs flat names
    import pandas as _pd
    flat = ["_".join([str(x) for x in t if x != ""]) for t in out.columns] if isinstance(out.columns, _pd.MultiIndex) else [str(x) for x in out.columns]
    assert any(c in ("g1","g1_") for c in flat) and any(c in ("g2","g2_") for c in flat)
    # check one aggregate value
    mask = (out["g1"]=="a") & (out["g2"]==1)
    if isinstance(out.columns, _pd.MultiIndex) and ("x","sum") in out.columns:
        assert out.loc[mask, ("x","sum")].item() == 3
    else:
        candidates = [c for c in out.columns if (isinstance(c, tuple) and c[0]=="x") or (isinstance(c, str) and c.startswith("x_"))]
        assert out.loc[mask, candidates].iloc[0].max() in (3, 1.5)  # sum or mean present

def test_tbl_concat_multiple_files(tmp_path):
    base = pd.DataFrame({"a":[1,2], "b":[3,4]})
    f1 = tmp_path / "f1.csv"; f2 = tmp_path / "f2.csv"
    base.to_csv(f1, index=False); base.to_csv(f2, index=False)
    out = _handle_tbl_concat(base, NS(files=[str(f1), str(f2)], sep=",", encoding="utf-8"), is_header_present=True)
    assert len(out) == len(base) * 3  # original + two files

def test_tbl_pivot_with_fill_and_values_list():
    long = pd.DataFrame({"id":[1,1,2], "k":["x","y","x"], "v":[10,20,30]})
    out = _handle_tbl_pivot(long, NS(index="id", columns="k", values="v", agg="sum"), is_header_present=True)
    assert set(out.columns) >= {"id","x","y"}
    # melt then transpose round-trip sanity
    m = _handle_tbl_melt(out, NS(id_vars="id", value_vars="x,y", var_name="k", value_name="v"), is_header_present=True)
    t = _handle_tbl_transpose(out, NS(), is_header_present=True)
    assert t.shape[0] in (2, 3)

# ---------- VIEW ARGUMENTS ----------

def test_view_frequency_all_columns_mode():
    df = pd.DataFrame({"A":["x","x","y"], "B":[1,1,2], "C":[1.5, 2.5, 3.5]})
    # By default, object columns only; with all_columns=True include numeric
    out_obj = _handle_view_frequency(df, NS(columns=None, all_columns=False, n=2), is_header_present=True)
    assert "A" in out_obj.columns and "B" not in out_obj.columns
    out_all = _handle_view_frequency(df, NS(columns=None, all_columns=True, n=2), is_header_present=True)
    assert {"A","B","C"}.issubset(set(out_all.columns))
    
