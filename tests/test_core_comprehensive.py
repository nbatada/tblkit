# tests/test_core_comprehensive.py  (full coverage for header/col/row/sort/tbl/view)
import pandas as pd
from types import SimpleNamespace as NS
from tblkit.core import (
    _handle_header_add, _handle_header_view, _handle_header_rename,
    _handle_col_add, _handle_col_cast, _handle_col_drop, _handle_col_eval, _handle_col_fillna,
    _handle_col_join, _handle_col_rename, _handle_col_split,
    _handle_row_filter, _handle_row_head, _handle_row_tail, _handle_row_unique,
    _handle_row_sample, _handle_row_shuffle,
    _handle_sort_header, _handle_sort_row,
    _handle_tbl_aggregate, _handle_tbl_concat, _handle_tbl_melt, _handle_tbl_pivot, _handle_tbl_transpose,
    _handle_view_frequency, _handle_view_tree
)

def test_header_add_view_rename():
    df = pd.DataFrame([[1,2],[3,4]])
    out_add = _handle_header_add(df.copy(), NS(prefix="c", start=1, force=False), is_header_present=False)
    assert list(out_add.columns) == ["c1","c2"]
    out_view = _handle_header_view(pd.DataFrame({"A":[1,2],"B":[3,4]}), NS(), is_header_present=True)
    assert list(out_view.columns) == ["#","header","sample_data_row_1"]
    assert out_view.iloc[0]["header"] == "A"
    out_ren = _handle_header_rename(pd.DataFrame({"A":[1], "B":[2]}), NS(map="A:X,B:Y", from_file=None), is_header_present=True)
    assert list(out_ren.columns) == ["X","Y"]

def test_columns_suite_all_ops():
    df = pd.DataFrame({"A": ["1", "2", "x"], "B": ["p", "q", "r"], "C": [10, 20, 30]})

    # add before column B
    out_add = _handle_col_add(df, NS(columns="B", new_header="Z", value="v"), is_header_present=True)
    assert out_add.columns.tolist()[:2] == ["A", "Z"]

    # cast numeric strings to int where possible, keep non-numeric as original
    out_cast = _handle_col_cast(df, NS(columns="A", to="int"), is_header_present=True)
    assert list(out_cast["A"]) == [1, 2, "x"]

    # drop column B
    out_drop = _handle_col_drop(df, NS(columns="B"), is_header_present=True)
    assert "B" not in out_drop.columns

    # eval to create new column D = C*2
    out_eval = _handle_col_eval(df, NS(expr="C * 2", output="D"), is_header_present=True)
    assert out_eval["D"].tolist() == [20, 40, 60]

    # fillna on A,B with '0' (compare as strings to allow type coercion)
    df_nan = pd.DataFrame({"A": [None, "2", None], "B": ["p", None, "r"], "C": [10, 20, 30]})
    out_fill = _handle_col_fillna(df_nan, NS(columns="A,B", value="0"), is_header_present=True)
    assert list(map(str, out_fill["A"].tolist())) == ["0", "2", "0"]
    assert list(map(str, out_fill["B"].tolist())) == ["p", "0", "r"]

    # join A and B into J with '-' keeping originals
    out_join = _handle_col_join(df, NS(columns="A,B", delimiter="-", keep=True, output="J"), is_header_present=True)
    assert out_join["J"].tolist() == ["1-p", "2-q", "x-r"]

    # split B on literal '|' into B_1, B_2 (fixed=True to avoid regex; maxsplit=1 to cap parts)
    df_split = pd.DataFrame({"B": ["p|q", "r|s", "t"]})
    out_split = _handle_col_split(
        df_split,
        NS(columns="B", delimiter="|", fixed=True, into=None, keep=False, maxsplit=1),
        is_header_present=True,
    )
    assert set(out_split.columns) == {"B_1", "B_2"}
    assert out_split["B_1"].tolist()[:2] == ["p", "r"]
    assert out_split["B_2"].isna().iloc[2]

    # col rename delegates to header rename
    out_colren = _handle_col_rename(
        pd.DataFrame({"A": [1], "B": [2]}),
        NS(map="A:X,B:Y", from_file=None, columns=None, exclude=None),
        is_header_present=True,
    )
    assert list(out_colren.columns) == ["X", "Y"]

def test_row_filter_head_tail_unique_sample_shuffle():
    df = pd.DataFrame({"A":[3,1,2,2], "B":["x","y","z","z"]})
    out_flt = _handle_row_filter(df, NS(expr="A > 1", invert=False), is_header_present=True)
    assert set(out_flt["A"]) == {2,3}
    out_inv = _handle_row_filter(df, NS(expr="A > 1", invert=True), is_header_present=True)
    assert out_inv["A"].tolist() == [1]
    assert len(_handle_row_head(df, NS(n=2), is_header_present=True)) == 2
    assert _handle_row_tail(df, NS(n=1), is_header_present=True)["A"].tolist() == [2]
    out_unique = _handle_row_unique(df, NS(columns="B", invert=False), is_header_present=True)
    assert out_unique["B"].tolist() == ["x","y","z"]
    out_dups = _handle_row_unique(df, NS(columns="B", invert=True), is_header_present=True)
    assert out_dups["B"].tolist() == ["z","z"]
    samp_n = _handle_row_sample(df, NS(n=2, f=None, seed=42, with_replacement=False), is_header_present=True)
    assert len(samp_n) == 2
    shuf = _handle_row_shuffle(df, NS(seed=0), is_header_present=True)
    assert set(shuf["A"]) == set(df["A"])

def test_sort_header_and_rows():
    df = pd.DataFrame({"B":[2,1], "A":[1,2]})
    out_hdr = _handle_sort_header(df, NS(natural=False), is_header_present=True)
    assert out_hdr.columns.tolist() == ["A","B"]
    df2 = pd.DataFrame({"A":[3,1,2], "B":[9,8,7]})
    out_rows = _handle_sort_row(df2, NS(by="A", descending=False, natural=False), is_header_present=True)
    assert out_rows["A"].tolist() == [1,2,3]
    out_rows_desc = _handle_sort_row(df2, NS(by="B", descending=True, natural=False), is_header_present=True)
    assert out_rows_desc["B"].tolist() == [9,8,7]

def test_tbl_aggregate_concat_melt_pivot_transpose(tmp_path):
    df = pd.DataFrame({"g":["a","a","b"], "x":[1,2,3], "y":[10,20,30]})
    out_agg = _handle_tbl_aggregate(df, NS(columns="x,y", funcs="sum", group="g"), is_header_present=True)
    # robust to MultiIndex vs flat
    import pandas as _pd
    flat = ["_".join([str(x) for x in t if x != ""]) for t in out_agg.columns] if isinstance(out_agg.columns, _pd.MultiIndex) else [str(x) for x in out_agg.columns]
    assert any(c in ("g","g_") for c in flat)
    # value for x sum when g=='a'
    if isinstance(out_agg.columns, _pd.MultiIndex) and ("x","sum") in out_agg.columns:
        v = out_agg.loc[out_agg["g"]=="a", ("x","sum")].item()
    else:
        xcols = [c for c in out_agg.columns if (isinstance(c, tuple) and c[0]=="x") or c=="x_sum" or (isinstance(c, str) and c.endswith("_sum") and c.startswith("x"))]
        v = out_agg.loc[out_agg["g"]=="a", xcols].iloc[0, -1]
    assert v == 3
    # concat: with file
    f = tmp_path / "part.csv"
    df.to_csv(f, index=False)
    out_cat = _handle_tbl_concat(df, NS(files=[str(f)], sep=",", encoding="utf-8"), is_header_present=True)
    assert len(out_cat) == len(df)*2
    # melt
    out_melt = _handle_tbl_melt(df, NS(id_vars="g", value_vars="x,y", var_name="k", value_name="v"), is_header_present=True)
    assert set(out_melt.columns) == {"g","k","v"} and len(out_melt) == 6
    # pivot back to wide
    out_piv = _handle_tbl_pivot(out_melt, NS(index="g", columns="k", values="v", agg="sum"), is_header_present=True)
    assert set(out_piv.columns) >= {"g","x","y"}
    # transpose
    left = pd.DataFrame({"id":[1,2], "x":[10,30], "y":[20,40]})
    out_tr = _handle_tbl_transpose(left, NS(), is_header_present=True)
    assert out_tr.shape == (2,3)

def test_view_frequency_and_tree(capsys):
    df = pd.DataFrame({"A":["x","x","y"], "B":["u","u","u"], "C":[1,2,3]})
    out_freq = _handle_view_frequency(df, NS(columns=None, all_columns=False, n=2), is_header_present=True)
    assert "rank" in out_freq.columns and set(out_freq.columns) >= {"rank","A","B"}
    _ = _handle_view_tree(df, NS(), is_header_present=True)
    captured = capsys.readouterr().out
    assert "header" in captured and "col" in captured and "tbl" in captured
