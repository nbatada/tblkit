import pandas as pd
from types import SimpleNamespace as NS

def _mk_df():
    return pd.DataFrame({
        0: ["a", "b", "c"],
        1: ["New York, NY", "Los Angeles, CA", "Austin, TX"],
        "val1": [1, 2, 3],
        "val2": [10, 20, 30],
        "tmp_A": ["x","y","z"],
        "tmp_B": ["p","q","r"],
    })

def test_parse_multi_cols_preserves_labels():
    from tblkit.utils.columns import parse_multi_cols
    df = _mk_df()
    assert parse_multi_cols("1-3", df.columns) == [0, 1, "val1"]
    assert parse_multi_cols("val*", df.columns) == ["val1", "val2"]
    assert parse_multi_cols("re:^tmp_", df.columns) == ["tmp_A", "tmp_B"]

def test_view_with_integer_columns():
    from tblkit.core import _handle_view
    df = _mk_df()
    args = NS(columns="1-3", max_cols=None, max_col_width=40, show_full=False)
    out = _handle_view(df, args, is_header_present=True)
    assert list(out.columns) == [0, 1, "val1"]
    assert out.shape == (3, 3)

def test_read_table_with_commas(tmp_path):
    from tblkit.utils.io import read_table, write_table
    p = tmp_path/"in.csv"
    df = _mk_df()
    write_table(df, str(p), sep=",", header=True)
    df2 = read_table(str(p), sep="auto", header=0)
    assert df2.iloc[0,1] == "New York, NY"

def test_col_replace_csv_aware():
    from tblkit.core import _handle_col_replace
    df = _mk_df()
    args = NS(columns="val1,val2", vals_from="1,10", vals_to="100,1000", na_only=False, regex=False)
    out = _handle_col_replace(df, args, is_header_present=True)
    assert out.at[0,"val1"] == 100 and out.at[0,"val2"] == 1000
