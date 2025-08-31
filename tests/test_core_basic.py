# tests/test_core_basic.py  (sanity smoke; uses handlers directly, NOT run_handler)
from types import SimpleNamespace as NS
import pandas as pd
from tblkit.core import (
    _handle_header_view,
    _handle_col_select, _handle_row_filter, _handle_tbl_melt
)

def test_header_view_dataframe():
    df = pd.DataFrame({"A":[1,2], "B":[3,4]})
    out = _handle_header_view(df, NS(), is_header_present=True)
    assert list(out.columns) == ["#","header","sample_data_row_1"]
    assert out.iloc[0]["header"] == "A"

def test_col_select_and_row_filter_and_melt():
    df = pd.DataFrame({"A":[3,1,2], "B":["x","y","z"]})
    out = _handle_col_select(df, NS(columns="A", type=None, invert=False), is_header_present=True)
    assert list(out.columns) == ["A"]
    out2 = _handle_row_filter(df, NS(expr="A > 1", invert=False), is_header_present=True)
    assert set(out2["A"]) == {2,3}
    df2 = pd.DataFrame({"id":[1,2], "x":[10,30], "y":[20,40]})
    out3 = _handle_tbl_melt(df2, NS(id_vars="id", value_vars="x,y", var_name="k", value_name="v"), is_header_present=True)
    assert set(out3.columns) == {"id","k","v"} and len(out3) == 4
