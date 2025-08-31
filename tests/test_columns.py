# tests/test_columns.py
import pandas as pd
from tblkit.utils import columns as U

def test_selectors():
    cols = ["A","B","C","tmp_1","tmp_2"]
    assert U.parse_single_col("A", cols) == "A"
    assert U.parse_single_col("2", cols) == "B"
    assert U.parse_multi_cols("A,C", cols) == ["A","C"]
    assert U.parse_multi_cols("2-4", cols) == ["B","C","tmp_1"]
    assert U.parse_multi_cols("tmp_*", cols) == ["tmp_1","tmp_2"]
    assert U.parse_multi_cols("re:^tmp_", cols) == ["tmp_1","tmp_2"]
    df = pd.DataFrame(columns=cols)
    # letter range (Excel-style) supported
    assert U.resolve_columns_advanced(df, ["B:D","A"]) == ["B","C","tmp_1","A"]
