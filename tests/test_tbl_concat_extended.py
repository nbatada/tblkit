import os
from pathlib import Path
import pandas as pd
from types import SimpleNamespace as NS
from tblkit.core import _handle_tbl_concat

def _w(p: Path, rows: list[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")

def test_concat_with_filelist_and_regex(tmp_path: Path):
    # Arrange directories: projA/sampleB and projC/sampleD
    f1 = tmp_path / "projA" / "sampleB" / "t1.csv"
    f2 = tmp_path / "projC" / "sampleD" / "t2.csv"
    _w(f1, ["id,val", "1,10"])
    _w(f2, ["id,val", "2,20"])

    flist = tmp_path / "files.txt"
    _w(flist, [str(f1), str(f2)])

    # Regex extracts last two folders as proj/sample anywhere in the path
    regex = r"(?P<proj>[^/]+)/(?P<sample>[^/]+)/[^/]+$"
    ns = NS(files=[], filelist=str(flist), extract_from_path=regex,
            ancestor_cols_to_include=None, sep=",", encoding="utf-8")

    out = _handle_tbl_concat(None, ns, is_header_present=True)
    assert set(["proj","sample"]).issubset(out.columns)
    assert {tuple(out.loc[i, ["proj","sample"]]) for i in out.index} == {("projA","sampleB"), ("projC","sampleD")}
    assert out.shape[0] == 2

def test_concat_with_ancestor_cols(tmp_path: Path):
    f = tmp_path / "P1" / "P2" / "t.csv"
    _w(f, ["a,b", "x,y"])
    ns = NS(files=[str(f)], filelist=None, extract_from_path=None,
            ancestor_cols_to_include="proj,sample", sep=",", encoding="utf-8")

    out = _handle_tbl_concat(None, ns, is_header_present=True)
    assert set(["proj","sample"]).issubset(out.columns)
    assert out.loc[0, "proj"] == "P1"
    assert out.loc[0, "sample"] == "P2"

def test_concat_includes_piped_df_and_skips_empty(tmp_path: Path):
    # One real file + one header-only (empty) file
    f_good = tmp_path / "g.csv"
    f_empty = tmp_path / "e.csv"
    _w(f_good, ["c", "z"])
    _w(f_empty, ["c"])  # header only -> empty DataFrame
    piped = pd.DataFrame({"c": ["piped"]})

    ns = NS(files=[str(f_good), str(f_empty)], filelist=None,
            extract_from_path=None, ancestor_cols_to_include=None,
            sep=",", encoding="utf-8")
    out = _handle_tbl_concat(piped, ns, is_header_present=True)
    # 2 rows: piped + good; empty is skipped
    assert out.shape[0] == 2
    assert out.iloc[0]["c"] == "piped"
    assert out.iloc[1]["c"] == "z"
    
