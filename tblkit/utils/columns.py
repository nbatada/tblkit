from __future__ import annotations
import re
from fnmatch import fnmatch
import pandas as pd
from typing import Iterable, List, Sequence, Optional

def parse_single_col(spec: str, available: Optional[Sequence] = None):
    """
    Resolve a single column spec without coercing labels to strings.
    Supports exact name, 1-based index, glob, and regex ("re:...").
    """
    s = spec.strip()
    if available is None:
        return s
    labels = list(available)
    names = [str(x) for x in labels]

    # 1-based index
    if s.isdigit():
        idx = int(s) - 1
        if 0 <= idx < len(labels):
            return labels[idx]
        raise IndexError(f"Column index out of range: {s}")
    # regex
    if s.startswith("re:"):
        import re as _re
        pat = _re.compile(s[3:])
        hits = [lab for lab, name in zip(labels, names) if pat.search(name)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise KeyError(f"No columns match regex: {s[3:]}")
        raise KeyError(f"Regex matches multiple columns: {s[3:]}")
    # glob
    if any(ch in s for ch in "*?"):
        from fnmatch import fnmatch as _fnmatch
        hits = [lab for lab, name in zip(labels, names) if _fnmatch(name, s)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise KeyError(f"No columns match glob: {s}")
        raise KeyError(f"Glob matches multiple columns: {s}")
    # exact by string projection
    try:
        j = names.index(s)
        return labels[j]
    except ValueError:
        raise KeyError(f"Unknown column: {s}")


def resolve_columns_advanced(df: pd.DataFrame, specs: Iterable[str]) -> List[str]:
    """
    Resolve column specifications to concrete column labels (order-preserving, de-duplicated).
    Supports:
      - Names:            "colA,B"
      - 1-based indices:  "1,3-5"  (inclusive range)
      - Excel letters:    "A:C", "AA"
      - Regex:            "re:^score_\\d+$"
      - Globs:            "l*"
    Raises KeyError with a clear message if any selection is out of bounds or unknown.
    """
    import re as _re
    from fnmatch import fnmatch as _fnmatch

    labels: List = list(df.columns)
    names: List[str] = [str(x) for x in labels]
    n = len(labels)
    out: List = []

    def _add_by_index_1based(idx1: int):
        if idx1 < 1 or idx1 > n:
            raise KeyError(f"Column index {idx1} out of range (1..{n}).")
        out.append(labels[idx1 - 1])

    tokens: List[str] = []
    for s in specs:
        if s is None: continue
        tokens += [t for t in _re.split(r"[,\s]+", str(s).strip()) if t]

    for tok in tokens:
        # 1-based numeric range "a-b"
        m = _re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            a, b = map(int, m.groups())
            lo, hi = (a, b) if a <= b else (b, a)
            for k in range(lo, hi + 1):
                _add_by_index_1based(k)
            continue
        # single 1-based index
        if _re.fullmatch(r"\d+", tok):
            _add_by_index_1based(int(tok)); continue
        # regex
        if tok.startswith("re:"):
            pat = tok[3:]
            hits = [lab for lab, nm in zip(labels, names) if _re.search(pat, nm)]
            if not hits:
                raise KeyError(f"No columns match regex: {pat}")
            out.extend(hits); continue
        # Excel letters range/name
        if ":" in tok and not tok.startswith("re:"):
            a, b = tok.split(":", 1)
            def xl_to_idx(s: str) -> int:
                s = s.strip().upper()
                if not _re.fullmatch(r"[A-Z]+", s):
                    return -1
                val = 0
                for ch in s:
                    val = val * 26 + (ord(ch) - ord('A') + 1)
                return val
            ia, ib = xl_to_idx(a), xl_to_idx(b)
            if ia > 0 and ib > 0:
                lo, hi = (ia, ib) if ia <= ib else (ib, ia)
                for k in range(lo, hi + 1):
                    _add_by_index_1based(k)
                continue
        # glob
        if any(ch in tok for ch in "*?"):
            hits = [lab for lab, nm in zip(labels, names) if _fnmatch(nm, tok)]
            if not hits:
                raise KeyError(f"No columns match glob: {tok}")
            out.extend(hits); continue
        # exact name
        if tok in names:
            out.append(labels[names.index(tok)]); continue
        raise KeyError(f"Unknown column: {tok}")

    # de-duplicate, preserve order
    seen = set(); uniq: List = []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq

def parse_multi_cols(spec: str, available: Optional[Sequence] = None) -> List:
    """
    Resolve comma-separated specs while preserving ORIGINAL labels (types).
    Supports:
      - Names                  e.g., "colA,B"
      - 1-based indices        e.g., "1,3"
      - Numeric ranges         e.g., "2-5" (inclusive, 1-based)
      - Name ranges A:C        e.g., "A:C" (Excel-style by position)
      - Globs                  e.g., "l*"
      - Regex                  e.g., "re:^score_\\d+$"
    """
    tokens = [s.strip() for s in str(spec).split(",") if s.strip()]
    if available is None:
        return tokens
    labels = list(available)
    names  = [str(x) for x in labels]
    n = len(labels)

    out: List = []
    import re as _re
    from fnmatch import fnmatch as _fnmatch
    import pandas as _pd

    for tok in tokens:
        # 1-based numeric range a-b
        if _re.fullmatch(r"\d+-\d+", tok):
            a, b = map(int, tok.split("-"))
            lo, hi = (a, b) if a <= b else (b, a)
            if lo < 1 or hi > n:
                raise KeyError(f"Column range {tok} out of bounds (1..{n}).")
            out.extend(labels[i-1] for i in range(lo, hi+1))
            continue
        # Name range A:C (by position)
        if ":" in tok and not tok.startswith("re:"):
            def xl_to_idx(s: str) -> int:
                s = s.strip().upper()
                if not _re.fullmatch(r"[A-Z]+", s): return -1
                v = 0
                for ch in s: v = v*26 + (ord(ch)-ord('A')+1)
                return v
            a, b = tok.split(":", 1)
            ia, ib = xl_to_idx(a), xl_to_idx(b)
            if ia > 0 and ib > 0:
                lo, hi = (ia, ib) if ia <= ib else (ib, ia)
                if lo < 1 or hi > n:
                    raise KeyError(f"Column range {tok} out of bounds (1..{n}).")
                out.extend(labels[i-1] for i in range(lo, hi+1))
                continue
        # regex
        if tok.startswith("re:"):
            pat = _re.compile(tok[3:])
            hits = [lab for lab, s in zip(labels, names) if pat.search(s)]
            if not hits:
                raise KeyError(f"No columns match regex: {tok[3:]}")
            out.extend(hits); continue
        # glob
        if any(ch in tok for ch in "*?"):
            hits = [lab for lab, s in zip(labels, names) if _fnmatch(s, tok)]
            if not hits:
                raise KeyError(f"No columns match glob: {tok}")
            out.extend(hits); continue
        # single 1-based index
        if tok.isdigit():
            idx = int(tok)
            if idx < 1 or idx > n:
                raise KeyError(f"Column index {tok} out of bounds (1..{n}).")
            out.append(labels[idx-1]); continue
        # exact name
        try:
            j = names.index(tok)
            out.append(labels[j]); continue
        except ValueError:
            raise KeyError(f"Unknown column: {tok}")

    # de-duplicate, preserve order
    seen = set(); uniq: List = []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq






