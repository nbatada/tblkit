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
    Resolve column specifications to concrete column names (order-preserving, de-duplicated).
    Supports:
      - Names:           "age,height"
      - Excel letters:   "A:C", "AA", mixes like "A:height"
      - Numeric indices: "0:3", "2", "-2:" (0-based; negatives count from end)
      - Regex:           "re:^score_\\d+$"
      - Commas in any spec string; multiple spec strings accepted via 'specs' iterable.
    Raises KeyError/IndexError for unknown/out-of-range references.
    """
    from typing import Iterable, List, Optional
    import re
    from tblkit.utils.logging import get_logger
    logger = get_logger(__name__)

    names = list(map(str, df.columns))
    name_to_idx = {n: i for i, n in enumerate(names)}
    n = len(names)

    def xl_to_idx(tok: str) -> Optional[int]:
        t = tok.strip().upper()
        if not re.fullmatch(r"[A-Z]+", t):
            return None
        val = 0
        for ch in t:
            val = val * 26 + (ord(ch) - 64)  # A=1
        return val - 1  # to 0-based

    def to_idx(tok: str) -> Optional[int]:
        tok = tok.strip()
        if tok in name_to_idx:
            return name_to_idx[tok]
        if re.fullmatch(r"-?\d+", tok):
            i = int(tok)
            return n + i if i < 0 else i
        x = xl_to_idx(tok)
        return x

    # Flatten tokens from all spec strings
    tokens: List[str] = []
    for s in specs or []:
        if s is None:
            continue
        tokens.extend([t.strip() for t in str(s).split(",") if t.strip()])

    idxs: List[int] = []
    for tok in tokens:
        if tok.startswith("re:"):
            pat = tok[3:]
            try:
                rx = re.compile(pat)
            except re.error as e:
                raise ValueError(f"Bad regex in column spec '{tok}': {e}") from e
            for i, name in enumerate(names):
                if rx.search(name):
                    idxs.append(i)
            continue

        if ":" in tok:
            a, b = tok.split(":", 1)
            ai = to_idx(a) if a else 0
            bi = to_idx(b) if b else (n - 1)
            if ai is None and a:
                raise KeyError(f"Unknown column bound: {a}")
            if bi is None and b:
                raise KeyError(f"Unknown column bound: {b}")
            # normalize bounds
            if not (0 <= ai < n) or not (0 <= bi < n):
                raise IndexError(f"Column range out of bounds: {tok}")
            lo, hi = (ai, bi) if ai <= bi else (bi, ai)
            idxs.extend(range(lo, hi + 1))
            continue

        i = to_idx(tok)
        if i is None or not (0 <= i < n):
            raise KeyError(f"Unknown column: {tok}")
        idxs.append(i)

    out: List[str] = []
    seen = set()
    for i in idxs:
        c = names[i]
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def parse_multi_cols(spec: str, available: Optional[Sequence] = None) -> List:
    """
    Resolve comma-separated specs while preserving ORIGINAL labels (types).
    Supports exact names, 1-based indices, numeric ranges "2-5", name ranges "A:C",
    glob, and regex via "re:...".
    """
    tokens = [s.strip() for s in str(spec).split(",") if s.strip()]
    if available is None:
        return tokens
    labels = list(available)
    names  = [str(x) for x in labels]

    out: List = []
    import re as _re
    from fnmatch import fnmatch as _fnmatch
    import pandas as _pd

    for tok in tokens:
        # 1-based numeric range
        if _re.fullmatch(r"\d+-\d+", tok):
            a, b = map(int, tok.split("-"))
            a -= 1; b -= 1
            rng = range(min(a, b), max(a, b) + 1)
            out.extend(labels[i] for i in rng); continue
        # name range A:C
        if ":" in tok and not tok.startswith("re:"):
            df = _pd.DataFrame(columns=names)
            sel = resolve_columns_advanced(df, [tok])  # returns string names
            s2l = {str(l): l for l in labels}
            out.extend(s2l[s] for s in sel if s in s2l); continue
        # regex
        if tok.startswith("re:"):
            pat = _re.compile(tok[3:])
            out.extend([lab for lab, s in zip(labels, names) if pat.search(s)]); continue
        # glob
        if any(ch in tok for ch in "*?"):
            out.extend([lab for lab, s in zip(labels, names) if _fnmatch(s, tok)]); continue
        # single 1-based index
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(labels):
                out.append(labels[idx]); continue
            raise IndexError(f"Column index out of range: {tok}")
        # exact by string projection
        try:
            j = names.index(tok)
            out.append(labels[j]); continue
        except ValueError:
            raise KeyError(f"Unknown column: {tok}")

    seen = set(); uniq: List = []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq





