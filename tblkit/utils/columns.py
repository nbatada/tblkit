from __future__ import annotations
import re
from fnmatch import fnmatch
import pandas as pd
from typing import Iterable, List, Sequence, Optional

def parse_single_col(spec: str, available: Optional[Sequence[str]] = None) -> str:
    """
    Resolve one column spec. Supports:
      - exact name,
      - 1-based index ("2"),
      - glob ("val*"),
      - regex ("re:^tmp_").
    If available is None, returns the trimmed spec without validation.
    """
    s = spec.strip()
    if available is None:
        return s
    names = [str(x) for x in available]

    if s.isdigit():
        idx = int(s) - 1
        if 0 <= idx < len(names):
            return names[idx]
        raise IndexError(f"Column index out of range: {s}")

    if s in names:
        return s

    if any(ch in s for ch in "*?"):
        matches = [n for n in names if fnmatch(n, s)]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"Ambiguous glob or no match: {s} -> {matches}")

    if s.startswith("re:"):
        pat = re.compile(s[3:])
        matches = [n for n in names if pat.search(n)]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"Ambiguous regex or no match: {s} -> {matches}")

    raise KeyError(f"Unknown column: {s}")

def resolve_columns_advanced(df: pd.DataFrame, specs: Iterable[str]) -> List[str]:
    """Resolve ranges using names (e.g., A:C) or Excel-style letters (e.g., B:D, AA:AC).
    Also accepts mixing (e.g., A:C where A is a name and C is a letter). Deduplicates, preserves order."""
    names = list(map(str, df.columns))
    name_to_idx = {n: i for i, n in enumerate(names)}

    def xl_to_idx(tok: str) -> Optional[int]:
        t = tok.strip().upper()
        if not re.fullmatch(r"[A-Z]+", t):
            return None
        val = 0
        for ch in t:
            val = val * 26 + (ord(ch) - 64)  # A=1
        return val - 1  # to 0-based

    out: List[str] = []
    for spec in specs:
        spec = spec.strip()
        if ":" in spec:
            start_tok, end_tok = [s.strip() for s in spec.split(":", 1)]
            i = name_to_idx.get(start_tok)
            j = name_to_idx.get(end_tok)

            if i is None:
                i = xl_to_idx(start_tok)
            if j is None:
                j = xl_to_idx(end_tok)

            if i is None or j is None:
                raise KeyError(f"Unknown column in range: {spec}")

            if not (0 <= i < len(names)) or not (0 <= j < len(names)):
                raise IndexError(f"Range out of bounds: {spec} over {len(names)} columns")

            rng = range(i, j + 1) if i <= j else range(j, i + 1)
            out.extend(names[k] for k in rng)
        else:
            if spec in name_to_idx:
                out.append(spec)
            else:
                k = xl_to_idx(spec)
                if k is None or not (0 <= k < len(names)):
                    raise KeyError(f"Unknown column: {spec}")
                out.append(names[k])

    # de-duplicate while preserving order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq

def parse_multi_cols(spec: str, available: Optional[Sequence[str]] = None) -> List[str]:
    """
    Resolve comma-separated specs. Supports:
      - exact names,
      - 1-based indices,
      - numeric ranges "2-5",
      - name ranges "A:C",
      - glob patterns,
      - regex via "re:pattern".
    Deduplicates while preserving order. If available is None, returns tokens trimmed.
    """
    tokens = [s.strip() for s in spec.split(",") if s.strip()]
    if available is None:
        return tokens

    names = [str(x) for x in available]
    out: List[str] = []
    for tok in tokens:
        # 1-based numeric range
        if re.fullmatch(r"\d+-\d+", tok):
            a, b = map(int, tok.split("-"))
            a -= 1; b -= 1
            rng = range(min(a, b), max(a, b) + 1)
            out.extend(names[i] for i in rng); continue
        # name range A:C (but not regex)
        if ":" in tok and not tok.startswith("re:"):
            df = pd.DataFrame(columns=names)
            out.extend(resolve_columns_advanced(df, [tok])); continue
        # regex
        if tok.startswith("re:"):
            pat = re.compile(tok[3:])
            out.extend([n for n in names if pat.search(n)]); continue
        # glob
        if any(ch in tok for ch in "*?"):
            out.extend([n for n in names if fnmatch(n, tok)]); continue
        # single 1-based index
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(names):
                out.append(names[idx]); continue
            raise IndexError(f"Column index out of range: {tok}")
        # exact
        if tok in names:
            out.append(tok); continue
        raise KeyError(f"Unknown column: {tok}")

    seen = set(); uniq: List[str] = []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq

