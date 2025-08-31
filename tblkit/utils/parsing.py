from __future__ import annotations
import argparse

def build_epilog(title: str, items: list[str]) -> str:
    if not items:
        return ""
    width = max(len(x) for x in items)
    lines = ["", title]
    for x in items:
        pad = " " * (width - len(x))
        lines.append(f"  {x}{pad}  ")
    return "\n".join(lines)

def add_common_io_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("I/O")
    g.add_argument("-i", "--input", help="Input table file (default: stdin).")
    # Use -O/--out-file for filesystem output to avoid colliding with per-command --output (column name)
    g.add_argument("-O", "--out-file", dest="out_file",
                   help="Output table file (default: stdout).")
    g.add_argument("--sep", default="\t",
                   help="Input field separator (default: TAB).")
    g.add_argument("--output-sep", dest="output_sep",
                   help="Output field separator (default: same as --sep).")
    g.add_argument("--encoding", default="utf-8",
                   help="Text encoding for I/O (default: utf-8).")
    g.add_argument("--na-values", nargs="+",
                   help="Additional strings to recognize as NA/NaN.")
    g.add_argument("--on-bad-lines", choices=("error", "warn", "skip"), default="error",
                   help="Malformed line behavior when reading (default: error).")
    g.add_argument("--na-rep", default="",
                   help="String to represent missing values on output (default: empty).")
    g.add_argument("--no-header", action="store_true",
                   help="Treat input as headerless.")
    g.add_argument("--pretty", action="store_true",
                   help="Preview result to stderr instead of writing.")
    g.add_argument("--quiet", action="store_true",
                   help="Suppress non-critical logs.")
    g.add_argument("--debug", action="store_true",
                   help="Verbose debug logging.")
    g.add_argument("--log-file", dest="log_file",
                   help="Optional path to write logs.")
    g.add_argument("--seed", type=int,
                   help="Random seed for commands that sample or shuffle.")
    
