from __future__ import annotations
import argparse
from tblkit.utils import formatters as UFMT


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
    g.add_argument("-O", "--out-file", dest="out_file", help="Output file (default: stdout).")

    # TSV by default; CSV only when requested
    g.add_argument("--sep", default="tsv",
                   help="Input field separator: tsv,csv,|,space,\\t (default: tsv).")
    g.add_argument("--output-sep", dest="output_sep",
                   help="Output field separator (default: TSV unless overridden).")

    g.add_argument("--encoding", default="utf-8")
    g.add_argument("--na-values", nargs="+")
    g.add_argument("--on-bad-lines", choices=("error","warn","skip"), default="error")
    g.add_argument("--no-header", action="store_true", help="Treat input as headerless.")
    g.add_argument("--pretty", action="store_true", help="Pretty-print result to stdout.")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--debug", action="store_true")
    g.add_argument("--log-file", dest="log_file")
    g.add_argument("--seed", type=int)
    g.add_argument("--commands", action=UFMT.CommandsAction,
                   help="Show the available commands as a tree and exit.")
