# Utility package public API for plugins and core to consume.
from __future__ import annotations

from . import io, parsing, columns, formatters
from . import logging as ULOG

class UtilsAPI:
    """Façade exposing helpers without binding methods."""
    def __init__(self) -> None:
        # I/O
        self.read_table = io.read_table
        self.write_table = io.write_table
        self.pretty_print = io.pretty_print
        # Parsing helpers
        self.add_common_io_args = parsing.add_common_io_args
        self.build_epilog = parsing.build_epilog
        # Column helpers
        self.resolve_columns_advanced = columns.resolve_columns_advanced
        self.parse_single_col = columns.parse_single_col
        self.parse_multi_cols = columns.parse_multi_cols
        # Help/formatters
        self.ActionParser = formatters.ActionParser
        self.CommandGroupHelpFormatter = formatters.CommandGroupHelpFormatter
        # Logger convenience
        self.logger = ULOG.get_logger("tblkit.plugins")

__all__ = ["UtilsAPI", "io", "parsing", "columns", "formatters", "ULOG"]
