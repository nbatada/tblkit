from __future__ import annotations
import argparse, shutil, os, sys
from typing import Dict, List

_TERM_WIDTH = shutil.get_terminal_size((100, 20)).columns

# Insert near top-level helpers (once):
def _is_subparsers_action(action: argparse.Action) -> bool:
    """
    Return True for the 'subparsers' action using only public APIs:
    a mapping of names -> ArgumentParser instances.
    """
    choices = getattr(action, "choices", None)
    if not isinstance(choices, dict) or not choices:
        return False
    return all(isinstance(p, argparse.ArgumentParser) for p in choices.values())


class EnhancedHelpFormatter(argparse.HelpFormatter):
    """
    Help formatter with wider output and a clearer subcommand (subparsers) listing.
    """

    def __init__(self, prog: str) -> None:
        # Wider columns; 32 aligns help text nicely for typical commands.
        super().__init__(prog, max_help_position=32, width=_TERM_WIDTH)

    def _format_action(self, action: argparse.Action) -> str:
        # Render subparser blocks in a compact "Actions:" table.
        if isinstance(action, argparse._SubParsersAction):
            return self._format_subparsers(action)
        return super()._format_action(action)

    def _format_subparsers(self, action: argparse._SubParsersAction) -> str:
        # Map subcommand name -> short help (taken from add_parser(..., help="...")).
        helps: Dict[str, str] = {}
        # argparse exposes a private iterator for the subactions; use if present.
        get_subactions = getattr(action, "_get_subactions", None)
        if callable(get_subactions):
            for subact in get_subactions():
                # Each subaction corresponds to a subcommand choice.
                name = getattr(subact, "dest", None)
                if name:
                    helps[name] = getattr(subact, "help", "") or ""

        rows = []
        for name in action.choices.keys():
            short = helps.get(name, "")
            rows.append((name, short))

        if not rows:
            return ""

        maxlen = max(len(n) for n, _ in rows)
        out_lines = ["\n", "Actions:\n"]
        for name, short in rows:
            pad = " " * (maxlen - len(name))
            out_lines.append(f"  {name}{pad}  {short}\n")
        return "".join(out_lines)



class CommandGroupHelpFormatter(argparse.HelpFormatter):
    """
    Help formatter that renders subparsers (e.g., "Action" commands) as a
    two-column table with a colored header and **right-justified** command names.
    """

    # --- ANSI color helpers kept local to avoid extra imports ---
    _ANSI_RESET = "\033[0m"
    _ANSI_BOLD = "\033[1m"
    _ANSI_CYAN = "\033[36m"
    _ANSI_ORANGE = "\033[33m"
    @staticmethod
    def _supports_color() -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        # color only if stdout is a TTY
        return sys.stdout.isatty()

    @classmethod
    def _c(cls, text: str) -> str:
        if not cls._supports_color():
            return text
        return f"{cls._ANSI_BOLD}{cls._ANSI_CYAN}{text}{cls._ANSI_RESET}"

    # Core: override subparser rendering
    def _format_action(self, action: argparse.Action) -> str:
        # Hide the default -h/--help entry everywhere
        if getattr(action, "option_strings", None) and any(s in ("-h", "--help") for s in action.option_strings):
            return ""

        # Intercept subparsers block using public duck-typing
        if _is_subparsers_action(action):
            helps: Dict[str, str] = {}

            # Prefer argparse’s choice actions if present
            for choice_action in getattr(action, "_choices_actions", []):
                name = getattr(choice_action, "dest", None)
                if name:
                    helps[name] = getattr(choice_action, "help", "") or ""

            # Fallback to .choices mapping
            if not helps:
                for name, sub in getattr(action, "choices", {}).items():
                    helps[name] = getattr(sub, "description", "") or sub.format_usage().strip()

            if not helps:
                return ""

            names = sorted(helps.keys())
            name_w = max(len(n) for n in names)

            # Header: uncolored
            label = (getattr(action, "metavar", None) or getattr(action, "dest", "") or "command")
            header_label = label.capitalize()

            out_lines: List[str] = []
            out_lines.append("")
            out_lines.append(f"  {header_label.rjust(name_w)}  Description")
            out_lines.append(f"  {'-'*name_w}  -----------")

            # Rows
            for n in names:
                desc = helps.get(n, "")

                # Color names only (cyan for actions, orange for groups)
                if label.lower() == "action" and self._supports_color():
                    colored = f"{self._ANSI_BOLD}{self._ANSI_CYAN}{n}{self._ANSI_RESET}"
                elif label.lower() == "group" and self._supports_color():
                    colored = f"{self._ANSI_BOLD}{self._ANSI_ORANGE}{n}{self._ANSI_RESET}"
                else:
                    colored = n

                # Right-justify by RAW name length; pad before coloring
                pad = " " * max(0, name_w - len(n))
                n_disp = f"{pad}{colored}"

                # Left-justified, vertically aligned descriptions
                wrapped = self._fill_text(desc, width=_TERM_WIDTH - (name_w + 4), indent="")
                wrapped_lines = wrapped.splitlines() or [""]
                if wrapped_lines:
                    out_lines.append(f"  {n_disp}  {wrapped_lines[0]}")
                    for cont in wrapped_lines[1:]:
                        out_lines.append(f"  {' ' * name_w}  {cont}")
                else:
                    out_lines.append(f"  {n_disp}")

            out_lines.append("")
            return "\n".join(out_lines)

        # Default behavior for non-subparsers actions
        return super()._format_action(action)
    
class MainHelpFormatter(CommandGroupHelpFormatter):
    """Formatter for the top-level parser: inherits colored Action header and right-justified action names."""
    pass

class CustomArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser that prints the command's help before reporting an error.
    This mirrors tools like git and samtools and is useful for missing-required-arg cases.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("formatter_class", EnhancedHelpFormatter)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        # Print help to stderr, then exit with code 2 and the error message.
        self.print_help(sys.stderr)
        self.exit(2, f"Error: {message}\n")

class ActionParser(argparse.ArgumentParser):
    """
    Subparser class for individual actions that ensures consistent formatting:
    - Uses CommandGroupHelpFormatter by default so the "Action" table stays styled.
    - Preserves colored section headings when showing help for a single action.
    """
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("formatter_class", CommandGroupHelpFormatter)
        super().__init__(*args, **kwargs)

    # Ensure sub-subparsers (if ever used) also inherit our formatter
    def add_subparsers(self, **kwargs):
        kwargs.setdefault("parser_class", ActionParser)
        return super().add_subparsers(**kwargs)


class PluginsAction(argparse.Action):
    """
    Implements a --plugins flag that can be wired to print basic plugin info and exit.
    Core can replace/augment this behavior if a plugin registry is present.
    """

    def __init__(self, option_strings, dest, nargs=0, **kwargs) -> None:
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        sys.stdout.write("Plugin system is active. Core will report loaded plugins at runtime.\n")
        parser.exit(0)

class CommandsAction(argparse.Action):
    """
    argparse Action: --commands → print command tree and exit(0).
    """
    def __init__(self, option_strings, dest, nargs=0, **kwargs) -> None:
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        # Detect color support like the old 'view tree'
        use_color = sys.stdout.isatty() and (os.getenv("NO_COLOR") is None)
        cyan = "\033[96m" if use_color else ""
        orange = "\033[33m" if use_color else ""
        reset = "\033[0m" if use_color else ""

        # Top-level subparsers
        subparsers_actions = [a for a in parser._actions if _is_subparsers_action(a)]
        if not subparsers_actions:
            sys.stdout.write("tblkit\n")
            parser.exit(0)

        top = subparsers_actions[0]
        choices = sorted(top.choices.items())
        help_map = {a.dest: (a.help or "") for a in getattr(top, "_choices_actions", [])}

        tree = ["tblkit"]
        for gi, (gname, gparser) in enumerate(choices):
            is_last_g = (gi == len(choices) - 1)
            gpfx = "└── " if is_last_g else "├── "
            gline = f"{gpfx}{cyan}{gname}{reset}"
            gpadw = len(gpfx + gname)
            ghelp = help_map.get(gname, "")
            # Child actions
            acts = [a for a in gparser._actions if _is_subparsers_action(a)]
            if acts:
                act = acts[0]
                action_help = {a.dest: (a.help or "") for a in getattr(act, "_choices_actions", [])}
                ach = sorted(act.choices.items())
                # Group header
                tree.append(f"{gline}{' ' * max(0, 30 - gpadw)}  ({ghelp})")
                for ai, (aname, _) in enumerate(ach):
                    is_last_a = (ai == len(ach) - 1)
                    childprefix = "    " if is_last_g else "│   "
                    apfx = childprefix + ("└── " if is_last_a else "├── ")
                    ahelp = action_help.get(aname, "")
                    tree.append(f"{apfx}{orange}{aname}{reset}")
                    if ahelp:
                        # align help
                        padw = len(apfx + aname)
                        tree[-1] += f"{' ' * max(0, 30 - padw)}  ({ahelp})"
            else:
                tree.append(f"{gline}{' ' * max(0, 30 - gpadw)}  ({ghelp})")

        sys.stdout.write("\n".join(tree) + "\n")
        parser.exit(0)
        
class CommandsAction(argparse.Action):
    """
    argparse Action: --commands → print command tree and exit(0).
    """
    def __init__(self, option_strings, dest, nargs=0, **kwargs) -> None:
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        use_color = sys.stdout.isatty() and (os.getenv("NO_COLOR") is None)
        cyan = "\033[96m" if use_color else ""
        orange = "\033[33m" if use_color else ""
        reset = "\033[0m" if use_color else ""

        subparsers_actions = [a for a in parser._actions if _is_subparsers_action(a)]
        if not subparsers_actions:
            sys.stdout.write("tblkit\n")
            parser.exit(0)

        top = subparsers_actions[0]
        choices = sorted(top.choices.items())
        help_map = {a.dest: (a.help or "") for a in getattr(top, "_choices_actions", [])}

        tree = ["tblkit"]
        for gi, (gname, gparser) in enumerate(choices):
            is_last_g = (gi == len(choices) - 1)
            gpfx = "└── " if is_last_g else "├── "
            gline = f"{gpfx}{cyan}{gname}{reset}"
            gpadw = len(gpfx + gname)
            ghelp = help_map.get(gname, "")
            acts = [a for a in gparser._actions if _is_subparsers_action(a)]
            if acts:
                act = acts[0]
                action_help = {a.dest: (a.help or "") for a in getattr(act, "_choices_actions", [])}
                ach = sorted(act.choices.items())
                tree.append(f"{gline}{' ' * max(0, 30 - gpadw)}  ({ghelp})")
                for ai, (aname, _) in enumerate(ach):
                    is_last_a = (ai == len(ach) - 1)
                    childprefix = "    " if is_last_g else "│   "
                    apfx = childprefix + ("└── " if is_last_a else "├── ")
                    ahelp = action_help.get(aname, "")
                    line = f"{apfx}{orange}{aname}{reset}"
                    if ahelp:
                        padw = len(apfx + aname)
                        line += f"{' ' * max(0, 30 - padw)}  ({ahelp})"
                    tree.append(line)
            else:
                tree.append(f"{gline}{' ' * max(0, 30 - gpadw)}  ({ghelp})")

        sys.stdout.write("\n".join(tree) + "\n")
        parser.exit(0)
