"""Build the full command and function listing shown by ``textinsight --help``.

Everything is read from the argument parser and the package itself, so the
listing cannot fall out of date when a command, option or function changes.
"""

from __future__ import annotations

import argparse
import inspect
import re
import textwrap

__all__ = ["COMMAND_GROUPS", "command_reference", "function_reference"]

COMMAND_GROUPS = {
    "Analyse text": [
        "sentiment",
        "aspects",
        "keywords",
        "summarize",
        "stats",
        "redact",
        "language",
        "clean",
    ],
    "Files and reports": ["analyze"],
    "Tools": ["info", "models", "benchmark", "help"],
}
_WIDTH = 13
_LINE = 96
_MARKUP = re.compile(r"``([^`]+)``")


def _usage(command: argparse.ArgumentParser) -> str:
    parts = []
    for action in command._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if action.option_strings:
            name = max(action.option_strings, key=len)
            needs_value = action.nargs != 0
            parts.append(f"[{name} {action.dest.upper()}]" if needs_value else f"[{name}]")
        elif action.choices:
            parts.append("{" + ",".join(str(c) for c in action.choices) + "}")
        else:
            name = action.dest.upper()
            parts.append({"*": f"[{name} ...]", "?": f"[{name}]"}.get(str(action.nargs), name))
    return " ".join(parts)


def command_reference(parser: argparse.ArgumentParser) -> str:
    """List every command with its description and options, grouped by purpose."""
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    descriptions = {choice.dest: choice.help for choice in subparsers._choices_actions}
    lines = ["commands (run 'textinsight <command> --help' for details):"]
    for group, names in COMMAND_GROUPS.items():
        lines.append(f"\n  {group}")
        for name in names:
            lines.append(f"    {name:<{_WIDTH}}{descriptions[name]}")
            usage = _usage(subparsers.choices[name])
            if usage:
                indent = " " * (4 + _WIDTH)
                lines.append(
                    textwrap.fill(usage, _LINE, initial_indent=indent, subsequent_indent=indent)
                )
    return "\n".join(lines)


def function_reference() -> str:
    """List the public Python functions with the first line of each docstring."""
    import textinsight

    lines = ["python functions (import textinsight as ti; help(ti.<name>) for details):"]
    for name in sorted(textinsight.__all__):
        value = getattr(textinsight, name)
        if inspect.isfunction(value):
            summary = (inspect.getdoc(value) or "").splitlines()[0]
            plain = _MARKUP.sub(r"\1", summary)
            lines.append(f"  ti.{name:<19}{plain}")
    lines.append(f"  ti.{'Pipeline':<19}A reusable analyze() with fixed tasks and settings.")
    lines.append(f"  ti.{'config':<19}Settings such as ti.config.offline = True.")
    return "\n".join(lines)
