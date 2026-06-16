# SPDX-License-Identifier: MIT
"""OCCLUDE ASCII art and decorative elements."""

from rich.align import Align
from rich.panel import Panel
from rich.text import Text

# Main logo — ANSI Shadow figlet font
OCCLUDE_LOGO = """
 ██████╗  ██████╗ ██████╗██╗     ██╗   ██╗██████╗ ███████╗
██╔═══██╗██╔════╝██╔════╝██║     ██║   ██║██╔══██╗██╔════╝
██║   ██║██║     ██║     ██║     ██║   ██║██║  ██║█████╗
██║   ██║██║     ██║     ██║     ██║   ██║██║  ██║██╔══╝
╚██████╔╝╚██████╗╚██████╗███████╗╚██████╔╝██████╔╝███████╗
 ╚═════╝  ╚═════╝ ╚═════╝╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝
"""

_GRADIENT = ["#FFFFFF", "#FFFFFF", "#DDDDDD", "#DDDDDD", "#AAAAAA", "#AAAAAA"]


def get_styled_logo() -> Text:
    """Return the OCCLUDE logo with a white-to-gray fade."""
    text = Text()
    for i, line in enumerate(OCCLUDE_LOGO.strip("\n").split("\n")):
        color = _GRADIENT[min(i, len(_GRADIENT) - 1)]
        text.append(line + "\n", style=color)
    return text


def get_header_panel() -> Panel:
    """Generate the complete header panel, centered."""
    return Panel(
        Align.center(get_styled_logo()),
        border_style="#FFFFFF",
        padding=(1, 2),
    )


# Smaller logo — thin ASCII style
OCCLUDE_LOGO_SMALL = """
               |             |
,---.,---.,---.|    .   .,---|,---.
|   ||    |    |    |   ||   ||---'
`---'`---'`---'`---'`---'`---'`---'
"""
