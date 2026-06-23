"""Shared rich console + UI helpers (banners, progress bars, colored notes)."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

THEME = Theme({
    "ok": "bold green",
    "warn": "bold yellow",
    "err": "bold red",
    "info": "bold cyan",
    "muted": "dim",
    "accent": "bold magenta",
})

console = Console(theme=THEME)

BRAND = "[accent]🎙  EGY-Emilia[/accent]"


def banner(stage: str, subtitle: str = "") -> None:
    body = f"{BRAND}  [info]{stage}[/info]"
    if subtitle:
        body += f"\n[muted]{subtitle}[/muted]"
    console.print(Panel(body, border_style="accent", expand=False))


def ok(msg: str) -> None:      console.print(f"[ok]✓[/ok] {msg}")
def info(msg: str) -> None:    console.print(f"[info]ℹ[/info] {msg}")
def warn(msg: str) -> None:    console.print(f"[warn]⚠[/warn] {msg}")
def err(msg: str) -> None:     console.print(f"[err]✗[/err] {msg}")
def note(msg: str) -> None:    console.print(f"[muted]   ↳ {msg}[/muted]")


def progress(transient: bool = False) -> Progress:
    """A consistent, pretty progress bar with spinner, %, count, ETA."""
    return Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("[muted]•[/muted]"),
        TimeElapsedColumn(),
        TextColumn("[muted]eta[/muted]"),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )
