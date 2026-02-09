"""Core commands for Sidecar OS."""

import typer
from rich.console import Console

console = Console()

def add(text: str) -> None:
    """Add a new task or note."""
    console.print(f"✓ Added: {text}", style="green")

def status() -> None:
    """Show current status."""
    console.print("📊 Sidecar OS Status", style="bold blue")
    console.print("• No tasks yet")
    console.print("• System ready")