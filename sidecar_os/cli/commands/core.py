"""Core commands for Sidecar OS."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sidecar_os.core.sidecar_core.events import EventStore, InboxCapturedEvent
from sidecar_os.core.sidecar_core.state import project_events_to_state

console = Console()

def add(text: str) -> None:
    """Add a new task or note."""
    # Strip whitespace for consistency
    trimmed_text = text.strip()

    # Create and store inbox captured event
    event = InboxCapturedEvent(
        payload={"text": trimmed_text, "priority": "normal"}
    )

    # Store the event
    store = EventStore()
    event_id = store.append(event)

    # Display confirmation with event ID
    console.print(f"✓ Added to inbox: {trimmed_text}", style="green")
    console.print(f"  Event ID: {event_id[:8]}...", style="dim")

def status() -> None:
    """Show current status."""
    console.print("📊 Sidecar OS Status", style="bold blue")

    # Load and project events to current state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    # Create status table
    status_table = Table(show_header=False, box=None, padding=(0, 2))
    status_table.add_column("Item", style="cyan")
    status_table.add_column("Count", justify="right", style="bold")

    # Add status rows
    status_table.add_row("📥 Inbox Items", str(state.stats.inbox_count))
    status_table.add_row("🔄 Unprocessed", str(state.stats.unprocessed_inbox_count))
    status_table.add_row("✅ Active Tasks", str(state.stats.active_tasks))
    status_table.add_row("🏁 Completed", str(state.stats.completed_tasks))
    status_table.add_row("📊 Total Events", str(state.stats.total_events))

    console.print(status_table)
    console.print()

    # Show recent inbox items if any
    if state.inbox_items:
        recent_items = state.get_recent_inbox_items(limit=5)

        inbox_panel = Panel(
            "\n".join([
                f"{'🔄' if not item.processed else '✓'} {item.text}"
                for item in recent_items
            ]) or "No items",
            title="Recent Inbox Items",
            border_style="blue"
        )
        console.print(inbox_panel)

    # Show active tasks if any
    if state.tasks:
        active_tasks = state.get_active_tasks()[:5]  # Limit to 5

        if active_tasks:
            tasks_panel = Panel(
                "\n".join([
                    f"• {task.title} ({task.status})"
                    for task in active_tasks
                ]) or "No active tasks",
                title="Active Tasks",
                border_style="green"
            )
            console.print(tasks_panel)

    if not state.inbox_items and not state.tasks:
        console.print("• System ready - no data yet", style="dim")
        console.print("• Try: sidecar add \"Your first item\"", style="dim")