"""Core commands for Sidecar OS."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from sidecar_os.core.sidecar_core.events import EventStore, InboxCapturedEvent, TaskCreatedEvent, TaskCompletedEvent
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


def task(inbox_id: str, title: Optional[str] = typer.Option(None, "--title", "-t", help="Custom title for the task")) -> None:
    """Convert an inbox item to a structured task."""
    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    # Find the inbox item
    inbox_item = None
    for item in state.inbox_items.values():
        if item.event_id.startswith(inbox_id) or item.text == inbox_id:
            inbox_item = item
            break

    if not inbox_item:
        console.print(f"❌ Inbox item not found: {inbox_id}", style="red")
        return

    if inbox_item.processed:
        console.print(f"⚠️  Item already processed: {inbox_item.text}", style="yellow")
        return

    # Create task title from inbox text if not provided
    task_title = title or inbox_item.text

    # Create task event
    task_event = TaskCreatedEvent(
        payload={
            "task_id": f"task_{len(state.tasks) + 1}",
            "title": task_title,
            "description": f"Created from inbox: {inbox_item.text}",
            "created_from_event": inbox_item.event_id,
            "priority": inbox_item.priority or "normal",
            "tags": inbox_item.tags
        }
    )

    # Store the event
    event_id = store.append(task_event)
    task_id = task_event.payload["task_id"]

    # Display confirmation
    console.print(f"✓ Created task: {task_title}", style="green")
    console.print(f"  Task ID: {task_id}", style="dim")
    console.print(f"  Event ID: {event_id[:8]}...", style="dim")


def list_items(show_all: bool = typer.Option(False, "--all", "-a", help="Show all items including completed")) -> None:
    """List inbox items and tasks."""
    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    console.print("📋 Items & Tasks", style="bold blue")
    console.print()

    # Show unprocessed inbox items
    unprocessed_inbox = state.get_unprocessed_inbox()
    if unprocessed_inbox:
        inbox_table = Table(title="📥 Unprocessed Inbox Items", show_header=True, header_style="bold cyan")
        inbox_table.add_column("ID", style="dim", width=12)
        inbox_table.add_column("Text", style="white")
        inbox_table.add_column("Priority", justify="center", width=10)
        inbox_table.add_column("Added", style="dim", width=16)

        for item in sorted(unprocessed_inbox, key=lambda x: x.timestamp, reverse=True):
            inbox_table.add_row(
                item.event_id[:8] + "...",
                item.text,
                item.priority or "normal",
                item.timestamp.strftime("%m-%d %H:%M")
            )
        console.print(inbox_table)
        console.print()

    # Show active tasks
    active_tasks = state.get_active_tasks()
    if active_tasks:
        tasks_table = Table(title="✅ Active Tasks", show_header=True, header_style="bold green")
        tasks_table.add_column("Task ID", style="cyan", width=12)
        tasks_table.add_column("Title", style="white")
        tasks_table.add_column("Status", justify="center", width=10)
        tasks_table.add_column("Priority", justify="center", width=10)
        tasks_table.add_column("Created", style="dim", width=16)

        for task in sorted(active_tasks, key=lambda x: x.created_at, reverse=True):
            status_style = "yellow" if task.status == "in_progress" else "white"
            tasks_table.add_row(
                task.task_id,
                task.title,
                f"[{status_style}]{task.status}[/{status_style}]",
                task.priority or "normal",
                task.created_at.strftime("%m-%d %H:%M")
            )
        console.print(tasks_table)
        console.print()

    # Show completed tasks if requested
    if show_all:
        completed_tasks = state.get_completed_tasks()
        if completed_tasks:
            completed_table = Table(title="🏁 Completed Tasks", show_header=True, header_style="bold dim")
            completed_table.add_column("Task ID", style="dim", width=12)
            completed_table.add_column("Title", style="dim")
            completed_table.add_column("Completed", style="dim", width=16)

            for task in sorted(completed_tasks, key=lambda x: x.completed_at or x.created_at, reverse=True)[:10]:
                completed_table.add_row(
                    task.task_id,
                    task.title,
                    task.completed_at.strftime("%m-%d %H:%M") if task.completed_at else "Unknown"
                )
            console.print(completed_table)
            console.print()

    # Show summary
    if not unprocessed_inbox and not active_tasks:
        console.print("• No active items or tasks", style="dim")
        console.print("• Try: sidecar add \"New item\"", style="dim")
    else:
        console.print(f"Summary: {len(unprocessed_inbox)} inbox • {len(active_tasks)} active tasks", style="dim")


def done(query: str) -> None:
    """Mark a task as completed."""
    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    # Find the task by ID or title
    task = None
    for t in state.tasks.values():
        if (t.task_id == query or
            t.task_id.startswith(query) or
            query.lower() in t.title.lower()):
            task = t
            break

    if not task:
        console.print(f"❌ Task not found: {query}", style="red")
        console.print("💡 Try: sidecar list  # to see available tasks", style="dim")
        return

    if task.status == "completed":
        console.print(f"⚠️  Task already completed: {task.title}", style="yellow")
        return

    # Create completion event
    completion_event = TaskCompletedEvent(
        payload={
            "task_id": task.task_id,
            "completion_note": f"Completed via CLI with query: {query}"
        }
    )

    # Store the event
    event_id = store.append(completion_event)

    # Display confirmation
    console.print(f"✓ Completed task: {task.title}", style="green")
    console.print(f"  Task ID: {task.task_id}", style="dim")
    console.print(f"  Event ID: {event_id[:8]}...", style="dim")