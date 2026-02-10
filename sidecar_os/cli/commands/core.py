"""Core commands for Sidecar OS."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from sidecar_os.core.sidecar_core.events import EventStore, InboxCapturedEvent, TaskCreatedEvent, TaskCompletedEvent, ProjectCreatedEvent, ProjectFocusedEvent
from sidecar_os.core.sidecar_core.state import project_events_to_state
from sidecar_os.core.sidecar_core.router import AdvancedPatternInterpreter

console = Console()

def add(text: str, smart: bool = typer.Option(True, "--smart/--no-smart", "-s/-n", help="Enable smart interpretation")) -> None:
    """Add a new task or note with optional smart interpretation."""
    # Strip whitespace for consistency
    trimmed_text = text.strip()

    # Create and store inbox captured event
    inbox_event = InboxCapturedEvent(
        payload={"text": trimmed_text, "priority": "normal"}
    )

    # Store the event
    store = EventStore()
    inbox_event_id = store.append(inbox_event)

    # Smart interpretation if enabled
    additional_events = []
    interpretation_result = None

    if smart:
        # Load current state for context
        events = store.read_all()
        state = project_events_to_state(events)

        # Interpret the input
        interpreter = AdvancedPatternInterpreter()
        interpretation_result = interpreter.interpret_text(trimmed_text, state)

        # Store additional events if confident enough
        if interpretation_result.confidence > 0.7:
            for event in interpretation_result.events:
                # Link task events to the inbox event
                if hasattr(event, 'payload') and 'created_from_event' in event.payload:
                    event.payload['created_from_event'] = inbox_event_id
                # Link clarification events to the inbox event
                elif hasattr(event, 'payload') and 'source_event_id' in event.payload:
                    event.payload['source_event_id'] = inbox_event_id

                additional_event_id = store.append(event)
                additional_events.append((event, additional_event_id))

    # Display confirmation with event ID
    console.print(f"✓ Added to inbox: {trimmed_text}", style="green")
    console.print(f"  Event ID: {inbox_event_id[:8]}...", style="dim")

    # Show interpretation results if available
    if interpretation_result and interpretation_result.confidence > 0.5:
        console.print(f"🧠 {interpretation_result.explanation}", style="cyan")

        if additional_events:
            for event, event_id in additional_events:
                event_type = type(event).__name__.replace('Event', '').replace('Created', '').replace('Focused', 'Focus')
                console.print(f"  → Generated {event_type} ({event_id[:8]}...)", style="dim cyan")

    # Show clarification questions if needed
    if interpretation_result and interpretation_result.needs_clarification:
        console.print("❓ Needs clarification:", style="yellow")
        for i, question in enumerate(interpretation_result.clarification_questions, 1):
            console.print(f"  {i}. {question}", style="dim yellow")

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
    status_table.add_row("📂 Projects", str(state.stats.project_count))
    status_table.add_row("❓ Clarifications", str(state.stats.pending_clarifications))
    status_table.add_row("📊 Total Events", str(state.stats.total_events))

    console.print(status_table)
    console.print()

    # Show current focus and recent projects
    if state.projects:
        projects_info = []

        # Show current focus project
        if state.current_focus_project and state.current_focus_project in state.projects:
            focus_project = state.projects[state.current_focus_project]
            projects_info.append(f"🎯 Current focus: {focus_project.name}")

        # Show recent projects
        recent_projects = state.get_recent_projects(limit=3)
        if recent_projects:
            project_names = [p.name for p in recent_projects if p.project_id != state.current_focus_project]
            if project_names:
                projects_info.append(f"📂 Recent: {', '.join(project_names[:2])}")

        if projects_info:
            projects_panel = Panel(
                "\n".join(projects_info),
                title="Projects",
                border_style="magenta"
            )
            console.print(projects_panel)

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


def focus(project_query: str) -> None:
    """Set focus on a project."""
    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    # Find the project
    project = None
    for p in state.projects.values():
        if (p.project_id.lower() == project_query.lower() or
            p.name.lower() == project_query.lower() or
            project_query.lower() in [alias.lower() for alias in p.aliases]):
            project = p
            break

    if not project:
        console.print(f"❌ Project not found: {project_query}", style="red")
        if state.projects:
            console.print("💡 Available projects:", style="dim")
            for p in state.projects.values():
                aliases_str = f" ({', '.join(p.aliases)})" if p.aliases else ""
                console.print(f"  • {p.name}{aliases_str}", style="dim")
        return

    # Create focus event
    focus_event = ProjectFocusedEvent(
        payload={"project_id": project.project_id}
    )

    # Store the event
    event_id = store.append(focus_event)

    # Display confirmation
    console.print(f"🎯 Focused on project: {project.name}", style="green")
    console.print(f"  Project ID: {project.project_id}", style="dim")
    console.print(f"  Event ID: {event_id[:8]}...", style="dim")


def triage() -> None:
    """Review and process unhandled inbox items and clarifications."""
    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    console.print("🔍 Triage Mode", style="bold blue")
    console.print()

    # Show pending clarifications first
    pending_clarifications = state.get_pending_clarifications()
    if pending_clarifications:
        console.print("❓ Pending Clarifications", style="bold yellow")
        for clarification in pending_clarifications[:3]:
            source_item = state.inbox_items.get(clarification.source_event_id)
            source_text = source_item.text if source_item else "Unknown item"

            console.print(f"📝 {source_text}", style="white")
            for i, question in enumerate(clarification.questions, 1):
                console.print(f"   {i}. {question}", style="dim")
            console.print()

    # Show unprocessed inbox items
    unprocessed_inbox = state.get_unprocessed_inbox()
    if unprocessed_inbox:
        console.print("📥 Unprocessed Inbox Items", style="bold cyan")

        # Use interpreter to suggest categorization
        interpreter = AdvancedPatternInterpreter()

        for item in unprocessed_inbox[:5]:  # Show top 5
            console.print(f"📝 {item.text}", style="white")

            # Get interpretation suggestion
            result = interpreter.interpret_text(item.text, state)
            if result.confidence > 0.5:
                console.print(f"   💡 Suggestion: {result.explanation} ({result.confidence:.0%} confidence)", style="dim green")
            else:
                console.print(f"   ❓ Unclear - {result.explanation}", style="dim yellow")
            console.print()

    # Summary
    if not pending_clarifications and not unprocessed_inbox:
        console.print("✨ All caught up! No items need triage.", style="green")
    else:
        console.print(f"📊 Triage Summary: {len(pending_clarifications)} clarifications, {len(unprocessed_inbox)} unprocessed items", style="dim")
        console.print("💡 Use 'sidecar task <id>' to convert inbox items to tasks", style="dim")


def project_add(name: str, alias: Optional[str] = typer.Option(None, "--alias", "-a", help="Project alias")) -> None:
    """Manually add a new project."""
    # Load current state to check for duplicates
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    # Check if project already exists
    existing = state.find_project_by_alias(name)
    if existing:
        console.print(f"⚠️  Project already exists: {existing.name}", style="yellow")
        return

    # Generate project ID
    project_id = name.lower().replace(' ', '-').replace('_', '-')

    # Create project event
    aliases = [alias] if alias else []
    project_event = ProjectCreatedEvent(
        payload={
            "project_id": project_id,
            "name": name,
            "aliases": aliases,
            "description": "Manually created project"
        }
    )

    # Store the event
    event_id = store.append(project_event)

    # Display confirmation
    console.print(f"✓ Created project: {name}", style="green")
    console.print(f"  Project ID: {project_id}", style="dim")
    console.print(f"  Event ID: {event_id[:8]}...", style="dim")

    if alias:
        console.print(f"  Alias: {alias}", style="dim")


def project_list() -> None:
    """List all projects."""
    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    if not state.projects:
        console.print("📂 No projects found", style="dim")
        console.print("💡 Try: sidecar project-add \"Project Name\"", style="dim")
        return

    console.print("📂 Projects", style="bold blue")
    console.print()

    # Create projects table
    projects_table = Table(show_header=True, header_style="bold cyan")
    projects_table.add_column("Name", style="white")
    projects_table.add_column("ID", style="dim", width=20)
    projects_table.add_column("Aliases", style="cyan", width=20)
    projects_table.add_column("Tasks", justify="center", width=8)
    projects_table.add_column("Focus", justify="center", width=8)

    for project in state.get_recent_projects():
        # Count tasks for this project
        task_count = len(state.get_tasks_for_project(project.project_id))

        # Focus indicator
        focus_indicator = "🎯" if state.current_focus_project == project.project_id else ""

        projects_table.add_row(
            project.name,
            project.project_id,
            ", ".join(project.aliases) if project.aliases else "",
            str(task_count) if task_count > 0 else "",
            focus_indicator
        )

    console.print(projects_table)