"""Core commands for Sidecar OS."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional, List, Dict

from sidecar_os.core.sidecar_core.events import EventStore, InboxCapturedEvent, TaskCreatedEvent, TaskCompletedEvent, TaskScheduledEvent, TaskDurationSetEvent, TaskPriorityUpdatedEvent, TaskStatusUpdatedEvent, ProjectCreatedEvent, ProjectFocusedEvent, ProjectFocusClearedEvent, ClarificationRequestedEvent, ClarificationResolvedEvent
from sidecar_os.core.sidecar_core.state import project_events_to_state
from sidecar_os.core.sidecar_core.state.models import SidecarState
from sidecar_os.core.sidecar_core.router import AdvancedPatternInterpreter, InterpreterConfig
from sidecar_os.core.sidecar_core.llm import LLMService, get_usage_tracker
from sidecar_os.core.sidecar_core.summaries import SummaryGenerator, SummaryStyle, SummaryPeriod
from sidecar_os.core.sidecar_core.projects import ProjectCleanupManager, EventLogMigrator

console = Console()

def add(text: str) -> None:
    """Add a new task or note with intelligent interpretation."""
    # Strip whitespace for consistency
    trimmed_text = text.strip()

    # Create and store inbox captured event
    inbox_event = InboxCapturedEvent(
        payload={"text": trimmed_text, "priority": "normal"}
    )

    # Store the event
    store = EventStore()
    inbox_event_id = store.append(inbox_event)

    # Always use intelligent interpretation with hybrid pattern + LLM approach
    additional_events = []
    interpretation_result = None

    # Load current state for context
    events = store.read_all()
    state = project_events_to_state(events)

    # Interpret the input with hybrid interpreter
    config = InterpreterConfig(
        use_llm=True,
        llm_confidence_threshold=0.6,
        immediate_clarification_threshold=0.3
    )
    interpreter = AdvancedPatternInterpreter(config=config)
    interpretation_result = interpreter.interpret_text(trimmed_text, state)

    # Store events based on type and confidence
    for event in interpretation_result.events:
        should_store = False

        # Always store clarification events regardless of confidence
        if isinstance(event, ClarificationRequestedEvent):
            should_store = True
            # Link clarification to the inbox event
            if hasattr(event, 'payload') and 'source_event_id' in event.payload:
                event.payload['source_event_id'] = inbox_event_id
        # Store other events only if confidence is high enough
        elif interpretation_result.confidence > 0.6:
            should_store = True
            # Link task events to the inbox event
            if hasattr(event, 'payload') and 'created_from_event' in event.payload:
                event.payload['created_from_event'] = inbox_event_id

        if should_store:
            additional_event_id = store.append(event)
            additional_events.append((event, additional_event_id))

    # Display confirmation with event ID
    console.print(f"✓ Added to inbox: {trimmed_text}", style="green")
    console.print(f"  Event ID: {inbox_event_id[:8]}...", style="dim")

    # Show interpretation results
    if interpretation_result:
        # Show analysis method and confidence
        method_icon = "🧠" if interpretation_result.used_llm else "🔍"
        method_name = interpretation_result.analysis_method.title()
        console.print(f"{method_icon} {method_name}: {interpretation_result.explanation}", style="cyan")

        # Show generated events
        if additional_events:
            for event, event_id in additional_events:
                event_type = type(event).__name__.replace('Event', '').replace('Created', '').replace('Focused', 'Focus')
                console.print(f"  → Generated {event_type} ({event_id[:8]}...)", style="dim cyan")

        # Handle immediate clarification questions (very low confidence)
        if interpretation_result.needs_clarification and interpretation_result.confidence < 0.3:
            console.print("❓ Immediate clarification needed:", style="bold yellow")
            for i, question in enumerate(interpretation_result.clarification_questions, 1):
                console.print(f"  {i}. {question}", style="dim yellow")
            console.print("💡 Please provide more details or use 'sidecar triage' later", style="dim")

        # Handle staged clarification (medium-low confidence)
        elif interpretation_result.needs_clarification:
            console.print("🤔 Added to triage queue for clarification", style="yellow")
            console.print("💡 Use 'sidecar triage' to provide additional details", style="dim")

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

    # Add temporal statistics
    overdue_tasks = state.get_overdue_tasks()
    due_today_tasks = state.get_tasks_due_today()
    if overdue_tasks:
        status_table.add_row("⚠️ Overdue", f"[red]{len(overdue_tasks)}[/red]")
    if due_today_tasks:
        status_table.add_row("📅 Due Today", f"[yellow]{len(due_today_tasks)}[/yellow]")

    status_table.add_row("🏁 Completed", str(state.stats.completed_tasks))
    status_table.add_row("📂 Projects", str(state.stats.project_count))
    status_table.add_row("❓ Clarifications", str(state.stats.pending_clarifications))
    status_table.add_row("📊 Total Events", str(state.stats.total_events))

    console.print(status_table)
    console.print()

    # Show LLM usage statistics from persistent tracker
    try:
        usage_tracker = get_usage_tracker()
        usage_summary = usage_tracker.get_usage_summary()

        # Try to get LLM service config info (provider, model, limits)
        try:
            llm_service = LLMService()
            llm_status = llm_service.get_status()
            provider = llm_status['provider'].title()
            model = llm_status['model']
            cost_limit = llm_status['cost_limit']
        except:
            provider = usage_summary.get('provider', 'Unknown').title()
            model = "claude-opus-4.6"  # Default
            cost_limit = 10.0  # Default

        # Show LLM status if there's any usage or for debugging
        if usage_summary['daily_requests'] > 0 or usage_summary['daily_cost'] > 0:
            llm_panel = Panel(
                f"🧠 Provider: {provider}\n"
                f"📞 Requests: {usage_summary['daily_requests']}\n"
                f"💰 Daily Cost: ${usage_summary['daily_cost']:.4f} / ${cost_limit:.2f}\n"
                f"📊 Model: {model}\n"
                f"🎯 Tokens: {usage_summary['daily_input_tokens']} in / {usage_summary['daily_output_tokens']} out",
                title="LLM Usage",
                border_style="cyan"
            )
            console.print(llm_panel)
        else:
            # Show minimal status when no usage
            llm_panel = Panel(
                f"🧠 Provider: {provider} (Ready)\n"
                f"📊 Model: {model}",
                title="LLM Status",
                border_style="dim"
            )
            console.print(llm_panel)
    except Exception as e:
        # Gracefully handle LLM service unavailable
        console.print(f"[dim]LLM status unavailable: {str(e)}[/dim]")
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
        active_tasks = state.get_active_tasks()[:10]  # Show more tasks

        if active_tasks:
            def format_task(task):
                """Format task with project prefix and temporal info if available."""
                # Build base task description
                if task.project_id and task.project_id in state.projects:
                    project_name = state.projects[task.project_id].name
                    base_text = f"[{project_name}] {task.title}"
                else:
                    base_text = task.title

                # Add temporal information
                temporal_info = []

                # Add due date info
                if task.scheduled_for:
                    from datetime import datetime
                    now = datetime.now(task.scheduled_for.tzinfo or None)
                    if task.scheduled_for < now:
                        temporal_info.append("⚠ Overdue")
                    elif task.scheduled_for.date() == now.date():
                        temporal_info.append("📅 Due today")
                    else:
                        days_diff = (task.scheduled_for.date() - now.date()).days
                        if days_diff <= 7:
                            temporal_info.append(f"Due {task.scheduled_for.strftime('%a')}")

                # Add duration info
                if task.duration_minutes and task.duration_minutes >= 60:
                    hours = task.duration_minutes // 60
                    temporal_info.append(f"{hours}h")
                elif task.duration_minutes:
                    temporal_info.append(f"{task.duration_minutes}m")

                # Combine everything
                if temporal_info:
                    return f"• {base_text} ({task.status}, {', '.join(temporal_info)})"
                else:
                    return f"• {base_text} ({task.status})"

            tasks_panel = Panel(
                "\n".join([
                    format_task(task)
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


def list_items(
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all items including completed"),
    due_today: bool = typer.Option(False, "--due-today", help="Show tasks due today")
) -> None:
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

    # Filter for due today if requested
    if due_today:
        active_tasks = state.get_tasks_due_today()
        table_title = "📅 Tasks Due Today"
    else:
        table_title = "✅ Active Tasks"

    if active_tasks:
        tasks_table = Table(title=table_title, show_header=True, header_style="bold green")
        tasks_table.add_column("Task ID", style="cyan", width=12)
        tasks_table.add_column("Title", style="white")
        tasks_table.add_column("Status", justify="center", width=10)
        tasks_table.add_column("Due Date", style="yellow", width=12)
        tasks_table.add_column("Duration", style="magenta", width=8)
        tasks_table.add_column("Priority", justify="center", width=8)
        tasks_table.add_column("Created", style="dim", width=12)

        # Sort tasks by due date (overdue first, then by proximity)
        if not due_today:  # Only sort by due date if not filtering for today
            active_tasks = state.get_tasks_sorted_by_due_date()
        else:
            # For due today, sort by due date time
            active_tasks = sorted(active_tasks, key=lambda x: x.scheduled_for or x.created_at)

        for task in active_tasks:
            status_style = "yellow" if task.status == "in_progress" else "white"

            # Format title with project prefix if available
            if task.project_id and task.project_id in state.projects:
                project_name = state.projects[task.project_id].name
                formatted_title = f"[{project_name}] {task.title}"
            else:
                formatted_title = task.title

            # Format due date with overdue styling
            if task.scheduled_for:
                from datetime import datetime
                now = datetime.now(task.scheduled_for.tzinfo or None)
                is_overdue = task.scheduled_for < now

                if is_overdue:
                    due_date_text = f"[red bold]⚠ {task.scheduled_for.strftime('%m-%d')}[/red bold]"
                elif task.scheduled_for.date() == now.date():
                    due_date_text = f"[yellow bold]📅 Today[/yellow bold]"
                else:
                    # Show day of week for dates within a week
                    days_diff = (task.scheduled_for.date() - now.date()).days
                    if days_diff <= 7:
                        due_date_text = f"[green]{task.scheduled_for.strftime('%a %m-%d')}[/green]"
                    else:
                        due_date_text = f"{task.scheduled_for.strftime('%m-%d')}"
            else:
                due_date_text = "[dim]--[/dim]"

            # Format duration
            if task.duration_minutes:
                if task.duration_minutes >= 60:
                    hours = task.duration_minutes // 60
                    minutes = task.duration_minutes % 60
                    if minutes == 0:
                        duration_text = f"{hours}h"
                    else:
                        duration_text = f"{hours}h{minutes}m"
                else:
                    duration_text = f"{task.duration_minutes}m"
            else:
                duration_text = "[dim]--[/dim]"

            tasks_table.add_row(
                task.task_id,
                formatted_title,
                f"[{status_style}]{task.status}[/{status_style}]",
                due_date_text,
                duration_text,
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


def update(
    natural_request: str = typer.Argument(help="Natural language task update (e.g., 'completed CA ZAP update to Abhi') or task ID for structured updates"),
    priority: Optional[str] = typer.Option(None, "--priority", "-p", help="Update priority (low, normal, high, urgent)"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Update status (pending, in_progress, completed, cancelled, on_hold)"),
    due_date: Optional[str] = typer.Option(None, "--due", "-d", help="Update due date (today, tomorrow, Friday, 2024-12-31)"),
    duration: Optional[str] = typer.Option(None, "--duration", "--dur", help="Update duration ([30min], [2h], [1.5h])")
) -> None:
    """Update task properties using natural language or structured options.

    Examples:
      sidecar update "completed CA ZAP update to Abhi"
      sidecar update "make the email task high priority and due tomorrow"
      sidecar update "mark LPD experiments as in progress"
      sidecar update abc123 --priority high --due tomorrow  # structured mode
    """
    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    # Check if this is natural language or structured mode
    has_structured_options = any([priority, status, due_date, duration])

    if not has_structured_options:
        # Natural language mode - use LLM to parse the request
        import asyncio
        asyncio.run(_handle_natural_language_update(natural_request, store, state))
        return
    else:
        # Structured mode - find task by ID or title match
        task = None
        for t in state.tasks.values():
            if (t.task_id == natural_request or
                t.task_id.startswith(natural_request) or
                natural_request.lower() in t.title.lower()):
                task = t
                break

        if not task:
            console.print(f"❌ Task not found: {natural_request}", style="red")
            console.print("💡 Try: sidecar list  # to see available tasks", style="dim")
            return

        # Continue with structured update logic
        import asyncio
        asyncio.run(_handle_structured_update(task, store, priority, status, due_date, duration))


async def _handle_natural_language_update(natural_request: str, store: EventStore, state) -> None:
    """Handle natural language task update requests using LLM parsing."""
    from sidecar_os.core.sidecar_core.llm import LLMService, LLMConfig

    # Prepare task data for LLM
    task_data = []
    for task in state.get_active_tasks()[:20]:  # Limit to prevent context overflow
        task_data.append({
            "task_id": task.task_id,  # Full task ID for matching
            "task_id_short": task.task_id[:8],  # Short ID for context
            "title": task.title,
            "project_name": state.projects[task.project_id].name if task.project_id and task.project_id in state.projects else "",
            "priority": task.priority or "normal",
            "status": task.status,
            "scheduled_for": task.scheduled_for.isoformat() if task.scheduled_for else None,
            "duration_minutes": task.duration_minutes
        })

    # Initialize LLM service
    llm_config = LLMConfig()
    llm_service = LLMService(llm_config)

    console.print(f"🧠 Parsing request: [bold]{natural_request}[/bold]")

    try:
        # Parse the natural language request
        parse_result = await llm_service.parse_task_update_request(
            natural_text=natural_request,
            available_tasks=task_data
        )

        if parse_result.get("confidence", 0) < 0.5:
            console.print("❌ Could not understand the request. Try being more specific.", style="red")
            console.print(f"   Error: {parse_result.get('explanation', 'Low confidence parsing')}", style="dim")
            return

        # Find matching tasks
        task_matches = parse_result.get("task_matches", [])
        if not task_matches:
            console.print("❌ Could not find any matching tasks.", style="red")
            console.print("💡 Try: sidecar list  # to see available tasks", style="dim")
            return

        # Use the best matching task
        best_match = task_matches[0]
        task_id = best_match.get("task_id")
        task = state.tasks.get(task_id)

        if not task:
            console.print(f"❌ Task not found: {task_id}", style="red")
            return

        console.print(f"✅ Found task: [bold]{task.title}[/bold] (confidence: {best_match.get('confidence', 0):.1f})")
        console.print(f"   Reason: {best_match.get('match_reason', 'Pattern match')}", style="dim")

        # Apply updates from LLM parsing
        updates = parse_result.get("updates", {})
        await _apply_updates_from_llm(task, store, updates)

    except Exception as e:
        console.print(f"❌ Failed to parse request: {str(e)}", style="red")
        console.print("💡 Try using structured options: sidecar update <task_id> --priority high", style="dim")


async def _apply_updates_from_llm(task, store: EventStore, updates: dict) -> None:
    """Apply updates parsed from LLM to a task."""
    events_created = []

    # Show current task info
    console.print()
    console.print(f"📋 Updating Task: [bold]{task.title}[/bold]")
    console.print(f"   Current Priority: {task.priority or 'normal'}")
    console.print(f"   Current Status: {task.status}")
    console.print(f"   Current Due Date: {task.scheduled_for.strftime('%a %b %d, %Y') if task.scheduled_for else 'Not set'}")
    console.print(f"   Current Duration: {task.duration_minutes}min" if task.duration_minutes else "   Current Duration: Not set")
    console.print()

    # Update Priority
    if updates.get("priority") and updates["priority"] != "null":
        priority = updates["priority"].lower()
        valid_priorities = ["low", "normal", "high", "urgent"]
        if priority in valid_priorities:
            priority_event = TaskPriorityUpdatedEvent(
                payload={
                    "task_id": task.task_id,
                    "priority": priority,
                    "previous_priority": task.priority or "normal"
                }
            )
            store.append(priority_event)
            events_created.append(f"Priority → {priority}")

    # Update Status
    if updates.get("status") and updates["status"] != "null":
        status = updates["status"].lower()
        valid_statuses = ["pending", "in_progress", "completed", "cancelled", "on_hold"]
        if status in valid_statuses:
            status_event = TaskStatusUpdatedEvent(
                payload={
                    "task_id": task.task_id,
                    "status": status,
                    "previous_status": task.status
                }
            )
            store.append(status_event)
            events_created.append(f"Status → {status}")

    # Update Due Date
    if updates.get("due_date") and updates["due_date"] != "null":
        due_date_text = updates["due_date"]

        # Handle relative dates
        if due_date_text.lower() in ['today', 'tomorrow']:
            from sidecar_os.core.sidecar_core.router.interpreter import AdvancedPatternInterpreter
            interpreter = AdvancedPatternInterpreter()
            due_date_iso = interpreter._convert_relative_date_to_iso(due_date_text.lower())
        else:
            # Try as ISO format
            try:
                from datetime import datetime
                parsed_date = datetime.fromisoformat(due_date_text.replace('Z', '+00:00'))
                due_date_iso = parsed_date.isoformat()
            except (ValueError, AttributeError):
                console.print(f"⚠️  Could not parse due date: {due_date_text}", style="yellow")
                due_date_iso = None

        if due_date_iso:
            scheduled_event = TaskScheduledEvent(
                payload={
                    "task_id": task.task_id,
                    "scheduled_for": due_date_iso
                }
            )
            store.append(scheduled_event)
            events_created.append(f"Due Date → {due_date_text}")

    # Update Duration
    if updates.get("duration_minutes") and isinstance(updates["duration_minutes"], int) and updates["duration_minutes"] > 0:
        duration_minutes = updates["duration_minutes"]
        duration_event = TaskDurationSetEvent(
            payload={
                "task_id": task.task_id,
                "duration_minutes": duration_minutes
            }
        )
        store.append(duration_event)

        # Format for display
        if duration_minutes >= 60:
            hours = duration_minutes // 60
            minutes = duration_minutes % 60
            if minutes == 0:
                duration_display = f"{hours}h"
            else:
                duration_display = f"{hours}h{minutes}m"
        else:
            duration_display = f"{duration_minutes}m"
        events_created.append(f"Duration → {duration_display}")

    # Display results
    if events_created:
        console.print("✅ Task updated successfully:", style="green")
        for update in events_created:
            console.print(f"   • {update}")
        console.print(f"   Task ID: {task.task_id}", style="dim")
    else:
        console.print("ℹ️  No updates were applied based on the request.", style="yellow")


async def _handle_structured_update(task, store: EventStore, priority: str, status: str, due_date: str, duration: str) -> None:
    """Handle structured task updates with explicit parameters."""
    # Show current task info
    console.print(f"📋 Updating Task: [bold]{task.title}[/bold]")
    console.print(f"   Current Priority: {task.priority or 'normal'}")
    console.print(f"   Current Status: {task.status}")
    console.print(f"   Current Due Date: {task.scheduled_for.strftime('%a %b %d, %Y') if task.scheduled_for else 'Not set'}")
    console.print(f"   Current Duration: {task.duration_minutes}min" if task.duration_minutes else "   Current Duration: Not set")
    console.print()

    events_created = []

    # Update Priority
    if priority:
        valid_priorities = ["low", "normal", "high", "urgent"]
        if priority.lower() not in valid_priorities:
            console.print(f"❌ Invalid priority: {priority}. Valid options: {', '.join(valid_priorities)}", style="red")
            return

        priority_event = TaskPriorityUpdatedEvent(
            payload={
                "task_id": task.task_id,
                "priority": priority.lower(),
                "previous_priority": task.priority or "normal"
            }
        )
        store.append(priority_event)
        events_created.append(f"Priority → {priority.lower()}")

    # Update Status
    if status:
        valid_statuses = ["pending", "in_progress", "completed", "cancelled", "on_hold"]
        if status.lower() not in valid_statuses:
            console.print(f"❌ Invalid status: {status}. Valid options: {', '.join(valid_statuses)}", style="red")
            return

        status_event = TaskStatusUpdatedEvent(
            payload={
                "task_id": task.task_id,
                "status": status.lower(),
                "previous_status": task.status
            }
        )
        store.append(status_event)
        events_created.append(f"Status → {status.lower()}")

    # Update Due Date
    if due_date:
        # Parse relative dates using our existing interpreter logic
        from sidecar_os.core.sidecar_core.router.interpreter import AdvancedPatternInterpreter
        interpreter = AdvancedPatternInterpreter()

        # Convert relative date to ISO format
        due_date_iso = None
        if due_date.lower() in ['today', 'tomorrow'] or due_date.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            due_date_iso = interpreter._convert_relative_date_to_iso(due_date.lower())
        else:
            # Try to parse as ISO date directly
            try:
                from datetime import datetime
                parsed_date = datetime.fromisoformat(due_date)
                due_date_iso = parsed_date.isoformat()
            except ValueError:
                console.print(f"❌ Invalid date format: {due_date}. Use: today, tomorrow, Friday, or YYYY-MM-DD", style="red")
                return

        if due_date_iso:
            scheduled_event = TaskScheduledEvent(
                payload={
                    "task_id": task.task_id,
                    "scheduled_for": due_date_iso
                }
            )
            store.append(scheduled_event)
            events_created.append(f"Due Date → {due_date}")

    # Update Duration
    if duration:
        # Parse duration using regex patterns
        import re
        duration_minutes = None

        # Parse [2h], [30min], [1.5h] formats
        if match := re.search(r'^\[?(\d+(?:\.\d+)?)\s*(?:hrs?|hours?)\]?$', duration, re.IGNORECASE):
            duration_minutes = int(float(match.group(1)) * 60)
        elif match := re.search(r'^\[?(\d+)\s*(?:mins?|minutes?)\]?$', duration, re.IGNORECASE):
            duration_minutes = int(match.group(1))
        elif match := re.search(r'^\[?(\d+(?:\.\d+)?)\s*h\]?$', duration, re.IGNORECASE):
            duration_minutes = int(float(match.group(1)) * 60)
        elif match := re.search(r'^\[?(\d+)\s*m\]?$', duration, re.IGNORECASE):
            duration_minutes = int(match.group(1))
        else:
            console.print(f"❌ Invalid duration format: {duration}. Use: [30min], [2h], [1.5h]", style="red")
            return

        if duration_minutes and duration_minutes > 0:
            duration_event = TaskDurationSetEvent(
                payload={
                    "task_id": task.task_id,
                    "duration_minutes": duration_minutes
                }
            )
            store.append(duration_event)

            # Format for display
            if duration_minutes >= 60:
                hours = duration_minutes // 60
                minutes = duration_minutes % 60
                if minutes == 0:
                    duration_display = f"{hours}h"
                else:
                    duration_display = f"{hours}h{minutes}m"
            else:
                duration_display = f"{duration_minutes}m"
            events_created.append(f"Duration → {duration_display}")

    # Display results
    if events_created:
        console.print("✅ Task updated successfully:", style="green")
        for update in events_created:
            console.print(f"   • {update}")
        console.print(f"   Task ID: {task.task_id}", style="dim")
    else:
        console.print("ℹ️  No updates specified. Use --priority, --status, --due, or --duration options.", style="yellow")
        console.print("💡 Example: sidecar update 8c5e --priority high --due tomorrow", style="dim")


def ask(question: str = typer.Argument(help="Natural language question about your tasks (e.g., 'what is due tomorrow?')")) -> None:
    """Ask natural language questions about your tasks and projects.

    Examples:
      sidecar ask "what is due tomorrow?"
      sidecar ask "show me the tasks for ca zap project"
      sidecar ask "what's overdue?"
      sidecar ask "how many high priority tasks do I have?"
      sidecar ask "what am I working on?"
    """
    import asyncio
    asyncio.run(_handle_natural_query(question))


async def _handle_natural_query(question: str) -> None:
    """Handle natural language queries about tasks and projects."""
    from sidecar_os.core.sidecar_core.llm import LLMService, LLMConfig
    from datetime import datetime, UTC

    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    console.print(f"🤔 [bold]{question}[/bold]")
    console.print()

    # Prepare task data for LLM
    task_data = []
    for task in state.tasks.values():
        task_data.append({
            "task_id": task.task_id,
            "title": task.title,
            "project_name": state.projects[task.project_id].name if task.project_id and task.project_id in state.projects else "",
            "priority": task.priority or "normal",
            "status": task.status,
            "scheduled_for": task.scheduled_for.isoformat() if task.scheduled_for else None,
            "duration_minutes": task.duration_minutes,
            "created_at": task.created_at.isoformat() if task.created_at else None
        })

    # Prepare project data for LLM
    project_data = []
    for project in state.projects.values():
        task_count = len(state.get_tasks_for_project(project.project_id))
        project_data.append({
            "project_id": project.project_id,
            "name": project.name,
            "aliases": project.aliases,
            "task_count": task_count
        })

    # Initialize LLM service
    llm_config = LLMConfig()
    llm_service = LLMService(llm_config)

    try:
        # Parse the natural language question
        parse_result = await llm_service.parse_natural_query(
            question=question,
            available_tasks=task_data,
            available_projects=project_data
        )

        if parse_result.get("confidence", 0) < 0.3:
            console.print("❌ I couldn't understand your question. Try being more specific.", style="red")
            console.print(f"   Error: {parse_result.get('explanation', 'Low confidence parsing')}", style="dim")
            return

        # Execute the parsed query
        await _execute_natural_query(parse_result, state, question)

    except Exception as e:
        console.print(f"❌ Failed to process question: {str(e)}", style="red")
        console.print("💡 Try asking questions like: 'what is due tomorrow?' or 'show me high priority tasks'", style="dim")


async def _execute_natural_query(parse_result: dict, state, original_question: str) -> None:
    """Execute a parsed natural language query and display results."""
    from datetime import datetime, UTC, timedelta

    query_type = parse_result.get("query_type", "list_tasks")
    filters = parse_result.get("filters", {})
    response_style = parse_result.get("response_style", "conversational")

    # Start with all tasks
    tasks = list(state.tasks.values())

    # Apply filters
    if filters.get("project_id"):
        tasks = [t for t in tasks if t.project_id == filters["project_id"]]
    elif filters.get("project_name"):
        # Find project by name (case insensitive)
        project_name = filters["project_name"].lower()
        project_id = None
        for proj in state.projects.values():
            if (proj.name.lower() == project_name or
                project_name in [alias.lower() for alias in proj.aliases] or
                project_name in proj.name.lower()):
                project_id = proj.project_id
                break
        if project_id:
            tasks = [t for t in tasks if t.project_id == project_id]

    if filters.get("priority"):
        tasks = [t for t in tasks if (t.priority or "normal") == filters["priority"]]

    if filters.get("status"):
        tasks = [t for t in tasks if t.status == filters["status"]]

    if filters.get("keywords"):
        keywords = [kw.lower() for kw in filters["keywords"]]
        tasks = [t for t in tasks if any(keyword in t.title.lower() for keyword in keywords)]

    # Apply date filters
    now_utc = datetime.now(UTC)
    today = now_utc.date()
    tomorrow = today + timedelta(days=1)
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    if filters.get("due_date_filter"):
        due_filter = filters["due_date_filter"]
        try:
            if due_filter == "today":
                tasks = [t for t in tasks if t.scheduled_for and t.scheduled_for.date() == today]
            elif due_filter == "tomorrow":
                tasks = [t for t in tasks if t.scheduled_for and t.scheduled_for.date() == tomorrow]
            elif due_filter == "overdue":
                tasks = [t for t in tasks if t.scheduled_for and t.scheduled_for.date() < today and t.status != "completed"]
            elif due_filter == "this_week":
                tasks = [t for t in tasks if t.scheduled_for and week_start <= t.scheduled_for.date() <= week_end]
        except Exception as e:
            console.print(f"⚠️  Date filter error: {str(e)}", style="yellow")
            # Continue without date filtering

    if filters.get("created_filter"):
        created_filter = filters["created_filter"]
        if created_filter == "today":
            tasks = [t for t in tasks if t.created_at and t.created_at.date() == today]
        elif created_filter == "this_week":
            tasks = [t for t in tasks if t.created_at and week_start <= t.created_at.date() <= week_end]

    # Display results based on query type
    if query_type == "count_tasks":
        count = len(tasks)
        if count == 0:
            console.print("📊 No tasks match your criteria.", style="yellow")
        elif count == 1:
            console.print(f"📊 [bold]1 task[/bold] matches your criteria.", style="green")
        else:
            console.print(f"📊 [bold]{count} tasks[/bold] match your criteria.", style="green")

        # Show a brief breakdown if there are results
        if count > 0 and response_style == "conversational":
            priorities = {}
            statuses = {}
            for task in tasks:
                priority = task.priority or "normal"
                priorities[priority] = priorities.get(priority, 0) + 1
                statuses[task.status] = statuses.get(task.status, 0) + 1

            if len(priorities) > 1:
                priority_breakdown = ", ".join([f"{count} {priority}" for priority, count in priorities.items()])
                console.print(f"   Priority: {priority_breakdown}", style="dim")

            if len(statuses) > 1:
                status_breakdown = ", ".join([f"{count} {status}" for status, count in statuses.items()])
                console.print(f"   Status: {status_breakdown}", style="dim")

    elif query_type == "list_tasks":
        if not tasks:
            console.print("📝 No tasks match your criteria.", style="yellow")
            return

        # Display tasks in a table format
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Task", style="white", width=50)
        table.add_column("Project", style="cyan", width=12)
        table.add_column("Priority", justify="center", width=8)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Due Date", style="yellow", width=12)
        table.add_column("Duration", style="dim", width=8)

        # Sort tasks by due date (overdue first, then by proximity)
        def sort_by_due_date(task):
            if not task.scheduled_for:
                return (1, datetime.max.replace(tzinfo=UTC))  # No due date goes to end
            try:
                overdue = task.scheduled_for.date() < today
                return (0 if overdue else 1, task.scheduled_for)
            except Exception as e:
                # Fallback: put problematic dates at the end
                return (2, datetime.max.replace(tzinfo=UTC))

        try:
            sorted_tasks = sorted(tasks, key=sort_by_due_date)
        except Exception as e:
            console.print(f"⚠️  Sorting error: {str(e)}", style="yellow")
            sorted_tasks = tasks  # Use unsorted if sorting fails

        for task in sorted_tasks[:20]:  # Limit to prevent overwhelming output
            # Format due date
            due_date_str = ""
            if task.scheduled_for:
                if task.scheduled_for.date() == today:
                    due_date_str = "📅 Today"
                elif task.scheduled_for.date() == tomorrow:
                    due_date_str = "📅 Tomorrow"
                elif task.scheduled_for.date() < today:
                    due_date_str = "🔴 Overdue"
                else:
                    due_date_str = task.scheduled_for.strftime("%a %m-%d")

            # Format duration
            duration_str = ""
            if task.duration_minutes:
                if task.duration_minutes >= 60:
                    hours = task.duration_minutes // 60
                    minutes = task.duration_minutes % 60
                    if minutes == 0:
                        duration_str = f"{hours}h"
                    else:
                        duration_str = f"{hours}h{minutes}m"
                else:
                    duration_str = f"{task.duration_minutes}m"

            # Get project name
            project_name = ""
            if task.project_id and task.project_id in state.projects:
                project_name = state.projects[task.project_id].name

            # Format task title with project prefix if no project column would show it
            title = task.title
            if project_name and not project_name:
                title = f"[{project_name}] {title}"

            table.add_row(
                title[:47] + "..." if len(title) > 50 else title,
                project_name,
                task.priority or "normal",
                task.status,
                due_date_str,
                duration_str
            )

        console.print(table)

        # Add conversational response
        if response_style == "conversational":
            count = len(tasks)
            if count > 20:
                console.print(f"💡 Showing first 20 of {count} tasks", style="dim")
            elif count == 1:
                console.print("💬 Here's the task that matches your question.", style="green")
            else:
                console.print(f"💬 Found {count} tasks that match your question.", style="green")

    else:
        console.print(f"❌ Query type '{query_type}' not yet implemented.", style="red")


def focus(
    project_query: Optional[str] = typer.Argument(None, help="Project name, ID, or alias to focus on"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear current focus")
) -> None:
    """Set focus on a project or clear current focus."""
    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    # Handle --clear option
    if clear:
        if not state.current_focus_project:
            console.print("💡 No project is currently focused", style="dim")
            return

        # Create focus cleared event
        clear_event = ProjectFocusClearedEvent(payload={})
        event_id = store.append(clear_event)

        console.print("🔄 Project focus cleared", style="green")
        console.print(f"  Event ID: {event_id[:8]}...", style="dim")
        return

    # Handle focus without argument - show current focus
    if not project_query:
        if state.current_focus_project:
            focus_project = state.projects.get(state.current_focus_project)
            if focus_project:
                console.print(f"🎯 Currently focused on: {focus_project.name}", style="cyan")
                console.print(f"  Project ID: {focus_project.project_id}", style="dim")
                aliases_str = f"  Aliases: {', '.join(focus_project.aliases)}" if focus_project.aliases else ""
                if aliases_str:
                    console.print(aliases_str, style="dim")
            else:
                console.print("⚠️  Focus project no longer exists", style="yellow")
        else:
            console.print("💡 No project is currently focused", style="dim")

        # Show available projects
        if state.projects:
            console.print("\n📂 Available projects:", style="cyan")
            for p in state.projects.values():
                focus_indicator = "🎯 " if p.project_id == state.current_focus_project else "   "
                aliases_str = f" ({', '.join(p.aliases)})" if p.aliases else ""
                console.print(f"{focus_indicator}{p.name}{aliases_str}", style="dim" if p.project_id != state.current_focus_project else "white")
        return

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
    """Interactive triage to resolve clarifications and process unhandled items."""
    import json
    from uuid import uuid4

    # Load events and project state
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    console.print("🔍 Interactive Triage Mode", style="bold blue")
    console.print("Press Ctrl+C to exit at any time", style="dim")
    console.print()

    # Handle pending clarifications first - they're highest priority
    pending_clarifications = state.get_pending_clarifications()
    if pending_clarifications:
        console.print("❓ Resolving Pending Clarifications", style="bold yellow")
        console.print()

        processed_any = False
        for clarification in pending_clarifications:
            processed_any = True
            source_item = state.inbox_items.get(clarification.source_event_id)
            source_text = source_item.text if source_item else "Unknown item"

            # Show the original item and questions
            console.print(Panel(
                f"Original: [white]{source_text}[/white]",
                title="🔍 Clarification Needed",
                border_style="yellow"
            ))

            console.print("Please answer these questions:", style="cyan")

            # Collect answers to all questions
            answers = {}
            try:
                for i, question in enumerate(clarification.questions, 1):
                    console.print(f"\n{i}. {question}", style="white")
                    answer = typer.prompt("  Answer", default="", show_default=False).strip()
                    if answer:  # Only store non-empty answers
                        answers[f"q{i}"] = answer
                        answers[f"question_{i}"] = question

                # Process the answers to generate appropriate events
                additional_events = _process_clarification_answers(
                    source_text, clarification.questions, answers, state, source_item.event_id if source_item else ""
                )

                # Store all generated events
                if additional_events:
                    console.print("\n✨ Generated from your answers:", style="green")
                    for event in additional_events:
                        event_id = store.append(event)
                        event_type = type(event).__name__.replace('Event', '').replace('Created', '').replace('Focused', 'Focus')
                        console.print(f"  → {event_type} ({event_id[:8]}...)", style="dim green")

                # Mark clarification as resolved
                resolved_event = ClarificationResolvedEvent(
                    payload={
                        "clarification_id": clarification.request_id,
                        "answers": answers,
                        "resolved_at": clarification.created_at.isoformat()
                    }
                )
                store.append(resolved_event)
                console.print(f"✓ Clarification resolved", style="green")

                # Ask if they want to continue
                if len(pending_clarifications) > 1:
                    console.print()
                    continue_triage = typer.confirm("Continue with next clarification?", default=True)
                    if not continue_triage:
                        break

            except (KeyboardInterrupt, typer.Abort):
                console.print("\n⚠️  Triage interrupted. Progress saved.", style="yellow")
                break

        if processed_any:
            console.print("\n" + "="*50)

        # Reload state after processing clarifications
        events = store.read_all()
        state = project_events_to_state(events)

    # Show remaining unprocessed inbox items with suggestions
    unprocessed_inbox = state.get_unprocessed_inbox()
    remaining_clarifications = state.get_pending_clarifications()

    if unprocessed_inbox:
        console.print("📥 Unprocessed Inbox Items (with AI suggestions)", style="bold cyan")
        console.print("💡 Use 'sidecar task <id>' to convert items to tasks", style="dim")
        console.print()

        # Use interpreter to suggest categorization
        config = InterpreterConfig(
            use_llm=True,
            llm_confidence_threshold=0.6,
            immediate_clarification_threshold=0.3
        )
        interpreter = AdvancedPatternInterpreter(config=config)

        for item in unprocessed_inbox[:5]:  # Show top 5
            console.print(f"📝 {item.text}", style="white")
            console.print(f"    ID: {item.event_id[:8]}...", style="dim")

            # Get interpretation suggestion
            result = interpreter.interpret_text(item.text, state)
            if result.confidence > 0.5:
                method_icon = "🧠" if result.used_llm else "🔍"
                console.print(f"    {method_icon} Suggestion: {result.explanation} ({result.confidence:.0%} confidence)", style="dim green")
            else:
                console.print(f"    ❓ Unclear - {result.explanation}", style="dim yellow")
            console.print()

    # Summary
    if not remaining_clarifications and not unprocessed_inbox:
        console.print("✨ All caught up! No items need triage.", style="green")
    else:
        summary_parts = []
        if remaining_clarifications:
            summary_parts.append(f"{len(remaining_clarifications)} clarifications")
        if unprocessed_inbox:
            summary_parts.append(f"{len(unprocessed_inbox)} unprocessed items")

        console.print(f"📊 Triage Summary: {', '.join(summary_parts)}", style="dim")


def _process_clarification_answers(
    original_text: str,
    questions: List[str],
    answers: Dict[str, str],
    state: SidecarState,
    source_event_id: str
) -> List:
    """Process clarification answers and generate appropriate events."""
    from uuid import uuid4
    events = []

    # Extract key information from answers
    project_mentioned = None
    task_mentioned = None
    item_type = None

    # Analyze only the actual user answers, not the questions
    user_answers_only = []
    for key, value in answers.items():
        if key.startswith('q') and not key.startswith('question_'):
            user_answers_only.append(value)

    all_answer_text = " ".join(user_answers_only).lower()

    # Detect item type - be more specific about task detection
    if any(phrase in all_answer_text for phrase in ["task", "todo", "need to do", "have to do"]):
        item_type = "task"
    elif any(word in all_answer_text for word in ["note", "notes", "information", "reference", "meeting", "status update"]):
        item_type = "note"
    elif any(word in all_answer_text for word in ["project", "initiative", "work on"]):
        item_type = "project"

    # Detect project references
    for answer in answers.values():
        # Check against existing projects
        for project in state.projects.values():
            if (project.name.lower() in answer.lower() or
                any(alias.lower() in answer.lower() for alias in project.aliases)):
                project_mentioned = project.project_id
                break

        # Check for new project indicators
        if not project_mentioned:
            import re
            # Look for project-like patterns in answers
            project_patterns = [
                r"project.+?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"([A-Z]{2,5})\s+project",
                r"working on\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"^([A-Z]{2,10}(?:\s+[A-Z]{2,10})*)\s*$",  # Simple acronyms like "EU UVP", "UVP EU"
                r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+project",  # "Project Name project"
                r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$"  # Simple project names like "Data Platform"
            ]
            for pattern in project_patterns:
                match = re.search(pattern, answer.strip(), re.IGNORECASE)
                if match:
                    project_name = match.group(1).strip()
                    if len(project_name) > 1:
                        # Create new project
                        project_id = project_name.lower().replace(' ', '-')
                        events.append(ProjectCreatedEvent(
                            payload={
                                "project_id": project_id,
                                "name": project_name,
                                "aliases": [],
                                "description": f"Created from clarification: {original_text}"
                            }
                        ))
                        project_mentioned = project_id
                        break

    # Generate task if appropriate - only if explicitly identified as a task
    if item_type == "task":
        # Try to extract task title from answers or use original text
        task_title = original_text

        # Look for specific task descriptions in answers
        for answer in answers.values():
            if len(answer) > 10 and any(word in answer.lower() for word in ["need to", "should", "must", "have to"]):
                task_title = answer
                break

        task_event = TaskCreatedEvent(
            payload={
                "task_id": str(uuid4()),
                "title": task_title,
                "description": f"Created from clarification. Original: {original_text}",
                "created_from_event": source_event_id,
                "project_id": project_mentioned
            }
        )
        events.append(task_event)

    # Focus on project if mentioned
    if project_mentioned and project_mentioned in state.projects:
        events.append(ProjectFocusedEvent(
            payload={"project_id": project_mentioned}
        ))

    return events


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


def weekly(style: str = typer.Option("exec", "--style", "-s", help="Summary style: exec (brief) or friendly (detailed)")) -> None:
    """Generate weekly activity summary with LLM analysis."""
    # Validate style
    try:
        summary_style = SummaryStyle(style)
    except ValueError:
        console.print(f"❌ Invalid style: {style}. Use 'exec' or 'friendly'", style="red")
        return

    # Generate summary
    try:
        console.print("🧠 Generating weekly summary...", style="dim")

        generator = SummaryGenerator()
        summary = generator.generate_summary(
            period=SummaryPeriod.WEEKLY,
            style=summary_style
        )

        console.print()
        console.print(summary)
        console.print()

        # Show usage info
        usage_tracker = get_usage_tracker()
        usage_summary = usage_tracker.get_usage_summary()
        if usage_summary['daily_requests'] > 0:
            console.print(f"[dim]Generated using {usage_summary['provider']} (${usage_summary['daily_cost']:.4f})[/dim]")

    except Exception as e:
        console.print(f"❌ Summary generation failed: {str(e)}", style="red")
        console.print("💡 Check your LLM configuration", style="dim")


def daily(style: str = typer.Option("exec", "--style", "-s", help="Summary style: exec (brief) or friendly (detailed)")) -> None:
    """Generate daily activity summary with LLM analysis."""
    # Validate style
    try:
        summary_style = SummaryStyle(style)
    except ValueError:
        console.print(f"❌ Invalid style: {style}. Use 'exec' or 'friendly'", style="red")
        return

    # Generate summary
    try:
        console.print("🧠 Generating daily summary...", style="dim")

        generator = SummaryGenerator()
        summary = generator.generate_summary(
            period=SummaryPeriod.DAILY,
            style=summary_style
        )

        console.print()
        console.print(summary)
        console.print()

        # Show usage info
        usage_tracker = get_usage_tracker()
        usage_summary = usage_tracker.get_usage_summary()
        if usage_summary['daily_requests'] > 0:
            console.print(f"[dim]Generated using {usage_summary['provider']} (${usage_summary['daily_cost']:.4f})[/dim]")

    except Exception as e:
        console.print(f"❌ Summary generation failed: {str(e)}", style="red")
        console.print("💡 Check your LLM configuration", style="dim")

def project_cleanup(
    analyze_only: bool = typer.Option(True, "--analyze-only", "-a", help="Only show cleanup suggestions without making changes"),
    confirm: bool = typer.Option(False, "--confirm", "-y", help="Skip confirmation prompts")
) -> None:
    """Clean up messy project data from testing phase."""

    cleanup_manager = ProjectCleanupManager()

    if analyze_only:
        console.print("🔍 Analyzing project cleanup opportunities...", style="dim")
        console.print()

        summary = cleanup_manager.get_cleanup_summary()
        console.print(summary)

        if "No cleanup needed" not in summary:
            console.print("\n💡 Use --no-analyze-only to apply these changes")
    else:
        # TODO: Implement actual cleanup execution
        # For now, just show what would be done
        console.print("🚧 Cleanup execution not yet implemented", style="yellow")
        console.print("📊 Showing analysis for now...")
        console.print()

        summary = cleanup_manager.get_cleanup_summary()
        console.print(summary)


def event_migrate(
    preview: bool = typer.Option(False, "--preview-only", "-p", help="Only preview without executing"),
    execute: bool = typer.Option(False, "--execute", "-x", help="Execute the migration (creates backup)"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompts")
) -> None:
    """Migrate event log by filtering out garbage data and keeping clean events."""

    migrator = EventLogMigrator()

    # Default behavior: show preview, then ask for execution
    # --preview-only: only show preview
    # --execute: skip preview and execute directly

    if preview and not execute:
        # Preview only mode
        console.print("🔍 Analyzing event log migration opportunities...", style="dim")
        console.print()

        preview_text = migrator.get_migration_preview()
        console.print(preview_text)
        return

    if not execute:
        # Default mode: show preview + instructions
        console.print("🔍 Analyzing event log migration opportunities...", style="dim")
        console.print()

        preview_text = migrator.get_migration_preview()
        console.print(preview_text)
        console.print("\n💡 Use --execute to perform the migration")
        return

    # Execute mode
    if not force:
        console.print("⚠️  This will modify your event log. A backup will be created.", style="yellow")
        console.print("Are you sure you want to proceed? [y/N]: ", end="")

        try:
            import sys
            confirmation = input().lower().strip()
            if confirmation not in ['y', 'yes']:
                console.print("❌ Migration cancelled", style="red")
                return
        except (EOFError, KeyboardInterrupt):
            console.print("\n❌ Migration cancelled", style="red")
            return

    console.print("🔄 Performing event log migration...", style="dim")

    try:
        result = migrator.migrate_to_clean_log(dry_run=False)

        console.print("✅ Migration completed successfully!", style="green")
        console.print()
        console.print(f"📊 Results:")
        console.print(f"  • Processed: {result.events_processed} events")
        console.print(f"  • Kept: {result.events_kept} clean events")
        console.print(f"  • Filtered: {result.events_filtered} garbage events")
        console.print(f"  • Deleted: {len(result.projects_deleted)} projects")
        console.print(f"  • Renamed: {len(result.projects_renamed)} projects")
        console.print()
        console.print(f"💾 Backup created: {result.backup_path}")
        console.print()
        console.print("🎉 Your event log is now clean! Try 'ss status' to see the results.")

    except Exception as e:
        console.print(f"❌ Migration failed: {str(e)}", style="red")
        console.print("💡 Your original data is safe - no changes were made")
