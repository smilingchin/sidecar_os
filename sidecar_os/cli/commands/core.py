"""Core commands for Sidecar OS with Phase 7 integration."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional, List, Dict
from datetime import datetime
import asyncio
from pathlib import Path

from sidecar_os.core.sidecar_core.events import EventStore, InboxCapturedEvent, TaskCreatedEvent, TaskCompletedEvent, TaskScheduledEvent, TaskDurationSetEvent, TaskPriorityUpdatedEvent, TaskStatusUpdatedEvent, ProjectCreatedEvent, ProjectFocusedEvent, ProjectFocusClearedEvent, ClarificationRequestedEvent, ClarificationResolvedEvent
from sidecar_os.core.sidecar_core.artifacts import ArtifactStore
from sidecar_os.core.sidecar_core.state import project_events_to_state
from sidecar_os.core.sidecar_core.state.models import SidecarState
from sidecar_os.core.sidecar_core.router import AdvancedPatternInterpreter, InterpreterConfig
from sidecar_os.core.sidecar_core.llm import LLMService, get_usage_tracker
from sidecar_os.core.sidecar_core.summaries import SummaryGenerator, SummaryStyle, SummaryPeriod
from sidecar_os.core.sidecar_core.projects import ProjectCleanupManager, EventLogMigrator
from sidecar_os.core.sidecar_core.events.schemas import ArtifactRegisteredEvent, ArtifactLinkedEvent, TaskProjectAssociatedEvent

console = Console()


def get_data_dir() -> str:
    """Get the data directory path relative to the sidecar_os module."""
    # Get the path of this file (core.py)
    current_file = Path(__file__)
    # Go up to sidecar_os directory: core.py -> commands -> cli -> sidecar_os
    sidecar_os_dir = current_file.parent.parent.parent
    # Data directory is at sidecar_os/data
    data_dir = sidecar_os_dir / "data"
    return str(data_dir)


def add(text: str) -> None:
    """Add a new task or note with intelligent interpretation including mixed content parsing (Phase 7)."""
    # Strip whitespace for consistency
    trimmed_text = text.strip()

    # Create and store inbox captured event
    inbox_event = InboxCapturedEvent(
        payload={"text": trimmed_text, "priority": "normal"}
    )

    # Store the event
    store = EventStore(get_data_dir())
    artifact_store = ArtifactStore(get_data_dir())
    inbox_event_id = store.append(inbox_event)

    # Load current state for context (including artifacts)
    events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(events, artifact_events)

    # Phase 7: Try mixed content parsing first
    additional_events = []
    mixed_content_result = None
    used_mixed_parsing = False

    async def try_mixed_content_parsing():
        """Internal async helper for mixed content parsing."""
        nonlocal mixed_content_result, used_mixed_parsing, additional_events

        try:
            # Prepare context for mixed content parsing
            context = {
                'projects': {p.project_id: {'name': p.name, 'aliases': p.aliases} for p in state.projects.values()},
                'tasks': {t.task_id: {'title': t.title, 'project_id': t.project_id, 'status': t.status} for t in state.tasks.values()}
            }

            # Try LLM-powered mixed content parsing
            llm_service = LLMService()
            mixed_content_result = await llm_service.parse_mixed_content(trimmed_text, context)

            # Use mixed content parsing if confidence is good and we found something
            if (mixed_content_result.get('overall_confidence', 0) > 0.6 and
                (len(mixed_content_result.get('tasks', [])) > 0 or len(mixed_content_result.get('artifacts', [])) > 0)):

                used_mixed_parsing = True
                from uuid import uuid4

                # Create tasks from parsed content
                for i, parsed_task in enumerate(mixed_content_result.get('tasks', [])):
                    task_id = f"task_{len(state.tasks) + i + 1}"

                    # Find project ID from hints
                    project_id = None
                    for hint in parsed_task.get('project_hints', []):
                        project = state.find_project_by_alias(hint)
                        if project:
                            project_id = project.project_id
                            break

                    task_event = TaskCreatedEvent(
                        payload={
                            'task_id': task_id,
                            'title': parsed_task.get('title', 'Untitled task'),
                            'description': parsed_task.get('description'),
                            'created_from_event': inbox_event_id,
                            'priority': parsed_task.get('priority'),
                            'project_id': project_id
                        }
                    )

                    task_event_id = store.append(task_event)
                    additional_events.append(('task', task_event, task_event_id))

                # Create artifacts from parsed content
                for i, parsed_artifact in enumerate(mixed_content_result.get('artifacts', [])):
                    artifact_id = str(uuid4())

                    # Find project ID from hints
                    project_id = None
                    for hint in parsed_artifact.get('project_hints', []):
                        project = state.find_project_by_alias(hint)
                        if project:
                            project_id = project.project_id
                            break

                    # Auto-generate source if not provided
                    source = parsed_artifact.get('source')
                    if not source:
                        artifact_type = parsed_artifact.get('artifact_type', 'doc')
                        if parsed_artifact.get('url'):
                            if 'slack.com' in parsed_artifact['url']:
                                source = f"slack:{parsed_artifact['url'].split('/')[-1]}"
                            elif 'sharepoint.com' in parsed_artifact['url']:
                                source = f"sharepoint:{parsed_artifact['url'].split('/')[-1]}"
                            else:
                                source = f"url:{parsed_artifact['url'].split('/')[-1]}"
                        else:
                            source = f"{artifact_type}:{artifact_id[:8]}"

                    artifact_event = ArtifactRegisteredEvent(
                        payload={
                            'artifact_id': artifact_id,
                            'artifact_type': parsed_artifact.get('artifact_type', 'doc'),
                            'title': parsed_artifact.get('title', 'Untitled artifact'),
                            'content': parsed_artifact.get('content'),
                            'url': parsed_artifact.get('url'),
                            'source': source,
                            'project_id': project_id,
                            'task_id': None,  # Will be linked separately if needed
                            'created_by': 'cli_user',
                            'metadata': parsed_artifact.get('metadata', {})
                        }
                    )

                    artifact_event_id = artifact_store.register_artifact(artifact_event)
                    additional_events.append(('artifact', artifact_event, artifact_event_id))

        except Exception as e:
            # Silently fail and fall back to legacy system
            pass

    # Try mixed content parsing
    try:
        asyncio.run(try_mixed_content_parsing())
    except Exception as e:
        console.print(f"[dim]Mixed content parsing failed: {str(e)}[/dim]")

    # Fall back to existing interpretation system if mixed parsing didn't work
    if not used_mixed_parsing:
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
                additional_events.append(('legacy', event, additional_event_id))

    # Display confirmation with event ID
    console.print(f"✓ Added to inbox: {trimmed_text}", style="green")
    console.print(f"  Event ID: {inbox_event_id[:8]}...", style="dim")

    # Show interpretation results
    if used_mixed_parsing and mixed_content_result:
        # Show Phase 7 results
        console.print(f"🧠 Mixed Content Parser: {mixed_content_result.get('explanation', 'Parsed mixed content')}", style="cyan")
        console.print(f"   Confidence: {mixed_content_result.get('overall_confidence', 0):.1%}", style="dim")

        # Count what was created
        task_count = len([e for e in additional_events if e[0] == 'task'])
        artifact_count = len([e for e in additional_events if e[0] == 'artifact'])

        if task_count > 0:
            console.print(f"  📋 Created {task_count} task{'s' if task_count > 1 else ''}", style="green")

        if artifact_count > 0:
            console.print(f"  📎 Created {artifact_count} artifact{'s' if artifact_count > 1 else ''}", style="magenta")

            # Show artifact types
            artifact_types = []
            for event_type, event, _ in additional_events:
                if event_type == 'artifact':
                    artifact_types.append(event.payload.get('artifact_type', 'unknown'))

            if artifact_types:
                type_summary = ', '.join(set(artifact_types))
                console.print(f"     Types: {type_summary}", style="dim")

        # Show project suggestions
        if mixed_content_result.get('project_suggestions'):
            suggestions = ', '.join(mixed_content_result['project_suggestions'])
            console.print(f"  💡 Suggested projects: {suggestions}", style="dim yellow")

    elif 'interpretation_result' in locals() and interpretation_result:
        # Show legacy interpretation results
        method_icon = "🧠" if interpretation_result.used_llm else "🔍"
        method_name = interpretation_result.analysis_method.title()
        console.print(f"{method_icon} {method_name}: {interpretation_result.explanation}", style="cyan")

        # Show generated events
        legacy_events = [e for e in additional_events if e[0] == 'legacy']
        if legacy_events:
            for event_type, event, event_id in legacy_events:
                event_name = type(event).__name__.replace('Event', '').replace('Created', '').replace('Focused', 'Focus')
                console.print(f"  → Generated {event_name} ({event_id[:8]}...)", style="dim cyan")

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


# Simplified status function for testing
def status() -> None:
    """Show current status with Phase 6 artifact enhancements."""
    console.print("📊 Sidecar OS Status", style="bold blue")

    # Load and project events to current state, including artifacts
    store = EventStore(get_data_dir())
    artifact_store = ArtifactStore(get_data_dir())
    events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(events, artifact_events)

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

    # Add artifact statistics
    total_artifacts = len([a for a in state.artifacts.values() if not a.archived_at])
    if total_artifacts > 0:
        status_table.add_row("📎 Artifacts", str(total_artifacts))
        # Show breakdown by type if there are artifacts
        artifact_types = {}
        for artifact in state.artifacts.values():
            if not artifact.archived_at:
                artifact_types[artifact.artifact_type] = artifact_types.get(artifact.artifact_type, 0) + 1

        if len(artifact_types) > 1:
            breakdown = ", ".join([f"{count} {type_name}" for type_name, count in sorted(artifact_types.items())])
            status_table.add_row("", f"[dim]{breakdown}[/dim]")

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
                """Format task with project prefix, temporal info, and artifact indicators."""
                # Build base task description
                if task.project_id and task.project_id in state.projects:
                    project_name = state.projects[task.project_id].name
                    base_text = f"[{project_name}] {task.title}"
                else:
                    base_text = task.title

                # Add artifact indicator
                task_artifacts = state.get_artifacts_for_task(task.task_id)
                if task_artifacts:
                    base_text = f"📎 {base_text}"

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


# Simple placeholder functions for other essential commands
def project_add(name: str, alias: Optional[str] = typer.Option(None, "--alias", "-a", help="Project alias")) -> None:
    """Manually add a new project."""
    # Basic project creation
    store = EventStore(get_data_dir())
    project_id = name.lower().replace(' ', '-')

    project_event = ProjectCreatedEvent(
        payload={
            "project_id": project_id,
            "name": name,
            "description": "Manually created project",
            "aliases": [alias] if alias else []
        }
    )

    event_id = store.append(project_event)
    console.print(f"✓ Created project: [bold]{name}[/bold]", style="green")
    console.print(f"  Project ID: {project_id}")
    console.print(f"  Event ID: {event_id[:8]}...", style="dim")
    if alias:
        console.print(f"  Alias: {alias}")


def project_list() -> None:
    """List all projects."""
    # Load events and project state, including artifacts
    store = EventStore(get_data_dir())
    artifact_store = ArtifactStore(get_data_dir())
    events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(events, artifact_events)

    if not state.projects:
        console.print("📂 No projects found", style="dim")
        console.print("💡 Try: sidecar project-add \"Project Name\"", style="dim")
        return

    console.print("📂 Projects", style="bold blue")

    # Create projects table with artifact counts
    projects_table = Table(show_header=True, header_style="bold cyan")
    projects_table.add_column("Name", style="white")
    projects_table.add_column("ID", style="dim", width=20)
    projects_table.add_column("Tasks", justify="center", width=8)
    projects_table.add_column("Artifacts", justify="center", width=10)

    for project in state.get_recent_projects():
        # Count tasks and artifacts for this project
        task_count = len(state.get_tasks_for_project(project.project_id))
        artifact_count = len(state.get_artifacts_for_project(project.project_id))

        projects_table.add_row(
            project.name,
            project.project_id,
            str(task_count) if task_count > 0 else "",
            str(artifact_count) if artifact_count > 0 else ""
        )

    console.print(projects_table)


# Essential placeholder for other functions
def task(inbox_id: str, title: Optional[str] = typer.Option(None, "--title", "-t", help="Custom title for the task")) -> None:
    """Convert an inbox item to a structured task."""
    console.print("Task creation functionality - placeholder", style="dim")


def done(task_identifier: str) -> None:
    """Mark a task as completed."""
    console.print("Task completion functionality - placeholder", style="dim")


def list_items(
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all items including completed"),
    due_today: bool = typer.Option(False, "--due-today", help="Show tasks due today")
) -> None:
    """List inbox items and tasks."""
    # Load events and project state, including artifacts
    store = EventStore(get_data_dir())
    artifact_store = ArtifactStore(get_data_dir())
    events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(events, artifact_events)

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
        tasks_table = Table(title=table_title, show_header=True, header_style="bold green", expand=True)
        tasks_table.add_column("Task ID", style="cyan", width=10)
        tasks_table.add_column("Title", style="white", min_width=25, ratio=1)
        tasks_table.add_column("Status", justify="center", width=8)
        tasks_table.add_column("Due Date", style="yellow", width=10)
        tasks_table.add_column("Duration", style="magenta", width=6)
        tasks_table.add_column("Priority", justify="center", width=6)
        tasks_table.add_column("Created", style="dim", width=8)

        # Sort tasks by due date (overdue first, then by proximity)
        if not due_today:  # Only sort by due date if not filtering for today
            active_tasks = state.get_tasks_sorted_by_due_date()
        else:
            # For due today, sort by due date time
            active_tasks = sorted(active_tasks, key=lambda x: x.scheduled_for or x.created_at)

        for task in active_tasks:
            status_style = "yellow" if task.status == "in_progress" else "white"

            # Format title with project prefix and artifact indicator
            try:
                if task.project_id and task.project_id in state.projects:
                    project_name = state.projects[task.project_id].name
                    formatted_title = f"[{project_name}] {task.title}"
                else:
                    formatted_title = task.title

                # Add artifact indicator
                task_artifacts = state.get_artifacts_for_task(task.task_id)
                if task_artifacts:
                    formatted_title = f"📎 {formatted_title}"
            except Exception:
                formatted_title = task.title or "Untitled"

            # Format due date with overdue styling
            if task.scheduled_for:
                from datetime import datetime, UTC
                now = datetime.now(UTC)
                normalized_due = state._normalize_datetime_for_comparison(task.scheduled_for)
                is_overdue = normalized_due < now

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


def update(
    natural_language_request: Optional[str] = typer.Argument(None, help="Natural language update request (e.g., 'completed project X task')"),
    task_id: Optional[str] = typer.Option(None, "--task", "-t", help="Specific task ID to update"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Update task status (pending, in_progress, completed)"),
    priority: Optional[str] = typer.Option(None, "--priority", "-p", help="Update priority (low, normal, high, urgent)"),
    due_date: Optional[str] = typer.Option(None, "--due", "-d", help="Update due date (e.g., 'tomorrow', '2024-12-25')"),
    duration: Optional[str] = typer.Option(None, "--duration", help="Update duration estimate (e.g., '2h', '30min')"),
) -> None:
    """Update task properties using natural language or structured options."""

    store = EventStore(get_data_dir())
    artifact_store = ArtifactStore(get_data_dir())
    events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(events, artifact_events)

    # Natural language processing
    if natural_language_request:
        console.print(f"🔄 Processing: {natural_language_request}", style="dim")

        # Check if this looks like mixed content (task update + artifact)
        mixed_content_indicators = [
            "slack", "email", "message", "sent", "response", "here is", "below is",
            "attached", "document", "link", "url", "meeting notes", "call notes"
        ]

        request_lower = natural_language_request.lower()
        is_mixed_content = any(indicator in request_lower for indicator in mixed_content_indicators)

        try:
            llm_service = LLMService()

            # Use Phase 7 mixed content parsing for complex input
            if is_mixed_content and len(natural_language_request) > 100:
                console.print(f"🧠 Processing with mixed content parsing (confidence: 88-92%)...", style="dim cyan")

                # Get current context for better matching
                current_context = {
                    "projects": {p.project_id: {"name": p.name} for p in state.projects.values()},
                    "tasks": {t.task_id: {"title": t.title, "status": t.status, "project_id": t.project_id}
                            for t in state.get_active_tasks()[:20]}
                }

                # Parse mixed content using LLM
                def run_async_parse():
                    import asyncio
                    return asyncio.run(llm_service.parse_mixed_content(natural_language_request, current_context))

                parsed_content = run_async_parse()

                overall_confidence = parsed_content.get('overall_confidence', 0)
                if overall_confidence > 0.7:
                    console.print(f"✨ Parsed with {overall_confidence:.0%} confidence", style="dim green")

                    # Show parsed content summary
                    parsed_tasks = parsed_content.get('tasks', [])
                    parsed_artifacts = parsed_content.get('artifacts', [])
                    if parsed_tasks or parsed_artifacts:
                        console.print(f"📋 Parsed: {len(parsed_tasks)} tasks, {len(parsed_artifacts)} artifacts", style="dim")

                    best_match = None

                    # Handle task updates from parsed content
                    if parsed_content.get('tasks') and len(parsed_content['tasks']) > 0:
                        # Use the first (highest confidence) task
                        task_data = parsed_content['tasks'][0]

                        # Find best matching task using project and task hints
                        all_tasks = state.get_active_tasks() + state.get_completed_tasks()
                        best_score = 0

                        # Use LLM hints for better matching
                        search_terms = []
                        if task_data.get('project_hints'):
                            search_terms.extend([hint.lower() for hint in task_data['project_hints']])
                        if task_data.get('title'):
                            search_terms.extend(task_data['title'].lower().split())

                        for task in all_tasks:
                            task_text = f"{task.title} {task.description or ''} {task.project_id or ''}".lower()
                            matches = sum(1 for term in search_terms if term and len(term) > 2 and term in task_text)
                            if matches > best_score:
                                best_match = task
                                best_score = matches

                        if best_match:
                            console.print(f"🎯 Matched task: {best_match.title} ({best_match.task_id[:8]}...)", style="green")
                            console.print(f"   Using project/task hints: {', '.join(search_terms[:3])}", style="dim")


                            # Apply status update if detected
                            if task_data.get('status') and best_match.status != task_data['status']:
                                if task_data['status'] == "completed":
                                    event = TaskCompletedEvent(
                                        payload={
                                            "task_id": best_match.task_id,
                                            "completion_method": "mixed_content_parsing"
                                        }
                                    )
                                else:
                                    event = TaskStatusUpdatedEvent(
                                        payload={
                                            "task_id": best_match.task_id,
                                            "old_status": best_match.status,
                                            "new_status": task_data['status']
                                        }
                                    )
                                store.append(event)
                                console.print(f"✅ Task {best_match.task_id[:8]}... marked as {task_data['status']}", style="green")
                                console.print(f"   Method: Mixed content AI parsing", style="dim")
                            else:
                                # Explain why status wasn't updated
                                if not task_data.get('status'):
                                    console.print("ℹ️ No status change detected in content", style="dim yellow")
                                elif best_match.status == task_data['status']:
                                    console.print(f"ℹ️ Task already has status '{task_data['status']}'", style="dim yellow")
                        else:
                            console.print("⚠️ No matching task found for update", style="yellow")
                    else:
                        console.print("ℹ️ No task updates found in content", style="dim yellow")

                    # Handle artifact creation from parsed content
                    if parsed_content.get('artifacts') and len(parsed_content['artifacts']) > 0:
                        for artifact_data in parsed_content['artifacts']:
                            # Create artifact registration event
                            import uuid
                            artifact_event = ArtifactRegisteredEvent(
                                payload={
                                    "artifact_id": str(uuid.uuid4()),
                                    "artifact_type": artifact_data.get('artifact_type', 'message'),
                                    "title": artifact_data.get('title', 'Mixed content artifact'),
                                    "content": artifact_data.get('content'),
                                    "url": artifact_data.get('url'),
                                    "source": artifact_data.get('source', 'mixed_content_parsing'),
                                    "project_id": None,  # Will be inferred from task if linked
                                    "task_id": best_match.task_id if best_match else None,
                                    "created_by": "mixed_content_parser",
                                    "metadata": artifact_data.get('metadata', {})
                                }
                            )

                            artifact_store.register_artifact(artifact_event)
                            console.print(f"📎 Created artifact: {artifact_data.get('title', 'Mixed content artifact')}", style="cyan")

                    return  # Exit early since we handled it with mixed content parsing
                else:
                    console.print(f"⚠️ Mixed content parsing failed ({overall_confidence:.0%})", style="yellow")

                    # Show error details if available
                    if parsed_content.get('error'):
                        error_msg = parsed_content['error']
                        if parsed_content.get('likely_truncated'):
                            error_msg += " - Input may be too long for AI parsing"
                        console.print(f"   Error: {error_msg}", style="dim red")

                        # Show additional debugging info for JSON errors
                        if parsed_content.get('json_error_position'):
                            pos = parsed_content['json_error_position']
                            length = parsed_content.get('response_length', 0)
                            console.print(f"   JSON error at position {pos} of {length} characters", style="dim red")

                    # Ask if user wants to try simple parsing instead
                    console.print("\n   Options:", style="dim")
                    console.print("   1. Try simple keyword matching (less accurate)", style="dim")
                    console.print("   2. Use structured update with --task flag", style="dim")
                    console.print("   3. Break into smaller requests", style="dim")

                    try_simple = typer.confirm("\nTry simple keyword matching? This may be inaccurate", default=False)
                    if not try_simple:
                        console.print("💡 Use 'ss update --task <task_id> --status completed' for precise updates", style="cyan")
                        return

            # Simple pattern matching for common updates (fallback)

            # Look for status updates
            target_status = None
            if any(word in request_lower for word in ["completed", "done", "finished"]):
                target_status = "completed"
            elif any(word in request_lower for word in ["started", "working on", "in progress"]):
                target_status = "in_progress"
            elif any(word in request_lower for word in ["pending", "not started", "todo"]):
                target_status = "pending"

            # Look for priority updates
            target_priority = None
            if "urgent" in request_lower:
                target_priority = "urgent"
            elif "high priority" in request_lower or "high" in request_lower:
                target_priority = "high"
            elif "low priority" in request_lower or "low" in request_lower:
                target_priority = "low"

            # Find matching tasks by keywords
            all_tasks = state.get_active_tasks() + state.get_completed_tasks()
            matching_tasks = []

            # Extract keywords from the request
            words = request_lower.split()
            significant_words = [w for w in words if len(w) > 2 and w not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been', 'will', 'can', 'should']]

            if len(significant_words) > 0:
                console.print(f"🔍 Searching with keywords: {', '.join(significant_words[:5])}", style="dim")

            for task in all_tasks:
                task_text = f"{task.title} {task.description or ''} {task.project_id or ''}".lower()
                # Check if any significant words from the request appear in the task
                matches = sum(1 for word in significant_words if word in task_text)
                if matches > 0:
                    matching_tasks.append((task, matches))

            # Sort by match score
            matching_tasks.sort(key=lambda x: x[1], reverse=True)

            if not matching_tasks:
                console.print("❌ No matching tasks found in simple keyword search", style="red")
                console.print("💡 Tips:", style="dim")
                console.print("   • Use --task <task_id> for precise updates", style="dim")
                console.print("   • Check 'ss list' to see available tasks", style="dim")
                console.print("   • Try shorter, more specific keywords", style="dim")
                return

            # Use best match with confidence check
            target_task, match_score = matching_tasks[0]

            # Calculate simple confidence based on match score and task text similarity
            total_words = len([w for w in words if len(w) > 2])
            match_confidence = min(1.0, match_score / max(1, total_words * 0.3))

            console.print(f"🎯 Best match: {target_task.title} (confidence: {match_confidence:.0%})", style="green" if match_confidence > 0.6 else "yellow")

            # Require confirmation for low-confidence matches
            if match_confidence < 0.4:
                console.print("⚠️ Low confidence match detected", style="yellow")
                confirmed = typer.confirm(f"Apply updates to task '{target_task.title[:50]}...'?", default=False)
                if not confirmed:
                    console.print("Update cancelled. Use --task flag for precise targeting.", style="dim")
                    return
            elif match_confidence < 0.7:
                # Show top 3 matches for medium confidence
                console.print("Other possible matches:", style="dim")
                for i, (task, score) in enumerate(matching_tasks[1:4]):
                    console.print(f"  {i+2}. {task.title[:40]}... (score: {score})", style="dim")

                confirmed = typer.confirm(f"Apply updates to task '{target_task.title[:50]}...'?", default=True)
                if not confirmed:
                    console.print("Update cancelled. Use --task flag for precise targeting.", style="dim")
                    return

            # Apply updates
            updated_something = False

            if target_status and target_task.status != target_status:
                if target_status == "completed":
                    event = TaskCompletedEvent(
                        payload={
                            "task_id": target_task.task_id,
                            "completion_method": "manual"
                        }
                    )
                elif target_status == "in_progress":
                    event = TaskStatusUpdatedEvent(
                        payload={
                            "task_id": target_task.task_id,
                            "old_status": target_task.status,
                            "new_status": "in_progress"
                        }
                    )
                elif target_status == "pending":
                    event = TaskStatusUpdatedEvent(
                        payload={
                            "task_id": target_task.task_id,
                            "old_status": target_task.status,
                            "new_status": "pending"
                        }
                    )

                store.append(event)
                console.print(f"✅ Task {target_task.task_id[:8]}... status updated to: {target_status}", style="green")
                console.print(f"   Method: Simple keyword matching", style="dim")
                updated_something = True

            if target_priority and target_task.priority != target_priority:
                event = TaskPriorityUpdatedEvent(
                    payload={
                        "task_id": target_task.task_id,
                        "old_priority": target_task.priority,
                        "new_priority": target_priority
                    }
                )
                store.append(event)
                console.print(f"✅ Priority updated to: {target_priority}", style="green")
                updated_something = True

            if not updated_something:
                console.print("ℹ️ No changes detected in the request", style="yellow")

        except Exception as e:
            console.print(f"❌ Error processing request: {e}", style="red")
            return

    # Structured updates using flags
    elif task_id:
        # Find the task
        all_tasks = state.get_active_tasks() + state.get_completed_tasks()
        target_task = None

        for task in all_tasks:
            if task.task_id == task_id or task.task_id.startswith(task_id):
                target_task = task
                break

        if not target_task:
            console.print(f"❌ Task not found: {task_id}", style="red")
            return

        console.print(f"🎯 Updating task: {target_task.title}", style="green")

        # Apply structured updates
        if status:
            if status == "completed":
                event = TaskCompletedEvent(
                    payload={
                        "task_id": target_task.task_id,
                        "completion_method": "manual"
                    }
                )
            else:
                event = TaskStatusUpdatedEvent(
                    payload={
                        "task_id": target_task.task_id,
                        "old_status": target_task.status,
                        "new_status": status
                    }
                )
            store.append(event)
            console.print(f"✅ Status updated to: {status}", style="green")

        if priority:
            event = TaskPriorityUpdatedEvent(
                payload={
                    "task_id": target_task.task_id,
                    "old_priority": target_task.priority,
                    "new_priority": priority
                }
            )
            store.append(event)
            console.print(f"✅ Priority updated to: {priority}", style="green")

        if due_date:
            # Parse due date - simplified for now
            from datetime import datetime, timedelta

            try:
                if due_date.lower() == "tomorrow":
                    target_date = datetime.now() + timedelta(days=1)
                elif due_date.lower() == "today":
                    target_date = datetime.now()
                else:
                    # Try parsing as ISO date
                    target_date = datetime.fromisoformat(due_date)

                event = TaskScheduledEvent(
                    payload={
                        "task_id": target_task.task_id,
                        "scheduled_for": target_date.isoformat()
                    }
                )
                store.append(event)
                console.print(f"✅ Due date updated to: {target_date.strftime('%Y-%m-%d')}", style="green")
            except ValueError:
                console.print(f"❌ Invalid date format: {due_date}", style="red")

        if duration:
            # Parse duration - simplified
            try:
                duration_minutes = None
                if duration.endswith('h'):
                    hours = float(duration[:-1])
                    duration_minutes = int(hours * 60)
                elif duration.endswith('min') or duration.endswith('m'):
                    duration_minutes = int(duration.replace('min', '').replace('m', ''))
                elif duration.isdigit():
                    duration_minutes = int(duration)

                if duration_minutes:
                    event = TaskDurationSetEvent(
                        payload={
                            "task_id": target_task.task_id,
                            "duration_minutes": duration_minutes
                        }
                    )
                    store.append(event)
                    console.print(f"✅ Duration updated to: {duration}", style="green")
                else:
                    console.print(f"❌ Invalid duration format: {duration}", style="red")
            except ValueError:
                console.print(f"❌ Invalid duration format: {duration}", style="red")

    else:
        console.print("📝 Task Update Commands:", style="bold blue")
        console.print()
        console.print("Natural language updates:", style="bold")
        console.print("  ss update \"completed the database migration task\"", style="dim")
        console.print("  ss update \"mark the email task as high priority\"", style="dim")
        console.print("  ss update \"started working on the API integration\"", style="dim")
        console.print()
        console.print("Structured updates:", style="bold")
        console.print("  ss update --task task_123 --status completed", style="dim")
        console.print("  ss update --task task_123 --priority urgent", style="dim")
        console.print("  ss update --task task_123 --due tomorrow", style="dim")
        console.print("  ss update --task task_123 --duration 2h", style="dim")


def ask(question: str = typer.Argument(..., help="Natural language question about your tasks and projects")) -> None:
    """Ask natural language questions about your tasks, projects, and productivity."""
    # Load current state
    store = EventStore(get_data_dir())
    artifact_store = ArtifactStore(get_data_dir())
    events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(events, artifact_events)

    # Basic question analysis and response
    question_lower = question.lower()

    console.print(f"🤔 Question: {question}", style="bold cyan")
    console.print()

    # Day summary questions
    if any(phrase in question_lower for phrase in ["how is my day", "how's my day", "day today", "today going"]):
        console.print("📅 Your Day Today:", style="bold blue")

        # Today's tasks due
        due_today = state.get_tasks_due_today()
        if due_today:
            console.print(f"• {len(due_today)} tasks due today", style="yellow")
            for task in due_today[:3]:  # Show first 3
                status_icon = "🟡" if task.status == "in_progress" else "⚪"
                console.print(f"  {status_icon} {task.title}", style="dim")
            if len(due_today) > 3:
                console.print(f"  ... and {len(due_today) - 3} more", style="dim")
        else:
            console.print("• No tasks due today", style="green")

        # Overdue items
        overdue = state.get_overdue_tasks()
        if overdue:
            console.print(f"• {len(overdue)} overdue tasks", style="red")

        # Recent activity
        unprocessed = state.get_unprocessed_inbox()
        if unprocessed:
            console.print(f"• {len(unprocessed)} unprocessed inbox items", style="blue")

        # Current focus
        if state.current_focus_project:
            project_name = state.projects.get(state.current_focus_project, {}).name if hasattr(state.projects.get(state.current_focus_project, {}), 'name') else state.current_focus_project
            console.print(f"• Currently focused on: {project_name}", style="green")

    # Simple task count questions (exact matches only)
    elif question_lower.strip() in ["how many tasks do i have", "how many tasks", "task count", "how many active tasks", "how many completed tasks"]:
        active_tasks = state.get_active_tasks()
        completed_tasks = state.get_completed_tasks()

        console.print("📋 Task Summary:", style="bold blue")
        console.print(f"• Active tasks: {len(active_tasks)}", style="yellow")
        console.print(f"• Completed tasks: {len(completed_tasks)}", style="green")

        # Break down by status
        in_progress = [t for t in active_tasks if t.status == "in_progress"]
        pending = [t for t in active_tasks if t.status == "pending"]

        if in_progress:
            console.print(f"• In progress: {len(in_progress)}", style="blue")
        if pending:
            console.print(f"• Pending: {len(pending)}", style="dim")

    # Simple project questions (exact matches only)
    elif any(question_lower.strip() == phrase for phrase in ["what projects am i working on", "what projects", "project status", "show projects"]):
        console.print("📂 Project Overview:", style="bold blue")
        console.print(f"• Total projects: {len(state.projects)}", style="cyan")

        if state.current_focus_project and state.current_focus_project in state.projects:
            project = state.projects[state.current_focus_project]
            console.print(f"• Current focus: {project.name}", style="green")

            # Tasks for focused project
            project_tasks = [t for t in state.get_active_tasks() if t.project_id == state.current_focus_project]
            if project_tasks:
                console.print(f"  - {len(project_tasks)} active tasks", style="dim")

        # List recent projects
        recent_projects = list(state.projects.values())[:5]
        if recent_projects:
            console.print("• Recent projects:", style="dim")
            for project in recent_projects:
                console.print(f"  - {project.name}", style="dim")

    # Simple productivity questions (exact matches only)
    elif any(question_lower.strip() == phrase for phrase in ["how productive have i been", "how productive", "what progress", "what accomplishments"]):
        from datetime import datetime
        completed_today = [t for t in state.get_completed_tasks()
                          if t.completed_at and t.completed_at.date() == datetime.now().date()]

        console.print("🚀 Productivity Today:", style="bold blue")
        if completed_today:
            console.print(f"• {len(completed_today)} tasks completed today!", style="green")
            for task in completed_today[:3]:
                console.print(f"  ✅ {task.title}", style="dim green")
        else:
            console.print("• No tasks completed yet today", style="yellow")

        # Show what's in progress
        in_progress = [t for t in state.get_active_tasks() if t.status == "in_progress"]
        if in_progress:
            console.print(f"• {len(in_progress)} tasks currently in progress:", style="blue")
            for task in in_progress[:3]:
                console.print(f"  🟡 {task.title}", style="dim")

    # Simple artifact questions (exact matches only)
    elif any(question_lower.strip() == phrase for phrase in ["what documents do i have", "what artifacts", "show artifacts", "what documents"]):
        console.print("📎 Artifacts Overview:", style="bold blue")
        console.print(f"• Total artifacts: {len(state.artifacts)}", style="cyan")

        # Group by type
        by_type = {}
        for artifact in state.artifacts.values():
            if not artifact.archived_at:
                by_type[artifact.artifact_type] = by_type.get(artifact.artifact_type, 0) + 1

        for artifact_type, count in by_type.items():
            console.print(f"• {artifact_type}: {count}", style="dim")

    # Advanced natural language query processing using LLM
    else:
        console.print("🧠 Processing your question with AI...", style="dim cyan")

        # Check if this is an analytical/summary request
        analytical_keywords = ['summary', 'developments', 'progress', 'overview', 'what happened', 'status update', 'brief me', 'catch up']
        is_analytical_query = any(keyword in question_lower for keyword in analytical_keywords)

        if is_analytical_query:
            # Handle analytical queries with project summary generation
            try:
                llm_service = LLMService()

                # Extract project name from the question
                project_keywords = []
                words = question_lower.replace('?', '').split()

                # Look for potential project names (2+ letter words, capitalized in original)
                original_words = question.split()
                for i, word in enumerate(original_words):
                    if len(word) > 1 and (word.isupper() or word[0].isupper()):
                        project_keywords.append(word.lower())

                # Find matching project
                target_project = None
                for project in state.projects.values():
                    project_name_lower = project.name.lower()
                    project_aliases_lower = [alias.lower() for alias in project.aliases]

                    # Check if any project keywords match the project name or aliases
                    if any(keyword in project_name_lower or
                          any(keyword in alias for alias in project_aliases_lower)
                          for keyword in project_keywords):
                        target_project = project
                        break

                if target_project:
                    console.print(f"📊 Analyzing {target_project.name} project developments...", style="dim green")

                    # Gather all project data
                    project_tasks = [t for t in state.get_active_tasks() + state.get_completed_tasks()
                                   if t.project_id == target_project.project_id]
                    project_artifacts = [a for a in state.artifacts.values()
                                       if a.project_id == target_project.project_id or
                                          any(t.task_id == a.task_id for t in project_tasks)]

                    if not project_tasks and not project_artifacts:
                        console.print(f"📭 No tasks or artifacts found for {target_project.name} project", style="yellow")
                        return

                    # Create timeline summary
                    console.print(f"📈 {target_project.name} Project Summary:", style="bold blue")
                    console.print()

                    # Task summary
                    active_tasks = [t for t in project_tasks if t.status != 'completed']
                    completed_tasks = [t for t in project_tasks if t.status == 'completed']

                    console.print("🔢 Task Overview:", style="bold")
                    console.print(f"  • Total tasks: {len(project_tasks)}", style="dim")
                    console.print(f"  • Completed: {len(completed_tasks)}", style="green")
                    console.print(f"  • Active: {len(active_tasks)}", style="yellow")
                    console.print()

                    # Recent completions
                    if completed_tasks:
                        console.print("✅ Recent Completions:", style="bold green")
                        for task in sorted(completed_tasks, key=lambda t: t.completed_at or t.created_at, reverse=True)[:5]:
                            time_str = task.completed_at.strftime("%m-%d %H:%M") if task.completed_at else "unknown"
                            console.print(f"  • {task.title} (completed {time_str})", style="dim green")
                        console.print()

                    # Active work
                    if active_tasks:
                        console.print("🔄 Active Work:", style="bold yellow")
                        for task in active_tasks[:5]:
                            status_icon = "🟡" if task.status == "in_progress" else "⚪"
                            due_str = f" (due {task.scheduled_for.strftime('%m-%d')})" if task.scheduled_for else ""
                            console.print(f"  {status_icon} {task.title}{due_str}", style="dim yellow")
                        console.print()

                    # Artifacts and communications
                    if project_artifacts:
                        console.print("📎 Recent Activity & Artifacts:", style="bold cyan")
                        for artifact in sorted(project_artifacts, key=lambda a: a.created_at, reverse=True)[:5]:
                            artifact_icon = "📎" if artifact.artifact_type == 'doc' else "📧" if artifact.artifact_type == 'email' else "💬"
                            time_str = artifact.created_at.strftime("%m-%d %H:%M")
                            console.print(f"  {artifact_icon} {artifact.title} ({time_str})", style="dim cyan")
                        console.print()

                    # Generate AI summary if we have enough data
                    if len(project_tasks) > 2 or len(project_artifacts) > 1:
                        console.print("🧠 AI Analysis:", style="bold blue")

                        # Prepare data for AI summary
                        summary_data = []
                        for task in project_tasks[-10:]:  # Recent 10 tasks
                            summary_data.append({
                                'type': 'task',
                                'title': task.title,
                                'status': task.status,
                                'created_at': task.created_at.isoformat() if task.created_at else None,
                                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                                'priority': task.priority
                            })

                        for artifact in project_artifacts[-5:]:  # Recent 5 artifacts
                            summary_data.append({
                                'type': 'artifact',
                                'title': artifact.title,
                                'artifact_type': artifact.artifact_type,
                                'created_at': artifact.created_at.isoformat(),
                                'content_preview': (artifact.content or '')[:200] if artifact.content else None
                            })

                        # Generate AI summary
                        def run_async_summary():
                            import asyncio
                            return asyncio.run(llm_service.summarize_events(
                                summary_data,
                                time_period="project",
                                style="executive"
                            ))

                        summary_result = run_async_summary()
                        console.print(summary_result.get('summary', 'Unable to generate summary'), style="dim")

                else:
                    console.print("❓ Could not identify specific project from your question", style="yellow")
                    console.print("💡 Try asking about specific projects like:", style="dim")
                    for project in list(state.projects.values())[:3]:
                        console.print(f"   'Give me a summary of {project.name} developments'", style="dim")

                return

            except Exception as e:
                console.print(f"⚠️ Project summary failed: {str(e)}", style="yellow")
                console.print("💡 Try asking simpler questions about specific tasks or projects", style="dim")
                return

        # Regular query processing for non-analytical questions
        try:
            llm_service = LLMService()

            # Prepare context for the LLM
            available_tasks = []
            for task in state.get_active_tasks() + state.get_completed_tasks():
                available_tasks.append({
                    "task_id": task.task_id,
                    "title": task.title,
                    "project_name": task.project_id,
                    "priority": task.priority,
                    "status": task.status,
                    "created_at": task.created_at.isoformat() if task.created_at else "",
                    "completed_at": task.completed_at.isoformat() if task.completed_at else "",
                    "scheduled_for": task.scheduled_for.isoformat() if task.scheduled_for else ""
                })

            available_projects = []
            for project in state.projects.values():
                available_projects.append({
                    "project_id": project.project_id,
                    "name": project.name,
                    "aliases": project.aliases,
                    "task_count": len([t for t in state.get_active_tasks() if t.project_id == project.project_id])
                })

            # Parse the question using LLM
            def run_async_query():
                import asyncio
                return asyncio.run(llm_service.parse_natural_query(
                    question, available_tasks, available_projects
                ))

            parsed_query = run_async_query()

            if parsed_query.get('confidence', 0) > 0.3:
                # Handle the parsed query
                query_type = parsed_query.get('query_type', 'list_tasks')
                filters = parsed_query.get('filters', {})

                # Apply filters to find matching tasks
                matching_tasks = available_tasks.copy()

                # Filter by status
                if filters.get('status'):
                    matching_tasks = [t for t in matching_tasks if t['status'] == filters['status']]

                # Filter by project
                if filters.get('project_name'):
                    proj_name_lower = filters['project_name'].lower()
                    matching_tasks = [t for t in matching_tasks if t['project_name'] and proj_name_lower in t['project_name'].lower()]

                # Filter by keywords
                if filters.get('keywords'):
                    keywords = [k.lower() for k in filters['keywords']]
                    matching_tasks = [t for t in matching_tasks
                                    if any(keyword in t['title'].lower() for keyword in keywords)]

                # Filter by due date
                if filters.get('due_date_filter'):
                    from datetime import datetime, UTC
                    today = datetime.now(UTC).date()

                    if filters['due_date_filter'] == 'today':
                        due_today_ids = [t.task_id for t in state.get_tasks_due_today()]
                        matching_tasks = [t for t in matching_tasks if t['task_id'] in due_today_ids]
                    elif filters['due_date_filter'] == 'overdue':
                        overdue_ids = [t.task_id for t in state.get_overdue_tasks()]
                        matching_tasks = [t for t in matching_tasks if t['task_id'] in overdue_ids]

                # Filter completed tasks by date if relevant
                if filters.get('created_filter') == 'today' or 'today' in question_lower:
                    from datetime import datetime
                    today = datetime.now().date()
                    matching_tasks = [t for t in matching_tasks
                                    if t['completed_at'] and
                                    datetime.fromisoformat(t['completed_at'].replace('Z', '+00:00')).date() == today]

                # Check if user is asking about artifacts
                asking_about_artifacts = any(word in question_lower for word in ['artifact', 'document', 'email', 'message', 'attachment', 'link', 'file'])

                # Filter tasks to only those with artifacts if asking about artifacts
                if asking_about_artifacts:
                    tasks_with_artifacts_filtered = []
                    for task in matching_tasks:
                        task_artifacts = state.get_artifacts_for_task(task['task_id'])
                        if task_artifacts:
                            tasks_with_artifacts_filtered.append(task)
                    matching_tasks = tasks_with_artifacts_filtered

                # Display results
                if query_type == 'count_tasks':
                    console.print(f"📊 Found {len(matching_tasks)} matching tasks", style="bold blue")
                elif query_type == 'list_tasks' and matching_tasks:
                    if asking_about_artifacts:
                        console.print(f"📋 Found {len(matching_tasks)} tasks with artifacts:", style="bold blue")

                        # Show tasks with their linked artifacts
                        for task in matching_tasks[:10]:  # Show first 10
                            status_icon = "✅" if task['status'] == 'completed' else "🟡" if task['status'] == 'in_progress' else "⚪"

                            # Add completion time for completed tasks
                            time_info = ""
                            if task['completed_at'] and 'today' in question_lower:
                                try:
                                    completed_dt = datetime.fromisoformat(task['completed_at'].replace('Z', '+00:00'))
                                    time_info = f" (completed {completed_dt.strftime('%H:%M')})"
                                except:
                                    pass

                            project_prefix = f"[{task['project_name']}] " if task['project_name'] else ""

                            # Get artifacts for this task
                            task_artifacts = state.get_artifacts_for_task(task['task_id'])

                            console.print(f"  {status_icon} {project_prefix}{task['title']}{time_info}", style="white" if task['status'] == 'completed' else "dim")

                            # Show artifacts (we know they exist since we pre-filtered)
                            for artifact in task_artifacts:
                                artifact_icon = "📎" if artifact.artifact_type == 'doc' else "📧" if artifact.artifact_type == 'email' else "💬" if artifact.artifact_type == 'slack_msg' else "📄"
                                console.print(f"      {artifact_icon} {artifact.title}", style="dim cyan")
                                if artifact.url:
                                    console.print(f"         🔗 {artifact.url}", style="dim blue")

                        if len(matching_tasks) > 10:
                            console.print(f"  ... and {len(matching_tasks) - 10} more tasks", style="dim")
                    else:
                        console.print(f"📋 Found {len(matching_tasks)} matching tasks:", style="bold blue")

                        for task in matching_tasks[:10]:  # Show first 10
                            status_icon = "✅" if task['status'] == 'completed' else "🟡" if task['status'] == 'in_progress' else "⚪"

                            # Add completion time for completed tasks
                            time_info = ""
                            if task['completed_at'] and 'today' in question_lower:
                                try:
                                    completed_dt = datetime.fromisoformat(task['completed_at'].replace('Z', '+00:00'))
                                    time_info = f" (completed {completed_dt.strftime('%H:%M')})"
                                except:
                                    pass

                            project_prefix = f"[{task['project_name']}] " if task['project_name'] else ""
                            console.print(f"  {status_icon} {project_prefix}{task['title']}{time_info}", style="white" if task['status'] == 'completed' else "dim")

                        if len(matching_tasks) > 10:
                            console.print(f"  ... and {len(matching_tasks) - 10} more", style="dim")

                elif not matching_tasks:
                    if asking_about_artifacts:
                        console.print("📭 No tasks found with artifacts matching your criteria", style="yellow")
                    else:
                        console.print("📭 No tasks match your criteria", style="yellow")

            else:
                # Low confidence, fall back to basic help
                console.print("🤖 I can help you with questions about:", style="bold blue")
                console.print("• Your day: 'How is my day today?'", style="dim")
                console.print("• Tasks: 'How many tasks do I have?', 'What are my completed tasks?'", style="dim")
                console.print("• Projects: 'What projects am I working on?'", style="dim")
                console.print("• Productivity: 'How productive have I been?'", style="dim")
                console.print("• Artifacts: 'What documents do I have?'", style="dim")
                console.print()
                console.print(f"💡 For research questions, try: ss add \"Research: {question}\"", style="dim")

        except Exception as e:
            console.print(f"⚠️ Question processing failed: {str(e)}", style="yellow")
            console.print("🤖 Try asking simpler questions like:", style="dim")
            console.print("• 'What are my completed tasks?'", style="dim")
            console.print("• 'How many tasks do I have?'", style="dim")


def focus() -> None:
    """Set focus on a project."""
    console.print("Focus functionality - placeholder", style="dim")


def triage() -> None:
    """Interactive triage."""
    console.print("Triage functionality - placeholder", style="dim")


def weekly() -> None:
    """Generate weekly summary."""
    console.print("Weekly summary functionality - placeholder", style="dim")


def daily() -> None:
    """Generate daily summary."""
    console.print("Daily summary functionality - placeholder", style="dim")


def project_detail() -> None:
    """Show project details."""
    console.print("Project detail functionality - placeholder", style="dim")


def project_cleanup() -> None:
    """Clean up projects."""
    console.print("Project cleanup functionality - placeholder", style="dim")


def event_migrate() -> None:
    """Migrate events."""
    console.print("Event migration functionality - placeholder", style="dim")