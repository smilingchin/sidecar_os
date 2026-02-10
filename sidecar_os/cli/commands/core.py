"""Core commands for Sidecar OS."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional, List, Dict

from sidecar_os.core.sidecar_core.events import EventStore, InboxCapturedEvent, TaskCreatedEvent, TaskCompletedEvent, ProjectCreatedEvent, ProjectFocusedEvent, ProjectFocusClearedEvent, ClarificationRequestedEvent, ClarificationResolvedEvent
from sidecar_os.core.sidecar_core.state import project_events_to_state
from sidecar_os.core.sidecar_core.state.models import SidecarState
from sidecar_os.core.sidecar_core.router import AdvancedPatternInterpreter, InterpreterConfig
from sidecar_os.core.sidecar_core.llm import LLMService, get_usage_tracker
from sidecar_os.core.sidecar_core.summaries import SummaryGenerator, SummaryStyle, SummaryPeriod

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