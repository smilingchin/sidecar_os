"""Artifact management commands for Sidecar OS."""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional
from uuid import uuid4

from sidecar_os.core.sidecar_core.events import EventStore, ArtifactRegisteredEvent, ArtifactLinkedEvent, ArtifactUnlinkedEvent, ArtifactArchivedEvent
from sidecar_os.core.sidecar_core.artifacts import ArtifactStore
from sidecar_os.core.sidecar_core.state import project_events_to_state

console = Console()


def artifact_add(
    artifact_type: str = typer.Argument(help="Type: slack_msg, email, doc, meeting_notes, gdrive"),
    title: str = typer.Argument(help="Human-readable title"),
    content: str = typer.Option(None, "--content", "-c", help="Full content/text"),
    url: str = typer.Option(None, "--url", "-u", help="External URL"),
    source: str = typer.Option(None, "--source", "-s", help="Source identifier"),
    task_id: str = typer.Option(None, "--task", "-t", help="Link to task ID"),
    project_id: str = typer.Option(None, "--project", "-p", help="Link to project ID")
) -> None:
    """Register a new project artifact (Slack message, document, email, etc.).

    Examples:
      sidecar artifact-add slack_msg "Weekly team update" \\
        --content "Hey team, here's this week's progress..." \\
        --project ca-zap --source "C1234567.1704067200.002345"

      sidecar artifact-add doc "Q1 Planning Spreadsheet" \\
        --url "https://company.sharepoint.com/sites/planning/q1-2024.xlsx" \\
        --project q1-planning
    """
    # Validate artifact type
    valid_types = ["slack_msg", "email", "doc", "meeting_notes", "gdrive", "sharepoint", "quip", "call_notes"]
    if artifact_type not in valid_types:
        console.print(f"❌ Invalid artifact type: {artifact_type}", style="red")
        console.print(f"Valid types: {', '.join(valid_types)}", style="dim")
        return

    # Generate artifact ID
    artifact_id = str(uuid4())

    # Auto-generate source if not provided
    if not source:
        if url:
            # Extract source from URL
            if "slack.com" in url:
                # Extract channel and timestamp from Slack URL
                parts = url.split("/")
                if len(parts) >= 3:
                    channel = parts[-2] if parts[-2].startswith("C") else "unknown"
                    timestamp = parts[-1] if parts[-1].startswith("p") else "unknown"
                    source = f"slack:{channel}.{timestamp}"
                else:
                    source = f"slack:unknown"
            elif "sharepoint.com" in url:
                source = f"sharepoint:{url.split('/')[-1]}"
            elif "quip.com" in url:
                source = f"quip:{url.split('/')[-1]}"
            else:
                source = f"url:{url.split('/')[-1]}"
        else:
            source = f"{artifact_type}:{artifact_id[:8]}"

    # Validate project/task references exist
    store = EventStore()
    events = store.read_all()
    state = project_events_to_state(events)

    if project_id and project_id not in state.projects:
        console.print(f"⚠️  Project '{project_id}' not found. Artifact will be created but not linked to project.", style="yellow")
        project_id = None

    if task_id and task_id not in state.tasks:
        console.print(f"⚠️  Task '{task_id}' not found. Artifact will be created but not linked to task.", style="yellow")
        task_id = None

    # Create artifact event
    artifact_event = ArtifactRegisteredEvent(
        payload={
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "title": title,
            "content": content,
            "url": url,
            "source": source,
            "task_id": task_id,
            "project_id": project_id,
            "created_by": "cli_user",
            "metadata": {}
        }
    )

    # Store artifact
    artifact_store = ArtifactStore()
    stored_id = artifact_store.register_artifact(artifact_event)

    # Display confirmation
    console.print(f"✅ Registered artifact: [bold]{title}[/bold]", style="green")
    console.print(f"   Type: {artifact_type}")
    console.print(f"   Source: {source}")
    console.print(f"   Artifact ID: {artifact_id[:8]}...", style="dim")

    if project_id:
        project_name = state.projects[project_id].name
        console.print(f"   🔗 Linked to project: [bold]{project_name}[/bold]")

    if task_id:
        task_title = state.tasks[task_id].title
        console.print(f"   🔗 Linked to task: [bold]{task_title[:50]}...[/bold]")

    if url:
        console.print(f"   🌐 URL: {url}")

    if content:
        content_preview = content[:100] + "..." if len(content) > 100 else content
        console.print(f"   📄 Content: {content_preview}")


def artifact_link(
    artifact_id: str = typer.Argument(help="Artifact ID to link"),
    task_id: str = typer.Option(None, "--task", "-t", help="Task to link to"),
    project_id: str = typer.Option(None, "--project", "-p", help="Project to link to")
) -> None:
    """Link existing artifact to task or project."""
    if not task_id and not project_id:
        console.print("❌ Must specify either --task or --project to link to", style="red")
        return

    # Load current state
    store = EventStore()
    artifact_store = ArtifactStore()

    main_events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(main_events, artifact_events)

    # Find artifact
    artifact = None
    for art in state.artifacts.values():
        if art.artifact_id == artifact_id or art.artifact_id.startswith(artifact_id):
            artifact = art
            break

    if not artifact:
        console.print(f"❌ Artifact not found: {artifact_id}", style="red")
        console.print("💡 Use: sidecar artifact-list to see available artifacts", style="dim")
        return

    # Validate references
    if task_id and task_id not in state.tasks:
        console.print(f"❌ Task not found: {task_id}", style="red")
        return

    if project_id and project_id not in state.projects:
        console.print(f"❌ Project not found: {project_id}", style="red")
        return

    # Create link event
    link_event = ArtifactLinkedEvent(
        payload={
            "artifact_id": artifact.artifact_id,
            "task_id": task_id,
            "project_id": project_id
        }
    )

    # Store link event
    artifact_store.register_artifact(link_event)

    # Display confirmation
    console.print(f"✅ Linked artifact: [bold]{artifact.title}[/bold]", style="green")

    if task_id:
        task_title = state.tasks[task_id].title
        console.print(f"   🔗 To task: [bold]{task_title}[/bold]")

    if project_id:
        project_name = state.projects[project_id].name
        console.print(f"   🔗 To project: [bold]{project_name}[/bold]")


def artifact_list(
    project_id: str = typer.Option(None, "--project", "-p", help="Filter by project"),
    artifact_type: str = typer.Option(None, "--type", "-t", help="Filter by type"),
    task_id: str = typer.Option(None, "--task", help="Filter by task")
) -> None:
    """List artifacts with optional filtering."""
    # Load all events
    store = EventStore()
    artifact_store = ArtifactStore()

    main_events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(main_events, artifact_events)

    # Apply filters
    artifacts = list(state.artifacts.values())

    if project_id:
        artifacts = state.get_artifacts_for_project(project_id)
        if not artifacts:
            console.print(f"📎 No artifacts found for project: {project_id}", style="yellow")
            return

    if task_id:
        artifacts = state.get_artifacts_for_task(task_id)
        if not artifacts:
            console.print(f"📎 No artifacts found for task: {task_id}", style="yellow")
            return

    if artifact_type:
        artifacts = [a for a in artifacts if a.artifact_type == artifact_type]
        if not artifacts:
            console.print(f"📎 No artifacts found of type: {artifact_type}", style="yellow")
            return

    # Remove archived artifacts
    artifacts = [a for a in artifacts if not a.archived_at]

    if not artifacts:
        console.print("📎 No artifacts found", style="dim")
        console.print("💡 Try: sidecar artifact-add to create artifacts", style="dim")
        return

    # Create artifacts table
    table = Table(show_header=True, header_style="bold cyan", title="📎 Project Artifacts")
    table.add_column("ID", style="dim", width=12)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Title", style="white", width=40)
    table.add_column("Project", style="green", width=15)
    table.add_column("Task", style="yellow", width=20)
    table.add_column("Created", style="dim", width=12)

    # Sort by creation date (most recent first)
    artifacts = sorted(artifacts, key=lambda a: a.created_at, reverse=True)

    for artifact in artifacts:
        # Get project name
        project_name = ""
        if artifact.project_id and artifact.project_id in state.projects:
            project_name = state.projects[artifact.project_id].name

        # Get task title
        task_title = ""
        if artifact.task_id and artifact.task_id in state.tasks:
            task_title = state.tasks[artifact.task_id].title
            if len(task_title) > 18:
                task_title = task_title[:15] + "..."

        table.add_row(
            artifact.artifact_id[:8] + "...",
            artifact.artifact_type,
            artifact.title[:37] + "..." if len(artifact.title) > 40 else artifact.title,
            project_name,
            task_title,
            artifact.created_at.strftime("%m-%d %H:%M")
        )

    console.print(table)
    console.print(f"💡 Use 'sidecar artifact-show <id>' to see full details", style="dim")


def artifact_show(artifact_id: str = typer.Argument(help="Artifact ID")) -> None:
    """Show detailed artifact information including full content."""
    # Load all events
    store = EventStore()
    artifact_store = ArtifactStore()

    main_events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(main_events, artifact_events)

    # Find artifact
    artifact = None
    for art in state.artifacts.values():
        if art.artifact_id == artifact_id or art.artifact_id.startswith(artifact_id):
            artifact = art
            break

    if not artifact:
        console.print(f"❌ Artifact not found: {artifact_id}", style="red")
        console.print("💡 Use: sidecar artifact-list to see available artifacts", style="dim")
        return

    # Display detailed artifact info
    console.print(f"📎 Artifact: [bold]{artifact.title}[/bold]")
    console.print(f"   ID: {artifact.artifact_id}")
    console.print(f"   Type: {artifact.artifact_type}")
    console.print(f"   Source: {artifact.source}")
    console.print(f"   Created: {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    if artifact.created_by:
        console.print(f"   Created by: {artifact.created_by}")

    if artifact.project_id and artifact.project_id in state.projects:
        project_name = state.projects[artifact.project_id].name
        console.print(f"   🔗 Project: [bold]{project_name}[/bold]")

    if artifact.task_id and artifact.task_id in state.tasks:
        task_title = state.tasks[artifact.task_id].title
        console.print(f"   🔗 Task: [bold]{task_title}[/bold]")

    if artifact.url:
        console.print(f"   🌐 URL: {artifact.url}")

    if artifact.content:
        console.print()
        console.print("📄 Content:", style="bold")
        console.print(artifact.content)

    if artifact.metadata:
        console.print()
        console.print("📊 Metadata:", style="bold")
        for key, value in artifact.metadata.items():
            console.print(f"   {key}: {value}")

    if artifact.archived_at:
        console.print()
        console.print(f"🗄️  Archived: {artifact.archived_at.strftime('%Y-%m-%d %H:%M:%S')}", style="red")


def artifact_archive(artifact_id: str = typer.Argument(help="Artifact ID")) -> None:
    """Archive an artifact (soft delete)."""
    # Load all events
    store = EventStore()
    artifact_store = ArtifactStore()

    main_events = store.read_all()
    artifact_events = artifact_store.read_all_artifact_events()
    state = project_events_to_state(main_events, artifact_events)

    # Find artifact
    artifact = None
    for art in state.artifacts.values():
        if art.artifact_id == artifact_id or art.artifact_id.startswith(artifact_id):
            artifact = art
            break

    if not artifact:
        console.print(f"❌ Artifact not found: {artifact_id}", style="red")
        console.print("💡 Use: sidecar artifact-list to see available artifacts", style="dim")
        return

    if artifact.archived_at:
        console.print(f"⚠️  Artifact already archived: {artifact.title}", style="yellow")
        return

    # Create archive event
    archive_event = ArtifactArchivedEvent(
        payload={
            "artifact_id": artifact.artifact_id,
            "reason": "Manual archive via CLI"
        }
    )

    # Store archive event
    artifact_store.register_artifact(archive_event)

    console.print(f"🗄️  Archived artifact: [bold]{artifact.title}[/bold]", style="yellow")
    console.print("   Artifact is now hidden from default views but remains in storage", style="dim")