import typer
from .commands.core import add, status, task, list_items, done, update, ask, focus, triage, project_add, project_list, project_detail, weekly, daily, project_cleanup, event_migrate
from .commands.artifacts import artifact_add, artifact_link, artifact_list, artifact_show, artifact_archive

app = typer.Typer(help="Sidecar OS - event-sourced productivity assistant")

@app.command()
def hello() -> None:
    """Sanity check command."""
    typer.echo("Sidecar OS is alive.")

# Register core commands
app.command()(add)
app.command()(status)
app.command()(task)
app.command(name="list")(list_items)  # "list" is a Python builtin, so use name mapping
app.command()(done)
app.command()(update)
app.command()(ask)

# Register project commands
app.command()(focus)
app.command()(triage)
app.command(name="project-add")(project_add)
app.command(name="project-list")(project_list)
app.command(name="project-detail")(project_detail)
app.command(name="project-cleanup")(project_cleanup)
app.command(name="event-migrate")(event_migrate)

# Register summary commands
app.command()(weekly)
app.command()(daily)

# Register artifact commands
app.command(name="artifact-add")(artifact_add)
app.command(name="artifact-link")(artifact_link)
app.command(name="artifact-list")(artifact_list)
app.command(name="artifact-show")(artifact_show)
app.command(name="artifact-archive")(artifact_archive)

# Entry point function for the CLI script
def cli() -> None:
    app()
