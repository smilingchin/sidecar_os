import typer
from .commands.core import add, status, task, list_items, done, focus, triage, project_add, project_list, weekly, daily, project_cleanup

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

# Register project commands
app.command()(focus)
app.command()(triage)
app.command(name="project-add")(project_add)
app.command(name="project-list")(project_list)
app.command(name="project-cleanup")(project_cleanup)

# Register summary commands
app.command()(weekly)
app.command()(daily)

# Entry point function for the CLI script
def cli() -> None:
    app()
