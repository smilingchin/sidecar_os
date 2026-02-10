import typer
from .commands.core import add, status, task, list_items, done, focus, triage, project_add, project_list

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

# Entry point function for the CLI script
def cli() -> None:
    app()
