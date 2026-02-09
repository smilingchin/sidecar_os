import typer
from .commands.core import add, status

app = typer.Typer(help="Sidecar OS - event-sourced productivity assistant")

@app.command()
def hello() -> None:
    """Sanity check command."""
    typer.echo("Sidecar OS is alive.")

# Register core commands
app.command()(add)
app.command()(status)

# Entry point function for the CLI script
def cli() -> None:
    app()
