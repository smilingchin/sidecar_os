import click

@click.group(help="Sidecar OS - event-sourced productivity assistant")
def cli() -> None:
    pass

@cli.command(help="Sanity check command.")
def hello() -> None:
    click.echo("Sidecar OS is alive.")
