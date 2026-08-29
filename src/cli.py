"""experience-compiler-lab CLI (thin dispatch layer; commands live in subpackages)."""

import typer

app = typer.Typer(name="exp", help="Agent Experience Compiler")


@app.callback()
def main() -> None:
    """Agent Experience Compiler — root group.

    NOTE: the callback exists so the app is ALWAYS built as a click Group.
    Typer 0.27 collapses single-command apps into that command, which breaks
    CliRunner tests (`invoke(app, ["version"])` → "unexpected extra
    arguments") and changes behavior as soon as a second command is added.
    """


@app.command()
def version() -> None:
    """Print version."""
    typer.echo("0.1.0")


if __name__ == "__main__":
    app()
