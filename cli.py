import shlex
import sys
from typing import Optional, List

import typer
from rich.console import Console

from science_agent.agent import Agent
from science_agent.tools.arxiv import ArxivTool
from science_agent.tools.pubmed import PubMedTool
from science_agent.tools.units import UnitsTool
from science_agent.tools.plot import PlotTool

app = typer.Typer(help="Science Tools Agent CLI")
console = Console()


def build_agent() -> Agent:
    tools = [ArxivTool(), PubMedTool(), UnitsTool(), PlotTool()]
    return Agent(tools)


@app.command()
def arxiv(query: str):
    """Search arXiv for papers."""
    tool = ArxivTool()
    result = tool.run_cli(query)
    console.print(result)


@app.command()
def pubmed(query: str):
    """Search PubMed for articles."""
    tool = PubMedTool()
    result = tool.run_cli(query)
    console.print(result)


units_app = typer.Typer(help="Unit conversion commands")
app.add_typer(units_app, name="units")


@units_app.command("convert")
def units_convert(args: List[str] = typer.Argument(..., help="e.g., 9.81 m/s^2 to ft/s^2")):
    """Convert between units using Pint."""
    tool = UnitsTool()
    joined = " ".join(args)
    result = tool.run_cli(f"convert {joined}")
    console.print(result)


@app.command()
def plot(
    expr: str = typer.Option(..., "--expr", help="Expression in x, e.g., 'sin(x)'")
    , xmin: float = typer.Option(-10.0, "--xmin")
    , xmax: float = typer.Option(10.0, "--xmax")
    , points: int = typer.Option(400, "--points")
    , outfile: Optional[str] = typer.Option(None, "--outfile", help="Save to file if provided")
):
    tool = PlotTool()
    args = f"--expr {shlex.quote(expr)} --xmin {xmin} --xmax {xmax} --points {points}"
    if outfile:
        args += f" --outfile {shlex.quote(outfile)}"
    result = tool.run_cli(args)
    console.print(result)


@app.command()
def chat():
    """Interactive chat with slash-commands for tools."""
    agent = build_agent()
    agent.print_banner()
    console.print(agent.help_text())
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            raise typer.Exit(0)
        if line.strip().lower() in {"/quit", "/exit"}:
            console.print("Goodbye!")
            raise typer.Exit(0)
        if line.strip().lower() in {"/help"}:
            console.print(agent.help_text())
            continue
        if line.strip().lower() in {"/tools"}:
            console.print(", ".join(agent.list_tools()))
            continue
        out = agent.handle_command(line)
        if out:
            console.print(out)


if __name__ == "__main__":
    app()