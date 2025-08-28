# Science Tools Agent

An extensible Python agent that can call science-related tools (arXiv, PubMed, unit conversion, plotting) via a simple CLI and an interactive chat.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py --help
```

## CLI Examples

- arXiv search:
```bash
python cli.py arxiv "graph neural networks chemistry"
```

- PubMed search and summaries:
```bash
python cli.py pubmed "CRISPR prime editing"
```

- Unit conversion:
```bash
python cli.py units convert 9.81 m/s^2 to ft/s^2
```

- Plot a quick function and save to a file:
```bash
python cli.py plot --expr "sin(x)" --xmin -6.28 --xmax 6.28 --outfile plot.png
```

## Interactive Agent

Start a chat loop that can call tools with slash-commands:

```bash
python cli.py chat
```

Commands inside chat:
- `/arxiv <query>`
- `/pubmed <query>`
- `/units convert <value> <from_unit> to <to_unit>`
- `/plot --expr <expr> [--xmin A --xmax B --outfile file.png]`

Natural language mapping is basic and keyword-based to keep the demo lightweight. Replace with your favorite LLM for smarter orchestration.

## Project Structure

```
science_agent/
  agent.py
  tools/
    __init__.py
    base.py
    arxiv.py
    pubmed.py
    units.py
    plot.py
cli.py
requirements.txt
README.md
```

## Notes
- PubMed access uses NCBI E-utilities. Be considerate with rate limits.
- Plotting evaluates a math expression over x using Python's `math` with safe globals.
- Extend the agent by adding new tools implementing `Tool` in `science_agent/tools/base.py`.
