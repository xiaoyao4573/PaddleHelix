from __future__ import annotations

from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from .tools.base import Tool


class Agent:
	"""Simple agent that routes commands to registered tools.

	This demo agent recognizes slash-commands and a few keyword heuristics.
	Replace `route_text_to_tool` with an LLM-based planner for more power.
	"""

	def __init__(self, tools: List[Tool]):
		self.name_to_tool: Dict[str, Tool] = {tool.name: tool for tool in tools}
		self.console = Console()

	def list_tools(self) -> List[str]:
		return sorted(self.name_to_tool.keys())

	def handle_command(self, line: str) -> str:
		line = line.strip()
		if not line:
			return ""

		if line.startswith("/"):
			# slash command: /tool args...
			parts = line[1:].split(maxsplit=1)
			tool_name = parts[0].lower()
			args = parts[1] if len(parts) > 1 else ""
			tool = self.name_to_tool.get(tool_name)
			if not tool:
				return f"Unknown tool '{tool_name}'. Known: {', '.join(self.list_tools())}"
			return tool.run_cli(args)

		# Basic heuristic routing
		tool = self.route_text_to_tool(line)
		if tool is None:
			return "I could not infer a tool. Use /help or a slash-command."
		return tool.run_cli(line)

	def route_text_to_tool(self, text: str) -> Optional[Tool]:
		lower = text.lower()
		if any(k in lower for k in ["arxiv", "preprint", "paper", "hep-th", "cs."]):
			return self.name_to_tool.get("arxiv")
		if any(k in lower for k in ["pubmed", "ncbi", "medline", "clinical"]):
			return self.name_to_tool.get("pubmed")
		if any(k in lower for k in ["convert", "unit", "units", "to "]):
			return self.name_to_tool.get("units")
		if any(k in lower for k in ["plot", "graph", "curve", "chart"]):
			return self.name_to_tool.get("plot")
		return None

	def print_banner(self) -> None:
		self.console.print(Panel.fit("Science Tools Agent - type /help for commands"))

	def help_text(self) -> str:
		return (
			"Commands:\n"
			"  /arxiv <query>\n"
			"  /pubmed <query>\n"
			"  /units convert <value> <from_unit> to <to_unit>\n"
			"  /plot --expr <expr> [--xmin A --xmax B --outfile file.png]\n"
			"  /help\n  /tools\n  /quit"
		)