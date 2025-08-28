from __future__ import annotations

import math
import shlex
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .base import Tool


# Build a safe globals that prefers NumPy vectorized functions.
SAFE_GLOBALS: Dict[str, object] = {
	"__builtins__": {},
	"np": np,
}
for _name in dir(np):
	if not _name.startswith("_"):
		SAFE_GLOBALS[_name] = getattr(np, _name)
# Add a few constants from math for convenience
SAFE_GLOBALS.update({"pi": math.pi, "e": math.e})


class PlotTool(Tool):
	name = "plot"
	description = "Plot y = f(x) over a range and optionally save to a file"

	def run_cli(self, args: str) -> str:
		# robust arg parsing that respects quotes
		tokens = shlex.split(args)
		expr = None
		xmin = -10.0
		xmax = 10.0
		points = 400
		outfile: Optional[str] = None
		i = 0
		while i < len(tokens):
			t = tokens[i]
			if t == "--expr" and i + 1 < len(tokens):
				expr = tokens[i + 1]
				i += 2
				continue
			if t == "--xmin" and i + 1 < len(tokens):
				xmin = float(tokens[i + 1])
				i += 2
				continue
			if t == "--xmax" and i + 1 < len(tokens):
				xmax = float(tokens[i + 1])
				i += 2
				continue
			if t == "--points" and i + 1 < len(tokens):
				points = int(tokens[i + 1])
				i += 2
				continue
			if t == "--outfile" and i + 1 < len(tokens):
				outfile = tokens[i + 1]
				i += 2
				continue
			i += 1

		if not expr:
			return "Usage: /plot --expr <expr> [--xmin A --xmax B --points N --outfile file.png]"

		# strip surrounding quotes if present (defensive)
		if (expr.startswith("'") and expr.endswith("'")) or (expr.startswith('"') and expr.endswith('"')):
			expr = expr[1:-1]

		x = np.linspace(xmin, xmax, points)
		local_vars = {"x": x}
		try:
			y = eval(expr, SAFE_GLOBALS, local_vars)
		except Exception as e:
			return f"Failed to evaluate expression: {e}"

		plt.figure(figsize=(6, 3.5))
		plt.plot(x, y)
		plt.title(f"y = {expr}")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.grid(True, alpha=0.3)
		if outfile:
			plt.savefig(outfile, bbox_inches="tight", dpi=150)
			plt.close()
			return f"Saved plot to {outfile}"
		else:
			plt.savefig("plot.png", bbox_inches="tight", dpi=150)
			plt.close()
			return "Saved plot to plot.png"