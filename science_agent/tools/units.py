from __future__ import annotations

from pint import UnitRegistry

from .base import Tool


class UnitsTool(Tool):
	name = "units"
	description = "Convert between units using Pint"

	def __init__(self) -> None:
		self.ureg = UnitRegistry()

	def run_cli(self, args: str) -> str:
		parts = args.strip().split()
		if len(parts) >= 5 and parts[0] == "convert" and parts[3] == "to":
			value = float(parts[1])
			from_unit = parts[2]
			to_unit = parts[4]
			qty = value * self.ureg(from_unit)
			converted = qty.to(to_unit)
			return f"{qty:~P} = {converted:~P}"
		return "Usage: /units convert <value> <from_unit> to <to_unit>"