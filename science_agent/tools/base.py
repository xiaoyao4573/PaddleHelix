from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
	name: str
	description: str

	@abstractmethod
	def run_cli(self, args: str) -> str:
		"""Run the tool given a CLI-like argument string."""
		raise NotImplementedError