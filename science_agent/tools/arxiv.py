from __future__ import annotations

import datetime as _dt
import html
import re
import xml.etree.ElementTree as ET

import requests

from .base import Tool

ARXIV_API = "https://export.arxiv.org/api/query"


def _strip_html(text: str) -> str:
	text = re.sub(r"<[^>]+>", " ", text)
	return re.sub(r"\s+", " ", text).strip()


def _search_arxiv(query: str, max_results: int = 5) -> str:
	params = {
		"search_query": query,
		"start": 0,
		"max_results": max_results,
		"sortBy": "relevance",
	}
	r = requests.get(ARXIV_API, params=params, timeout=20)
	r.raise_for_status()
	root = ET.fromstring(r.text)
	ns = {"a": "http://www.w3.org/2005/Atom"}
	lines = []
	for entry in root.findall("a:entry", ns):
		title = entry.findtext("a:title", default="", namespaces=ns) or "Untitled"
		title = html.unescape(_strip_html(title))
		link = entry.find("a:id", ns)
		url = link.text if link is not None else ""
		published = entry.findtext("a:published", default="", namespaces=ns)
		date = published.split("T")[0] if published else ""
		auths = [a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)]
		first_author = auths[0] if auths else "Unknown"
		lines.append(f"- {title} ({date}) by {first_author}\n  {url}")
	if not lines:
		return "No results found."
	return "\n".join(lines)


class ArxivTool(Tool):
	name = "arxiv"
	description = "Search arXiv and list top results (Atom API)"

	def run_cli(self, args: str) -> str:
		query = args.strip()
		if not query:
			return "Usage: /arxiv <query>"
		return _search_arxiv(query, max_results=5)