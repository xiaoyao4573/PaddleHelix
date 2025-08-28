from __future__ import annotations

import html
import textwrap
from typing import List

import requests

from .base import Tool

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

def _pubmed_search(query: str, retmax: int = 5) -> List[str]:
	params = {
		"db": "pubmed",
		"term": query,
		"retmode": "json",
		"retmax": str(retmax),
	}
	r = requests.get(ESEARCH, params=params, timeout=20)
	r.raise_for_status()
	data = r.json()
	return data.get("esearchresult", {}).get("idlist", [])


def _pubmed_summaries(ids: List[str]) -> List[str]:
	if not ids:
		return []
	params = {
		"db": "pubmed",
		"id": ",".join(ids),
		"retmode": "json",
	}
	r = requests.get(ESUMMARY, params=params, timeout=20)
	r.raise_for_status()
	data = r.json().get("result", {})
	uids = data.get("uids", [])
	lines: List[str] = []
	for uid in uids:
		item = data.get(uid, {})
		title = html.unescape(item.get("title", "Untitled"))
		journal = item.get("fulljournalname") or item.get("source") or ""
		year = item.get("pubdate", "").split(" ")[0]
		authors = item.get("authors", [])
		first_author = authors[0]["name"] if authors else "Unknown"
		pmid = item.get("uid", uid)
		ul = f"- {title} ({year}) {journal} by {first_author}\n  https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
		lines.append(textwrap.shorten(ul, width=400))
	return lines


class PubMedTool(Tool):
	name = "pubmed"
	description = "Search PubMed and list top results"

	def run_cli(self, args: str) -> str:
		query = args.strip()
		if not query:
			return "Usage: /pubmed <query>"
		ids = _pubmed_search(query, retmax=5)
		lines = _pubmed_summaries(ids)
		if not lines:
			return "No results found."
		return "\n".join(lines)