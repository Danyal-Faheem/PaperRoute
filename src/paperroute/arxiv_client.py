from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Protocol
from urllib.parse import quote_plus

from .models import Paper

try:
    import httpx
except ImportError:  # optional for unit tests
    httpx = None  # type: ignore


class ArxivTransport(Protocol):
    async def get(self, url: str, **kwargs): ...


class ArxivClient:
    _FIELD_QUERY = re.compile(
        r"(?<![A-Za-z0-9_])(?:all|ti|au|abs|co|jr|cat|rn|id|submittedDate|lastUpdatedDate):",
        re.IGNORECASE,
    )

    def __init__(self, base_url: str = "https://export.arxiv.org/api/query", cache_dir: str | Path = "data/papers",
                 timeout: float = 45, transport: ArxivTransport | None = None) -> None:
        self.base_url, self.cache_dir, self.timeout = base_url, Path(cache_dir), timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transport = transport

    async def search(self, query: str, max_results: int = 20, categories: list[str] | None = None) -> list[Paper]:
        max_results = min(max_results, 20)
        topic = query.strip() if self._FIELD_QUERY.search(query) else f"all:{query}"
        search = f"({topic})"
        if categories:
            search += " AND (" + " OR ".join(f"cat:{c}" for c in categories) + ")"
        url = f"{self.base_url}?search_query={quote_plus(search)}&start=0&max_results={max_results}&sortBy=relevance"
        response = await self._get(url)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        text = response.text if hasattr(response, "text") else str(response)
        return self._parse_feed(text)

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        # ``id_list`` is intentionally used here: routing an ID through
        # ``search_query=all:`` treats the identifier as free text and can
        # return unrelated papers. Preserve legacy slash IDs and versions.
        safe = self._sanitize_id(arxiv_id)
        if not safe:
            return None
        result = await self.get_papers([safe])
        return result.get(safe) or next(iter(result.values()), None)

    async def get_papers(self, arxiv_ids: list[str]) -> dict[str, Paper]:
        """Fetch metadata for a case in one id_list request."""
        safe_ids = [self._sanitize_id(value) for value in arxiv_ids]
        safe_ids = [value for value in safe_ids if value]
        if not safe_ids:
            return {}
        url = f"{self.base_url}?id_list={quote_plus(','.join(safe_ids))}&max_results={len(safe_ids)}"
        response = await self._get(url)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        text = response.text if hasattr(response, "text") else str(response)
        found = {paper.arxiv_id: paper for paper in self._parse_feed(text)}
        # Feeds may return the latest version even when an unversioned ID was
        # requested. Match exact IDs first, then their base identifier while
        # preserving the caller's requested key.
        by_base = {re.sub(r"v\d+$", "", identifier): paper for identifier, paper in found.items()}
        result: dict[str, Paper] = {}
        for requested in safe_ids:
            paper = found.get(requested) or by_base.get(re.sub(r"v\d+$", "", requested))
            if paper:
                # Downstream labels/rankings use the manifest's unversioned
                # ID. Keep the latest feed URL/metadata but canonicalize the
                # identity used by the evaluation joins.
                result[requested] = paper.model_copy(update={"arxiv_id": requested})
        return result

    @staticmethod
    def _sanitize_id(arxiv_id: str) -> str:
        safe = arxiv_id.strip()
        safe = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", safe, flags=re.I)
        return re.sub(r"[^A-Za-z0-9._/:-]", "", safe)

    async def download_pdf(self, paper: Paper) -> Path:
        filename = hashlib.sha256(paper.arxiv_id.encode()).hexdigest()[:24] + ".pdf"
        target = self.cache_dir / filename
        if target.exists() and target.stat().st_size:
            return target
        response = await self._get(paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}")
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        data = response.content if hasattr(response, "content") else bytes(response)
        target.write_bytes(data)
        return target

    async def _get(self, url: str):
        if self.transport:
            try:
                return await self.transport.get(url, timeout=self.timeout)
            except TypeError:
                # Minimal test transports often only accept the URL.
                return await self.transport.get(url)
        if httpx is None:
            raise RuntimeError("httpx is required for live arXiv access")
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            return await client.get(url)

    @staticmethod
    def _parse_feed(xml: str) -> list[Paper]:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml)
        papers: list[Paper] = []
        for entry in root.findall("a:entry", ns):
            link_values = {x.attrib.get("type"): x.attrib.get("href", "") for x in entry.findall("a:link", ns)}
            raw_id = (entry.findtext("a:id", "", ns)).rstrip("/").split("/")[-1]
            papers.append(Paper(arxiv_id=raw_id, title=" ".join((entry.findtext("a:title", "", ns)).split()),
                                abstract=" ".join((entry.findtext("a:summary", "", ns)).split()),
                                authors=[a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)],
                                categories=[x.attrib.get("term", "") for x in entry.findall("a:category", ns)],
                                published=entry.findtext("a:published", None, ns),
                                updated=entry.findtext("a:updated", None, ns),
                                abs_url=link_values.get("text/html", f"https://arxiv.org/abs/{raw_id}"),
                                pdf_url=link_values.get("application/pdf", f"https://arxiv.org/pdf/{raw_id}")))
        unique: dict[str, Paper] = {p.arxiv_id: p for p in papers}
        return list(unique.values())


class OfflineArxivClient:
    """Deterministic six-paper fixture for explicit, no-network demo mode."""

    def __init__(self, cache_dir: str | Path = "data/papers") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.papers = [
            Paper(arxiv_id="demo.00001", title="Evidence-Grounded Language Models",
                  abstract="A practical study of retrieval, grounding, and factuality in language models.", categories=["cs.CL"]),
            Paper(arxiv_id="demo.00002", title="Retrieval Augmented Generation for Research",
                  abstract="We evaluate retrieval augmented generation methods for trustworthy answers.", categories=["cs.AI"]),
            Paper(arxiv_id="demo.00003", title="Calibration of Neural Predictors",
                  abstract="Methods for uncertainty and calibration of neural network predictions.", categories=["cs.LG"]),
            Paper(arxiv_id="demo.00004", title="Efficient Attention in Transformers",
                  abstract="An empirical analysis of efficient attention for long-context models.", categories=["cs.LG"]),
            Paper(arxiv_id="demo.00005", title="A Survey of Machine Learning Systems",
                  abstract="A broad survey of machine learning systems and reproducibility practices.", categories=["cs.AI"]),
            Paper(arxiv_id="demo.00006", title="Graph Representations for Relational Data",
                  abstract="Graph neural representations for relational learning tasks.", categories=["cs.LG"]),
        ]

    async def search(self, query: str, max_results: int = 20, categories: list[str] | None = None) -> list[Paper]:
        terms = {term.casefold() for term in query.split() if len(term) > 3}
        ranked = sorted(self.papers, key=lambda paper: -sum(term in (paper.title + " " + paper.abstract).casefold() for term in terms))
        normalized_categories = {category.strip() for category in (categories or []) if category.strip()}
        # The landing page's default ``cs.AI, cs.CL`` scope is a broad demo
        # scope: retain all six fixtures so the local walkthrough exercises
        # the full shortlist. Explicit/custom scopes still filter normally.
        if normalized_categories and normalized_categories != {"cs.AI", "cs.CL"}:
            scoped = [paper for paper in ranked if normalized_categories & set(paper.categories)]
            ranked = scoped or ranked
        return ranked[:min(max_results, 20)]

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        return next((paper for paper in self.papers if paper.arxiv_id == arxiv_id.strip()), None)

    async def download_pdf(self, paper: Paper) -> Path:
        target = self.cache_dir / (paper.arxiv_id.replace("/", "_") + ".pdf")
        if not target.exists():
            target.write_bytes(_demo_pdf_bytes(paper))
        return target


def _demo_pdf_bytes(paper: Paper) -> bytes:
    """Create a tiny valid one-page PDF without requiring a PDF generator."""
    text = f"{paper.title}. Evidence one: {paper.abstract[:140]}. Evidence two: this fixture supports reproducible triage."
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    output, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(output)
    output += b"xref\n0 6\n0000000000 65535 f \n"
    output += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    return output + b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF\n"
