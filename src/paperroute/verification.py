from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .models import Evidence

_DASHES = "‐‑‒–—―−"
_QUOTES = "‘’‚‛“”„‟"


def normalize_text(value: str) -> str:
    """Normalize common PDF extraction artifacts without inventing wording."""
    value = unicodedata.normalize("NFKC", value).translate(
        str.maketrans({**dict.fromkeys(_DASHES, "-"), **dict.fromkeys(_QUOTES, "\""), "\u00a0": " "})
    )
    # PDF extraction often splits a word at a line boundary as ``evi-\ndence``.
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def canonical_text(value: str) -> str:
    """Return punctuation-insensitive text for exact quote comparison."""
    return "".join(char for char in normalize_text(value) if char.isalnum())


def extract_pages(pdf_path: str | Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    try:
        return [(page.extract_text() or "") for page in PdfReader(str(pdf_path)).pages]
    except Exception:
        return []


def verify_evidence(evidence: list[Evidence], pages: list[str]) -> list[Evidence]:
    verified: list[Evidence] = []
    for item in evidence:
        idx = item.page - 1
        quotation = canonical_text(item.quotation)
        page = canonical_text(pages[idx]) if 0 <= idx < len(pages) else ""
        found = bool(quotation) and quotation in page
        item.verified = found
        item.verification_note = "Exact quotation found on page." if found else "Quotation not found in extracted page text."
        verified.append(item)
    return verified
