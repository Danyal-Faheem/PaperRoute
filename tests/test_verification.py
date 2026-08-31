from paperroute.models import Evidence
from paperroute.verification import extract_pages, normalize_text, verify_evidence


def test_normalization_handles_whitespace_and_case():
    assert normalize_text("  A\nsmall   RESULT ") == "a small result"


def test_verify_evidence_checks_the_declared_page_and_marks_failures():
    evidence = [
        Evidence(claim="result", quotation="The model improves accuracy.", page=2),
        Evidence(claim="bad", quotation="not present", page=1),
        Evidence(claim="missing", quotation="anything", page=4),
    ]
    result = verify_evidence(evidence, ["intro", "The model\n improves accuracy."])
    assert [item.verified for item in result] == [True, False, False]
    assert result[0].verification_note == "Exact quotation found on page."


def test_verify_evidence_handles_unicode_pdf_artifacts_and_hyphenation():
    evidence = [Evidence(claim="exact", quotation='The “efficient—model” uses ligatures.', page=1)]
    result = verify_evidence(evidence, ["The e\ufb03cient-model uses ligatures."])
    assert result[0].verified is True

    evidence = [Evidence(claim="exact", quotation="The efficient-model is robust.", page=1)]
    result = verify_evidence(evidence, ["The efficient-\nmodel is robust."])
    assert result[0].verified is True


def test_verify_evidence_rejects_paraphrase():
    evidence = [Evidence(claim="paraphrase", quotation="The method substantially improves results.", page=1)]
    result = verify_evidence(evidence, ["The method improves accuracy."])
    assert result[0].verified is False


def test_extract_pages_returns_empty_for_missing_or_malformed_pdf(tmp_path):
    assert extract_pages(tmp_path / "missing.pdf") == []
    malformed = tmp_path / "bad.pdf"
    malformed.write_bytes(b"not a pdf")
    assert extract_pages(malformed) == []
