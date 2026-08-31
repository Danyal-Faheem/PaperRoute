from __future__ import annotations

from .models import Run


def report_json(run: Run) -> str:
    return run.model_dump_json(indent=2)


def report_markdown(run: Run) -> str:
    lines = ["# PaperRoute report\n", f"**Run:** `{run.id}`  ", f"**Status:** {run.status.value}  ",
             f"**Question:** {run.request.research_question}\n", "## Ranked papers\n"]
    if not run.assessments:
        lines.append("No assessments are available yet.\n")
    for i, item in enumerate(run.assessments, 1):
        lines += [f"### {i}. {item.paper.title}", f"- **Decision:** {item.decision.value}",
                  f"- **Score:** {item.total_score:.2f}/100", f"- **arXiv:** [{item.paper.arxiv_id}]({item.paper.abs_url})",
                  f"- {item.summary or 'No summary provided.'}"]
        if item.limitations:
            lines.append(f"- **Limitations:** {'; '.join(item.limitations)}")
        if item.evidence:
            lines.append("- **Evidence:**")
            lines.extend(f"  - p. {e.page} ({'verified' if e.verified else 'unverified'}): {e.quotation}" for e in item.evidence)
        lines.append("")
    return "\n".join(lines)
