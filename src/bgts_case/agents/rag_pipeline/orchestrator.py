"""Orchestrator for the RAG ingestion pipeline.

Walks every PDF under the configured knowledge-base directory and runs the
two pipeline steps for each one in order:

    step 1 — chunk : PDF -> Documents (``step1_chunk.process_pdf``)
    step 2 — index : Documents -> Qdrant hybrid index (``step2_qdrant_index.upsert_chunks``)

Per-PDF failures are isolated: a single bad PDF logs an error and triggers a
Slack alert, but the remaining PDFs continue to be indexed. A final summary
alert reports the totals.
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from bgts_case.agents.alerts import mock_slack_alert
from bgts_case.agents.rag_pipeline import step1_chunk, step2_qdrant_index

# Project layout: src/bgts_case/agents/rag_pipeline/orchestrator.py -> parents[4] = repo root.
DEFAULT_KB_DIR = Path(__file__).resolve().parents[4] / "static" / "knowledge_base"


def _index_one(pdf_path: Path) -> int:
    """Chunk + index a single PDF. Returns the number of chunks indexed."""
    chunks = step1_chunk.process_pdf(pdf_path=pdf_path, alert_fn=mock_slack_alert)
    logger.info(f"orchestrator: {pdf_path.name} -> {len(chunks)} chunks")
    step2_qdrant_index.upsert_chunks(chunks=chunks)
    return len(chunks)


def run_pipeline(kb_dir: str | Path = DEFAULT_KB_DIR) -> dict:
    """Index every PDF under ``kb_dir`` into Qdrant.

    Returns a dict with ``indexed``, ``failed``, ``total_chunks``.
    """
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    pdf_paths = sorted(kb_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning(f"orchestrator: no PDFs found in {kb_dir}")
        mock_slack_alert(
            "RAG ingestion pipeline found no PDFs.",
            channel="rag-ingest",
            level="warning",
            kb_dir=str(kb_dir),
        )
        return {"indexed": 0, "failed": 0, "total_chunks": 0}

    logger.info(f"orchestrator: indexing {len(pdf_paths)} PDFs from {kb_dir}")

    indexed = 0
    failed: list[str] = []
    total_chunks = 0
    for pdf_path in pdf_paths:
        try:
            total_chunks += _index_one(pdf_path)
            indexed += 1
        except Exception as e:
            logger.exception(f"orchestrator: failed on {pdf_path.name}: {e}")
            failed.append(pdf_path.name)
            mock_slack_alert(
                f"RAG ingestion failed for {pdf_path.name}.",
                channel="alerts",
                level="error",
                pdf_name=pdf_path.name,
                error_type=type(e).__name__,
            )

    level = "error" if failed else "info"
    mock_slack_alert(
        f"RAG ingestion finished: {indexed}/{len(pdf_paths)} PDFs indexed, "
        f"{total_chunks} chunks total.",
        channel="rag-ingest",
        level=level,
        kb_dir=str(kb_dir),
        indexed=indexed,
        failed=failed,
        total_chunks=total_chunks,
    )
    logger.success(
        f"orchestrator: done. indexed={indexed} failed={len(failed)} "
        f"chunks={total_chunks}"
    )
    return {"indexed": indexed, "failed": len(failed), "total_chunks": total_chunks}


@click.command()
@click.option(
    "--kb-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_KB_DIR,
    show_default=True,
    help="Directory containing the knowledge-base PDFs to index.",
)
def run_rag_pipeline(kb_dir: Path) -> None:
    """Console entry point: index every PDF in ``kb_dir`` into Qdrant."""
    run_pipeline(kb_dir=kb_dir)


if __name__ == "__main__":
    run_rag_pipeline()
