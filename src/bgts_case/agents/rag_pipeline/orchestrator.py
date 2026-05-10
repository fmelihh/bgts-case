"""Orchestrator for the RAG ingestion pipeline.

Composes the pipeline steps in order:

    step 1 — chunk : PDF -> Documents (``step1_chunk.process_pdf``)
    step 2 — index : Documents -> Qdrant hybrid index (``step2_qdrant_index.upsert_chunks``)
"""

from __future__ import annotations

from loguru import logger

from bgts_case.agents.alerts import mock_slack_alert
from bgts_case.agents.rag_pipeline import step1_chunk, step2_qdrant_index


def run_pipeline():
    pdf_path = "/Users/furkanmelih/personal_projects/bgts-case/static/knowledge_base/KB-01_BGP_Troubleshooting.pdf"
    try:
        chunks = step1_chunk.process_pdf(pdf_path=pdf_path, alert_fn=mock_slack_alert)
        logger.info(f"Pipeline produced {len(chunks)} chunks from {pdf_path}")
        step2_qdrant_index.upsert_chunks(chunks=chunks)
        logger.info("Pipeline completed indexing.")
    except Exception as e:
        logger.exception(f"Pipeline failed for {pdf_path}: {e}")
        mock_slack_alert(
            f"RAG ingestion pipeline failed: {e}",
            channel="alerts",
            level="error",
            pdf_path=pdf_path,
            error_type=type(e).__name__,
        )
        raise e


__all__ = ["run_pipeline"]
