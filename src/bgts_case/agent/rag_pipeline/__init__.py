"""RAG ingestion pipeline package.

Public entry points:

- :func:`run_pipeline` — orchestrated full pipeline (chunk + index).
- :func:`process_pdf` — step 1 (chunking) only; exported for tests/notebooks.
"""

from bgts_case.agent.rag_pipeline.orchestrator import run_pipeline
from bgts_case.agent.rag_pipeline.step1_chunk import process_pdf

__all__ = ["run_pipeline", "process_pdf"]
