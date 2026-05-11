"""End-to-end Ragas + MLflow evaluation for the network ops RAG agent.

Scores ONLY the agent's final natural-language answer; retrieval behaviour
is captured via MLflow Traces (langchain autolog) rather than scored as a
context metric.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from mlflow.langchain import autolog as enable_langchain_autolog
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, ResponseRelevancy

from bgts_case.agent.interfaces import (
    DEFAULT_CHAT_MODEL,
    EVAL_EMBEDDING_MODEL,
    EVAL_MODEL,
    make_chat_model,
    make_embeddings,
)
from bgts_case.agent.main import graph
from bgts_case.secret import secrets


def project_root() -> Path:
    """Anchor on the project-named directory in the current working dir.

    Walks ``Path.cwd()`` upward and stops at the first segment equal to
    ``bgts-case``. Works regardless of where the caller is installed
    (editable, site-packages, etc.) as long as it's invoked from inside
    the repo.
    """
    parts = Path.cwd().resolve().parts
    if "bgts-case" not in parts:
        raise RuntimeError(f"cwd {Path.cwd()} is not inside a 'bgts-case' directory")
    idx = parts.index("bgts-case")
    return Path(*parts[: idx + 1])


DATASET_PATH = project_root() / "static" / "eval_cases.json"
MLFLOW_EXPERIMENT = "network-ops-agent-eval"
MLFLOW_RUN_NAME = "ragas-eval-v1"

# Columns ragas adds for the input data, not metrics. Everything else in
# `to_pandas()` output is a metric score.
_RAGAS_INPUT_COLUMNS = {
    "user_input",
    "response",
    "reference",
    "retrieved_contexts",
    "reference_contexts",
}


def _extract_response(messages: list[Any]) -> str:
    """Return the text of the last AIMessage that did not emit tool calls.

    AIMessage.content may be ``str`` or ``list[dict]`` (block format). For
    blocks we concatenate every ``type == "text"`` segment in order. Anything
    else (thinking, tool_use blocks) is ignored.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = msg.content
        if isinstance(content, str):
            return content
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)
    raise RuntimeError("no final AIMessage without tool_calls found in graph output")


async def _run_agent(user_input: str) -> str:
    state = await graph.ainvoke({"messages": [HumanMessage(content=user_input)]})
    return _extract_response(state["messages"])


async def _collect_responses(
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Phase 1 — run the agent against every case.

    Kept fully async and isolated from Ragas. Sequential to stay under
    Fireworks rate limits — ``asyncio.gather`` would burst the API.
    """
    raw_outputs: list[dict[str, Any]] = []
    eval_rows: list[dict[str, str]] = []
    for case in cases:
        case_id = case["case_id"]
        user_input = case["user_input"]
        print(f"[run_eval] {case_id}: invoking agent")
        response = await _run_agent(user_input)
        raw_outputs.append(
            {
                "case_id": case_id,
                "user_input": user_input,
                "response": response,
            }
        )
        eval_rows.append(
            {
                "user_input": user_input,
                "response": response,
                "reference": case["reference"],
            }
        )
    return raw_outputs, eval_rows


def _slice_means(
    df: pd.DataFrame, slice_col: str, metric_cols: list[str]
) -> pd.DataFrame:
    groups = df.groupby(slice_col, dropna=False)
    return groups[metric_cols].mean().assign(count=groups.size()).reset_index()


def _sanitize_metric_name(col: str) -> str:
    """MLflow metric names reject ``(``, ``)``, ``=`` and whitespace.

    Ragas emits column names like ``factual_correctness(mode=f1)`` —
    flatten them to ``factual_correctness_mode_f1``.
    """
    return col.replace("(", "_").replace(")", "").replace("=", "_").replace(" ", "")


def run(limit: int = 5) -> None:
    """Sync entry point for the ``run-eval`` console script.

    Two phases on purpose:
      1. ``asyncio.run`` wraps ONLY the agent invocations. Inside this
         loop we are talking to the LangGraph agent + Fireworks via
         LangChain's async client.
      2. Ragas ``evaluate()`` is called AFTER the async loop has fully
         closed. Ragas spawns its own asyncio executor internally; if
         it lands inside an already-running loop (compounded by
         ``nest_asyncio`` that LangGraph pulls in transitively) the
         httpx clients in the workers fail with ``APIConnectionError``
         on every job. Keeping the two phases sequential and at the
         sync layer avoids the nesting entirely.
    """
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"][:limit]

    mlflow.set_tracking_uri(secrets.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # Autolog BEFORE invoking the graph so each ainvoke produces a trace
    # in the MLflow UI — used for debugging cases that score poorly.
    enable_langchain_autolog()

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run_ctx:
        mlflow.log_param("dataset_name", payload["name"])
        mlflow.log_param("schema_version", payload["ragas_schema_version"])
        mlflow.log_param("total_cases", len(cases))
        mlflow.log_param("primary_model", DEFAULT_CHAT_MODEL)
        mlflow.log_param("eval_model", EVAL_MODEL)
        mlflow.log_param("eval_embedding_model", EVAL_EMBEDDING_MODEL)

        # Phase 1 — agent invocations. Async only.
        raw_outputs, eval_rows = asyncio.run(_collect_responses(cases))
        mlflow.log_dict({"outputs": raw_outputs}, "agent_outputs.json")

        # Phase 2 — Ragas. Pure sync entry point; no surrounding loop.
        eval_dataset = EvaluationDataset.from_list(eval_rows)
        evaluator_llm = LangchainLLMWrapper(make_chat_model(model=EVAL_MODEL))
        evaluator_embeddings = LangchainEmbeddingsWrapper(make_embeddings())
        metrics = [ResponseRelevancy(), FactualCorrectness()]

        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        # evaluate() only returns Executor when return_executor=True, which
        # we do not pass. Narrow for the type checker.
        assert isinstance(result, EvaluationResult)
        print(result)

        per_case_df = result.to_pandas()

        # Discover metric column names from the DataFrame itself. Reading
        # them off ``metric.name`` is wrong because Ragas appends the
        # metric's config to the column (e.g. ``factual_correctness``
        # becomes ``factual_correctness(mode=f1)``) — that mismatch is
        # what produced the earlier ``KeyError: 'factual_correctness'``.
        metric_cols = [c for c in per_case_df.columns if c not in _RAGAS_INPUT_COLUMNS]

        # Merge metadata + case_id back in by row order — ragas preserves
        # the input order in EvaluationDataset.from_list.
        per_case_df.insert(0, "case_id", pd.Series([c["case_id"] for c in cases]))
        per_case_df["difficulty"] = pd.Series(
            [c["metadata"]["difficulty"] for c in cases]
        )
        per_case_df["question_type"] = pd.Series(
            [c["metadata"]["question_type"] for c in cases]
        )
        per_case_df["source_kbs"] = pd.Series(
            [",".join(c["metadata"]["source_kbs"]) for c in cases]
        )
        per_case_df["related_tickets"] = pd.Series(
            [",".join(c["metadata"]["related_tickets"]) for c in cases]
        )

        aggregate_means: dict[str, float] = {}
        for col in metric_cols:
            mean = float(per_case_df[col].mean())
            aggregate_means[_sanitize_metric_name(col)] = mean
            mlflow.log_metric(_sanitize_metric_name(col), mean)

        mlflow.log_table(per_case_df, "ragas_per_case.json")

        difficulty_slice = _slice_means(per_case_df, "difficulty", metric_cols)
        qtype_slice = _slice_means(per_case_df, "question_type", metric_cols)
        mlflow.log_table(difficulty_slice, "ragas_by_difficulty.json")
        mlflow.log_table(qtype_slice, "ragas_by_question_type.json")

        # Flat per-slice metrics so they appear as columns in the
        # Compare Runs parallel-coordinates view.
        for _, row in difficulty_slice.iterrows():
            bucket = row["difficulty"]
            for col in metric_cols:
                mlflow.log_metric(
                    f"{_sanitize_metric_name(col)}__difficulty_{bucket}",
                    float(row[col]),
                )
        for _, row in qtype_slice.iterrows():
            bucket = row["question_type"]
            for col in metric_cols:
                mlflow.log_metric(
                    f"{_sanitize_metric_name(col)}__question_type_{bucket}",
                    float(row[col]),
                )

        print(f"mlflow run_id: {run_ctx.info.run_id}")
