import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI

# Add the current directory to the path so we can import rag module when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from rag import default_rag_client

BASE_DIR = Path(__file__).parent
EVALS_DIR = BASE_DIR / "evals"
LOG_DIR = str((EVALS_DIR / "logs").resolve())

MODE = os.getenv("RAG_EVAL_MODE", "rag").strip().lower()  # "rag" | "no_rag"

MODEL_NAME = os.getenv("RAG_EVAL_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("RAG_EVAL_BASE_URL", "http://localhost:11434/v1")

# Noise Sensitivity config: add N random irrelevant docs and re-generate once per question.
NOISE_DOCS = int(os.getenv("RAG_EVAL_NOISE_DOCS", "2"))
NOISE_ENABLE = os.getenv("RAG_EVAL_NOISE_ENABLE", "1") == "1"

client = OpenAI(
    api_key="ollama",  # Ollama doesn't require a real key
    base_url=OLLAMA_BASE_URL,
)
rag_client = default_rag_client(llm_client=client, logdir=LOG_DIR) if MODE == "rag" else None


def load_dataset() -> List[Dict[str, str]]:
    dataset_file = Path(__file__).resolve().parents[2] / "evals" / "datasets" / "test_dataset_no_rag.csv"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    rows: List[Dict[str, str]] = []
    with dataset_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            question, grading_notes = line.split(",", 1)
            rows.append(
                {
                    "question": question.strip(),
                    "grading_notes": grading_notes.strip(),
                }
            )
    return rows


def _chat_text(prompt: str, system: str = "You are a strict evaluator.") -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def grade_correctness(response_text: str, grading_notes: str) -> str:
    prompt = (
        "Check if the response contains points mentioned in grading notes and return only 'pass' or 'fail'.\n"
        f"Response: {response_text}\n"
        f"Grading Notes: {grading_notes}"
    )
    out = _chat_text(prompt).lower()
    if "pass" in out:
        return "pass"
    if "fail" in out:
        return "fail"
    # Fall back to a conservative decision if the model doesn't comply.
    return "fail"


def grade_relevance(question: str, context: str) -> str:
    prompt = (
        "Based on the given question and retrieved documents, is the retrieved information relevant to the question? "
        "Return only 'yes' or 'no'.\n"
        f"Question: {question}\n"
        f"Documents: {context}\n"
        "Is the information relevant (yes/no)?"
    )
    out = _chat_text(prompt).lower()
    if out.startswith("y"):
        return "yes"
    if out.startswith("n"):
        return "no"
    return "no"


def _parse_grading_items(grading_notes: str) -> List[str]:
    # Expected format: "- item1 - item2 - item3"
    raw = [p.strip() for p in grading_notes.split("-") if p.strip()]
    seen = set()
    items: List[str] = []
    for item in raw:
        lowered = item.lower()
        if lowered not in seen:
            seen.add(lowered)
            items.append(item)
    return items


def context_recall(grading_notes: str, contexts: Iterable[str]) -> Tuple[float, List[str]]:
    """
    Context Recall (0..1):
    fraction of grading-note items that appear in retrieved contexts.
    Returns (score, missing_items).
    """
    items = _parse_grading_items(grading_notes)
    if not items:
        return 1.0, []

    haystacks = [(c or "").lower() for c in contexts]
    missing: List[str] = []
    hit = 0
    for item in items:
        needle = item.lower()
        if any(needle in h for h in haystacks):
            hit += 1
        else:
            missing.append(item)
    return hit / len(items), missing


def _build_context_for_generation(doc_texts: List[str]) -> str:
    parts = [f"Document {i+1}:\n{doc}" for i, doc in enumerate(doc_texts)]
    context = "\n\n".join(parts).strip()
    return context or "No relevant documents were retrieved."


def generate_with_context(question: str, doc_texts: List[str]) -> str:
    # Match the ExampleRAG prompting pattern for a fair "noise" comparison.
    context = _build_context_for_generation(doc_texts)
    prompt = rag_client.system_prompt.format(query=question, context=context)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": rag_client.system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def noise_sensitivity(
    question: str,
    grading_notes: str,
    retrieved_doc_texts: List[str],
    clean_response_text: str,
) -> Tuple[str, str]:
    """
    Noise Sensitivity (yes/no):
    add irrelevant documents, re-generate once, and check whether correctness degrades.

    Returns (noise_sensitivity_score, noisy_correctness_score).
    """
    clean_correctness = grade_correctness(clean_response_text, grading_notes)

    if not NOISE_ENABLE or NOISE_DOCS <= 0 or not getattr(rag_client, "documents", None):
        return "n/a", ""

    retrieved_set = {t.strip() for t in retrieved_doc_texts if t and t.strip()}
    candidates = [d for d in rag_client.documents if d.strip() and d.strip() not in retrieved_set]
    rng = random.Random(hash(question) & 0xFFFFFFFF)
    noise = rng.sample(candidates, k=min(NOISE_DOCS, len(candidates))) if candidates else []

    noisy_answer = generate_with_context(question, retrieved_doc_texts + noise)
    noisy_correctness = grade_correctness(noisy_answer, grading_notes)

    degraded = (clean_correctness == "pass") and (noisy_correctness == "fail")
    return ("no" if degraded else "yes"), noisy_correctness


def run():
    (EVALS_DIR / "experiments").mkdir(parents=True, exist_ok=True)
    (EVALS_DIR / "logs").mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    started = time.perf_counter()

    rows: List[Dict[str, Any]] = []
    for row in dataset:
        q = row["question"]
        grading = row["grading_notes"]

        row_start = time.perf_counter()
        if MODE == "rag":
            response = rag_client.query(q)
            retrieved_docs = response.get("retrieved_docs", []) or []
            retrieved_texts = [d.get("content", "") for d in retrieved_docs]
            retrieved_docs_content = "\n\n".join(retrieved_texts)
            answer_text = response.get("answer", "")
            log_file = response.get("logs", "")
            metrics = response.get("metrics", {}) or {}
        else:
            # No-RAG baseline: answer directly, no retrieval.
            response = {}
            retrieved_texts = []
            retrieved_docs_content = ""
            answer_text = _chat_text(q, system="Answer briefly and accurately.")
            log_file = ""
            metrics = {}
        rag_elapsed_ms = round((time.perf_counter() - row_start) * 1000, 2)

        score_start = time.perf_counter()
        correctness = grade_correctness(answer_text, grading)
        score_elapsed_ms = round((time.perf_counter() - score_start) * 1000, 2)

        relevance_start = time.perf_counter()
        relevance = grade_relevance(q, retrieved_docs_content) if MODE == "rag" else "n/a"
        relevance_elapsed_ms = round((time.perf_counter() - relevance_start) * 1000, 2)

        recall_start = time.perf_counter()
        if MODE == "rag":
            recall_score, missing_items = context_recall(grading, retrieved_texts)
        else:
            missing_items = _parse_grading_items(grading)
            recall_score = 0.0 if missing_items else 1.0
        recall_elapsed_ms = round((time.perf_counter() - recall_start) * 1000, 2)

        noise_start = time.perf_counter()
        if MODE == "rag":
            noise_score, noisy_correctness = noise_sensitivity(q, grading, retrieved_texts, answer_text)
        else:
            noise_score, noisy_correctness = "n/a", ""
        noise_elapsed_ms = round((time.perf_counter() - noise_start) * 1000, 2)

        query_delta = metrics.get("query_resource_delta", {}) or {}
        llm_metrics = metrics.get("llm_metrics", {}) or {}
        llm_delta = llm_metrics.get("llm_resource_delta", {}) or {}

        rows.append(
            {
                "question": q,
                "grading_notes": grading,
                "response": answer_text,
                "score": correctness,
                "relevance_score": relevance,
                "context_recall_score": round(recall_score, 4),
                "context_recall_missing": " | ".join(missing_items),
                "noise_sensitivity_score": noise_score,
                "noise_correctness_score": noisy_correctness,
                "log_file": log_file,
                "rag_elapsed_ms": rag_elapsed_ms,
                "score_elapsed_ms": score_elapsed_ms,
                "relevance_elapsed_ms": relevance_elapsed_ms,
                "context_recall_elapsed_ms": recall_elapsed_ms,
                "noise_sensitivity_elapsed_ms": noise_elapsed_ms,
                "query_latency_ms": metrics.get("query_latency_ms", ""),
                "llm_latency_ms": llm_metrics.get("llm_latency_ms", ""),
                "proc_rss_mb_delta": query_delta.get("proc_rss_mb_delta", ""),
                "proc_cpu_time_s_delta": query_delta.get("proc_cpu_time_s_delta", ""),
                "system_cpu_percent_delta": query_delta.get("system_cpu_percent_delta", ""),
                "gpu_util_percent_delta": query_delta.get("gpu_util_percent_delta", ""),
                "llm_proc_rss_mb_delta": llm_delta.get("proc_rss_mb_delta", ""),
            }
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    print(f"Evaluation completed in {elapsed_ms} ms")

    def _sanitize_filename_component(value: str) -> str:
        out: List[str] = []
        for ch in (value or ""):
            if ch.isalnum() or ch in {".", "-", "_"}:
                out.append(ch)
            else:
                out.append("_")
        sanitized = "".join(out).strip("._-")
        return sanitized or "unknown"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    disable_chunking = os.getenv("RAG_EVAL_DISABLE_CHUNKING", "0") == "1"
    if MODE == "no_rag":
        mode_tag = "no_rag"
    else:
        mode_tag = "rag_nochunk" if disable_chunking else "rag_chunked"
    model_tag = _sanitize_filename_component(MODEL_NAME)
    out_csv = EVALS_DIR / "experiments" / f"eval_results_{mode_tag}_{model_tag}_{stamp}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to: {out_csv.resolve()}")


if __name__ == "__main__":
    run()
