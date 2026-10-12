import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from math import log, sqrt
from typing import Any, Dict, List, Optional

from openai import OpenAI
try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
except Exception:
    pynvml = None

MODEL_NAME = "mistral"


def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Split long text into overlapping chunks for better retrieval quality."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def _tokenize_terms(text: str) -> List[str]:
    """Tokenize mixed-language terms for term-frequency based chunking."""
    lowered = text.lower()
    word_tokens = re.findall(r"[a-z0-9_]{2,}", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = [
        cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
    ]
    return word_tokens + cjk_bigrams


def _chunk_text_by_term_frequency(
    text: str, min_chunk_chars: int = 600, max_chunk_chars: int = 1500
) -> List[str]:
    """
    Split text into chunks using sentence boundaries and term-frequency cues.
    """
    if not text:
        return []
    if len(text) <= max_chunk_chars:
        return [text]

    units = [
        p.strip()
        for p in re.split(r"(?<=[。！？!?\.])\s+|\n+", text)
        if p and p.strip()
    ]
    if not units:
        return _chunk_text(text, chunk_size=max_chunk_chars, overlap=200)

    tf = Counter(_tokenize_terms(text))
    top_terms = {t for t, _ in tf.most_common(120)}

    unit_scores: List[int] = []
    for unit in units:
        terms = _tokenize_terms(unit)
        score = sum(tf[t] for t in terms if t in top_terms)
        unit_scores.append(score)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    current_score = 0

    for unit, score in zip(units, unit_scores):
        unit_len = len(unit)
        if current and current_len + unit_len > max_chunk_chars:
            chunks.append("\n".join(current).strip())
            current = []
            current_len = 0
            current_score = 0

        current.append(unit)
        current_len += unit_len + 1
        current_score += score

        avg_score = (current_score / max(1, len(current)))
        if current_len >= min_chunk_chars and score < avg_score * 0.35:
            chunks.append("\n".join(current).strip())
            current = []
            current_len = 0
            current_score = 0

    if current:
        chunks.append("\n".join(current).strip())

    return [c for c in chunks if c]


# Default corpus: non-legal text for RAG evaluation examples.
DEFAULT_DOCUMENTS = [
    "RAG combines retrieval and generation: retrieve relevant context, then generate an answer grounded in that context.",
    "A practical RAG pipeline has four steps: indexing, retrieval, prompt assembly, and response generation.",
    "GraphRAG uses graph structures to connect entities and relationships, which can improve multi-hop reasoning.",
    "Evaluation for RAG commonly checks groundedness, relevance, correctness, and completeness.",
    "Chunking strategy affects retrieval quality. Smaller chunks increase precision, while larger chunks preserve context.",
    "Overlapping chunks help avoid cutting key facts at boundaries.",
    "Keyword retrieval is simple and fast but can miss semantic matches. Embedding retrieval is usually more robust.",
    "Logs and traces are useful for debugging failed answers in RAG systems.",
]


def _load_documents() -> List[str]:
    """
    Load documents for RAG.
    By default uses non-legal built-in docs.
    Set RAG_EVAL_USE_SAMPLE_TXT=1 to load ../input/sample.txt instead.
    """
    use_sample = os.getenv("RAG_EVAL_USE_SAMPLE_TXT", "0") == "1"
    disable_chunking = os.getenv("RAG_EVAL_DISABLE_CHUNKING", "0") == "1"
    if not use_sample:
        if disable_chunking:
            return ["\n".join(DEFAULT_DOCUMENTS)]
        return DEFAULT_DOCUMENTS

    try:
        sample_path = os.path.join(
            os.path.dirname(__file__), "..", "input", "sample.txt"
        )
        sample_path = os.path.normpath(sample_path)
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                content = f.read()
            if disable_chunking:
                return [content]

            # Prefer TF-guided chunking; fallback to fixed-size overlap chunking.
            chunks = _chunk_text_by_term_frequency(content)
            if len(chunks) < 2:
                chunks = _chunk_text(content)

            # Parallel normalize chunks during preload.
            with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 1))) as ex:
                chunks = list(ex.map(lambda c: c.strip(), chunks))
            chunks = [c for c in chunks if c]
            if chunks:
                return chunks
    except Exception:
        pass
    return DEFAULT_DOCUMENTS


DOCUMENTS = _load_documents()


@dataclass
class TraceEvent:
    """Single event in the RAG application trace"""

    event_type: str
    component: str
    data: Dict[str, Any]


class BaseRetriever:
    """
    Base class for retrievers.
    Subclasses should implement the fit and get_top_k methods.
    """

    def __init__(self):
        self.documents = []

    def fit(self, documents: List[str]):
        """Store the documents"""
        self.documents = documents

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k most relevant documents for the query."""
        raise NotImplementedError("Subclasses should implement this method.")


class SimpleKeywordRetriever(BaseRetriever):
    """Ultra-simple keyword matching retriever"""

    def __init__(self):
        super().__init__()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize mixed-language text for simple lexical matching."""
        lowered = text.lower()
        word_tokens = re.findall(r"[a-z0-9_]+", lowered)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        cjk_bigrams = [
            cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
        ]
        return word_tokens + cjk_chars + cjk_bigrams

    def _count_keyword_matches(self, query: str, document: str) -> int:
        """Count how many query tokens appear in the document."""
        query_tokens = set(self._tokenize(query))
        document_tokens = set(self._tokenize(document))
        if not query_tokens or not document_tokens:
            return 0
        return len(query_tokens.intersection(document_tokens))

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Get top k documents by keyword match count"""
        scores = []

        for i, doc in enumerate(self.documents):
            match_count = self._count_keyword_matches(query, doc)
            scores.append((i, match_count))

        # Sort by match count (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]


class SimpleVectorRetriever(BaseRetriever):
    """Simple TF-IDF vector-space retriever with cosine similarity."""

    def __init__(self):
        super().__init__()
        self.doc_vectors: List[Dict[str, float]] = []
        self.doc_norms: List[float] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        lowered = text.lower()
        word_tokens = re.findall(r"[a-z0-9_]+", lowered)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        cjk_bigrams = [
            cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
        ]
        return word_tokens + cjk_chars + cjk_bigrams

    def fit(self, documents: List[str]):
        super().fit(documents)
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]

        df = Counter()
        for tokens in tokenized_docs:
            for token in set(tokens):
                df[token] += 1

        total_docs = max(1, len(tokenized_docs))
        self.idf = {
            token: log(total_docs / (doc_freq + 1)) + 1.0
            for token, doc_freq in df.items()
        }

        self.doc_vectors = []
        self.doc_norms = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            vec: Dict[str, float] = {}
            for token, freq in tf.items():
                vec[token] = float(freq) * self.idf.get(token, 1.0)
            norm = sqrt(sum(v * v for v in vec.values()))
            self.doc_vectors.append(vec)
            self.doc_norms.append(norm)

    def _build_query_vector(self, query: str) -> Dict[str, float]:
        tf = Counter(self._tokenize(query))
        vec: Dict[str, float] = {}
        for token, freq in tf.items():
            if token in self.idf:
                vec[token] = float(freq) * self.idf[token]
        return vec

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        qvec = self._build_query_vector(query)
        qnorm = sqrt(sum(v * v for v in qvec.values()))
        if qnorm == 0:
            return []

        scores = []
        for i, dvec in enumerate(self.doc_vectors):
            dnorm = self.doc_norms[i]
            if dnorm == 0:
                scores.append((i, 0.0))
                continue

            common = set(qvec.keys()).intersection(dvec.keys())
            dot = sum(qvec[token] * dvec[token] for token in common)
            cosine = dot / (qnorm * dnorm)
            scores.append((i, cosine))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class ExampleRAG:


    def __init__(
        self,
        llm_client,
        retriever: Optional[BaseRetriever] = None,
        system_prompt: Optional[str] = None,
        logdir: str = "logs",
    ):

        self.llm_client = llm_client
        self.retriever = retriever or SimpleKeywordRetriever()
        self.system_prompt = (
            system_prompt
            or """Answer the following question based on the provided documents:
                                Question: {query}
                                Documents:
                                {context}
                                Answer:
                            """
        )
        self.documents = []
        self.is_fitted = False
        self.traces = []
        self.logdir = logdir
        self._last_llm_metrics: Dict[str, Any] = {}

        # Create log directory if it doesn't exist
        os.makedirs(self.logdir, exist_ok=True)

        # Initialize tracing
        self.traces.append(
            TraceEvent(
                event_type="init",
                component="rag_system",
                data={
                    "retriever_type": type(self.retriever).__name__,
                    "system_prompt_length": len(self.system_prompt),
                    "logdir": self.logdir,
                },
            )
        )

    def _resource_snapshot(self) -> Dict[str, Any]:
        """Collect lightweight process/system resource usage snapshot."""
        snapshot: Dict[str, Any] = {"timestamp": datetime.now().isoformat()}

        if psutil is not None:
            try:
                process = psutil.Process(os.getpid())
                cpu_times = process.cpu_times()
                vm = psutil.virtual_memory()
                snapshot.update(
                    {
                        "proc_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
                        "proc_vms_mb": round(process.memory_info().vms / (1024 * 1024), 2),
                        "proc_cpu_time_s": round(cpu_times.user + cpu_times.system, 4),
                        "system_cpu_percent": psutil.cpu_percent(interval=None),
                        "system_mem_percent": vm.percent,
                    }
                )
            except Exception:
                pass

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                snapshot.update(
                    {
                        "gpu_util_percent": util.gpu,
                        "gpu_mem_percent": round((mem.used / mem.total) * 100, 2),
                        "gpu_mem_used_mb": round(mem.used / (1024 * 1024), 2),
                    }
                )
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return snapshot

    def _resource_delta(
        self, start_snapshot: Dict[str, Any], end_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute numeric deltas between two resource snapshots."""
        deltas: Dict[str, Any] = {}
        for key in end_snapshot:
            if key in start_snapshot and isinstance(end_snapshot[key], (int, float)) and isinstance(start_snapshot[key], (int, float)):
                deltas[f"{key}_delta"] = round(end_snapshot[key] - start_snapshot[key], 4)
        return deltas

    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base"""
        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "add_documents",
                    "num_new_documents": len(documents),
                    "total_documents_before": len(self.documents),
                    "document_lengths": [len(doc) for doc in documents],
                },
            )
        )

        self.documents.extend(documents)
        # Refit retriever with all documents
        self.retriever.fit(self.documents)
        self.is_fitted = True

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def add_documents_parallel(self, documents: List[str], max_workers: Optional[int] = None):
        """Add documents with parallel preprocessing during preload."""
        workers = max_workers or min(8, max(1, os.cpu_count() or 1))

        def _normalize(doc: str) -> str:
            return re.sub(r"\s+", " ", doc).strip()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            normalized = list(ex.map(_normalize, documents))
        normalized = [doc for doc in normalized if doc]
        self.add_documents(normalized)

    def set_documents(self, documents: List[str]):
        """Set documents (replacing any existing ones)"""
        old_doc_count = len(self.documents)

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "set_documents",
                    "num_new_documents": len(documents),
                    "old_document_count": old_doc_count,
                    "document_lengths": [len(doc) for doc in documents],
                },
            )
        )

        self.documents = documents
        self.retriever.fit(self.documents)
        self.is_fitted = True

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for the query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries containing document info
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_start",
                    "query": query,
                    "query_length": len(query),
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        top_docs = self.retriever.get_top_k(query, k=top_k)

        retrieved_docs = []
        for idx, score in top_docs:
            if score > 0:  # Only include documents with positive similarity scores
                retrieved_docs.append(
                    {
                        "content": self.documents[idx],
                        "similarity_score": score,
                        "document_id": idx,
                    }
                )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_complete",
                    "num_retrieved": len(retrieved_docs),
                    "scores": [doc["similarity_score"] for doc in retrieved_docs],
                    "document_ids": [doc["document_id"] for doc in retrieved_docs],
                },
            )
        )

        return retrieved_docs

    def generate_response(self, query: str, top_k: int = 3) -> str:
        """
        Generate response to query using retrieved documents

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Generated response
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")

        context = "\n\n".join(context_parts)
        if not context:
            context = "No relevant documents were retrieved."

        # Generate response using LLM client
        prompt = self.system_prompt.format(query=query, context=context)

        self.traces.append(
            TraceEvent(
                event_type="llm_call",
                component="openai_api",
                data={
                    "operation": "generate_response",
                    "model": MODEL_NAME,
                    "query": query,
                    "prompt_length": len(prompt),
                    "context_length": len(context),
                    "num_context_docs": len(retrieved_docs),
                },
            )
        )

        llm_start = time.perf_counter()
        llm_start_resource = self._resource_snapshot()
        try:
            response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            response_text = response.choices[0].message.content.strip()
            llm_end_resource = self._resource_snapshot()
            llm_latency_ms = round((time.perf_counter() - llm_start) * 1000, 2)
            llm_resource_delta = self._resource_delta(llm_start_resource, llm_end_resource)
            self._last_llm_metrics = {
                "llm_latency_ms": llm_latency_ms,
                "llm_resource_start": llm_start_resource,
                "llm_resource_end": llm_end_resource,
                "llm_resource_delta": llm_resource_delta,
            }

            self.traces.append(
                TraceEvent(
                    event_type="llm_response",
                    component="openai_api",
                    data={
                        "operation": "generate_response",
                        "response_length": len(response_text),
                        "usage": (
                            response.usage.model_dump() if response.usage else None
                        ),
                        "model": MODEL_NAME,
                        "llm_latency_ms": llm_latency_ms,
                        "llm_resource_start": llm_start_resource,
                        "llm_resource_end": llm_end_resource,
                        "llm_resource_delta": llm_resource_delta,
                    },
                )
            )

            return response_text

        except Exception as e:
            llm_end_resource = self._resource_snapshot()
            llm_latency_ms = round((time.perf_counter() - llm_start) * 1000, 2)
            llm_resource_delta = self._resource_delta(llm_start_resource, llm_end_resource)
            self._last_llm_metrics = {
                "llm_latency_ms": llm_latency_ms,
                "llm_resource_start": llm_start_resource,
                "llm_resource_end": llm_end_resource,
                "llm_resource_delta": llm_resource_delta,
            }
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="openai_api",
                    data={
                        "operation": "generate_response",
                        "error": str(e),
                        "llm_latency_ms": llm_latency_ms,
                        "llm_resource_start": llm_start_resource,
                        "llm_resource_end": llm_end_resource,
                        "llm_resource_delta": llm_resource_delta,
                    },
                )
            )
            return f"Error generating response: {str(e)}"

    def query(
        self, question: str, top_k: int = 3, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response

        Args:
            question: User question
            top_k: Number of documents to retrieve
            run_id: Optional run ID for tracing (auto-generated if not provided)

        Returns:
            Dictionary containing response and retrieved documents
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"
        query_start = time.perf_counter()
        query_start_resource = self._resource_snapshot()

        # Reset traces for this query
        self.traces = []

        self.traces.append(
            TraceEvent(
                event_type="query_start",
                component="rag_system",
                data={
                    "run_id": run_id,
                    "question": question,
                    "question_length": len(question),
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        try:
            retrieved_docs = self.retrieve_documents(question, top_k)
            response = self.generate_response(question, top_k)
            query_end_resource = self._resource_snapshot()
            query_latency_ms = round((time.perf_counter() - query_start) * 1000, 2)
            query_resource_delta = self._resource_delta(
                query_start_resource, query_end_resource
            )
            query_metrics = {
                "query_latency_ms": query_latency_ms,
                "query_resource_start": query_start_resource,
                "query_resource_end": query_end_resource,
                "query_resource_delta": query_resource_delta,
                "llm_metrics": self._last_llm_metrics,
            }

            result = {"answer": response, "run_id": run_id, "metrics": query_metrics}

            self.traces.append(
                TraceEvent(
                    event_type="query_complete",
                    component="rag_system",
                    data={
                        "run_id": run_id,
                        "success": True,
                        "response_length": len(response),
                        "num_retrieved": len(retrieved_docs),
                        "query_latency_ms": query_latency_ms,
                        "query_resource_start": query_start_resource,
                        "query_resource_end": query_end_resource,
                        "query_resource_delta": query_resource_delta,
                        "llm_metrics": self._last_llm_metrics,
                    },
                )
            )

            logs_path = self.export_traces_to_log(run_id, question, result)
            return {
                "answer": response,
                "run_id": run_id,
                "logs": logs_path,
                "metrics": query_metrics,
                "retrieved_docs": retrieved_docs,
            }

        except Exception as e:
            query_end_resource = self._resource_snapshot()
            query_latency_ms = round((time.perf_counter() - query_start) * 1000, 2)
            query_resource_delta = self._resource_delta(
                query_start_resource, query_end_resource
            )
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="rag_system",
                    data={
                        "run_id": run_id,
                        "operation": "query",
                        "error": str(e),
                        "query_latency_ms": query_latency_ms,
                        "query_resource_start": query_start_resource,
                        "query_resource_end": query_end_resource,
                        "query_resource_delta": query_resource_delta,
                    },
                )
            )

            # Return error result
            logs_path = self.export_traces_to_log(run_id, question, None)
            return {
                "answer": f"Error processing query: {str(e)}",
                "run_id": run_id,
                "logs": logs_path,
                "retrieved_docs": [],
            }

    def export_traces_to_log(
        self,
        run_id: str,
        query: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ):
        """Export traces to a log file with run_id"""
        timestamp = datetime.now().isoformat()
        log_filename = (
            f"rag_run_{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        )
        log_filepath = os.path.join(self.logdir, log_filename)

        log_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "query": query,
            "result": result,
            "num_documents": len(self.documents),
            "traces": [asdict(trace) for trace in self.traces],
        }

        with open(log_filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"RAG traces exported to: {log_filepath}")
        return log_filepath


def default_rag_client(llm_client, logdir: str = "logs") -> ExampleRAG:
    """
    Create a default RAG client with OpenAI LLM and optional retriever.

    Args:
        retriever: Optional retriever instance (defaults to SimpleKeywordRetriever)
        logdir: Directory for trace logs
    Returns:
        ExampleRAG instance
    """
    retriever_name = os.getenv("RAG_RETRIEVER", "vector").strip().lower()
    if retriever_name == "keyword":
        retriever = SimpleKeywordRetriever()
    else:
        retriever = SimpleVectorRetriever()
    client = ExampleRAG(llm_client=llm_client, retriever=retriever, logdir=logdir)
    client.add_documents_parallel(DOCUMENTS)  # Add default documents using parallel preload
    return client


if __name__ == "__main__":
    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_openai_api_key'")
        exit(1)

    # Initialize RAG system with tracing enabled
    llm = OpenAI(api_key=api_key)
    r = SimpleKeywordRetriever()
    rag_client = ExampleRAG(llm_client=llm, retriever=r, logdir="logs")

    # Add documents (this will be traced)
    rag_client.add_documents(DOCUMENTS)

    # Run query with tracing
    query = "What is Ragas"
    print(f"Query: {query}")
    response = rag_client.query(query, top_k=3)

    print("Response:", response["answer"])
    print(f"Run ID: {response['logs']}")
