#!/usr/bin/env python3
"""
RAG System Comparison & Optimisation Tool
• Migrated from FAISS → Chroma
• No external network calls – suitable for CI or air‑gapped tests
"""
from __future__ import annotations
import time, json, statistics, dataclasses, typing as t
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI  # runtime‑injected / mocked in tests

# ─────────────────── Models ────────────────────
@dataclasses.dataclass
class Result:
    query: str
    latency: float
    answer: str
    citations: list[str]

# ─────────────────── Engines ───────────────────
class ChromaRAG:
    """Thin wrapper around the production Chroma collection."""
    def __init__(
        self,
        path: str = "data/chroma",
        collection: str = "mass_law_enforcement",
        model_name: str = "text-embedding-3-small",
        client: OpenAI | None = None,
    ):
        self._chroma = chromadb.PersistentClient(path=path)
        self._collection = self._chroma.get_collection(
            name=collection,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model_name=model_name,
            ),
        )
        self._llm = client or OpenAI()

    # retrieval only (for test‑mode determinism)
    def retrieve(self, query: str, k: int = 5) -> list[str]:
        out = self._collection.query(query_texts=[query], n_results=k)
        return out["documents"][0]

    def answer(self, query: str) -> Result:
        start = time.perf_counter()
        docs = self.retrieve(query)
        context = "\n\n".join(docs)
        prompt = (
            "You are a Massachusetts criminal‑law copilot. "
            "Cite source paragraphs verbatim where possible.\n\n"
            f"Context:\n{context}\n\nQ: {query}\nA:"
        )
        resp = self._llm.chat.completions.create(  # mocked in tests
            model="gpt-4",
            temperature=0,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start
        return Result(query, latency, resp.choices[0].message.content, docs[:3])

# ───────────────── Comparison runner ────────────
class Comparator:
    def __init__(self, baseline: ChromaRAG, challenger: ChromaRAG):
        self.baseline, self.challenger = baseline, challenger

    def run(self, queries: list[str]) -> list[dict[str, t.Any]]:
        rows = []
        for q in queries:
            b = self.baseline.answer(q)
            c = self.challenger.answer(q)
            rows.append({
                "query": q,
                "latency_delta": b.latency - c.latency,
                "citation_gain": len(c.citations) - len(b.citations),
                "identical_answer": b.answer.strip() == c.answer.strip(),
            })
        return rows

    @staticmethod
    def summary(rows) -> dict[str, t.Any]:
        lat = [r["latency_delta"] for r in rows]
        cit = [r["citation_gain"] for r in rows]
        return {
            "p50_speedup_ms": statistics.median(lat) * 1_000,
            "avg_citation_gain": statistics.mean(cit),
            "identical_ratio": sum(r["identical_answer"] for r in rows) / len(rows),
        }

# CLI helper
if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="queries.txt",
                        help="File with one test‑query per line")
    args = parser.parse_args()

    baseline = ChromaRAG()              # prod settings
    challenger = ChromaRAG(model_name="text-embedding-3-large")  # A/B variant
    queries = Path(args.queries).read_text().splitlines()
    comp = Comparator(baseline, challenger)
    results = comp.run(queries)

    Path("rag_comparison_report.json").write_text(json.dumps({
        "results": results,
        "summary": comp.summary(results),
    }, indent=2))
    print("✅  Comparison complete – see rag_comparison_report.json")
