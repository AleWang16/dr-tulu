"""
Barebones paper-retrieval pipeline built from central dr_agent components.

This module intentionally avoids external APIs and MCP servers. It demonstrates:
- `BaseTool` + `DocumentToolOutput` for tool contracts
- `ChainedTool` for multi-step tool composition
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Dict, List, Union

from dr_agent.tool_interface import BaseTool, ChainedTool, Document, DocumentToolOutput
from dr_agent.tool_interface.data_types import ToolInput, ToolOutput


SAMPLE_PAPERS = [
    {
        "id": "paper-1",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/abs/1706.03762",
        "abstract": "The Transformer replaces recurrence with self-attention mechanisms.",
        "full_text": (
            "We propose the Transformer, a model architecture based only on attention "
            "mechanisms. Self-attention enables efficient sequence modeling."
        ),
    },
    {
        "id": "paper-2",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "url": "https://arxiv.org/abs/1810.04805",
        "abstract": "BERT introduces bidirectional pre-training for language representations.",
        "full_text": (
            "BERT pre-trains deep bidirectional representations by jointly conditioning "
            "on both left and right context in all layers."
        ),
    },
    {
        "id": "paper-3",
        "title": "Graph Neural Networks: A Review of Methods and Applications",
        "url": "https://arxiv.org/abs/1812.08434",
        "abstract": "A review of graph neural network methods and practical applications.",
        "full_text": (
            "Graph neural networks operate on graph-structured data and are widely used "
            "for node classification, graph classification, and recommendation."
        ),
    },
]


def _extract_query(tool_input: Union[str, ToolInput, ToolOutput]) -> str:
    if isinstance(tool_input, str):
        return tool_input
    if isinstance(tool_input, ToolOutput):
        return tool_input.output
    if isinstance(tool_input, dict):
        return str(tool_input.get("query", ""))
    raise ValueError(f"Unsupported input type: {type(tool_input)}")


class InMemorySearchTool(BaseTool):
    """Keyword search over an in-memory paper catalog."""

    def __init__(self, papers: List[Dict[str, str]], top_k: int = 3, **kwargs):
        super().__init__(tool_parser="null", name="in_memory_search", **kwargs)
        self.papers = papers
        self.top_k = top_k

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        call_id = self._generate_call_id()
        start = time.time()
        query = _extract_query(tool_input).strip()
        if not query:
            return DocumentToolOutput(
                tool_name=self.name,
                output="",
                called=False,
                error="Query is empty.",
                runtime=time.time() - start,
                call_id=call_id,
                documents=[],
            )

        terms = set(query.lower().split())
        scored_docs = []
        for paper in self.papers:
            haystack = f"{paper['title']} {paper['abstract']}".lower()
            score = float(sum(1 for term in terms if term in haystack))
            if score > 0:
                scored_docs.append((score, paper))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        selected = scored_docs[: self.top_k]
        documents = [
            Document(
                id=paper["id"],
                title=paper["title"],
                url=paper["url"],
                snippet=paper["abstract"],
                score=score,
            )
            for score, paper in selected
        ]

        output = "\n\n".join(doc.stringify() for doc in documents)
        return DocumentToolOutput(
            tool_name=self.name,
            output=output,
            called=True,
            error="",
            runtime=time.time() - start,
            call_id=call_id,
            documents=documents,
            query=query,
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for papers"}
            },
            "required": ["query"],
        }


class InMemoryBrowseTool(BaseTool):
    """Fetches full text for previously retrieved in-memory papers."""

    def __init__(self, url_to_full_text: Dict[str, str], **kwargs):
        super().__init__(tool_parser="null", name="in_memory_browse", **kwargs)
        self.url_to_full_text = url_to_full_text

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        call_id = self._generate_call_id()
        start = time.time()

        if not isinstance(tool_input, DocumentToolOutput) or not tool_input.documents:
            return DocumentToolOutput(
                tool_name=self.name,
                output="",
                called=False,
                error="Browse tool expects DocumentToolOutput with documents.",
                runtime=time.time() - start,
                call_id=call_id,
                documents=[],
            )

        enriched_documents = []
        for doc in tool_input.documents:
            full_text = self.url_to_full_text.get(doc.url, "") or (doc.snippet or "")
            enriched_documents.append(
                Document(
                    id=doc.id,
                    title=doc.title,
                    url=doc.url,
                    snippet=doc.snippet,
                    text=full_text,
                    score=doc.score,
                )
            )

        output = "\n\n".join(doc.simple_stringify() for doc in enriched_documents)
        return DocumentToolOutput(
            tool_name=self.name,
            output=output,
            called=True,
            error="",
            runtime=time.time() - start,
            call_id=call_id,
            documents=enriched_documents,
            query=tool_input.query,
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "Documents to fetch full text for",
                }
            },
            "required": ["documents"],
        }


class BarebonesPaperWorkflow:
    """Minimal orchestrator that demonstrates search -> browse composition."""

    def __init__(self, top_k: int = 2):
        self.top_k = top_k
        self.search_tool = InMemorySearchTool(SAMPLE_PAPERS, top_k=top_k)
        self.browse_tool = InMemoryBrowseTool(
            {paper["url"]: paper["full_text"] for paper in SAMPLE_PAPERS}
        )
        self.pipeline = ChainedTool(
            tools=[self.search_tool, self.browse_tool],
            output_formatting="last",
            tool_parser="null",
            name="barebones_pipeline",
        )

    async def __call__(self, query: str, **kwargs) -> DocumentToolOutput:
        search_output = await self.search_tool({"query": query})
        if search_output.error:
            return search_output
        return await self.browse_tool(search_output)

    async def run_chained(self, query: str) -> ToolOutput:
        return await self.pipeline({"query": query})


async def run_barebones_demo(query: str, top_k: int = 2) -> DocumentToolOutput:
    workflow = BarebonesPaperWorkflow(top_k=top_k)
    return await workflow(query)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a barebones in-memory paper pipeline.")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=2, help="Number of papers to return")
    args = parser.parse_args()

    result = asyncio.run(run_barebones_demo(args.query, top_k=args.top_k))
    if result.error:
        print(f"Error: {result.error}")
        return 1

    print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
