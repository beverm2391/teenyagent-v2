"""
Minimalist refactor of memgpt-v2 with simpler memory management and unified scoring/reranking logic.
This approach merges memory scoring and document reranking into a single function, removing duplicative code.
It also centralizes the chat flow, so new facts are processed immediately after retrieval, which can reduce complexity.
"""

from collections import deque
import numpy as np
from anthropic import Anthropic
from cohere import ClientV2 as Cohere
import os
from pydantic import BaseModel
import tiktoken
from typing import List, Dict, Tuple, Any


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    meta: dict
    billed_units: dict


class RerankResponse(BaseModel):
    id: str
    results: list[RerankResult]


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY must be set")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY must be set")


class SimpleMemGPT:
    def __init__(self, context_window=128000):
        self.chat_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.relevance_client = Cohere(api_key=COHERE_API_KEY)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.context_window = context_window
        self.max_response_tokens = 4096
        self.working_memory: Dict[str, float] = {}
        self.conversation = deque(maxlen=20)
        self.archive: List[Tuple[str, float]] = []
        self.pressure_threshold = 0.8
        self.working_memory_threshold = 0.7
        self.archive_threshold = 0.3

    def chat(self, user_input: str) -> str:
        """
        Main user interaction. Checks memory size, possibly handles pressure,
        retrieves context, constructs prompt, calls model, updates memories.
        """
        if (
            self._get_current_context_size()
            > self.pressure_threshold * self.context_window
        ):
            self._handle_memory_pressure()
        relevant_context = self._get_relevant_memories(user_input)
        prompt = self._build_prompt(relevant_context, user_input)
        response = self._query_llm(prompt)
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": response})
        new_facts = self._extract_facts(user_input, response)
        self._update_memories(new_facts)
        return response

    def _build_prompt(self, relevant_context, user_input):
        """
        Combines working memory facts, retrieved context, and conversation into the prompt.
        """
        facts_text = "Key Facts:\n" + "\n".join(k for k in self.working_memory.keys())
        archive_text = "Relevant Past Context:\n" + "\n".join(
            mem[0] for mem in relevant_context
        )
        recent_chat = "Recent Messages:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in self.conversation
        )
        return f"{facts_text}\n\n{archive_text}\n\n{recent_chat}\n\nUser: {user_input}"

    def _get_current_context_size(self) -> int:
        """
        Simple token counting of working memory plus conversation.
        """
        mem_str = "\n".join(f"{k}: {v}" for k, v in self.working_memory.items())
        conv_str = "\n".join(f"{m['role']}: {m['content']}" for m in self.conversation)
        return len(self.tokenizer.encode(mem_str)) + len(
            self.tokenizer.encode(conv_str)
        )

    def _query_llm(self, prompt: str) -> str:
        """
        Single function to call the model, discarding extraneous logic.
        """
        try:
            response = self.chat_client.messages.create(
                model="claude-3-5-sonnet-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_response_tokens,
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"LLM query failed: {e}")

    def _extract_facts(self, user_input: str, response: str) -> List[str]:
        """
        Uses the model to extract new facts. If it fails, returns empty.
        """
        extraction_prompt = (
            "Extract important facts from this conversation. "
            "Return a Python list of short strings. "
            f"User: {user_input}\nAssistant: {response}"
        )
        try:
            extraction = self.chat_client.messages.create(
                model="claude-3-5-sonnet-latest",
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=1000,
            )
            return eval(extraction.content[0].text)
        except:
            return []

    def _update_memories(self, facts: List[str]):
        """
        Scores new facts and distributes to working memory or archive.
        Uses a single rerank call with an importance prompt.
        """
        if not facts:
            return
        scored = self._rerank_documents("What is most important?", facts)
        for fact, score in scored:
            if score > self.working_memory_threshold:
                self.working_memory[fact] = score
            elif score > self.archive_threshold:
                self.archive.append((fact, score))

    def _handle_memory_pressure(self):
        """
        Moves less important items from working memory to archive.
        """
        if not self.working_memory:
            return
        scored = self._rerank_documents(
            "Most critical to keep", list(self.working_memory.keys())
        )
        for fact, score in scored:
            if score < self.working_memory_threshold:
                self.archive.append((fact, score))
                del self.working_memory[fact]

    def _get_relevant_memories(self, query: str) -> List[Tuple[str, float]]:
        """
        Reranks the archive for relevance to the query.
        """
        if not self.archive:
            return []
        docs = [mem[0] for mem in self.archive]
        return self._rerank_documents(query, docs)

    def _rerank_documents(
        self, query: str, documents: list[str]
    ) -> list[tuple[str, float]]:
        """
        Single rerank API call. On failure returns neutral scores.
        """
        try:
            result = self.relevance_client.rerank(
                model="rerank-v3.5", query=query, documents=documents
            )
            return [(documents[r.index], r.relevance_score) for r in result.results]
        except Exception as e:
            return [(doc, 0.5) for doc in documents]
