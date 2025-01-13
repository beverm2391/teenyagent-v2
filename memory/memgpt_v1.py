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
from rich.panel import Panel
from rich.box import ROUNDED
from rich.syntax import Syntax
from rich.rule import Rule
from rich.console import Console
from dotenv import load_dotenv
from enum import Enum
from pydantic import Field
from openai import OpenAI


# ! Env Vars
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY must be set")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY must be set")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set")


# ! Debug Levels
class DebugLevel(Enum):
    NONE = 0
    ERRORS = 1
    BASIC = 2
    DEBUG = 3


# ! Rich Console
console = Console()


# ! Pydantic Models
class RerankResult(BaseModel):
    index: int
    relevance_score: float
    meta: dict
    billed_units: dict


class RerankResponse(BaseModel):
    id: str
    results: list[RerankResult]


class ObserverResponse(BaseModel):
    results: list[str] = Field(
        description="A list of strings that are the facts that the observer extracted from the conversation."
    )


# ! MemGPT
class MemGPT:
    def __init__(self, context_window=200000, debug=DebugLevel.DEBUG):
        # ! Models
        self.chat_model = "claude-3-5-sonnet-latest"
        self.observer_model = "gpt-4o"
        self.relevance_model = "rerank-v3.5"

        # ! Clients
        self.chat_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.relevance_client = Cohere(api_key=COHERE_API_KEY)
        self.observer_client = OpenAI(api_key=OPENAI_API_KEY)

        # ! Tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # ! Context Window
        self.context_window = context_window
        self.max_response_tokens = 4096

        # ! Memories
        self.working_memory: Dict[str, float] = {}
        self.conversation = deque(maxlen=20)
        self.archive: List[Tuple[str, float]] = []

        # ! Thresholds
        self.pressure_threshold = 0.8 # 80% of the context window
        self.working_memory_threshold = 0.65
        self.archive_threshold = 0.2

        # ! Debug
        self.debug = debug

        # Log everything we did
        updates = []
        updates.append("MemGPT initialized")
        updates.append(f"Chat Model: {self.chat_model}")
        updates.append(f"Observer Model: {self.observer_model}")
        updates.append(f"Relevance Model: {self.relevance_model}")
        updates.append(f"Context Window: {self.context_window} tokens")
        updates.append(f"Max Response Tokens: {self.max_response_tokens} tokens")
        updates.append(f"Working Memory Threshold: {self.working_memory_threshold}")
        updates.append(f"Archive Threshold: {self.archive_threshold}")
        updates.append(f"Pressure Threshold: {self.pressure_threshold}")
        self._rich_basic("\n".join(updates), "Initialization", "green")

    def _rich_basic(
        self,
        content: str,
        title: str,
        style: str = "blue",
        level: DebugLevel = DebugLevel.BASIC,
    ):
        if self.debug.value >= level.value:
            console.print(
                Panel(
                    str(content),
                    title=f"[{style}]{title}[/{style}]",
                    border_style=style,
                    box=ROUNDED,
                    width=100,
                )
            )

    def chat(self, user_input: str) -> str:
        """
        Main user interaction. Checks memory size, possibly handles pressure,
        retrieves context, constructs prompt, calls model, updates memories.
        """
        self._rich_basic(f"{user_input[:500]}...", "User Input")

        current_size = self._get_current_context_size()
        if current_size > self.pressure_threshold * self.context_window:
            print("Triggering memory pressure")
            self._rich_basic(
                f"Memory pressure: {current_size}/{self.context_window} tokens",
                "Memory Warning",
                "red",
                DebugLevel.ERRORS,
            )
            self._handle_memory_pressure()

        print("getting relevant memories")
        relevant_context = self._get_relevant_memories(user_input)
        if len(relevant_context) > 0:
            self._rich_basic(
                "\n".join(f"- {mem[0]} ({mem[1]:.2f})" for mem in relevant_context),
                "Retrieved Memories",
                "green",
            )

        print("building prompt")
        prompt = self._build_prompt(relevant_context, user_input)
        print("querying llm")
        response = self._query_llm(prompt)

        # updaet the conversation so that when we call _update_memories, we have the most recent messages
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": response})

        # TODO make this async somehow??
        print("extracting facts from the response")
        new_facts = self._extract_facts(user_input, response)
        print("updating memories")
        self._update_memories(new_facts)
        print("logging assistant response")
        self._rich_basic(response, "Assistant Response")
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
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_response_tokens,
            )
            return response.content[0].text
        except Exception as e:
            self._rich_basic(str(e), "LLM Error", "red", DebugLevel.ERRORS)
            raise RuntimeError(f"LLM query failed: {e}")

    def _extract_facts(self, user_input: str, response: str) -> List[str]:
        """
        Uses the model to extract new facts. If it fails, returns empty.
        """
        extraction_prompt = (
            "Extract important facts from this conversation that you don't already know."
            "Return a Python list of short strings. "
            f"User: {user_input}\nAssistant: {response}"
        )
        try:
            # call the parse method
            extraction = self.observer_client.beta.chat.completions.parse(
                model=self.observer_model,
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=1000,
                response_format=ObserverResponse,
            )
            # get the results from the parsed response model
            facts: List[str] = extraction.choices[0].message.parsed.results
            # log the facts
            self._rich_basic(
                f"Extracted facts:\n" + "\n".join(f"- {f}" for f in facts),
                "Memory Update",
                "green",
            )
            return facts
        except Exception as e:
            self._rich_basic(f"Fact extraction failed: {e}", "Error", "red", DebugLevel.ERRORS)
            return []

    def _update_memories(self, facts: List[str]):
        """
        Scores new facts and distributes to working memory or archive.
        Uses a single rerank call with an importance prompt.
        """
        if not facts:
            print("in _update_memories, no facts to update")
            return
        
        dump_conversation = "\n".join(f"{m['role']}: {m['content']}" for m in self.conversation)
        scored = self._rerank_documents(f"What is most important based on the current conversation? Apply more weight to the most recent messages. CONVERSTION:\n\n{dump_conversation}\n\nEND CONVERSTION", facts)
        updates = []
        print("scored facts: ", scored)
        for fact, score in scored:
            print("fact: ", fact, "score: ", score)
            if score > self.working_memory_threshold:
                print("adding to working memory")
                self.working_memory[fact] = score
                updates.append(f"Working Memory: {fact} ({score:.2f})")
            elif score > self.archive_threshold:
                print("adding to archive")
                self.archive.append((fact, score))
                updates.append(f"Archive: {fact} ({score:.2f})")
            else:
                print("discarding fact due to low score of: ", score)

        if updates:
            self._rich_basic("\n".join(updates), "Memory Updates", "green")

    def _handle_memory_pressure(self):
        """
        Moves less important items from working memory to archive.
        """
        if not self.working_memory:
            print("in _handle_memory_pressure, no working memory to handle")
            return
        
        dump_conversation = "\n".join(f"{m['role']}: {m['content']}" for m in self.conversation)

        self._rich_basic("Handling memory pressure...", "Memory Management", "green")
        scored = self._rerank_documents(
            f"Most critical to keep in working memory based on the current conversation. Apply more weight to the most recent messages. CONVERSTION:\n\n{dump_conversation}\n\nEND CONVERSTION", list(self.working_memory.keys())
        )

        moved = []
        for fact, score in scored:
            if score < self.working_memory_threshold:
                self.archive.append((fact, score))
                del self.working_memory[fact]
                moved.append(f"Moved to archive: {fact} ({score:.2f})")

        if moved:
            self._rich_basic("\n".join(moved), "Memory Pressure Resolution", "green")

    def _get_relevant_memories(self, query: str) -> List[Tuple[str, float]]:
        """
        Reranks the archive for relevance to the query.
        """
        if not self.archive:
            print("in _get_relevant_memories, no archive to get relevant memories from. returning empty list")
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
                model=self.relevance_model, query=query, documents=documents
            )
            return [(documents[r.index], r.relevance_score) for r in result.results]
        except Exception as e:
            self._rich_basic(f"Reranking failed: {e}", "Error", "red", DebugLevel.ERRORS)
            return [(doc, 0.5) for doc in documents]

    def show_memory_status(self):
        """
        Shows how close we are to hitting memory pressure.
        This prints basic info on the context window, working memory usage, conversation usage, and thresholds.
        """
        # Builds a string of working memory items so we can tokenize it to measure how many tokens they occupy
        mem_str = "\n".join(k for k in self.working_memory.keys())
        # Builds a string of conversation items for token counting
        conv_str = "\n".join(f"{m['role']}: {m['content']}" for m in self.conversation)

        # Measures token usage for working memory
        working_memory_tokens = len(self.tokenizer.encode(mem_str))
        # Measures token usage for conversation
        conversation_tokens = len(self.tokenizer.encode(conv_str))
        # Calculates the total used tokens
        total_used_tokens = working_memory_tokens + conversation_tokens
        # Calculates how many tokens remain before we exceed our context window
        tokens_remaining = self.context_window - total_used_tokens
        # Calculates the overall pressure ratio
        overall_pressure = total_used_tokens / self.context_window

        # Constructs a message to display in a Rich panel
        status_message = (
            f"Total context window: {self.context_window}\n"
            f"Working memory tokens: {working_memory_tokens}\n"
            f"Conversation tokens: {conversation_tokens}\n"
            f"Tokens remaining: {tokens_remaining}\n"
            f"Overall pressure: {overall_pressure:.2f}\n"
            f"Pressure threshold: {self.pressure_threshold}"
        )

        # Uses the existing log mechanism to display the information
        self._rich_basic(status_message, "Memory Status", "cyan", level=DebugLevel.BASIC)
