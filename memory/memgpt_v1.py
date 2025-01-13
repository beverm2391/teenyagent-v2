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
from openai import AsyncOpenAI
import asyncio


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


extraction_system_prompt = """
    You are an expert at extracting important information from conversations.
    You are given a conversation between a user and an assistant.
    You are tasked with extracting important information from the conversation that the assistant doesn't already know.
    You prioritize technical knowledge like how something was implemented, code, or other technical details.
    You always return a Python list of "blurbs" of the information you extracted. It is important that you include all important information in each blurb, so that someone can refer back to them later without the original conversation and understand the context. Make sure your blurbs are descriptive of all core information but not verbose in style.
"""

extraction_user_prompt = """
    Extract important information from this conversation that you don't already know.
    User: {user_input}\nAssistant: {response}
"""


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
        self.observer_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        # ! Tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # ! Context Window
        self.context_window = context_window
        self.max_response_tokens = 8192

        # ! Memories
        self.working_memory: Dict[str, float] = {}
        self.conversation: deque = deque()
        self.archive: List[Tuple[str, float]] = []
        self.current_task: str = ""

        # ! Thresholds and sizes
        buffer = 1000 # general buffer for tokens
        # max input tokens is the context window minus the max response tokens and buffer
        self.max_input_tokens = self.context_window - self.max_response_tokens - buffer
        self.conversation_size = 0.6 * self.max_input_tokens # 60% of the max input tokens
        self.working_memory_size = (1-(self.conversation_size / self.max_input_tokens)) * self.context_window # the rest of the max input tokens

        # Relevance thresholds (when should we add a fact to working memory or archive)
        self.working_memory_relevance_threshold = 0.5
        self.archive_relevance_threshold = 0.15

        # ! Debug
        self.debug = debug
        self.console = console

        # Log everything we did
        updates = []
        updates.append("MemGPT initialized")
        updates.append(f"Chat Model: {self.chat_model}")
        updates.append(f"Observer Model: {self.observer_model}")
        updates.append(f"Relevance Model: {self.relevance_model}")
        updates.append(f"Context Window: {self.context_window} tokens")
        updates.append(f"Max Response Tokens: {self.max_response_tokens} tokens")
        updates.append(f"Working Memory Relevance Threshold: {self.working_memory_relevance_threshold}")
        updates.append(f"Archive Relevance Threshold: {self.archive_relevance_threshold}")
        updates.append(f"Memory Pressure Threshold: {self.memory_pressure_threshold}")
        self._rich_block("\n".join(updates), "Initialization", "green")

    def _rich_block(
        self,
        content: str,
        title: str,
        style: str = "blue",
        level: DebugLevel = DebugLevel.BASIC,
    ):
        if self.debug.value >= level.value:
            self.console.print(
                Panel(
                    str(content),
                    title=f"[{style}]{title}[/{style}]",
                    border_style=style,
                    box=ROUNDED,
                    width=100,
                )
            )

    def _rich_log(
        self,
        content: str,
        style: str = "white",
        level: DebugLevel = DebugLevel.BASIC,
    ):
        if self.debug.value >= level.value:
            self.console.log(f"[{style}]{content}[/{style}]")

    async def chat(self, user_input: str) -> str:
        """
        Main user interaction. Checks memory size, possibly handles pressure,
        retrieves context, constructs prompt, calls model, updates memories.
        """
        self._rich_block(f"{user_input[:500]}...", "User Input")

        if self.over_max_input_tokens:
            self.console.log("Triggering memory pressure", "yellow")
            self._show_memory_status() # show the memory status
            self._handle_memory_pressure() # handle the memory pressure

        # Enter the main loop
        self.console.log("Getting relevant memories")
        relevant_context = self._get_relevant_memories(user_input)
        if len(relevant_context) > 0:
            self._rich_block(
                "\n".join(f"- {mem[0]} ({mem[1]:.2f})" for mem in relevant_context),
                "Retrieved Memories",
                "green",
            )

        self.console.log("Building prompt")
        prompt = self._build_prompt(relevant_context, user_input)
        self.console.log("Querying LLM")
        response = self._query_llm(prompt)

        # updaet the conversation so that when we call _update_memories, we have the most recent messages
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": response})

        # TODO make this async somehow??
        self.console.log("Extracting facts from response")

        # get new facts from the output
        new_facts = await self._extract_facts(user_input, response)

        # update the current task based on the new messages
        await self._update_current_task()

        self.console.log("Updating memories")
        self._update_memories(new_facts)
        self.console.log("Logging assistant response")
        self._rich_block(response, "Assistant Response")
        return response
    
    @property
    def over_max_input_tokens(self) -> bool:
        """
        Returns true if the current context size is over the max input tokens
        """
        return (self.working_memory_tokens + self.conversation_tokens) > self.max_input_tokens

    @property
    def conversation_tokens(self) -> int:
        """
        Updates the conversation tokens based on the messages.
        """
        # if no messages yet, return 0
        if len(self.conversation) == 0:
            return 0
        
        # else, count it
        dump_conversation = "\n".join(f"{m['role']}: {m['content']}" for m in self.conversation)
        return sum(len(self.tokenizer.encode(dump_conversation)))

    @property
    def working_memory_tokens(self) -> int:
        """
        Returns the total amount of tokens in the working memory.
        """
        # if no working memory yet, return 0
        if len(self.working_memory) == 0:
            return 0

        # else, count it
        mem_str = "\n".join(f"{k}: {v}" for k, v in self.working_memory.items())
        return sum(len(self.tokenizer.encode(mem_str)))

    # TODO make a conversation class with methods on it
    def _trim_conversation(self, chunk_size: int = 1000, depth: int = 0):
        """
        Trims the conversation to the given number of tokens in chunks of chunk_size.
        """
        if self.conversation_size > self.conversation_tokens:
            if depth > 0: # if we trimmed anything, log it
                print(f"Trimmed conversation to {self.conversation_tokens} tokens")
            else: # if we didn't trim anything, log that
                print("No need to trim conversation")
            return
        else:
            # trim by chunk_size
            oldest_message = self.conversation.popleft()
            oldest_message_tokens = len(self.tokenizer.encode(oldest_message["content"]))
            if oldest_message_tokens > chunk_size:
                oldest_message["content"] = self.tokenizer.decode(self.tokenizer.encode(oldest_message["content"])[:chunk_size])
            return self._trim_conversation(chunk_size, depth + 1)

    # TODO make a working memory class with methods on it
    def _trim_working_memory(self, chunk_size: int = 1000, depth: int = 0):
        """
        Trims the working memory to the given number of tokens in chunks of chunk_size.
        """
        if self.working_memory_size > self.working_memory_tokens:
            if depth > 0: # if we trimmed anything, log it
                print(f"Trimmed working memory to {self.working_memory_tokens} tokens")
            else: # if we didn't trim anything, log that
                print("No need to trim working memory")
            return
        else:
            least_relevant_fact = min(self.working_memory.items(), key=lambda x: x[1])
            # move the least relevant fact to the archive
            self.archive.append(least_relevant_fact)
            del self.working_memory[least_relevant_fact[0]]
            return self._trim_working_memory(chunk_size, depth + 1)

    async def _update_current_task(self):
        """
        Get the current task from the conversation.
        """
        dump_conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.conversation
        )
        prompt = f"Figure out the current task from the conversation. Return a concise task description. CONVERSATION: {dump_conversation}"
        response = await self.observer_client.chat.completions.create(
            model=self.observer_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        current_task = response.choices[0].message.content
        self._rich_block(current_task, "Current Task", "yellow")
        self.current_task = current_task
        return current_task

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
            self._rich_block(str(e), "LLM Error", "red", DebugLevel.ERRORS)
            raise RuntimeError(f"LLM query failed: {e}")

    async def _extract_facts(self, user_input: str, response: str) -> List[str]:
        """
        Uses the model to extract new facts. If it fails, returns empty.
        """
        try:
            # call the parse method
            extraction = await self.observer_client.beta.chat.completions.parse(
                model=self.observer_model,
                messages=[
                    {
                        "role": "system",
                        "content": extraction_system_prompt,
                    },
                    {
                        "role": "user",
                        "content": extraction_user_prompt.format(
                            user_input=user_input, response=response
                        ),
                    },
                ],
                max_tokens=1000,
                response_format=ObserverResponse,
            )
            # get the results from the parsed response model
            facts: List[str] = extraction.choices[0].message.parsed.results
            # log the facts
            self._rich_block(
                f"Extracted facts:\n" + "\n".join(f"- {f}" for f in facts),
                "Memory Update",
                "green",
            )
            return facts
        except Exception as e:
            self._rich_block(
                f"Fact extraction failed: {e}", "Error", "red", DebugLevel.ERRORS
            )
            return []

    def _update_memories(self, facts: List[str]):
        """
        Scores new facts and distributes to working memory or archive.
        Uses a single rerank call with an importance prompt.
        """
        if not facts:
            print("in _update_memories, no facts to update")
            return

        scored = self._rerank_documents(self.current_task, facts)
        updates = []
        self.console.log("scored facts: ", scored)
        for fact, score in scored:
            if score > self.working_memory_relevance_threshold:
                self.console.log(
                    "adding to working memory: ", fact, "with score: ", score
                )
                self.working_memory[fact] = score
                updates.append(f"Working Memory: {fact} ({score:.2f})")
            elif score > self.archive_relevance_threshold:
                self.console.log("adding to archive: ", fact, "with score: ", score)
                self.archive.append((fact, score))
                updates.append(f"Archive: {fact} ({score:.2f})")
            else:
                self.console.log(
                    "discarding fact: ", fact, "due to low score of: ", score
                )

        if updates:
            self._rich_block("\n".join(updates), "Memory Updates", "green")


    def _handle_memory_pressure(self):
        """
        This does a few things:
        - Trims the conversation to the conversation size set in the constructor
        - Moves less important items from working memory to archive
        - Then, if still needed, trims the working memory to the working memory size set in the constructor
        """
        # We need to check two things, the conversation size and the working memory size

        # If our conversation is too big, trim it with our _trim_conversation method
        if self.conversation_tokens > self.conversation_size:
            self._trim_conversation()

        # If our working memory is too big, handle it by reranking and moving less important items to archive
        if self.conversation_tokens > self.working_memory_size:

            # First we rerank the working memory to see what is important
            # Since we only want to trim the least important items to the current tasks
            # TODO figure out if it'll alrady be ranked and this is redundant (aka update_memory was just called)
            self._rich_block("Handling memory pressure...", "Memory Management", "green")
            scored = self._rerank_documents(
                self.current_task, list(self.working_memory.keys())
            )
            moved = []
            for fact, score in scored:
                if score < self.working_memory_relevance_threshold:
                    self.archive.append((fact, score))
                    del self.working_memory[fact]
                    moved.append(f"Moved to archive: {fact} ({score:.2f})")

            # Now we trim the working memory to the size we want, if its over
            self._trim_working_memory()

            if moved:
                self._rich_block("\n".join(moved), "Memory Pressure Resolution", "green")

    def _get_relevant_memories(self, query: str) -> List[Tuple[str, float]]:
        """
        Reranks the archive for relevance to the query.
        """
        if not self.archive:
            self.console.log(
                "in _get_relevant_memories, no archive to get relevant memories from. returning empty list"
            )
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
            self._rich_block(
                f"Reranking failed: {e}", "Error", "red", DebugLevel.ERRORS
            )
            return [(doc, 0.5) for doc in documents]

    def _show_memory_status(self):
        """
        Shows memory status
        """
        status_message = (
            f"Total context window: {self.context_window}\n"
            f"Max Input Tokens: {self.max_input_tokens}\n"
            f"Working memory tokens: {self.working_memory_tokens} / {self.working_memory_size}\n"
            f"Conversation tokens: {self.conversation_tokens} / {self.conversation_size}\n"
        )

        # Uses the existing log mechanism to display the information
        self._rich_block(
            status_message, "Memory Status", "cyan", level=DebugLevel.BASIC
        )
