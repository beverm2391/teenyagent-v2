from collections import deque
import numpy as np
from anthropic import Anthropic
from cohere import ClientV2 as Cohere
import os
from pydantic import BaseModel
import tiktoken
from dotenv import load_dotenv
load_dotenv()

# ! Env Vars
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY must be set")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY must be set")

# ! Reranker Models
class RerankResult(BaseModel):
    # ? see https://docs.cohere.com/docs/rerank-overview
    index: int
    relevance_score: float
    meta: dict
    billed_units: dict

class RerankResponse(BaseModel):
    id: str
    results: list[RerankResult]


# ! Simple MemGPT
class SimpleMemGPT:
    def __init__(self, context_window=128000):
        # clients
        self.chat_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.relevance_client = Cohere(api_key=COHERE_API_KEY)

        # chat model context window
        self.context_window = context_window
        # max response tokens
        self.max_response_tokens = 4096
        # memory
        self.working_memory = {}  # Important facts/context
        self.conversation = deque(maxlen=20)  # Recent messages FIFO
        self.archive = []  # Long-term storage
        self.pressure_threshold = 0.8 # when we start to archive
        
    def chat(self, user_input: str):
        # 1. Check memory pressure
        if (
            self._get_current_context_size()
            > self.pressure_threshold * self.context_window
        ):
            self._handle_memory_pressure()

        # 2. Get relevant context from archive
        relevant_context = self._get_relevant_memories(user_input)

        # 3. Build prompt with active context
        prompt = self._build_prompt(
            relevant_context=relevant_context,
            working_memory=self.working_memory,
            conversation=self.conversation,
            current_input=user_input,
        )

        # 4. Get LLM response
        response = self.chat_client.messages.create(
            model="claude-3-5-sonnet-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_response_tokens,

        )

        # 5. Update memories
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": response})

        # 6. Extract and store important info
        new_facts = self._extract_facts(user_input, response)
        importance_scores = self._rank_importance(new_facts)

        for fact, score in importance_scores:
            if score > 0.8:  # Important enough for working memory
                self.working_memory[fact] = score
            elif score > 0.3:  # Worth archiving
                self.archive.append((fact, score))

        return response
    
    def rerank(self, query: str, documents: list):
        results: RerankResponse = self.relevance_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
        )
        # Type of Results:

    def _get_current_context_size(self):
        # Use tiktoken or similar to count tokens
        # Need to count working_memory + conversation
        pass
    
    def _handle_memory_pressure(self):
        # Move less relevant items from working memory to archive
        scores = self._rank_importance(self.working_memory)
        for fact, score in scores:
            if score < 0.7:  # Threshold for keeping in working memory
                self.archive.append((fact, score))
                del self.working_memory[fact]

    def _get_relevant_memories(self, query):
        # Get embeddings similarity scores
        scores = [(mem, self._similarity(query, mem[0])) for mem in self.archive]

        # Return top k most relevant memories
        return sorted(scores, key=lambda x: x[1], reverse=True)[:5]

    def _rank_importance(self, facts):
        # Could use a real reranker here, but simple scoring for now
        return [(fact, self._compute_importance(fact)) for fact in facts]
