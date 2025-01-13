from memory.memgpt_v1 import MemGPT
import pytest


@pytest.mark.asyncio
async def test_memgpt():
    memgpt = MemGPT()
    await memgpt.chat("My name is ben and i really like potatoes also the sky is blue. also you need to remember my name please")

    assert len(memgpt.working_memory) > 0, "nothing got stored in working memory"
    assert len(memgpt.archive) > 0, "nothing got stored in archive"
