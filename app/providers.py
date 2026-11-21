import asyncio, os, time, uuid
import httpx


class MockProvider:
    """Deterministic fake LLM with realistic latency, for demos and load tests."""
    async def complete(self, req) -> dict:
        q = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
        await asyncio.sleep(float(os.getenv("MOCK_LATENCY", "0.6")))
        text = f"[mock answer] {q}"
        pt, ct = sum(len(str(m["content"])) for m in req.messages) // 4, len(text) // 4 + 40
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:10]}", "object": "chat.completion", "created": int(time.time()),
                "model": req.model, "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}


class OpenAICompatProvider:
    """Works for OpenAI and Ollama (PROVIDER_BASE_URL=http://localhost:11434/v1)."""
    def __init__(self):
        self.c = httpx.AsyncClient(base_url=os.getenv("PROVIDER_BASE_URL", "https://api.openai.com/v1"),
                                   headers={"Authorization": f"Bearer {os.getenv('PROVIDER_API_KEY', '')}"}, timeout=120)

    async def complete(self, req) -> dict:
        body = {"model": req.model, "messages": req.messages, "temperature": req.temperature}
        if req.max_tokens: body["max_tokens"] = req.max_tokens
        if req.tools: body["tools"] = req.tools
        r = await self.c.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()
# Anthropic: add a third class that maps messages -> /v1/messages and back to this response shape.


def get_provider():
    return OpenAICompatProvider() if os.getenv("PROVIDER", "mock") == "openai" else MockProvider()
