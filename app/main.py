import time, uuid
from collections import Counter as C
from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import metrics as M
from .cache import Cache, cost_usd, normalize
from .embeddings import get_embedder
from .policy import resolve
from .providers import get_provider

app = FastAPI(title="Semantic Cache Gateway")
cache, embedder, provider = Cache(), get_embedder(), get_provider()
stats = C()


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 1.0
    max_tokens: int | None = None
    tools: list[dict] | None = None
    metadata: dict = Field(default_factory=dict)   # feature, user_id, tags[], cache_control, group


async def judge(new_q: str, cached_q: str, cached_answer: str) -> bool:
    """Borderline validation. Swap in a small LLM call (e.g. gpt-4o-mini) for production.
    Here: strict lexical-overlap check so the demo runs offline."""
    a, b = set(new_q.split()), set(cached_q.split())
    return len(a & b) / max(1, len(a | b)) >= 0.6


def wrap(resp: dict, **cache_meta) -> dict:
    return {**resp, "cache": cache_meta}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    t0 = time.perf_counter()
    md = req.metadata
    pre = resolve(md.get("feature", "default"), md.get("tags", []))
    nm = normalize(req, pre.per_user)
    pol = pre
    bypass = (not pol.enabled or md.get("cache_control") == "no-cache" or (pol.per_user and not nm.user)
              or not nm.query)
    status, entry, sim, reason, emb = "bypass", None, 0.0, "", None

    if not bypass:
        res = await run_in_threadpool(cache.lookup, nm, pol, embedder.embed)
        status, entry, sim, reason, emb = res.status, res.entry, res.sim, res.reason, res.emb
        if status == "judge_needed":
            ok = await judge(nm.query, entry.prompt, "")
            status, reason = ("semantic_hit", "judge_approved") if ok else ("blocked", "judge_rejected")
        if res.near_miss:
            M.NEAR.inc()
        if status == "blocked":
            M.BLOCKED.labels(reason).inc()
            status_for_store = "miss"
        if status == "miss":
            status_for_store = "miss"
        if status in ("exact_hit", "semantic_hit"):
            entry.hits += 1
            elapsed = time.perf_counter() - t0
            M.COST_SAVED.inc(entry.cost); M.LAT_SAVED.inc(max(0, entry.latency_s - elapsed))
            return finish(status, md, t0, wrap(entry.response, status=status, similarity=round(sim, 4),
                          cached_response_id=entry.response["id"], entry_id=entry.id,
                          matched_prompt=entry.prompt, original_prompt=nm.raw_query, reason=reason,
                          matched_metadata=entry.metadata, age_s=round(time.time() - entry.created, 1)))

    # miss / blocked / bypass -> provider
    tp = time.perf_counter()
    resp = await provider.complete(req)
    lat = time.perf_counter() - tp
    c = cost_usd(req.model, resp.get("usage", {}))
    M.COST_SPENT.inc(c)
    out_status = status if status in ("bypass", "blocked") else "miss"
    if not bypass and pol.enabled:
        if emb is None and pol.semantic:
            emb = await run_in_threadpool(embedder.embed, nm.query)
        cache.store(nm, pol, emb, resp, lat, c)
        M.SIZE.set(cache.size())
    return finish(out_status, md, t0, wrap(resp, status=out_status, similarity=round(sim, 4), reason=reason,
                  original_prompt=nm.raw_query, cached_response_id=None))


def finish(status, md, t0, body):
    dt = time.perf_counter() - t0
    M.REQS.labels(status, md.get("feature", "default")).inc()
    M.LAT.labels(status).observe(dt)
    stats[status] += 1
    body["cache"]["latency_ms"] = round(dt * 1000, 1)
    return body


class Invalidate(BaseModel):
    model: str | None = None
    system_prompt_hash: str | None = None
    feature: str | None = None
    user_id: str | None = None
    tag: str | None = None
    created_after: float | None = None
    created_before: float | None = None


@app.post("/v1/cache/invalidate")
def invalidate(f: Invalidate):
    if not any(v is not None for v in f.model_dump().values()):
        raise HTTPException(400, "Provide at least one filter (use /v1/cache/flush to clear everything)")
    def pred(e):
        return all([f.model is None or e.model == f.model,
                    f.system_prompt_hash is None or e.system_hash == f.system_prompt_hash,
                    f.feature is None or e.feature == f.feature,
                    f.user_id is None or e.user == f.user_id,
                    f.tag is None or f.tag in e.tags,
                    f.created_after is None or e.created >= f.created_after,
                    f.created_before is None or e.created <= f.created_before])
    n = cache.evict(pred); M.SIZE.set(cache.size())
    return {"evicted": n}


@app.post("/v1/cache/flush")
def flush():
    n = cache.evict(lambda e: True); M.SIZE.set(0)
    return {"evicted": n}


class Report(BaseModel):
    entry_id: str


@app.post("/v1/cache/report_wrong_hit")
def report(r: Report):
    """Evict the offending entry and count it; feeds the false-hit-rate panel."""
    M.WRONG.inc()
    return {"evicted": cache.evict(lambda e: e.id == r.entry_id)}


@app.get("/v1/cache/debug")
def debug(kind: str | None = None, limit: int = 100):
    rows = [d for d in cache.debug if kind is None or d["kind"] == kind]
    return rows[-limit:]


@app.get("/v1/cache/stats")
def get_stats():
    total = sum(stats.values()) or 1
    hits = stats["exact_hit"] + stats["semantic_hit"]
    return {"requests": sum(stats.values()), "by_status": dict(stats), "hit_rate": hits / total,
            "entries": cache.size()}


@app.get("/metrics")
def prom():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
