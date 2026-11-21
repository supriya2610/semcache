import hashlib, json, re, threading, time, uuid
from dataclasses import dataclass, field
import numpy as np
from .policy import Policy

PRICES = {"default": (0.0005, 0.0015), "gpt-4o": (0.0025, 0.01), "gpt-4o-mini": (0.00015, 0.0006)}  # $/1k in,out


def cost_usd(model: str, usage: dict) -> float:
    pi, po = PRICES.get(model, PRICES["default"])
    return usage.get("prompt_tokens", 0) / 1000 * pi + usage.get("completion_tokens", 0) / 1000 * po


def _txt(c) -> str:
    return c if isinstance(c, str) else json.dumps(c, sort_keys=True)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def sha(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class Normalized:
    ctx: str            # compatibility hash: model, system, temp, tools, feature, history, scope
    system_hash: str
    query: str          # last user message, normalized
    raw_query: str
    model: str
    feature: str
    user: str | None
    tags: list
    metadata: dict


def normalize(req, per_user: bool) -> Normalized:
    msgs = req.messages
    system = "\n".join(_txt(m["content"]) for m in msgs if m["role"] == "system")
    non_sys = [m for m in msgs if m["role"] != "system"]
    last = next((m for m in reversed(non_sys) if m["role"] == "user"), None)
    raw = _txt(last["content"]) if last else ""
    history = non_sys[:-1] if non_sys and non_sys[-1] is last else non_sys  # earlier turns must match exactly
    md = req.metadata or {}
    feature, user = md.get("feature", "default"), md.get("user_id")
    sys_h = sha(system)
    ctx = sha(req.model, sys_h, round(req.temperature, 3), req.max_tokens, req.tools, feature,
              [(m["role"], norm_text(_txt(m["content"]))) for m in history], user if per_user else None)
    return Normalized(ctx, sys_h, norm_text(raw), raw, req.model, feature, user, md.get("tags", []), md)


NUM = re.compile(r"\d+(?:[.,]\d+)?")
NEG = re.compile(r"\b(not|no|never|without|except|don't|dont|cannot|can't)\b")


def guard_ok(a: str, b: str) -> tuple[bool, str]:
    """Cheap deterministic guard: numbers and negation must agree."""
    if set(NUM.findall(a)) != set(NUM.findall(b)):
        return False, "numbers_differ"
    if bool(NEG.search(a)) != bool(NEG.search(b)):
        return False, "negation_differs"
    return True, ""


@dataclass
class Entry:
    id: str
    ctx: str
    key: str
    prompt: str
    emb: np.ndarray
    response: dict
    created: float
    expires: float
    feature: str
    model: str
    system_hash: str
    user: str | None
    tags: list
    metadata: dict
    latency_s: float
    cost: float
    hits: int = 0


@dataclass
class Lookup:
    status: str                 # exact_hit | semantic_hit | miss | blocked | judge_needed
    entry: Entry | None = None
    sim: float = 0.0
    reason: str = ""
    emb: np.ndarray | None = None
    near_miss: bool = False


class Cache:
    def __init__(self):
        self.by_ctx: dict[str, list[Entry]] = {}
        self.exact: dict[str, Entry] = {}
        self.by_id: dict[str, Entry] = {}
        self.lock = threading.Lock()
        self.debug: list[dict] = []   # near misses, blocks, etc.

    def size(self):
        return len(self.by_id)

    def log(self, **kw):
        self.debug.append({"ts": time.time(), **kw})
        del self.debug[:-5000]

    def lookup(self, nm: Normalized, pol: Policy, embed) -> Lookup:
        now = time.time()
        key = sha(nm.ctx, nm.query)
        with self.lock:
            e = self.exact.get(key)
            if e and e.expires > now:
                return Lookup("exact_hit", e, 1.0)
            cands = [c for c in self.by_ctx.get(nm.ctx, []) if c.expires > now]
        if not pol.semantic or not cands:
            return Lookup("miss", emb=None)
        q = embed(nm.query)
        sims = np.stack([c.emb for c in cands]) @ q
        i = int(np.argmax(sims)); s = float(sims[i]); best = cands[i]
        if s >= pol.threshold:
            ok, why = guard_ok(nm.query, best.prompt)
            if not ok:
                self.log(kind="blocked", reason=why, sim=s, prompt=nm.raw_query, matched=best.prompt)
                return Lookup("blocked", best, s, why, q)
            if s < pol.threshold + pol.judge_band:
                return Lookup("judge_needed", best, s, "borderline", q)
            return Lookup("semantic_hit", best, s, emb=q)
        near = s >= pol.threshold - pol.near_miss_band
        if near:
            self.log(kind="near_miss", sim=s, prompt=nm.raw_query, matched=best.prompt)
        return Lookup("miss", best, s, emb=q, near_miss=near)

    def store(self, nm: Normalized, pol: Policy, emb, response: dict, latency_s: float, cost: float) -> Entry:
        now = time.time()
        e = Entry(uuid.uuid4().hex[:12], nm.ctx, sha(nm.ctx, nm.query), nm.query, emb, response, now,
                  now + pol.ttl, nm.feature, nm.model, nm.system_hash, nm.user, nm.tags, nm.metadata,
                  latency_s, cost)
        with self.lock:
            self.exact[e.key] = e
            self.by_id[e.id] = e
            self.by_ctx.setdefault(e.ctx, []).append(e)
        return e

    def evict(self, pred) -> int:
        with self.lock:
            dead = [e for e in self.by_id.values() if pred(e)]
            for e in dead:
                self.by_id.pop(e.id, None); self.exact.pop(e.key, None)
                self.by_ctx[e.ctx] = [x for x in self.by_ctx.get(e.ctx, []) if x.id != e.id]
        return len(dead)

    def purge_expired(self):
        now = time.time()
        return self.evict(lambda e: e.expires <= now)
