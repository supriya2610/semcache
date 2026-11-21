# Semantic Cache Gateway

POST /v1/chat/completions   OpenAI-style body + `metadata: {feature, user_id, tags[], cache_control}`
Response adds `cache: {status: miss|exact_hit|semantic_hit|blocked|bypass, similarity, matched_prompt, ...}`

POST /v1/cache/invalidate   filters: model, system_prompt_hash, feature, user_id, tag, created_after/before
POST /v1/cache/report_wrong_hit   {entry_id}
GET  /v1/cache/stats | /v1/cache/debug?kind=near_miss|blocked | /metrics

Run: `pip install -r requirements.txt && uvicorn app.main:app`  (or `docker compose up`)

Decision pipeline: policy -> normalize (ctx hash) -> exact -> ANN within same ctx -> threshold
-> deterministic guard (numbers/negation) -> judge band -> hit; else miss -> provider -> store.

Not yet implemented (next steps): streaming, Redis/Qdrant backend (swap `Cache` internals),
LLM judge, Anthropic provider, Grafana dashboard JSON.
# semcache
