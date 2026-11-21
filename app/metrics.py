from prometheus_client import Counter, Gauge, Histogram

REQS = Counter("semcache_requests_total", "Requests", ["status", "feature"])
LAT = Histogram("semcache_latency_seconds", "End-to-end latency", ["status"],
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10))
COST_SAVED = Counter("semcache_cost_saved_usd_total", "Estimated $ saved")
COST_SPENT = Counter("semcache_cost_spent_usd_total", "Provider $ spent")
LAT_SAVED = Counter("semcache_latency_saved_seconds_total", "Estimated seconds saved")
NEAR = Counter("semcache_near_misses_total", "Prompts just below threshold")
BLOCKED = Counter("semcache_blocked_hits_total", "Candidate hits rejected", ["reason"])
WRONG = Counter("semcache_wrong_hit_reports_total", "User-reported wrong hits")
SIZE = Gauge("semcache_entries", "Cache entries")
