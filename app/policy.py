from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Policy:
    enabled: bool = True
    ttl: int = 3600
    threshold: float = 0.95
    per_user: bool = False       # scope cache entries to a single user
    semantic: bool = True        # False = exact-match only
    judge_band: float = 0.03     # hits within threshold..threshold+band get judge validation
    near_miss_band: float = 0.05

DEFAULT = Policy()

# feature-specific overrides
FEATURES = {
    "faq": Policy(ttl=86400, threshold=0.92),
    "support_chat": Policy(ttl=900, threshold=0.96),
}

# tag policies: combined conservatively (strictest wins)
TAGS = {
    "user_specific": Policy(ttl=300, threshold=0.97, per_user=True),
    "current_events": Policy(ttl=60, semantic=False),
    "legal_sensitive": Policy(enabled=False),
    "creative": Policy(enabled=False),
}


def resolve(feature: str, tags: list[str]) -> Policy:
    p = FEATURES.get(feature, DEFAULT)
    for t in tags:
        tp = TAGS.get(t)
        if not tp:
            continue
        p = replace(p, enabled=p.enabled and tp.enabled, ttl=min(p.ttl, tp.ttl) if tp.ttl != DEFAULT.ttl else p.ttl,
                    threshold=max(p.threshold, tp.threshold) if tp.threshold != DEFAULT.threshold else p.threshold,
                    per_user=p.per_user or tp.per_user, semantic=p.semantic and tp.semantic)
    return p
