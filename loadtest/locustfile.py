"""2,000-request simulation: repeats, paraphrases, and 'trap' pairs that differ in a key detail.
Run: locust -f loadtest/locustfile.py --headless -u 20 -r 10 -t 2m -H http://localhost:8000
False hit = semantic_hit whose matched group != this request's group."""
import random
from locust import HttpUser, task, between, events

GROUPS = {
 "pw_reset": ["How do I reset my password?", "how can i reset my password", "What's the way to reset my password?"],
 "refund": ["What is your refund policy?", "what's your refund policy", "Can you explain the refund policy?"],
 "hours": ["What are your support hours?", "what are your support hours", "When is support available?"],
 "ship_5": ["Does shipping take 5 days?", "does shipping take 5 days"],
 "ship_7": ["Does shipping take 7 days?", "does shipping take 7 days"],   # trap vs ship_5
 "export": ["How do I export my data to CSV?", "how to export my data to csv"],
 "no_export": ["How do I use the app without exporting data?"],
}
FALSE_HITS = {"n": 0, "semantic": 0}


class U(HttpUser):
    wait_time = between(0.01, 0.1)

    @task
    def ask(self):
        g = random.choice(list(GROUPS)) if random.random() < 0.7 else f"unique_{random.randint(0, 10**6)}"
        q = random.choice(GROUPS[g]) if g in GROUPS else f"Tell me about topic {g}"
        with self.client.post("/v1/chat/completions", name="chat", catch_response=True, json={
            "model": "gpt-4o-mini", "temperature": 0, "messages": [
                {"role": "system", "content": "You are a support bot."}, {"role": "user", "content": q}],
            "metadata": {"feature": "faq", "group": g}}) as r:
            c = r.json()["cache"]
            if c["status"] == "semantic_hit":
                FALSE_HITS["semantic"] += 1
                if c["matched_metadata"].get("group") != g:
                    FALSE_HITS["n"] += 1
                    self.client.post("/v1/cache/report_wrong_hit", json={"entry_id": c["entry_id"]})
                    r.failure("false hit")


@events.quitting.add_listener
def _(environment, **kw):
    s = max(1, FALSE_HITS["semantic"])
    print(f"semantic hits={FALSE_HITS['semantic']} false hits={FALSE_HITS['n']} false-hit rate={FALSE_HITS['n']/s:.2%}")
