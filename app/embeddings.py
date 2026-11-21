"""Pluggable embedders. EMBEDDER=hash (offline, demo only) | st | openai"""
import hashlib, os
import numpy as np


class HashEmbedder:
    """Char-trigram hashing. Offline and deterministic; catches near-duplicates, not true paraphrases."""
    dim = 512

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = f"  {text.lower()}  "
        for i in range(len(t) - 2):
            h = int(hashlib.md5(t[i:i + 3].encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


class STEmbedder:
    def __init__(self, name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def embed(self, text: str) -> np.ndarray:
        return self.m.encode(text, normalize_embeddings=True).astype(np.float32)


class OpenAIEmbedder:
    def __init__(self):
        import httpx
        self.c = httpx.Client(base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                              headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, timeout=30)

    def embed(self, text: str) -> np.ndarray:
        r = self.c.post("/embeddings", json={"model": "text-embedding-3-small", "input": text})
        r.raise_for_status()
        v = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
        return v / np.linalg.norm(v)


def get_embedder():
    return {"hash": HashEmbedder, "st": STEmbedder, "openai": OpenAIEmbedder}[os.getenv("EMBEDDER", "hash")]()
