
import json, requests
from typing import List, Dict, Any, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def embeddings(self, model: str, inputs: List[str]) -> List[List[float]]:
        try:
            out = self._post("/api/embed", {"model": model, "input": inputs}, timeout=600)
            return out.get("embeddings", [])
        except Exception:
            embs = []
            for s in inputs:
                out = self._post("/api/embeddings", {"model": model, "prompt": s}, timeout=300)
                embs.append(out.get("embedding", []))
            return embs

    def generate(self, model: str, prompt: str, images_b64: Optional[List[str]] = None, timeout: int = 300) -> str:
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if images_b64:
            payload["images"] = images_b64
        out = self._post("/api/generate", payload, timeout=timeout)
        return out.get("response", "")

    def chat(self, model: str, messages: List[Dict[str, str]], images_b64: Optional[List[str]] = None, timeout: int = 300) -> str:
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if images_b64:
            payload["images"] = images_b64
        out = self._post("/api/chat", payload, timeout=timeout)
        return out.get("message", {}).get("content", "")
