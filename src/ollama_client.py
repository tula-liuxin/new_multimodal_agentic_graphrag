import requests, time, json, threading
from typing import List, Dict, Any
from .utils import get_logger

class OllamaClient:
    def __init__(self, base_url: str, timeout: int=120, keep_alive: str="10m"):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type":"application/json"})
        self.keep_alive = keep_alive
        self.logger = get_logger()

    def embeddings(self, model: str, inputs: List[str]) -> List[List[float]]:
        # 批量调用 /api/embeddings
        url = f"{self.base}/api/embeddings"
        payload = {"model": model, "prompt": inputs, "keep_alive": self.keep_alive}
        r = self.session.post(url, data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # 兼容数组/单条返回
        if isinstance(data.get("embedding"), list) and all(isinstance(x, (int,float)) for x in data["embedding"]):
            return [data["embedding"]]
        if "embeddings" in data:
            return data["embeddings"]
        if "data" in data and isinstance(data["data"], list):
            return [row.get("embedding", []) for row in data["data"]]
        return []

    def generate(self, model: str, prompt: str, temperature: float=0.3, system: str=None) -> str:
        url = f"{self.base}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": self.keep_alive, "options":{"temperature":temperature}}
        if system:
            payload["system"] = system
        r = self.session.post(url, data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response","")
