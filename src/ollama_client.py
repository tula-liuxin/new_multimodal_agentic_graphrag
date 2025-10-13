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
        \"\"\"批量嵌入：优先使用 /api/embed (input 支持 list)，失败再回退 /api/embeddings。\"\"\"
        url_embed = f"{self.base}/api/embed"
        payload = {"model": model, "input": inputs if isinstance(inputs, list) else [inputs], "keep_alive": self.keep_alive}
        try:
            r = self.session.post(url_embed, data=json.dumps(payload), timeout=self.timeout)
            if r.status_code < 300:
                data = r.json()
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    return data["embeddings"]
        except Exception as e:
            self.logger.debug(f"/api/embed 调用失败，将回退 /api/embeddings：{e}")
        # 回退逐条
        url_old = f"{self.base}/api/embeddings"
        out = []
        for x in (inputs if isinstance(inputs, list) else [inputs]):
            payload = {"model": model, "prompt": x, "keep_alive": self.keep_alive}
            r = self.session.post(url_old, data=json.dumps(payload), timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data.get("embedding"), list):
                out.append(data["embedding"])
            elif "data" in data and isinstance(data["data"], list) and len(data["data"]) and "embedding" in data["data"][0]:
                out.append(data["data"][0]["embedding"])
            else:
                raise RuntimeError("未知的 /api/embeddings 返回格式")
        return out

    def generate(self, model: str, prompt: str, temperature: float=0.2, system: str=None) -> str:
        url = f"{self.base}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": self.keep_alive, "options":{"temperature":temperature}}
        if system:
            payload["system"] = system
        r = self.session.post(url, data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response","")
