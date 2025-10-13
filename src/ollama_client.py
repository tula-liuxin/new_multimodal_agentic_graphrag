import json
import requests
from typing import List, Any
from .utils import get_logger

class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 120, keep_alive: str = "10m"):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.logger = get_logger()

    def embeddings(self, model: str, inputs: List[str]) -> List[List[float]]:
        """
        批量嵌入：优先使用 /api/embed（字段 input，支持 list），失败则回退到旧接口 /api/embeddings（逐条 prompt）。
        """
        # Try new endpoint: /api/embed
        url_embed = f"{self.base}/api/embed"
        payload = {
            "model": model,
            "input": inputs if isinstance(inputs, list) else [inputs],
            "keep_alive": self.keep_alive,
        }
        try:
            r = self.session.post(url_embed, data=json.dumps(payload), timeout=self.timeout)
            if r.ok:
                data = r.json()
                if isinstance(data.get("embeddings"), list):
                    return data["embeddings"]
        except Exception as e:
            self.logger.debug(f"/api/embed 调用失败，将回退 /api/embeddings：{e}")

        # Fallback: /api/embeddings (single prompt per call)
        url_old = f"{self.base}/api/embeddings"
        outputs: List[List[float]] = []
        batch = inputs if isinstance(inputs, list) else [inputs]
        for x in batch:
            rp = {"model": model, "prompt": x, "keep_alive": self.keep_alive}
            r = self.session.post(url_old, data=json.dumps(rp), timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data.get("embedding"), list):
                outputs.append(data["embedding"])
            elif "data" in data and isinstance(data["data"], list) and data["data"] and "embedding" in data["data"][0]:
                outputs.append(data["data"][0]["embedding"])
            else:
                raise RuntimeError("未知的 /api/embeddings 返回格式")
        return outputs

    def generate(self, model: str, prompt: str, temperature: float = 0.3, system: str | None = None) -> str:
        url = f"{self.base}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        r = self.session.post(url, data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
