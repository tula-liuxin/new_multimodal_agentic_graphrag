# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time
from typing import Any, Dict, List, Optional
import requests

def _normalize_host(h: Optional[str]) -> str:
    if not h:
        return "http://127.0.0.1:11434"
    h = h.strip()
    if not h.startswith("http://") and not h.startswith("https://"):
        h = "http://" + h
    return h.rstrip("/")

class OllamaClient:
    """
    轻量 Ollama HTTP 封装：
    - 禁用系统代理避免把本机流量劫持到 7897/8889 等端口
    - 多 host 兜底：优先 CLI/配置 > 环境 OLLAMA_HOST > 127.0.0.1:11434 > localhost:11434
    - 合理默认 options：num_predict 限制、temperature 降低、keep_alive 维持热启动
    - 每次调用前做一次 /api/version 轻探测（非致命）
    - 支持 embeddings 分批提交
    """
    def __init__(self, host: Optional[str] = None, timeout: int = 60, retries: int = 1, logger=None):
        self.logger = logger
        self.timeout = int(timeout)
        self.retries = max(0, int(retries))
        # 禁用代理
        self.sess = requests.Session()
        self.sess.trust_env = False
        self.sess.proxies.update({"http": None, "https": None})
        # host candidates
        candidates = []
        if host:
            candidates.append(_normalize_host(host))
        env_h = os.environ.get("OLLAMA_HOST")
        if env_h:
            candidates.append(_normalize_host(env_h))
        candidates.extend(["http://127.0.0.1:11434", "http://localhost:11434"])
        # 去重保序
        seen = set(); self.hosts = []
        for h in candidates:
            if h not in seen:
                self.hosts.append(h); seen.add(h)

    def _pick_host(self) -> str:
        return self.hosts[0]

    def _rotate_host(self):
        if len(self.hosts) > 1:
            self.hosts = self.hosts[1:] + self.hosts[:1]

    def _ping(self, host: str):
        try:
            self.sess.get(host + "/api/version", timeout=5)
        except Exception:
            pass

    def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err = None
        for attempt in range(self.retries + 1):
            host = self._pick_host()
            try:
                self._ping(host)
                url = host + path
                resp = self.sess.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                if self.logger:
                    try:
                        self.logger.warning(f"Ollama 调用失败({host+path})：{e}")
                    except Exception:
                        pass
                self._rotate_host()
                time.sleep(min(1 + attempt, 3))
        raise last_err if last_err else RuntimeError("Ollama 调用失败")

    def generate(self, model: str, prompt: str, images=None,
                 stream: bool = False, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        opts = dict(options or {})
        # 合理默认：限制输出、降低温度、保持热启动
        opts.setdefault("num_predict", 256)
        opts.setdefault("temperature", 0.1)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": bool(stream),
            "options": opts,
            "keep_alive": "15m",
        }
        if images:
            payload["images"] = images
        return self._request("/api/generate", payload)

    def embeddings(self, model: str, texts) -> list:
        # texts: List[str]
        # 逐批避免过长请求
        if not texts:
            return []
        all_vecs = []
        bs = 32
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            payload = {"model": model, "input": chunk, "keep_alive": "15m"}
            jr = self._request("/api/embeddings", payload)
            vecs = [d.get("embedding") for d in jr.get("data", []) if "embedding" in d]
            all_vecs.extend(vecs)
        return all_vecs
