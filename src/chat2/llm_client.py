import json, urllib.request
from typing import Dict, Any

def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 45) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        txt = resp.read().decode("utf-8", "replace")
        try:
            return json.loads(txt)
        except Exception:
            return {"raw": txt}

def ollama_generate(host: str, model: str, prompt: str, timeout: int = 45, num_ctx: int = 4096, temperature: float = 0.0) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_ctx": num_ctx}}
    out = _http_post_json(url, payload, timeout=timeout)
    return out.get("response", "") or out.get("raw", "") or ""
