
import json
import base64
import concurrent.futures as cf
from typing import List, Dict, Any, Tuple
import requests

def _ollama_session(base_url: str, timeout: int = 15):
    s = requests.Session()
    s.headers.update({"Content-Type":"application/json"})
    s._ollama_base = base_url.rstrip("/")
    s._timeout = timeout
    return s

def _judge_one(session, model: str, question: str, text: str, temperature: float=0.0) -> bool:
    """Return True if related. Prompt forces YES/NO."""
    prompt = f"""You are a precise filter. Decide if the PASSAGE contains information related to the QUESTION.
Answer strictly with YES or NO.

QUESTION:
{question}

PASSAGE:
{text}
"""
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    try:
        r = session.post(session._ollama_base + "/api/generate", data=json.dumps(payload), timeout=session._timeout)
        r.raise_for_status()
        out = r.json().get("response","").strip().lower()
        return out.startswith("y")  # YES
    except Exception:
        return False

def judge_batch(base_url: str, model: str, question: str, passages: List[str], workers: int=6, timeout: int=15) -> List[bool]:
    session = _ollama_session(base_url, timeout=timeout)
    results = [False]*len(passages)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = { ex.submit(_judge_one, session, model, question, p): i for i,p in enumerate(passages) }
        for f in cf.as_completed(futs):
            i = futs[f]
            try:
                results[i]=bool(f.result())
            except Exception:
                results[i]=False
    return results
