import json, urllib.request

def _http_post_json(url, payload, timeout=45):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        txt = resp.read().decode("utf-8","replace")
        try: return json.loads(txt)
        except: return {"raw": txt}

def ollama_generate(host, model, prompt, timeout=45, num_ctx=4096, temperature=0.0):
    url = host.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_ctx": num_ctx}}
    out = _http_post_json(url, payload, timeout=timeout)
    return out.get("response","") or out.get("raw","") or ""
