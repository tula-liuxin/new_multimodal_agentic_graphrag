from typing import List, Dict, Any

def pronouns_from_cfg(cfg) -> List[str]:
    arr = cfg.get("coref_pronouns") or ["她","他","TA","ta","她们","他们","两者","两人","他们两个"]
    return arr

def contains_any(q: str, arr: List[str]) -> bool:
    return any(p in q for p in arr)

def rewrite_with_coref(question: str, parsed_coref: Dict[str, Any], fallback_entities: List[str]) -> str:
    if not parsed_coref:
        return question
    pron = parsed_coref.get("pronoun") or ""
    tgt = parsed_coref.get("target")
    tgts = parsed_coref.get("targets")
    q = question
    if isinstance(tgts, list) and len(tgts) >= 2 and pron:
        rep = f"{tgts[0]}和{tgts[1]}"
        return q.replace(pron, rep, 1) + f"（注：{pron}指代：{rep}）"
    if tgt and pron:
        return q.replace(pron, tgt, 1) + f"（注：{pron}指代：{tgt}）"
    if not tgts and len(fallback_entities) >= 2 and pron:
        rep = f"{fallback_entities[-2]}和{fallback_entities[-1]}"
        return q.replace(pron, rep, 1) + f"（注：{pron}指代：{rep}）"
    return question
