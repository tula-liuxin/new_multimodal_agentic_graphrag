def rewrite_with_coref(question, parsed_coref, fallback_entities):
    if not parsed_coref: return question
    pron=parsed_coref.get("pronoun") or ""
    tgt=parsed_coref.get("target"); tgts=parsed_coref.get("targets")
    q=question
    if isinstance(tgts,list) and len(tgts)>=2 and pron:
        rep=f"{tgts[0]}和{tgts[1]}"; return q.replace(pron,rep,1)+f"（注：{pron}指代：{rep}）"
    if tgt and pron: return q.replace(pron,tgt,1)+f"（注：{pron}指代：{tgt}）"
    if not tgts and len(fallback_entities)>=2 and pron:
        rep=f"{fallback_entities[-2]}和{fallback_entities[-1]}"; return q.replace(pron,rep,1)+f"（注：{pron}指代：{rep}）"
    return question
