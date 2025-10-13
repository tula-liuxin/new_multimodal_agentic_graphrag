import re
def normalize_zh(text: str) -> str:
    """
    简繁归一 + 标点归一 + 拉丁字母小写。若本机无 opencc，退化为仅标点/大小写归一。
    """
    if not text:
        return ""
    t = text
    # 半角转全角/常见标点统一：这里只做最必要，避免引入重依赖
    t = t.replace("，", ",").replace("。", ".").replace("；", ";").replace("：", ":")
    t = t.replace("？", "?").replace("！", "!").replace("（", "(").replace("）", ")")
    # 小写化 ASCII
    t = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), t)
    # 简繁转换（可选）
    try:
        from opencc import OpenCC
        cc = OpenCC('t2s')
        t = cc.convert(t)
    except Exception:
        pass
    return t

def is_cjk_name(q: str) -> bool:
    # 简单判断：2-4 连续汉字且不含空格/数字
    return bool(re.fullmatch(r"[一-龥]{2,4}", q or ""))
