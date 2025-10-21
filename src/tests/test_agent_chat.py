# 轻量占位测试：确保模块可导入，主入口可被发现
def test_imports():
    import src.agent_chat as agent_chat
    import src.intent_router as intent_router
    import src.chat_record_manager as crm
    import src.rag_b as ragb
    assert hasattr(agent_chat, "main")
    assert hasattr(intent_router, "route_intent")
    assert hasattr(crm, "start_chat_session")
    assert hasattr(ragb, "RAGBIndex")
