"""查询流程状态类型定义

定义完整的查询状态结构和辅助函数。
"""

from typing import TypedDict, List, Literal
import copy


class QueryGraphState(TypedDict):
    """
    Represents the state of our query graph.
    Attributes:
    各个属性的结构
    """
    session_id: str # 会话ID
    task_id:str # 任务ID
    original_query: str # 原始查询
    display_query: str # 会话中展示和持久化的用户查询
    query_type: Literal["text", "image", "multimodal"] # 查询类型
    retrieval_query: str # 文本或图片经语义增强后用于检索的查询
    query_image_bytes: bytes # 用户上传的查询图片
    query_image_mime_type: str # 查询图片的 MIME 类型
    image_query_description: str # VLM 生成的图片语义描述
    embedding_chunks: list # 已向量化的切片
    hyde_embedding_chunks: list # 已向量化的假设性问题切片
    bm25_chunks: list # BM25关键词检索切片
    rrf_chunks: list # rrf排序后的切片
    rerank_candidates: list  # BGE-Reranker 完整排序候选
    reranked_docs: list  # 排序后的文档
    conflict_judge_triggered: bool  # 是否触发选择性 LLM 仲裁
    conflict_judge_decisions: list  # 冲突候选仲裁记录
    expanded_docs: list  # 基于Rerank锚点扩展后的层级上下文
    answer: str #答案
    theme_names: List[str] # 兼容 Milvus 现有 theme_name 字段
    document_mentions: List[str]  # 用户明确提到的文档/实体/型号
    document_candidates: list  # 文档注册表召回的候选
    selected_documents: list  # 已精确确认的文档档案
    hard_filter_doc_ids: List[str]  # 只有精确唯一匹配时才设置
    soft_filter_doc_ids: List[str]  # 多候选软路由，仅提升优先级
    document_route_mode: Literal["global", "soft", "hard"]
    vector_route_fallback: bool
    hyde_route_fallback: bool
    bm25_route_fallback: bool
    rewritten_query: str  #重写答案
    retrieval_queries: List[str]  # 原问题及比较/多事实子问题
    query_decomposed: bool
    history: list   # 历史对话
    is_stream: bool # 是否流式输出


# ==================== 默认状态 ====================

DEFAULT_STATE: QueryGraphState = {
    "session_id": "",               # 会话ID
    "task_id": "",               # 任务ID
    "original_query": "",           # 原始查询
    "display_query": "",            # 会话展示文本
    "query_type": "text",           # 查询类型
    "retrieval_query": "",          # 用于主题识别与检索的查询
    "query_image_bytes": b"",       # 查询图片原始数据
    "query_image_mime_type": "",    # 查询图片 MIME 类型
    "image_query_description": "",  # 图片语义描述
    "embedding_chunks": [],         # 已向量化的切片
    "hyde_embedding_chunks": [],    # 已向量化的假设性问题切片
    "bm25_chunks": [],              # BM25关键词检索切片
    "rrf_chunks": [],               # rrf排序后的切片
    "rerank_candidates": [],        # BGE-Reranker 完整排序候选
    "reranked_docs": [],            # 排序后的文档
    "conflict_judge_triggered": False,
    "conflict_judge_decisions": [],
    "expanded_docs": [],            # 层级上下文扩展结果
    "answer": "",                   # 答案
    "theme_names": [],               # 兼容 Milvus 现有字段
    "document_mentions": [],        # 明确文档/实体/型号表达
    "document_candidates": [],      # 软路由候选
    "selected_documents": [],      # 唯一精确文档
    "hard_filter_doc_ids": [],      # 硬过滤doc_id
    "soft_filter_doc_ids": [],      # 软路由候选doc_id
    "document_route_mode": "global",  # 默认全库检索
    "vector_route_fallback": False,
    "hyde_route_fallback": False,
    "bm25_route_fallback": False,
    "rewritten_query": "",          # 重写查询
    "retrieval_queries": [],         # 原问题及子问题
    "query_decomposed": False,
    "history": [],                  # 历史对话
    "is_stream": False,             # 是否流式输出 (默认设为 False)
}

def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖。

    Args:
        **overrides: 要覆盖的字段键值对。

    Returns:
        新的状态实例，包含默认值和覆盖值。

    """
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state
