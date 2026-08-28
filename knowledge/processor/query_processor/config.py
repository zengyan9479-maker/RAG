"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。所有属性均采用懒加载模式。
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from knowledge.core.settings import get_settings

_app_settings = get_settings()


@dataclass
class QueryConfig:
    """查询流程配置。"""

    # ==================== 文本处理配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )
    max_query_image_bytes: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_QUERY_IMAGE_BYTES", str(10 * 1024 * 1024))
        )
    )
    vl_model: str = field(
        default_factory=lambda: _app_settings.vl_model
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "5"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.08"))
    )
    rerank_coverage_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "RERANK_COVERAGE_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
    )
    rerank_coverage_query_rank_limit: int = field(
        default_factory=lambda: max(
            1,
            int(os.getenv("RERANK_COVERAGE_QUERY_RANK_LIMIT", "5")),
        )
    )
    rerank_coverage_min_score: float = field(
        default_factory=lambda: float(
            os.getenv("RERANK_COVERAGE_MIN_SCORE", "0.10")
        )
    )

    # ==================== 选择性 LLM 冲突仲裁 ====================
    conflict_judge_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CONFLICT_JUDGE_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
    )
    conflict_judge_rrf_rank_limit: int = field(
        default_factory=lambda: max(
            1,
            int(os.getenv("CONFLICT_JUDGE_RRF_RANK_LIMIT", "5")),
        )
    )
    conflict_judge_max_pairs: int = field(
        default_factory=lambda: max(
            0,
            int(os.getenv("CONFLICT_JUDGE_MAX_PAIRS", "2")),
        )
    )
    conflict_judge_min_confidence: float = field(
        default_factory=lambda: min(
            1.0,
            max(
                0.0,
                float(os.getenv("CONFLICT_JUDGE_MIN_CONFIDENCE", "0.80")),
            ),
        )
    )
    conflict_judge_document_max_chars: int = field(
        default_factory=lambda: max(
            200,
            int(os.getenv("CONFLICT_JUDGE_DOCUMENT_MAX_CHARS", "1800")),
        )
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "20"))
    )
    rrf_direct_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_DIRECT_WEIGHT", "0.50"))
    )
    rrf_hyde_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_HYDE_WEIGHT", "0.30"))
    )
    rrf_bm25_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_BM25_WEIGHT", "0.20"))
    )
    rrf_min_exclusive_per_branch: int = field(
        default_factory=lambda: max(
            0, int(os.getenv("RRF_MIN_EXCLUSIVE_PER_BRANCH", "0"))
        )
    )

    # ==================== 层级上下文扩展 ====================
    context_expansion_top_k: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_EXPANSION_TOP_K", "3"))
    )
    context_expansion_window: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_EXPANSION_WINDOW", "1"))
    )
    context_expansion_max_chars: int = field(
        default_factory=lambda: int(
            os.getenv("CONTEXT_EXPANSION_MAX_CHARS", "4000")
        )
    )

    # ==================== 检索配置 ====================
    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "12"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "12"))
    )
    bm25_search_limit: int = field(
        default_factory=lambda: int(os.getenv("BM25_SEARCH_LIMIT", "12"))
    )
    hybrid_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("HYBRID_DENSE_WEIGHT", "0.5"))
    )
    hybrid_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.5"))
    )
    hyde_use_sparse: bool = field(
        default_factory=lambda: os.getenv(
            "HYDE_USE_SPARSE", "false"
        ).lower() in ("true", "1", "yes")
    )
    soft_route_weight: float = field(
        default_factory=lambda: float(os.getenv("SOFT_ROUTE_WEIGHT", "0.35"))
    )

    # ==================== 文档路由 ====================
    document_route_limit: int = field(
        default_factory=lambda: int(os.getenv("DOCUMENT_ROUTE_LIMIT", "10"))
    )
    document_route_max_options: int = field(
        default_factory=lambda: int(
            os.getenv("DOCUMENT_ROUTE_MAX_OPTIONS", "3")
        )
    )
    document_registry_scan_limit: int = field(
        default_factory=lambda: int(
            os.getenv("DOCUMENT_REGISTRY_SCAN_LIMIT", "1000")
        )
    )

    # ==================== Milvus 配置 ====================
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    bm25_collection: str = field(
        default_factory=lambda: (
            os.getenv("BM25_COLLECTION")
            or os.getenv("CHUNKS_COLLECTION", "")
        )
    )
    bm25_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "BM25_ENABLED", "false"
        ).lower() in ("true", "1", "yes")
    )
    document_registry_collection: str = field(
        default_factory=lambda: (
            os.getenv("DOCUMENT_REGISTRY_COLLECTION")
            or "kb_document_registry_v1"
        )
    )
    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。

        Returns:
            配置实例。
        """
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。

    Returns:
        全局配置实例。
    """
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config
