"""检索结果规整与分支内RRF工具。"""

from typing import Any, Iterable


CHUNK_OUTPUT_FIELDS = [
    "chunk_id", "content", "theme_name", "title", "doc_id",
    "canonical_title", "primary_subject", "document_type",
    "document_summary", "parent_title", "file_title", "section_id",
    "section_path", "parent_summary", "chunk_index",
    "section_chunk_index",
]


def hit_to_chunk(hit: Any) -> dict[str, Any]:
    getter = hit.get if hasattr(hit, "get") else lambda _key: None
    hit_id = getattr(hit, "id", None)
    if isinstance(hit, dict):
        hit_id = hit.get("id", hit_id)
    distance = getattr(hit, "distance", None)
    if isinstance(hit, dict):
        distance = hit.get("distance", hit.get("score", distance))
    chunk_id = getter("chunk_id") or hit_id
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "score": float(distance) if isinstance(distance, (int, float)) else None,
        **{field: getter(field) for field in CHUNK_OUTPUT_FIELDS if field != "chunk_id"},
    }


def merge_ranked_chunk_lists(
        ranked_lists: Iterable[tuple[str, list[dict[str, Any]], float]],
        limit: int,
        rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """按chunk_id融合多个排名列表，并保留各列表名次。"""
    scores: dict[Any, float] = {}
    documents: dict[Any, dict[str, Any]] = {}
    for list_name, chunks, weight in ranked_lists:
        for rank, chunk in enumerate(chunks or [], start=1):
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id is None:
                continue
            if chunk_id not in documents:
                documents[chunk_id] = dict(chunk)
            else:
                for key, value in chunk.items():
                    if documents[chunk_id].get(key) in (None, "", []):
                        documents[chunk_id][key] = value
            documents[chunk_id].setdefault("branch_ranks", {})[list_name] = rank
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)

    ordered = sorted(scores, key=lambda item: scores[item], reverse=True)
    if limit > 0:
        ordered = ordered[:limit]
    return [
        {**documents[chunk_id], "branch_fusion_score": scores[chunk_id]}
        for chunk_id in ordered
    ]
