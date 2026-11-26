"""
Метрика    / Что измеряет                                                  / Где используем
NDCG@10	   / Качество порядка ранжирования с учётом степени релевантности  / Основная метрика качества
Recall@100 / Долю известных релевантных товаров, найденных поиском	       / Retrieval
MRR@10	   / Насколько высоко расположен первый релевантный товар          / Ranking
Judged@10  / Долю результатов, для которых существует ручная оценка        / Диагностика полноты разметки
"""
from collections.abc import Sequence, Mapping, Iterable
from math import log2

def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _validate_relevances(
    relevances: Iterable[int],
) -> None:
    if any(rel not in (0, 1, 2) or not isinstance(rel, int) for rel in relevances):
        raise ValueError("Relevances must be 0, 1 or 2")


def _validate_min_relevance(
    min_relevance: int,
) -> None:
    if not isinstance(min_relevance, int) or min_relevance not in (1, 2):
        raise ValueError("min_relevance must be 1 or 2")

def _validate_ranking(
    ranked_ids: Sequence[int],
    judgments: Mapping[int, int],
    k: int,
) -> None:
    _validate_k(k)
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked_ids must be unique")
    _validate_relevances(judgments.values())

def dcg_at_k(
    relevances: Sequence[int],
    k: int,
) -> float:
    """Compute discounted cumulative gain at K."""
    _validate_k(k)
    _validate_relevances(relevances)
    
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += (2**rel - 1) / (log2(i + 1))
        
    return dcg

def ndcg_at_k(
    ranked_ids: Sequence[int],
    judgments: Mapping[int, int],
    k: int,
) -> float:
    """Compute normalized discounted cumulative gain at K."""
    _validate_ranking(ranked_ids, judgments, k)
    
    relevances = [judgments.get(ranked_id, 0) for ranked_id in ranked_ids]
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(judgments.values(), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0.0: return 0.0
    
    return dcg / idcg


def recall_at_k(
    ranked_ids: Sequence[int],
    judgments: Mapping[int, int],
    k: int,
    min_relevance: int = 1,
) -> float:
    """Compute recall over known relevant products at K."""
    _validate_ranking(ranked_ids, judgments, k)
    _validate_min_relevance(min_relevance)
    
    count_ranked_relevances = sum(judgments.get(ranked_id, 0) >= min_relevance
                                  for ranked_id in ranked_ids[:k])
    count_all_relevances = sum(rel >= min_relevance
                               for rel in judgments.values())
    if count_all_relevances == 0: return 0.0
    
    return count_ranked_relevances / count_all_relevances


def reciprocal_rank_at_k(
    ranked_ids: Sequence[int],
    judgments: Mapping[int, int],
    k: int,
    min_relevance: int = 1,
) -> float:
    """Compute reciprocal rank of the first relevant product at K."""
    _validate_ranking(ranked_ids, judgments, k)
    _validate_min_relevance(min_relevance)
        
    for rank, ranked_id in enumerate(ranked_ids[:k], start=1):
        if judgments.get(ranked_id, 0) >= min_relevance:
            return 1 / rank
        
    return 0.0


def judged_at_k(
    ranked_ids: Sequence[int],
    judgments: Mapping[int, int],
    k: int,
) -> float:
    """Compute the fraction of top-K slots containing judged products."""
    _validate_ranking(ranked_ids, judgments, k)
    judged = sum(
        ranked_id in judgments
        for ranked_id in ranked_ids[:k]
    )
    return judged / k