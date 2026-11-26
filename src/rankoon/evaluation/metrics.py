"""
Метрика    / Что измеряет                                                  / Где используем
NDCG@10	   / Качество порядка ранжирования с учётом степени релевантности  / Основная метрика качества
Recall@100 / Долю известных релевантных товаров, найденных поиском	       / Retrieval
MRR@10	   / Насколько высоко расположен первый релевантный товар          / Ranking
Judged@10  / Долю результатов, для которых существует ручная оценка        / Диагностика полноты разметки
"""
from collections.abc import Sequence, Mapping
from math import log2


def dcg_at_k(
    relevances: Sequence[int],
    k: int,
) -> float:
    """Compute discounted cumulative gain at K."""
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if any(rel not in (0, 1, 2) for rel in relevances):
        raise ValueError("Relevances must be 0, 1 or 2")
    
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
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked_ids must be unique")
    if any(rel not in (0, 1, 2) for rel in judgments.values()):
            raise ValueError("judgments.relevants must be 0, 1 or 2")
    
    relevances = [judgments.get(ranked_id, 0) for ranked_id in ranked_ids]
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(judgments.values(), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0.0: return 0.0
    
    return dcg / idcg