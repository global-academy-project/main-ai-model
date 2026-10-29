"""검색 도구 출력이 규격(지침 6.4)을 지키는지 검사한다. A가 자기 테스트에서 그대로 쓸 수 있다.

    problems = check_search_output(output, k=5)      # 빈 리스트면 통과
    assert not problems, problems

mock이든 실제 RAG든 같은 검사를 통과해야 에이전트 코드는 바뀌지 않는다.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

HIT_FIELDS = ("id", "doc_id", "source", "text", "score")


def check_search_output(output: Any, *, k: int | None = None, min_score: float | None = None) -> list[str]:
    if not isinstance(output, Mapping) or not isinstance(output.get("hits"), list):
        return ["출력이 {'hits': [...]} 형태가 아니다"]
    problems: list[str] = []
    hits = output["hits"]
    if k is not None and len(hits) > k:
        problems.append(f"hits가 k={k}보다 많다 ({len(hits)}개)")
    scores: list[float] = []
    for n, hit in enumerate(hits):
        if not isinstance(hit, Mapping):
            problems.append(f"hits[{n}]이 객체가 아니다")
            continue
        for name in HIT_FIELDS:
            if hit.get(name) is None:
                problems.append(f"hits[{n}].{name}이 없다")
        for name in ("id", "doc_id", "source", "text"):
            if hit.get(name) is not None and not isinstance(hit[name], str):
                problems.append(f"hits[{n}].{name}이 문자열이 아니다")
        if isinstance(hit.get("id"), str):
            try:
                uuid.UUID(hit["id"])
            except ValueError:
                problems.append(f"hits[{n}].id가 uuid가 아니다 ({hit['id']!r})")
        score = hit.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            if score is not None:
                problems.append(f"hits[{n}].score가 숫자가 아니다")
        else:
            scores.append(float(score))
            if min_score is not None and score < min_score:
                problems.append(f"hits[{n}].score={score}가 min_score={min_score} 미만이다")
    if scores != sorted(scores, reverse=True):
        problems.append("점수 높은 순이 아니다")
    return problems
