# eval/ — 검색 정답 세트

`retrieval_law_v0.jsonl`: 법령(산업안전보건기준에 관한 규칙) 검색 평가용 질문 46개. 한 줄에 질문 하나.

| 필드 | 뜻 |
| --- | --- |
| `id` | 질문 번호 (L=사람 문장, C=현장 구어체, A=에이전트가 실제로 만드는 질의형) |
| `kind` | `human` / `colloquial` / `agent` |
| `category` | 화기·밀폐공간·고소·전기·양중·기계·굴착 |
| `query` | 검색 문장 |
| `expect_articles` | 정답 조문. **이 중 하나라도** 검색 결과에 나오면 맞은 것 (예: `제241조의2`) |
| `confidence` | 기대 조문 확신도 high/medium/low (low는 법령 용어와 현장 용어가 어긋나 정답 자체가 애매한 질문) |
| `status` | `candidate` = 안전 전문가 확인 전. 확인되면 `confirmed`로 바꾼다 |
| `note` | 근거나 주의 |

- `agent` 질문 7개는 `app/agent/evidence_collector/vocab.py`의 `DEFAULT_REQUIREMENTS`(guide 자료) 재질의 문장 그대로다. 실제 LLM이 만드는 계획 질의는 로그에서 뽑아 `A08`부터 추가한다.
- 질문을 추가·수정하면 `python -m scripts.eval_retrieval --validate`로 형식과 기대 조문 존재를 먼저 검사한다.
- 기대 조문을 바꾼 뒤에는 이전 결과와 직접 비교하지 말고 기준선을 다시 잰다.
