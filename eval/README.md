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

---

# 문장 추출 LLM 정답 세트

`rule_input_extract_v2.json`: 작업 문장 → 측정값·작업 종류·감시자 추출(LLM, 시나리오 "B") 평가용 142문항. 학습 데이터 생성기와 따로 직접 쓴 문장이다 (학습에 넣지 않는다).

| 필드 | 뜻 |
| --- | --- |
| `id` | 문항 번호 |
| `tag` | 유형 (기본·여러 값·영문 약어·단위 표기·CO·CO2 구분·측정 아닌 숫자·값 없음·단위 없음·단위 틀림·값 두 번·감시인·작업 종류·STT 출력형·어순·긴 문장 등) |
| `text` | 작업 문장 |
| `workTypes` | 정답 작업 종류 (HOT·CONFINED·HEIGHT). 층수만으로는 고소로 보지 않는다, 높이 2m 이상부터 |
| `measurement` | LLM 이 뽑아야 하는 값 `{항목: 값}`. 단위가 없거나 틀린 값도 적힌 숫자 그대로 |
| `duplicates` | 같은 항목이 두 번 `{항목: [값, 값]}` → 두 값 모두 뽑아야 맞음 |
| `form.watcher_name` | 감시자 이름 (이름이 앞이든 뒤든). 없으면 필드 없음 |
| `spring` | Spring 이 어떻게 처리할지 (단위 없음 → 항목 단위로 채우고 안내, 단위 틀림 → 비움, 값 두 번 → "하나만 적어 주세요"). 채점에는 안 쓴다 |

- 기준 (11/11): LLM 은 적힌 대로 뽑고, 단위·중복 판단은 Spring(`LlmExtractionVerifier`)이 한다.
- 채점: `python -m scripts.eval_rule_input_extract eval/rule_input_extract_v2.json --show`
- 문항을 고치면 `python -m pytest tests/test_extract_gold_v2.py`로 형식을 먼저 검사한다.
- Spring 의 `gold-v1.json`(20문항)은 사전 추출 회귀 테스트용이라 그대로 둔다 (LLM 평가에는 v2 를 쓴다).
