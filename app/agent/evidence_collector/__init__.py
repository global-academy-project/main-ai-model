"""작업 내용 → 근거 수집 (계획 · 검색 · 충분성 판단 단계).

계획서 7장 루프(계획 → 검색 → 충분성 판단 → 판단 → 자기검증 → 출력) 중 앞의 세 단계만 맡는다.
위험요인 판단이나 서류 작성은 하지 않고, 다음 단계가 쓸 "근거 묶음(EvidencePack)"을 만든다.

    작업 내용(WorkContext)
     → ① 계획    planner.py     LLM이 작업·물질·장비를 뽑고 자료 종류별 검색 질의를 만든다 (실패 시 1회 재시도, 그래도 안 되면 규칙 기반 계획)
     → ② 검색    searcher.py    질의마다 A의 검색 도구를 호출한다. 같은 청크는 하나로 합친다
     → ③ 충분성  sufficiency.py 작업유형별 필수 근거 체크리스트(코드)와 비교한다
     → ④ 재질의  sufficiency.py 빠진 항목만 질의를 바꿔 다시 찾는다 (새 근거가 없으면 일찍 멈춘다)
     → EvidencePack (근거 목록 · 항목별 충족 여부 · 끝내 못 찾은 항목 · 검색 실패 기록)

파일 구성
    collector.py   EvidenceCollector (순서와 멈추는 조건)
    planner.py     프롬프트, Plan, Planner
    searcher.py    Searcher, SearchState (검색 도구 규격은 이 파일 맨 위)
    sufficiency.py 필수 근거 확인, 재질의 문장
    vocab.py       작업유형, Corpus, EvidenceRequirement와 기본값
    models.py      WorkContext, EvidencePack 등 입출력 타입
    textutil.py    extract_json_object 등 문자열 도우미
    recorder.py    on_step 콜백 전달

EvidenceCollector는 엔진이 아니다. LLMClient와 ToolInvoker를 생성자로 받아 쓰는 쪽이며, app.rag도 모른다.
규칙 조항 확인은 이 모듈의 일이 아니다 (C의 check_rules). MSDS는 RAG가 아니다 (EvidencePack.materials를 lookup_msds에 넘긴다).
DEFAULT_REQUIREMENTS의 항목·키워드는 임시값이며 안전 전문가·정답 세트로 확정한다.
"""

from .collector import EvidenceCollector
from .models import (
    EvidenceItem,
    EvidencePack,
    RequirementStatus,
    SearchFailure,
    SearchQuery,
    SearchToolOutput,
    StepRecord,
    StepRecorder,
    ToolHit,
    WorkContext,
)
from .planner import Plan, build_planner_input, build_planner_prompt, response_text
from .textutil import extract_json_object
from .vocab import (
    DEFAULT_CORPORA,
    DEFAULT_REQUIREMENTS,
    MAX_PLANNED_QUERIES,
    MAX_QUERY_CHARS,
    MAX_SUMMARY_CHARS,
    WORK_TYPES,
    Corpus,
    EvidenceRequirement,
    normalize_work_type,
)


