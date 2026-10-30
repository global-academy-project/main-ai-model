"""작업 등록 에이전트 API 요청·응답 (POST /api/v1/agent/work-register).

요청 (request.py)
  WorkRegisterRequest
    message, mode, runId, siteId, permitId
    slots: WorkSlots ─ 되묻기 항목. 응답의 slots를 다음 요청에 그대로 다시 보낸다

응답 (result.py)
  WorkRegisterResult
    runStatus        WAITING_INPUT | DONE
    slots            WorkSlots                    request.py
    ask              AskBack → MissingField[]      result.py
    classification   Classification               classification.py
    evidence         EvidenceSummary              evidence.py
    msds             MsdsDto[]                    evidence.py
    ruleCheck        RuleCheckDto                 rules.py
    drafts           Drafts → forms.py 양식 4종    result.py
    citations        Citation[]                   forms.py
    approvalGate     ApprovalGate                 result.py
    verification     VerificationIssue[]          result.py
    guidance         GuidanceNote[]               result.py
    llmSources       LlmSources                   result.py
    steps            AgentStepDto[]               steps.py 
    """