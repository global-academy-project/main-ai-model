import json

from app.agent.audit_report import AuditReportAgent
from app.llm.llm_base import LLMClient
from app.llm.llm_types import LLMResponse
from app.schemas.audit_report import AuditReportInput


class FakeAuditLLM(LLMClient):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        return LLMResponse(text=self.text)


def data():
    return AuditReportInput.model_validate({
        "reportId": 1001,
        "siteId": 1,
        "dateFrom": "2026-11-01",
        "dateTo": "2026-11-13",
        "items": [{
            "queryType": "risk_assessment_rate",
            "title": "위험성평가 실시율",
            "stats": {"total": 20, "completed": 18, "rate": 90.0},
            "evidence": [{"type": "RISK", "id": 7, "label": "위험성평가 #7"}],
        }],
    })


def test_audit_report_matches_stats_numbers():
    llm = FakeAuditLLM(json.dumps({"items": [{
        "queryType": "risk_assessment_rate",
        "summary": "위험성평가는 총 20건 중 18건이 실시되어 실시율은 90.0%입니다.",
    }]}, ensure_ascii=False))
    result = AuditReportAgent(llm=llm).run(data())
    assert result.overall_verification == "PASS"
    assert result.status == "REVIEW_REQUIRED"


def test_audit_report_rejects_mismatched_summary_number():
    llm = FakeAuditLLM(json.dumps({"items": [{
        "queryType": "risk_assessment_rate",
        "summary": "위험성평가는 총 20건 중 17건이 실시되어 실시율은 85.0%입니다.",
    }]}, ensure_ascii=False))
    result = AuditReportAgent(llm=llm).run(data())
    assert result.overall_verification == "FAIL"
    assert result.status == "FAILED"
    assert result.items[0].verification.status == "MISMATCH"


def test_audit_report_requires_all_input_items():
    llm = FakeAuditLLM(json.dumps({"items": []}))
    result = AuditReportAgent(llm=llm).run(data())
    assert result.overall_verification == "FAIL"
