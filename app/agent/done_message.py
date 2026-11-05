"""되묻기 항목이 모두 채워져 초안이 완료됐을 때 ask.message에 넣는 완료 문구 한 줄.

초안은 사용자에게 확정을 묻는 물음이기도 해서 DONE 응답도 ask(message, missing_fields=[])로 돌려준다.
문구는 초안에서 센 개수로 코드가 만든다 (LLM은 쓰지 않는다). 승인 차단 사유·확인 목록은 문구에 넣지 않는다:
approvalGate.reasons와 verification에 이미 있다.
mode가 GUIDE(시나리오 F)면 쉬운 말 문구를 쓴다.
"""

from __future__ import annotations

from app.schemas.agent_work import Drafts


def build_done_message(drafts: Drafts, mode: str) -> str:
    parts = [f"위험성평가 위험요인 {len(drafts.risk_assessment.items)}건"]
    if drafts.permits:
        parts.append(f"작업허가서 {len(drafts.permits)}건({', '.join(p.title for p in drafts.permits)})")
    parts.append(f"점검표 {len(drafts.inspection_checklist.items)}항목")
    parts.append(f"TBM 교육 요점 {len(drafts.tbm.points)}개")
    head = "오늘 작업에 필요한 서류 초안을 만들었습니다." if mode == "GUIDE" else "초안을 만들었습니다."
    return f"{head} {', '.join(parts)}."
