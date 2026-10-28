"""[연습용 mock] 검색 도구가 찾을 가짜 문서. A의 실제 RAG가 들어오면 이 파일은 쓰지 않는다.

내용은 공공 자료를 흉내 낸 연습용 문장이며 실제 재해조사보고서·KOSHA GUIDE 원문이 아니다.
수치 기준 중 밀폐공간 적정공기(산소 18% 이상 23.5% 미만 등)만 업무 DB 규칙 시드(산안규칙 제618조)와 같고,
나머지 수치·문구는 확인되지 않은 예시다. 출처 표시에 모두 '[mock]'을 붙인다.

계획서 9장 데모 문장(설비 번호·장소 표현)은 넣지 않는다 (시연·정답 세트 평가 오염 방지).
충분성 체크리스트(evidence_collector.DEFAULT_REQUIREMENTS)의 키워드가 들어가도록 썼다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MockDoc:
    source_type: str  # ACCIDENT_REPORT, SIF, NEAR_MISS, KOSHA_GUIDE, RA_GUIDE
    external_id: str
    title: str
    text: str
    tags: dict[str, list[str]] = field(default_factory=dict)  # 5축 태그 (work_type, accident_type, risk_factor)


ACCIDENT_DOCS: tuple[MockDoc, ...] = (
    MockDoc("ACCIDENT_REPORT", "ACC-0001", "[mock] 저장소 인근 연마 작업 중 화재",
            "인화성 용제 드럼이 놓인 저장소 옆에서 그라인더로 강재를 연마하던 중 불티가 튀어 용제 증기에 착화, 화재가 발생했다. "
            "작업 전 가연성가스 농도를 측정하지 않았고 불티 비산방지포와 화재감시자가 없었다. "
            "대책: 작업 반경 내 가연물 제거, 작업 전·중 가스 농도 측정, 비산방지포 설치, 화재감시자 배치.",
            {"work_type": ["화기"], "accident_type": ["화재"], "risk_factor": ["화학적"]}),
    MockDoc("ACCIDENT_REPORT", "ACC-0002", "[mock] 빈 드럼 용접 중 폭발",
            "내용물을 비웠다고 판단한 드럼의 뚜껑 부위를 용접하던 중 내부에 남은 인화성 증기가 폭발해 작업자가 중상을 입었다. "
            "드럼 내부 세정과 가스 측정을 하지 않았다. 대책: 용기 내부 세정·불활성가스 치환 후 측정, 화기작업 허가 절차 준수.",
            {"work_type": ["화기"], "accident_type": ["폭발·파열"], "risk_factor": ["화학적"]}),
    MockDoc("SIF", "SIF-0101", "[mock] 배관 절단 중 잔류물 분출",
            "운전 정지 후 배관을 토치로 절단하던 중 배관 안에 남아 있던 가연성 액체가 흘러나와 불이 붙었다. "
            "밸브 차단과 잠금, 배관 퍼지·세정, 맹판 설치가 이루어지지 않았다. 고위험요인: 잔류물 누출, 화재·폭발.",
            {"work_type": ["화기"], "accident_type": ["누출·접촉", "화재"], "risk_factor": ["화학적"]}),
    MockDoc("NEAR_MISS", "NM-0007", "[mock] 사내 아차사고: 연마석 파손",
            "고착된 부품을 그라인더로 절단하던 중 연마석이 반동으로 튀며 일부가 깨졌다. 방호덮개가 벗겨진 상태였다. "
            "대책: 방호덮개 부착 확인, 규격 연마석 사용, 보안경·안면보호구 착용.",
            {"work_type": ["화기", "일반"], "accident_type": ["절단·베임"], "risk_factor": ["기계적"]}),
    MockDoc("ACCIDENT_REPORT", "ACC-0201", "[mock] 저장조 세척 중 질식",
            "저장조 내부 세척을 위해 들어간 작업자가 산소결핍으로 쓰러졌고, 구조하러 들어간 동료도 질식했다. "
            "작업 전 산소 농도 측정과 환기를 하지 않았고 감시인이 없었다. 대책: 측정·환기, 송기마스크, 감시인 배치, 구조 장비 비치.",
            {"work_type": ["밀폐공간"], "accident_type": ["산소결핍"], "risk_factor": ["작업환경"]}),
    MockDoc("ACCIDENT_REPORT", "ACC-0202", "[mock] 맨홀 내부 황화수소 중독",
            "하수 맨홀 내부 점검 중 황화수소가 발생해 작업자가 의식을 잃었다. 유해가스 측정기가 없었다. "
            "대책: 출입 전 산소·황화수소 측정, 연속 환기, 감시인 상주.",
            {"work_type": ["밀폐공간"], "accident_type": ["산소결핍", "누출·접촉"], "risk_factor": ["작업환경"]}),
    MockDoc("ACCIDENT_REPORT", "ACC-0301", "[mock] 이동식 비계에서 떨어짐",
            "높이 3m 이동식 비계 위에서 조명을 교체하던 작업자가 균형을 잃고 떨어졌다. 안전난간이 없었고 안전대를 착용하지 않았다. "
            "대책: 작업발판·안전난간 설치, 안전대 부착설비 확보, 바퀴 고정.",
            {"work_type": ["고소"], "accident_type": ["떨어짐"], "risk_factor": ["작업환경"]}),
    MockDoc("ACCIDENT_REPORT", "ACC-0302", "[mock] 슬레이트 지붕 채광창 추락",
            "공장 지붕에서 설비를 설치하던 작업자가 채광창을 밟아 추락했다. 채광창 덮개와 추락방호망이 없었다. "
            "대책: 채광창 덮개, 추락방호망, 안전대 착용, 작업발판 설치.",
            {"work_type": ["고소"], "accident_type": ["떨어짐"], "risk_factor": ["작업환경"]}),
    MockDoc("ACCIDENT_REPORT", "ACC-0401", "[mock] 분전반 작업 중 감전",
            "분전반 차단기를 교체하던 작업자가 전원을 차단하지 않은 상태에서 충전부에 접촉해 감전되었다. "
            "대책: 정전 후 잠금·표지, 검전기로 무전압 확인, 절연 보호구 착용.",
            {"work_type": ["전기"], "accident_type": ["감전"], "risk_factor": ["전기적"]}),
    MockDoc("ACCIDENT_REPORT", "ACC-0501", "[mock] 크레인 인양물 낙하",
            "크레인으로 펌프를 인양하던 중 줄걸이 슬링이 끊어져 인양물이 낙하, 아래에 있던 작업자가 깔렸다. "
            "대책: 줄걸이 용구 점검, 신호수 배치, 인양물 하부 출입 통제.",
            {"work_type": ["중량물"], "accident_type": ["맞음", "깔림"], "risk_factor": ["기계적"]}),
    MockDoc("ACCIDENT_REPORT", "ACC-0601", "[mock] 펌프 정비 중 회전체 끼임",
            "펌프 커플링을 점검하던 중 다른 작업자가 전원을 넣어 손이 끼였다. 대책: 기동 장치 잠금(LOTO), 작업 중 표지.",
            {"work_type": ["일반"], "accident_type": ["끼임"], "risk_factor": ["기계적"]}),
)

GUIDE_DOCS: tuple[MockDoc, ...] = (
    MockDoc("KOSHA_GUIDE", "G-HOT-01", "[mock] 화기작업 안전 지침 - 가스 측정",
            "화기작업 전에는 작업 장소와 주변의 가연성가스 농도를 측정하고, 가스 농도가 폭발하한계(LEL)의 기준 미만일 때만 작업을 허가한다. "
            "작업 중에도 주기적으로 측정하며 기준을 넘으면 즉시 작업을 중지한다.",
            {"work_type": ["화기"], "accident_type": ["화재", "폭발·파열"], "risk_factor": ["화학적"]}),
    MockDoc("KOSHA_GUIDE", "G-HOT-02", "[mock] 화기작업 안전 지침 - 화재감시자",
            "용접·용단·연마 등 불티가 발생하는 작업에는 화재감시자를 배치한다. 화재감시자는 소화기를 갖추고 작업 중과 작업 후 일정 시간 동안 "
            "불티가 남아 있는지 확인한다.",
            {"work_type": ["화기"], "accident_type": ["화재"], "risk_factor": ["인적·관리"]}),
    MockDoc("KOSHA_GUIDE", "G-HOT-03", "[mock] 화기작업 안전 지침 - 가연물 관리",
            "작업 반경 안의 가연물은 치우고, 옮길 수 없으면 불연성 덮개나 불티 비산방지포로 덮는다. 인화성 물질 용기는 밀폐해 격리한다.",
            {"work_type": ["화기"], "accident_type": ["화재"], "risk_factor": ["화학적"]}),
    MockDoc("KOSHA_GUIDE", "G-PIPE-01", "[mock] 설비 개방 작업 지침",
            "배관·용기를 열기 전에는 내부 물질을 배출하고 퍼지·세정한 뒤 맹판을 설치한다. 관련 밸브는 차단·잠금하고 표지를 붙인다.",
            {"work_type": ["일반", "화기"], "accident_type": ["누출·접촉"], "risk_factor": ["화학적"]}),
    MockDoc("KOSHA_GUIDE", "G-CS-01", "[mock] 밀폐공간 작업 지침 - 적정공기",
            "밀폐공간에 들어가기 전에 산소와 유해가스 농도를 측정한다. 적정공기는 산소 18% 이상 23.5% 미만, 탄산가스 1.5% 미만, "
            "일산화탄소 30ppm 미만, 황화수소 10ppm 미만인 공기를 말한다. 기준에 맞지 않으면 환기 후 다시 측정한다.",
            {"work_type": ["밀폐공간"], "accident_type": ["산소결핍"], "risk_factor": ["작업환경"]}),
    MockDoc("KOSHA_GUIDE", "G-CS-02", "[mock] 밀폐공간 작업 지침 - 감시인",
            "밀폐공간 작업 중에는 외부에 감시인을 상주시키고, 작업자와 상시 연락한다. 구조 시에는 송기마스크를 착용한 사람만 들어간다.",
            {"work_type": ["밀폐공간"], "accident_type": ["산소결핍"], "risk_factor": ["인적·관리"]}),
    MockDoc("KOSHA_GUIDE", "G-HT-01", "[mock] 고소작업 지침 - 추락 방지",
            "높이 2m 이상에서 작업할 때는 작업발판과 안전난간을 설치하고, 설치가 어려우면 추락방호망과 안전대 부착설비를 갖춘다. "
            "작업자는 안전대를 착용한다.",
            {"work_type": ["고소"], "accident_type": ["떨어짐"], "risk_factor": ["작업환경"]}),
    MockDoc("KOSHA_GUIDE", "G-EL-01", "[mock] 전기작업 지침 - 정전 절차",
            "전기설비 작업은 정전 후 전원 차단기를 잠금하고 표지를 붙인다. 검전기로 무전압을 확인하고 잔류 전하를 방전한다.",
            {"work_type": ["전기"], "accident_type": ["감전"], "risk_factor": ["전기적"]}),
    MockDoc("KOSHA_GUIDE", "G-LF-01", "[mock] 크레인 인양 작업 지침",
            "크레인 인양 전에 줄걸이 용구의 손상 여부를 점검하고, 신호수를 지정한다. 인양물 아래에는 사람이 들어가지 않도록 통제한다.",
            {"work_type": ["중량물"], "accident_type": ["맞음", "깔림"], "risk_factor": ["기계적"]}),
    MockDoc("RA_GUIDE", "RA-01", "[mock] 위험성평가 안내 - 연마 공구",
            "휴대용 연삭기(그라인더)는 방호덮개를 부착한 상태로 쓰고, 연마석의 균열을 확인한다. 보안경과 장갑을 착용한다.",
            {"work_type": ["일반", "화기"], "accident_type": ["절단·베임"], "risk_factor": ["기계적"]}),
    MockDoc("RA_GUIDE", "RA-02", "[mock] 위험성평가 안내 - 비상 대응",
            "작업 전 TBM에서 비상 시 작업중지 신호와 대피 경로, 소화기 위치를 공유한다.",
            {"work_type": ["일반"], "accident_type": ["기타"], "risk_factor": ["인적·관리"]}),
)
