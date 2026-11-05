from app.rag.tags import classify_regulations


def test_classify_regulations_maps_section_15_rules_and_keeps_evidence():
    regulations = [
        "가.산업안전보건법에 의한 규제: 작업환경측정대상물질 (측정주기: 6개월)|관리대상유해물질|특수건강진단대상물질|공정안전보고서(PSM) 제출 대상물질",
        "나.화학물질관리법에 의한 규제: 사고대비물질",
        "다.위험물안전관리법에 의한 규제: 4류 제1석유류(비수용성) (200L)",
        "라.폐기물관리법에 의한 규제: 지정폐기물",
        "마.화학물질의 등록 및 평가 등에 관한 법률 규제: 기존화학물질",
        "바.기타 국내 규제: 해당없음",
    ]

    tags = classify_regulations(regulations)
    by_code = {tag["code"]: tag for tag in tags}

    assert set(by_code) == {
        "WORKPLACE_MEASUREMENT",
        "CONTROLLED_HAZARDOUS_SUBSTANCE",
        "SPECIAL_HEALTH_EXAM",
        "PROCESS_SAFETY_REPORT",
        "ACCIDENT_PREPARED_SUBSTANCE",
        "DANGEROUS_SUBSTANCE_REGULATION",
        "DESIGNATED_WASTE",
        "EXISTING_CHEMICAL_SUBSTANCE",
    }
    assert by_code["WORKPLACE_MEASUREMENT"]["law"] == "산업안전보건법"
    assert by_code["WORKPLACE_MEASUREMENT"]["evidence"] == [regulations[0]]
    assert "해당없음" not in by_code


def test_classify_regulations_ignores_empty_and_non_applicable_values():
    assert classify_regulations([
        "산업안전보건법에 의한 규제: 해당없음",
        "화학물질관리법에 의한 규제: 자료없음",
        "",
    ]) == []


def test_classify_regulations_deduplicates_tags_and_preserves_distinct_evidence():
    regulations = [
        "산업안전보건법: 작업환경측정대상물질",
        "산업안전보건법: 작업환경측정대상물질 (측정주기: 6개월)",
    ]

    tags = classify_regulations(regulations)

    assert len(tags) == 1
    assert tags[0]["code"] == "WORKPLACE_MEASUREMENT"
    assert tags[0]["evidence"] == regulations