"""시험용 데모 데이터. 저장소에 samples 폴더를 두지 않도록 시험 안에 담았다.

SIF_REQ / LAW_REQ 는 요청 본문(dict), SAMPLE 은 임시 폴더에 만든 청크 JSONL 경로다.
SAMPLE 은 시험 프로세스가 끝나면 지워진다.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from pathlib import Path

_SIF_JSON = r'''{
  "source": {
    "source_type": "SIF",
    "external_id": "SIF-DEMO-001",
    "title": "거푸집 자재반입, 가공, 운반 – 감전 (2018)",
    "url": "https://www.data.go.kr/data/15161362/openapi.do",
    "publisher": "한국산업안전보건공단",
    "license": "이용허락범위 제한 없음",
    "content_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "flagged": false
  },
  "chunks": [
    {
      "chunk_index": 0,
      "text": "[재해개요] 2018년 거푸집 자재 반입·가공·운반 과정에서 감전 사고가 발생했습니다.",
      "embedding_text": "거푸집 자재반입, 가공, 운반 – 감전\n[재해개요] 2018년 거푸집 자재 반입·가공·운반 과정에서 감전 사고가 발생했습니다.",
      "locator": "재해개요·유발요인·감소대책(예시)",
      "token_count": 20,
      "payload": {
        "severity": "FATAL",
        "accident_type": [
          "ELECTRIC"
        ],
        "hazard": [
          "ELECTRICAL"
        ],
        "work_type": [
          "ELECTRICAL"
        ],
        "cas": [],
        "ghs": [],
        "year": 2018,
        "natural_key": {
          "dataset": "sif",
          "record": "SIF-DEMO-001",
          "section": "case",
          "part": 0
        },
        "tier": "B",
        "cate": "건설업",
        "disaster_type": "감전",
        "situation": null
      }
    }
  ]
}'''

_LAW_JSON = r'''{
  "source": {
    "source_type": "LAW",
    "external_id": "LAW-DEMO-001",
    "title": "산업안전보건기준에 관한 규칙(데모)",
    "file_path": "raw/데모용_법령.hwp",
    "flagged": false
  },
  "chunks": [
    {
      "chunk_index": 1,
      "text": "제20조(출입의 금지 등) 사업주는 다음 각 호의 작업 또는 장소에 울타리를 설치하는 등 ...",
      "embedding_text": "산업안전보건기준에 관한 규칙 > 제1편 총칙 > 제2장 작업장 제20조(출입의 금지 등) ...",
      "locator": "제20조 (2/3)",
      "payload": {
        "natural_key": {
          "law_key": "LAW-DEMO-001",
          "article_key": "20",
          "part": 1
        },
        "article_label": "제20조",
        "article_title": "출입의 금지 등",
        "parts_total": 3,
        "repeated_lead_lines": 2,
        "oversize": false,
        "path": [
          "제1편 총칙",
          "제2장 작업장"
        ],
        "hierarchy": {
          "편": "제1편 총칙",
          "장": "제2장 작업장"
        },
        "hazard": [],
        "work_type": [],
        "cas": [],
        "year": 2025,
        "article_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "law_meta": {
          "effective_date": "2026-03-02",
          "promulgation_no": "고용노동부령 제450호",
          "promulgation_date": "2025-09-01",
          "amendment_type": "일부개정",
          "source_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        }
      }
    }
  ]
}'''

_CHUNKS_JSONL = r'''{"source_type": "SIF", "external_id": "SIF-DEMO-001", "title": "거푸집 자재반입, 가공, 운반 – 감전 (2018)", "url": "https://www.data.go.kr/data/15161362/openapi.do", "publisher": "한국산업안전보건공단", "license": "이용허락범위 제한 없음", "content_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "flagged": false, "chunk_index": 0, "text": "[재해개요] 2018년 거푸집 자재 반입·가공·운반 과정에서 감전 사고가 발생했습니다.", "embedding_text": "거푸집 자재반입, 가공, 운반 – 감전\n[재해개요] 2018년 거푸집 자재 반입·가공·운반 과정에서 감전 사고가 발생했습니다.", "locator": "재해개요·유발요인·감소대책(예시)", "token_count": 20, "payload": {"severity": "FATAL", "accident_type": ["ELECTRIC"], "hazard": ["ELECTRICAL"], "work_type": ["ELECTRICAL"], "cas": [], "ghs": [], "year": 2018, "natural_key": {"dataset": "sif", "record": "SIF-DEMO-001", "section": "case", "part": 0}, "tier": "B", "cate": "건설업", "disaster_type": "감전", "situation": null}}
{"source_type": "LAW", "external_id": "LAW-DEMO-001", "title": "산업안전보건기준에 관한 규칙(데모)", "file_path": "raw/데모용_법령.hwp", "flagged": false, "chunk_index": 1, "text": "제20조(출입의 금지 등) 사업주는 다음 각 호의 작업 또는 장소에 울타리를 설치하는 등 ...", "embedding_text": "산업안전보건기준에 관한 규칙 > 제1편 총칙 > 제2장 작업장 제20조(출입의 금지 등) ...", "locator": "제20조 (2/3)", "payload": {"natural_key": {"law_key": "LAW-DEMO-001", "article_key": "20", "part": 1}, "article_label": "제20조", "article_title": "출입의 금지 등", "parts_total": 3, "repeated_lead_lines": 2, "oversize": false, "path": ["제1편 총칙", "제2장 작업장"], "hierarchy": {"편": "제1편 총칙", "장": "제2장 작업장"}, "hazard": [], "work_type": [], "cas": [], "year": 2025, "article_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "law_meta": {"effective_date": "2026-03-02", "promulgation_no": "고용노동부령 제450호", "promulgation_date": "2025-09-01", "amendment_type": "일부개정", "source_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"}}}
{"source_type": "LAW", "external_id": "LAW-DEMO-001", "chunk_index": 2, "text": "제21조(데모) 데모용 조문 본문이다.", "embedding_text": "산업안전보건기준에 관한 규칙(데모) > 제1편 총칙 > 제2장 작업장 제21조(데모)", "locator": "제21조", "payload": {"natural_key": {"law_key": "LAW-DEMO-001", "article_key": "21", "part": 0}, "article_label": "제21조", "article_title": "데모", "parts_total": 1, "repeated_lead_lines": 0, "oversize": false, "path": ["제1편 총칙", "제2장 작업장"], "hierarchy": {"편": "제1편 총칙", "장": "제2장 작업장"}, "hazard": [], "work_type": [], "cas": [], "year": 2025, "article_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "law_meta": {"effective_date": "2026-03-02", "promulgation_no": "고용노동부령 제450호", "promulgation_date": "2025-09-01", "amendment_type": "일부개정", "source_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"}}}
'''

SIF_REQ = json.loads(_SIF_JSON)
LAW_REQ = json.loads(_LAW_JSON)

_TMP = Path(tempfile.mkdtemp(prefix="demo_chunks_"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
SAMPLE = _TMP / "chunks_demo.jsonl"
SAMPLE.write_text(_CHUNKS_JSONL, encoding="utf-8", newline="\n")
