"""S1 읽기 단계: 파일 형식마다 읽기 함수 하나.

모든 reader는 `SourceDoc`(메타 + 문단 목록)을 돌려준다. 그 뒤 단계(parse 이후)는 파일 형식을 모른다.

다른 형식(예: JSON 응답)을 더하려면:
  (XML은 이미 `readers/law_xml.py`에 있다. 아래는 새 형식을 더할 때의 규약이다.)
  1. `readers/<형식>.py`에 `read_<형식>(path) -> SourceDoc`를 만든다.
  2. 메타(법령명·시행일·공포번호)는 XML 태그에서 직접 채운다.
  3. 문단은 hwpx와 같은 글자 모양으로 내보낸다. 한 문단이 한 줄이다.
       편·장·절·관 제목  "제3장 통로", 조 "제21조(통로의 조명) ① …", 항 "② …"(앞 공백 가능),
       호 "1. …", 목 "가. …", 개정 주기 "[본조신설 2017. 3. 3.]", 부칙 "부칙 <…>"
  4. 아래 READERS에 확장자를 등록한다.
  이렇게 하면 parse·clean·chunk·fields·validate는 고치지 않는다.
"""

from __future__ import annotations

from pathlib import Path

from ..models import SourceDoc
from .hwpx import ReaderError, read_hwpx
from .law_xml import read_law_xml

READERS = {
    ".hwpx": read_hwpx,
    ".xml": read_law_xml,  # 국가법령정보센터 Open API 본문 응답(lawService.do ... type=XML)
}


def read_source(path: str | Path) -> SourceDoc:
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ReaderError(
            f"지원하지 않는 형식입니다: {path.suffix or '(확장자 없음)'} "
            f"(지원: {', '.join(sorted(READERS))})"
        )
    return reader(path)


__all__ = ["READERS", "ReaderError", "read_source"]
