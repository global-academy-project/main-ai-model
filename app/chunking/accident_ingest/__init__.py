"""사고 데이터(API) → 임베딩 직전 청크 파이프라인.

표준 라이브러리만 쓴다. 네트워크 코드와 앱 코드(import app...)를 넣지 않는다.
수집(fetch)과 적재(load)는 scripts/ 쪽 몫이고, 여기는 받아 둔 원본 페이지를 읽어
read → map → clean → chunk → fields → validate 까지 만든다.
사용법은 README_적용안내.md 참고.
"""

__all__ = ["stages", "cli"]
