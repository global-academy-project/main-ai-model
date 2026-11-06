"""법령 파일(.hwpx, 나중에 .xml) → 임베딩 직전 청크(JSONL).

단계: read(S0·S1) → parse(S2) → clean(S3) → chunk(S4) → fields(S5) → validate(S6)
id 발급과 /ingest/chunks 호출은 이 패키지의 범위 밖이다(뒤로 미룸). 설계: claude/법령_hwpx_파싱_청킹_설계.md
"""
