-- agent_kb V1: 지식 데이터 원본·청크, MSDS 캐시
-- 소유: FastAPI (agent-model). 담당: A (작성 C)
-- 이 DB는 업무 DB와 완전히 분리된다. 서로 FK·조회 없음. 적용 계정: kb_owner (DDL 전용)
--
-- 접근 계정 (infra/db/init/10-agent-kb.sh에서 생성)
--   kb_owner  : 이 마이그레이션만 (DDL)
--   kb_ingest : 적재 작업 (SELECT, INSERT, UPDATE, DELETE)
--   kb_reader : 에이전트 실행 (SELECT만) → 프롬프트 인젝션으로 조작돼도 지식 데이터를 바꿀 수 없음
--
-- 재적재는 (source_id, chunk_index) 기준 upsert → 청크 id·Qdrant 점 id가 바뀌지 않아 업무 DB의 인용이 유지된다.

CREATE TABLE kb_sources (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    source_type     VARCHAR(20)  NOT NULL,
    external_id     VARCHAR(100) NOT NULL,                  -- 가이드 번호, 보고서 번호, 아차사고면 업무 DB near_misses.id
    site_id         BIGINT       NULL,                      -- 업무 DB sites.id 값 (FK 아님). 공공 데이터는 NULL
    title           VARCHAR(300) NOT NULL,
    url             VARCHAR(500) NULL,
    license         VARCHAR(50)  NULL,                      -- 예: KOGL-3 (공공누리 제3유형)
    publisher       VARCHAR(100) NULL,                      -- 예: 고용노동부, 안전보건공단
    published_on    DATE         NULL,
    file_path       VARCHAR(500) NULL,
    content_sha256  CHAR(64)     NULL,                      -- 원본이 바뀌었는지 (같으면 재파싱 생략)
    status          VARCHAR(20)  NOT NULL DEFAULT 'PENDING',
    chunk_count     INT          NOT NULL DEFAULT 0,
    embedding_model VARCHAR(60)  NULL,
    flagged         BOOLEAN      NOT NULL DEFAULT FALSE,    -- 지시문처럼 보이는 문장 발견 (프롬프트 인젝션 의심) → 검토 전 검색 제외
    error_message   TEXT         NULL,
    created_at      DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at      DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_kb_src (source_type, external_id),
    KEY idx_kb_status (status),
    KEY idx_kb_site (site_id),
    CONSTRAINT ck_kb_type   CHECK (source_type IN ('ACCIDENT_REPORT', 'SIF', 'KOSHA_GUIDE', 'RISK_GUIDE', 'CHECKLIST', 'LAW', 'NEAR_MISS')),
    CONSTRAINT ck_kb_status CHECK (status IN ('PENDING', 'PARSED', 'EMBEDDED', 'FAILED')),
    CONSTRAINT ck_kb_site   CHECK ((source_type = 'NEAR_MISS') = (site_id IS NOT NULL)),   -- 사내 데이터만 현장, 공공 데이터는 현장 없음
    CONSTRAINT ck_kb_site_v CHECK (site_id IS NULL OR site_id > 0),                       -- Qdrant에서 0은 "공공"이라 실제 현장 id로 못 씀
    CONSTRAINT ck_kb_embed  CHECK (status <> 'EMBEDDED' OR (embedding_model IS NOT NULL AND chunk_count > 0)),
    CONSTRAINT ck_kb_failed CHECK (status <> 'FAILED' OR error_message IS NOT NULL),
    CONSTRAINT ck_kb_count  CHECK (chunk_count >= 0),
    CONSTRAINT ck_kb_public CHECK (source_type = 'NEAR_MISS' OR url IS NOT NULL OR file_path IS NOT NULL),  -- 공공 데이터는 출처 필수
    CONSTRAINT ck_kb_sha    CHECK (content_sha256 IS NULL OR content_sha256 REGEXP '^[0-9a-f]{64}$')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE kb_chunks (
    id              BIGINT       NOT NULL AUTO_INCREMENT,
    source_id       BIGINT       NOT NULL,
    chunk_index     INT          NOT NULL,
    collection_name VARCHAR(40)  NOT NULL,
    qdrant_point_id CHAR(36)     NOT NULL,                  -- uuid5(source_type:external_id:chunk_index)
    locator         VARCHAR(100) NULL,                      -- 조항, 페이지
    text            MEDIUMTEXT   NOT NULL,
    token_count     INT          NULL,
    created_at      DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at      DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_chunk (source_id, chunk_index),
    UNIQUE KEY uk_chunk_point (qdrant_point_id),
    CONSTRAINT fk_chunk_src FOREIGN KEY (source_id) REFERENCES kb_sources (id) ON DELETE CASCADE,
    CONSTRAINT ck_chunk_coll  CHECK (collection_name IN ('kb_accidents', 'kb_guides')),
    CONSTRAINT ck_chunk_index CHECK (chunk_index >= 0),
    CONSTRAINT ck_chunk_uuid  CHECK (qdrant_point_id REGEXP '^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'),
    CONSTRAINT ck_chunk_text  CHECK (CHAR_LENGTH(text) > 0),
    CONSTRAINT ck_chunk_tok   CHECK (token_count IS NULL OR token_count > 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE msds_cache (
    cas_no     VARCHAR(20)  NOT NULL,
    name_ko    VARCHAR(200) NULL,
    ghs        JSON         NULL,
    hazards    JSON         NULL,
    first_aid  TEXT         NULL,
    raw_json   JSON         NOT NULL,
    fetched_at DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (cas_no),
    CONSTRAINT ck_msds_cas CHECK (cas_no REGEXP '^[0-9]{2,7}-[0-9]{2}-[0-9]$')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
