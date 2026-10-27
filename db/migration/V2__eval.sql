-- agent_kb V2: 정답 세트와 채점 결과 (3주차 튜닝)
-- 소유: FastAPI. 계획서 "각자 업무 문항 7~8개" 정답 세트를 여기 넣고, 프롬프트·모델을 바꿀 때마다 돌려 점수를 쌓는다.
-- 평가 스크립트는 kb_ingest 계정으로 쓴다 (에이전트 실행용 kb_reader는 쓰기 불가).

CREATE TABLE eval_cases (
    id          BIGINT       NOT NULL AUTO_INCREMENT,
    case_code   VARCHAR(40)  NOT NULL,                      -- 예: C-PERMIT-CONFINED-O2-LOW
    owner       CHAR(1)      NOT NULL,                      -- 담당 개발자 A | B | C | D
    task_type   VARCHAR(30)  NOT NULL,                      -- 업무 DB agent_runs.task_type 과 같은 값
    scenario    CHAR(1)      NULL,                          -- 시연 시나리오 A~F (해당 시)
    input_json  JSON         NOT NULL,                      -- 에이전트에 넣을 요청
    expected    JSON         NOT NULL,                      -- 기대 결과: 판정(BLOCK 등), 반드시 나와야 할 위험요인, 반드시 인용할 근거 등
    legal_basis VARCHAR(200) NULL,                          -- 정답의 근거 조항 (정부 지침 기반 원칙)
    active      BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at  DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_ec_code (case_code),
    KEY idx_ec_owner (owner, active),
    CONSTRAINT ck_ec_code     CHECK (case_code REGEXP '^[A-D]-[A-Z0-9-]+$'),
    CONSTRAINT ck_ec_owner    CHECK (owner IN ('A', 'B', 'C', 'D')),
    CONSTRAINT ck_ec_code_own CHECK (LEFT(case_code, 1) = owner),
    CONSTRAINT ck_ec_scenario CHECK (scenario IS NULL OR scenario IN ('A', 'B', 'C', 'D', 'E', 'F')),
    CONSTRAINT ck_ec_task     CHECK (task_type IN ('WORK_REGISTER', 'RISK_DRAFT', 'PERMIT_DRAFT', 'INSPECTION_DRAFT', 'TBM_DRAFT',
                                                   'NEAR_MISS_TAG', 'PHOTO_CHECK', 'RECORD_QUERY', 'DAILY_LOG', 'AUDIT_REPORT')),
    CONSTRAINT ck_ec_json     CHECK (JSON_TYPE(input_json) = 'OBJECT' AND JSON_TYPE(expected) = 'OBJECT')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 평가 1회 (어떤 프롬프트·모델·코드로 돌렸나)
CREATE TABLE eval_runs (
    id             BIGINT       NOT NULL AUTO_INCREMENT,
    label          VARCHAR(100) NOT NULL,                   -- 예: prompt-v3 재검색 조건 완화
    model          VARCHAR(60)  NOT NULL,
    prompt_version VARCHAR(30)  NOT NULL,
    git_commit     CHAR(40)     NULL,
    network_mode   VARCHAR(10)  NOT NULL,
    status         VARCHAR(10)  NOT NULL DEFAULT 'RUNNING',
    total          INT          NOT NULL DEFAULT 0,
    passed         INT          NOT NULL DEFAULT 0,
    started_at     DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    finished_at    DATETIME(6)  NULL,
    PRIMARY KEY (id),
    KEY idx_er_time (started_at),
    CONSTRAINT ck_er_commit  CHECK (git_commit IS NULL OR git_commit REGEXP '^[0-9a-f]{40}$'),
    CONSTRAINT ck_er_network CHECK (network_mode IN ('OPEN', 'CLOSED')),
    CONSTRAINT ck_er_status  CHECK (status IN ('RUNNING', 'DONE', 'FAILED')),
    CONSTRAINT ck_er_counts  CHECK (total >= 0 AND passed BETWEEN 0 AND total),
    CONSTRAINT ck_er_finish  CHECK ((status = 'RUNNING') = (finished_at IS NULL))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE eval_results (
    id             BIGINT        NOT NULL AUTO_INCREMENT,
    eval_run_id    BIGINT        NOT NULL,
    case_id        BIGINT        NOT NULL,
    passed         BOOLEAN       NOT NULL,
    score          DECIMAL(5,4)  NOT NULL,                  -- 0~1 (부분 점수: 위험요인 몇 개 맞췄나 등)
    latency_ms     INT           NULL,
    input_tokens   INT           NULL,
    output_tokens  INT           NULL,
    output_json    JSON          NULL,
    failure_reason VARCHAR(500)  NULL,
    created_at     DATETIME(6)   NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_eres (eval_run_id, case_id),
    KEY idx_eres_case (case_id, passed),                    -- 문항별 통과 추이
    CONSTRAINT fk_eres_run  FOREIGN KEY (eval_run_id) REFERENCES eval_runs (id) ON DELETE CASCADE,
    CONSTRAINT fk_eres_case FOREIGN KEY (case_id)     REFERENCES eval_cases (id),
    CONSTRAINT ck_eres_score CHECK (score BETWEEN 0 AND 1),
    CONSTRAINT ck_eres_fail  CHECK (passed OR failure_reason IS NOT NULL),     -- 실패면 이유 필수
    CONSTRAINT ck_eres_nums  CHECK (COALESCE(latency_ms, 0) >= 0 AND COALESCE(input_tokens, 0) >= 0 AND COALESCE(output_tokens, 0) >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
