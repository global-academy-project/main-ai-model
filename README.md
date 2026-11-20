# Main AI Model

현장 안전관리 서비스에서 사용하는 FastAPI 기반 AI, 임베딩, 지식 데이터 적재
서비스와 Qdrant 초기화 작업을 제공합니다. AI API는 Spring Boot와 연동되며,
RAG 검색을 위해 임베딩 API와 Qdrant를 사용할 수 있습니다.

## 서비스

| 구성 | 진입점 / Docker target | 포트 | 역할 |
|---|---|---:|---|
| AI API | `app.main:app` / `ai` | 60100 | 에이전트, 아차사고 태깅, 규칙 입력 추출, 사진 인식, 음성 전사 |
| 임베딩 API | `app.rag.embedding_api:create_app` / `embedding` | 60110 | BGE-M3 임베딩 생성 |
| 지식 적재 API | `app.ingest_main:app` / `ingest` | 8001 | 지식 청크와 벡터 데이터 적재 |
| Qdrant 초기화 | `qdrant-init` | 일회성 작업 | Qdrant 컬렉션과 인덱스 초기화 |

## 로컬 개발과 테스트

Python 3.12를 사용합니다. PowerShell에서 가상 환경을 만들고 개발 의존성을
설치합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.dev.txt
```

테스트는 다음과 같이 실행합니다.

```powershell
python -m pytest
```

AI API를 로컬에서 실행하려면 `.env.example`을 참고해 `.env`를 만들고,
필요한 모델 파일과 연동 서비스를 준비한 뒤 실행합니다.

```powershell
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 60100
```

`.env.example`의 값은 설정 예시이며 그대로 실행 가능한 자격 증명이 아닙니다.
실행 설정에 따라 내부 API 주소와 토큰, MSDS 서비스 키, 로컬 모델 경로,
Qdrant 및 임베딩 연결 정보를 채워야 합니다. 의존 서비스가 없는 개발 환경에서는
규칙 확인과 MSDS 연동을 mock으로 설정하고, 외부망이 필요한 모델은 사전에 내려받아
두세요. `.env`와 실제 비밀 값은 커밋하지 마세요.

## Docker 이미지

[Dockerfile](Dockerfile)은 CPU 전용 런타임 이미지를 제공합니다. AI API target의
이름은 `api`가 아니라 `ai`입니다. 저장소 루트에서 필요한 이미지를 빌드합니다.

```powershell
docker build --target ai -t main-ai-model:ai .
docker build --target embedding -t main-ai-model:embedding .
docker build --target ingest -t main-ai-model:ingest .
docker build --target qdrant-init -t main-ai-model:qdrant-init .
```

AI API와 임베딩 API는 모델 다운로드/캐시와 연동 API 설정이 필요합니다.
임베딩 API는 초기화 완료 후 `/ready`, AI API는 `/health`, 지식 적재 API는
`/ready`에서 상태를 확인할 수 있습니다. 통합 Docker Compose 구성과 필요한 환경
변수는 `main-infra-manage` 저장소의 Docker 실행 가이드를 참고하세요.

## 의존성 파일

| 파일 | 용도 |
|---|---|
| `requirements.txt` | AI API의 직접 의존성 |
| `requirements.release.txt` | CPU PyTorch를 포함한 배포 의존성 |
| `requirements.dev.txt` | 개발 및 테스트 의존성 |
| `requirements.ingest.txt` | 지식 적재 API 의존성 |
| `requirements.qdrant-init.txt` | Qdrant 초기화 작업 의존성 |
| `requirements.lock` | 고정된 전체 의존성 버전 |

## 주요 디렉터리

| 경로 | 내용 |
|---|---|
| `app/agent/`, `app/llm/` | 에이전트 시나리오와 언어 모델 연동 |
| `app/api/` | AI API 라우터 |
| `app/rag/`, `app/chunking/` | RAG 검색, 임베딩, 청크 처리 |
| `app/ingest/` | 지식 데이터 적재 라우터 |
| `adapters/` | 이미지에 포함할 모델 어댑터와 ONNX 파일 |
| `scripts/` | 모델 다운로드, Qdrant 초기화, 적재 및 평가 도구 |
| `tests/` | 자동화 테스트 |
| `eval/` | 검색, 태깅, 추출 평가 데이터 |

## 평가 및 도메인 문서

평가 데이터의 형식과 실행 방법은 [`eval/README.md`](eval/README.md)를 참고하세요.
사내 아차사고 규칙 태깅과 RAG 검색 계약은
[`docs/NEAR_MISS_AI_TAGGING-v5.0.0.md`](docs/NEAR_MISS_AI_TAGGING-v5.0.0.md)에
정리되어 있습니다.
