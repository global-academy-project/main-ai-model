# Custom GitHub Template

이 저장소는 GitHub Pull Request 자동화 템플릿을 제공하는 예제 프로젝트입니다. 한국어 기반으로 운영되는 협업 흐름을 고려해 구성되었으며, PR 생성 시 템플릿 본문을 자동으로 보강하고 라벨과 담당자를 자동으로 지정할 수 있도록 설계되었습니다.

## 소개

오픈 소스 프로젝트나 팀 저장소에서 PR 템플릿은 단순히 양식 문서를 정리하는 수준을 넘어서, 실제 리뷰 품질과 작업 추적성을 높이는 핵심 도구가 됩니다. 이 템플릿은 다음 문제를 해결하도록 구성되어 있습니다.

- PR 생성 시 커밋 요약이 자동으로 템플릿에 반영됨
- 변경 파일 유형에 따라 문서/기능 라벨을 자동 분류함
- 브랜치 이름에 따라 refactor, bug, release 라벨을 자동 부여함
- 담당자(assignee)를 자동 지정해 작업자의 책임 범위를 명확히 함
- 한국어 중심의 PR 문화를 유지하면서도 GitHub Actions 기반으로 자동화함

## 주요 기능

### 1. 커밋 요약 자동 반영

PR 템플릿 내부에 `__COMMIT_SHA__` placeholder가 포함되어 있으면, GitHub Actions 스크립트가 PR의 커밋 목록을 조회한 뒤 각 커밋을 한 줄씩 정리해 본문에 삽입합니다.

이 방식은 단순히 마지막 커밋 SHA 한 개만 넣는 대신, 여러 커밋이 있는 경우에도 각각의 변경 내용을 정리해서 보존합니다.

### 2. 변경 파일 기반으로 라벨 자동 분류

PR에 포함된 파일 목록을 확인해 다음 규칙을 적용합니다.

- 문서 파일만 변경된 경우: `document`
- 코드 파일이 포함된 경우: `enhancement`

문서와 코드가 섞여 있으면 코드 쪽으로 우선 판별합니다.

### 3. 브랜치명 기반 라벨 추가

브랜치 이름을 확인해 다음 라벨을 보조적으로 추가합니다.

- `refactor` : 브랜치명에 `refactor` 포함
- `bug` : 브랜치명에 `bug`, `fix`, `hotfix`, `bugfix` 포함
- `release` : 브랜치명에 `release` 포함

기본적으로는 파일 유형 기반 라벨을 우선 두고, 브랜치 이름에 맞는 추가 라벨을 덧붙이는 구조입니다.

### 4. Assignee 자동 지정

GitHub Actions 스크립트는 기본적으로 PR 작성자를 assignee로 지정합니다. 필요하면 환경 변수 `PR_ASSIGNEES`를 통해 다른 담당자를 직접 지정할 수 있습니다.

예시:

```yaml
env:
  PR_ASSIGNEES: geonwoo8873
```

또는 여러 사람에게 넘길 수도 있습니다.

```yaml
env:
  PR_ASSIGNEES: geonwoo8873,team-lead
```

## 저장소 구조

```text
.github/
├── PULL_REQUEST_TEMPLATE/
│   └── pull_request_template.md
├── scripts/
│   └── pull-request-temp-apply.sh
├── workflows/
│   └── config.yml
README.md
```

## PR 템플릿 예시

PR 템플릿은 한국어 문맥에 맞춰 구성되어 있습니다. 핵심 구조는 다음과 같습니다.

```md
## 주요 변경 사항 (Key Changes)
<!-- 이번 PR의 핵심 내용을 간단히 작성해주세요. -->
- [핵심 변경 1]
- [핵심 변경 2]
- [핵심 변경 3]

#### 커밋 요약:
__COMMIT_SHA__

#### 변경 내용:
<!-- 변경 이유, 영향 범위, 주의사항 등을 작성해주세요. -->
```

이 템플릿은 사람에게는 읽기 쉬운 PR 설명을 제공하고, 기계적으로는 커밋 요약을 주입할 수 있는 구조를 갖습니다.

## 자동화 동작 흐름

1. `pull_request` 이벤트가 발생하면 GitHub Actions 워크플로가 실행됩니다.
2. 스크립트가 `GITHUB_EVENT_PATH`와 GitHub API 정보를 읽어 PR 정보를 확보합니다.
3. PR의 커밋 목록을 조회해 요약 문자열을 생성합니다.
4. 변경 파일 목록을 확인해 `document` 또는 `enhancement`를 판별합니다.
5. 브랜치명을 분석해 `refactor`, `bug`, `release` 여부를 확인합니다.
6. 필요한 라벨을 생성하거나 기존 라벨과 중복 없이 적용합니다.
7. 템플릿의 `__COMMIT_SHA__`를 실제 커밋 요약으로 대체해 PR 본문을 업데이트합니다.
8. PR 작성자를 기본 assignee로 지정합니다.

## 라벨 규칙

기본 규칙은 다음과 같습니다.

- 문서만 변경: `document`
- 코드 변경: `enhancement`
- 브랜치명에 `refactor` 포함: `refactor`
- 브랜치명에 `bug` 계열 키워드 포함: `bug`
- 브랜치명에 `release` 포함: `release`

동일 라벨이 중복 추가되지 않도록 중복 제거 처리도 함께 적용됩니다.

## GitHub Actions 설정

워크플로는 다음 환경 값을 전달합니다.

- `GITHUB_TOKEN`
- `GITHUB_EVENT_PATH`
- `GITHUB_REPOSITORY`
- `GITHUB_API_URL`
- `PR_ASSIGNEES` (선택)

실제 실행은 [.github/workflows/config.yml](.github/workflows/config.yml) 에서 관리되며, 본문 수정과 라벨/assignee 적용 로직은 [.github/scripts/pull-request-temp-apply.sh](.github/scripts/pull-request-temp-apply.sh) 에 있습니다.

## 사용 방법

1. 저장소에 PR 생성
2. 템플릿이 자동 로드됨
3. GitHub Actions가 커밋 요약과 라벨을 계산함
4. PR 본문과 라벨 상태를 확인함
5. 필요한 경우 수정 내용을 다시 정리해 최종 리뷰 진행

## 주의 사항

- PR 본문에서 설명은 사람이 직접 작성하는 것이 가장 정확합니다.
- 자동 생성되는 커밋 요약은 추적용 메타데이터로 활용하는 것이 적합합니다.
- 실제 GitHub Actions 동작은 토큰 권한과 브랜치 조건에 따라 달라질 수 있습니다.
- 공개 저장소에서는 `pull_request_target` 사용 시 보안 위험이 있으므로, 동작 범위와 권한을 신중히 설정하는 것이 좋습니다.

## 라이선스

이 템플릿은 팀 협업을 위해 설계된 예시 구조이며, 필요에 따라 자유롭게 커스터마이징해 사용할 수 있습니다.

본 저장소는 공개용 예시 템플릿으로 활용할 수 있도록 구성되어 있으며, 각 팀의 운영 정책에 맞게 문구, 라벨, 브랜치 규칙을 조정하는 것을 권장합니다.
