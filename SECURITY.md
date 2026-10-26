# Security Policy

## Supported Versions

아래는 현재 보안 업데이트를 제공하는 버전입니다.

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

> 참고:
> - `main` 브랜치는 다음 마이너/메이저 릴리스 준비를 위한 개발 브랜치입니다.
> - 패치 릴리스는 기존 호환 범위를 유지하면서 보안 수정 위주로 배포합니다.

## Security Guidelines

### 1. Dependency and Supply Chain Security
- 의존성은 신뢰 가능한 공식 레지스트리만 사용합니다.
- 버전 고정(lockfile) 상태를 유지하고, 정기적으로 취약점 스캔을 수행합니다.
- 사용하지 않는 패키지/권한은 제거합니다.

### 2. Secret and Credential Management
- API 키, 토큰, 비밀번호, 인증서 등 비밀값은 저장소에 커밋하지 않습니다.
- 비밀값은 환경 변수 또는 시크릿 매니저를 통해 주입합니다.
- 로그/에러 메시지에 민감정보가 출력되지 않도록 마스킹합니다.

### 3. Secure Development Practices
- 입력값 검증(Validation)과 출력 인코딩을 기본으로 적용합니다.
- 인증/인가 로직은 최소 권한 원칙(Least Privilege)을 따릅니다.
- 보안 관련 설정은 기본적으로 안전한 값(Secure by Default)으로 유지합니다.

### 4. Vulnerability Reporting
취약점 제보는 공개 이슈가 아닌 비공개 채널로 전달해 주세요.

- Contact: ``
- Expected response time: 영업일 기준 3일 이내 1차 회신
- 제보 시 포함 권장 사항:
  - 영향 받는 버전/환경
  - 재현 단계(Proof of Concept)
  - 예상 영향도 및 완화 제안

### 5. Disclosure Process
- 접수 → 재현/분석 → 수정 개발 → 패치 배포 → 공지 순으로 대응합니다.
- 사용자 보호를 위해 패치 배포 전 세부 취약점 내용은 제한 공개할 수 있습니다.

### 6. 권장 사항
- 커밋과 릴리스 태그에 GPG 서명을 적용해 작성자와 배포 출처를 검증할 수 있도록 합니다.
- 서명된 커밋과 태그를 확인한 뒤 병합하거나 배포합니다.
- GPG 개인 키와 암호문구는 안전하게 보관하고, 저장소나 코드에 커밋하지 않습니다.