#!/usr/bin/env bash

set -euo pipefail

fail() {
  echo "::error::$1"
  exit 1
}

warn() {
  echo "::warning::$1"
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    fail "Required command not found: $command_name"
  fi
}

require_command jq
require_command curl

for var in GITHUB_TOKEN GITHUB_REPOSITORY GITHUB_EVENT_PATH GITHUB_API_URL; do
  if [[ -z "${!var:-}" ]]; then
    fail "Environment variable $var is not set."
  fi
done

EVENT_PATH="${GITHUB_EVENT_PATH}"
API_URL="${GITHUB_API_URL:-https://api.github.com}"
REPO="${GITHUB_REPOSITORY}"
AUTH_HEADER="Authorization: Bearer ${GITHUB_TOKEN}"

if [[ ! -f "$EVENT_PATH" ]]; then
  fail "GitHub event payload file does not exist: $EVENT_PATH"
fi

ISSUE_NUMBER="$(jq -r '.issue.number // empty' "$EVENT_PATH")"
AUTHOR_LOGIN="$(jq -r '.issue.user.login // empty' "$EVENT_PATH")"
ISSUE_TITLE="$(jq -r '.issue.title // empty' "$EVENT_PATH")"
ISSUE_BODY="$(jq -r '.issue.body // empty' "$EVENT_PATH")"

if [[ -z "$ISSUE_NUMBER" || -z "$AUTHOR_LOGIN" ]]; then
  fail "No issue payload found or required fields are missing."
fi

# 이슈 제목 키워드로 라벨 결정 (PR의 브랜치 이름 기반 분류와 동일한 라벨 체계)
ISSUE_TITLE_LOWER="$(printf '%s' "$ISSUE_TITLE" | tr '[:upper:]' '[:lower:]')"
LABELS_TO_ADD=()

case "$ISSUE_TITLE_LOWER" in
  *bug*|*fix*|*hotfix*|*error*|*버그*|*오류*|*에러*) LABELS_TO_ADD+=("bug") ;;
  *refactor*|*리팩터*|*리팩토*) LABELS_TO_ADD+=("refactor") ;;
  *doc*|*readme*|*문서*) LABELS_TO_ADD+=("document") ;;
  *release*|*릴리스*|*릴리즈*) LABELS_TO_ADD+=("release") ;;
  *) LABELS_TO_ADD+=("enhancement") ;;
esac

declare -A SEEN=()
FINAL_LABELS=()
for label in "${LABELS_TO_ADD[@]}"; do
  if [[ -n "$label" && -z "${SEEN[$label]+x}" ]]; then
    FINAL_LABELS+=("$label")
    SEEN["$label"]=1
  fi
done

ensure_label() {
  local label_name="$1"
  local label_color="$2"
  local exists

  # Curl을 통해 리포지토리내 라벨 이름이 존재 하는지 확인
  exists="$(curl -fsSL \
    -H "Accept: application/vnd.github+json" \
    -H "$AUTH_HEADER" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "$API_URL/repos/$REPO/labels/$label_name" 2>/dev/null || true)"

  if [[ -z "$exists" ]]; then
    curl -fsSL -X POST \
      -H "Accept: application/vnd.github+json" \
      -H "$AUTH_HEADER" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      -H "Content-Type: application/json" \
      "$API_URL/repos/$REPO/labels" \
      -d "$(jq -n --arg name "$label_name" --arg color "$label_color" '{name: $name, color: $color}')" >/dev/null 2>&1 || true
  fi
}

# 라벨이 등록되어 있는지 확인하고 없으면 생성
# 필요한 라벨 목록: document, enhancement, refactor, bug, release
for label_name in document enhancement refactor bug release; do
  case "$label_name" in
    document) ensure_label "$label_name" "0075ca" ;;
    enhancement) ensure_label "$label_name" "a2eeef" ;;
    refactor) ensure_label "$label_name" "d4c5f9" ;;
    bug) ensure_label "$label_name" "d73a4a" ;;
    release) ensure_label "$label_name" "0e8a16" ;;
  esac
done

LABELS_JSON="$(printf '%s\n' "${FINAL_LABELS[@]}" | jq -R . | jq -s -c .)"

curl -fsSL -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "$AUTH_HEADER" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -H "Content-Type: application/json" \
  "$API_URL/repos/$REPO/issues/$ISSUE_NUMBER/labels" \
  -d "$(jq -n --argjson labels "$LABELS_JSON" '{labels: $labels}')" >/dev/null

ASSIGNEE_INPUT="${ISSUE_ASSIGNEES:-$AUTHOR_LOGIN}"
ASSIGNEES_CLEAN=()
declare -A ASSIGNEE_SEEN=()
IFS=',' read -r -a ASSIGNEES_RAW <<< "$ASSIGNEE_INPUT"
for assignee in "${ASSIGNEES_RAW[@]}"; do
  assignee="${assignee//[[:space:]]/}"
  [[ -n "$assignee" ]] || continue
  if [[ -z "${ASSIGNEE_SEEN[$assignee]+x}" ]]; then
    ASSIGNEE_SEEN["$assignee"]=1
    ASSIGNEES_CLEAN+=("$assignee")
  fi
done

if [[ ${#ASSIGNEES_CLEAN[@]} -gt 0 ]]; then
  ASSIGNEES_JSON="$(printf '%s\n' "${ASSIGNEES_CLEAN[@]}" | jq -R . | jq -s -c .)"
  curl -fsSL -X PATCH \
    -H "Accept: application/vnd.github+json" \
    -H "$AUTH_HEADER" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -H "Content-Type: application/json" \
    "$API_URL/repos/$REPO/issues/$ISSUE_NUMBER" \
    -d "$(jq -n --argjson assignees "$ASSIGNEES_JSON" '{assignees: $assignees}')" >/dev/null
fi

TEMPLATE_PATH=".github/ISSUE_TEMPLATE/bug_report.md"
BODY_UPDATED=false

# 작성자가 입력한 본문을 덮어쓰지 않도록 본문이 비어 있을 때만 템플릿 적용
if [[ -z "${ISSUE_BODY//[[:space:]]/}" ]]; then
  if [[ ! -f "$TEMPLATE_PATH" ]]; then
    fail "Issue template not found: $TEMPLATE_PATH"
  fi
  if [[ ! -r "$TEMPLATE_PATH" ]]; then
    fail "Issue template is not readable: $TEMPLATE_PATH"
  fi

  # 템플릿 상단의 YAML front matter(--- ... ---) 제거
  NEW_BODY="$(awk '
    NR == 1 && /^---[[:space:]]*$/ { in_fm = 1; next }
    in_fm && /^---[[:space:]]*$/ { in_fm = 0; next }
    !in_fm { print }
  ' "$TEMPLATE_PATH" | sed '/./,$!d')"

  PAYLOAD="$(jq -n --arg body "$NEW_BODY" '{body: $body}')"

  curl -fsSL -X PATCH \
    -H "Accept: application/vnd.github+json" \
    -H "$AUTH_HEADER" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -H "Content-Type: application/json" \
    "$API_URL/repos/$REPO/issues/$ISSUE_NUMBER" \
    -d "$PAYLOAD" >/dev/null
  BODY_UPDATED=true
else
  warn "Issue body is not empty; skipping template apply to preserve author content."
fi

echo "author=$AUTHOR_LOGIN, issue=#$ISSUE_NUMBER, template='$TEMPLATE_PATH', body_updated=$BODY_UPDATED, labels=${FINAL_LABELS[*]}"