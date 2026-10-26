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

PULL_REQUEST_NUMBER="$(jq -r '.pull_request.number // empty' "$EVENT_PATH")"
AUTHOR_LOGIN="$(jq -r '.pull_request.user.login // empty' "$EVENT_PATH")"
HEAD_SHA="$(jq -r '.pull_request.head.sha // empty' "$EVENT_PATH")"
BRANCH_NAME="$(jq -r '.pull_request.head.ref // empty' "$EVENT_PATH")"

if [[ -z "$PULL_REQUEST_NUMBER" || -z "$AUTHOR_LOGIN" || -z "$HEAD_SHA" || -z "$BRANCH_NAME" ]]; then
  fail "No pull_request payload found or required fields are missing."
fi

COMMITS_RESPONSE="$(curl -fsSL \
  -H "Accept: application/vnd.github+json" \
  -H "$AUTH_HEADER" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "$API_URL/repos/$REPO/pulls/$PULL_REQUEST_NUMBER/commits")"

COMMIT_SUMMARY="$(printf '%s' "$COMMITS_RESPONSE" | jq -r '.[] | [.sha, (.commit.message | split("\n")[0])] | @tsv' | awk -F $'\t' '{print $1 " : " $2}' | paste -sd '\n' -)"
if [[ -z "$COMMIT_SUMMARY" ]]; then
  fail "Failed to load commit summary for the pull request."
fi

CHANGED_FILES_RESPONSE="$(curl -fsSL \
  -H "Accept: application/vnd.github+json" \
  -H "$AUTH_HEADER" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "$API_URL/repos/$REPO/pulls/$PULL_REQUEST_NUMBER/files?per_page=100")"

DOC_ONLY=true
if [[ -n "$CHANGED_FILES_RESPONSE" && "$CHANGED_FILES_RESPONSE" != "[]" ]]; then
  while IFS= read -r file_name; do
    [[ -n "$file_name" ]] || continue
    case "$file_name" in
      *.md|*.mdx|*.markdown) ;;
      *) DOC_ONLY=false ;;
    esac
  done <<< "$(printf '%s' "$CHANGED_FILES_RESPONSE" | jq -r '.[].filename // empty')"
fi

BRANCH_NAME_LOWER="$(printf '%s' "$BRANCH_NAME" | tr '[:upper:]' '[:lower:]')"
LABELS_TO_ADD=()

if [[ "$DOC_ONLY" == true ]]; then
  LABELS_TO_ADD+=("document")
else
  LABELS_TO_ADD+=("enhancement")
fi

case "$BRANCH_NAME_LOWER" in
  *refactor*) LABELS_TO_ADD+=("refactor") ;;
  *bug*|*fix*|*hotfix*|*bugfix*) LABELS_TO_ADD+=("bug") ;;
  *release*) LABELS_TO_ADD+=("release") ;;
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
  "$API_URL/repos/$REPO/issues/$PULL_REQUEST_NUMBER/labels" \
  -d "$(jq -n --argjson labels "$LABELS_JSON" '{labels: $labels}')" >/dev/null

ASSIGNEE_INPUT="${PR_ASSIGNEES:-$AUTHOR_LOGIN}"
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
    "$API_URL/repos/$REPO/issues/$PULL_REQUEST_NUMBER" \
    -d "$(jq -n --argjson assignees "$ASSIGNEES_JSON" '{assignees: $assignees}')" >/dev/null
fi

TEMPLATE_PATH=".github/PULL_REQUEST_TEMPLATE/pull_request_template.md"
if [[ ! -f "$TEMPLATE_PATH" ]]; then
  fail "PR template not found: $TEMPLATE_PATH"
fi
if [[ ! -r "$TEMPLATE_PATH" ]]; then
  fail "Pull request template is not readable: $TEMPLATE_PATH"
fi

# PR 템플릿 내용을 읽어와서 변수에 저장
PR_BODY="$(cat "$TEMPLATE_PATH")"
# PR 템플릿 내 __COMMIT_SHA__를 실제 커밋 요약으로 치환
PR_BODY="${PR_BODY//__COMMIT_SHA__/$COMMIT_SUMMARY}"

# PR 템플릿 내 __BRANCH_NAME__를 실제 브랜치 이름으로 치환
PAYLOAD="$(jq -n --arg body "$PR_BODY" '{body: $body}')"

curl -fsSL -X PATCH \
  -H "Accept: application/vnd.github+json" \
  -H "$AUTH_HEADER" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -H "Content-Type: application/json" \
  "$API_URL/repos/$REPO/pulls/$PULL_REQUEST_NUMBER" \
  -d "$PAYLOAD" >/dev/null

echo "author=$AUTHOR_LOGIN, template='$TEMPLATE_PATH', branch=$BRANCH_NAME, labels=${FINAL_LABELS[*]}"
