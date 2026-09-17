#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

set -eu
umask 077

if [ "$#" -ne 18 ] \
    || [ "$1" != "--scope-manifest" ] \
    || [ "$2" != "/review-input/scope-manifest.json" ] \
    || [ "$3" != "--execution-profile" ] \
    || [ "$4" != "untrusted-container" ] \
    || [ "$5" != "--model" ] \
    || [ "$6" != "opus" ] \
    || [ "$7" != "--effort" ] \
    || [ "$8" != "max" ] \
    || [ "$9" != "--concurrency" ] \
    || [ "${10}" != "9" ] \
    || [ "${11}" != "--agent-timeout" ] \
    || [ "${12}" != "1200" ] \
    || [ "${13}" != "--phase-timeout" ] \
    || [ "${14}" != "3600" ] \
    || [ "${15}" != "--claude-path" ] \
    || [ "${16}" != "/usr/local/bin/claude" ] \
    || [ "${17}" != "--run-dir" ] \
    || [ "${18}" != "/review-run" ]; then
    echo "invalid fixed review invocation" >&2
    exit 64
fi

exec /usr/bin/env -i \
    HOME=/home/reviewer \
    USER=reviewer \
    LOGNAME=reviewer \
    SHELL=/bin/sh \
    PATH=/usr/local/bin:/usr/bin:/bin \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TMPDIR=/tmp \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
    CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1 \
    ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:?missing model auth token}" \
    ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:?missing model gateway URL}" \
    /usr/bin/python3 \
    /opt/review-engine/.claude/skills/flydsl-code-review/scripts/run_review.py \
    "$@"
