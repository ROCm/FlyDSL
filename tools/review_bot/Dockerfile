# syntax=docker/dockerfile:1

# linux/amd64 manifest for node:22-bookworm-slim, resolved 2026-09-17.
ARG BASE_IMAGE=node:22-bookworm-slim@sha256:4d676821dff059fd00d277ee4261ef34ea712317fed0737c03941481b5760c96
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG CLAUDE_CODE_VERSION=2.1.274
ARG CLAUDE_CODE_INTEGRITY=sha512-UI8TGoOO0fYT38VSoAjtu9C0EQkOwgwA4+ETFCgQhTO9NZvpgCERANL0UzioEW89b2zbcoL1B3z+7W37QdBxOA==
ARG SANDBOX_RUNTIME_VERSION=0.0.76
ARG SANDBOX_RUNTIME_INTEGRITY=sha512-wIC++TvCopRPJcX/v/PCp/SjYOWk/cAVq+VF9v6E3WwkriFP9uu2wvPJ2E+wMrNa3i3h3ew9zTpHcmyGLBKo0w==

USER root

RUN node -e \
      'for (const value of process.argv.slice(1)) if (!/^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$/.test(value)) process.exit(1)' \
      "${CLAUDE_CODE_VERSION}" "${SANDBOX_RUNTIME_VERSION}" \
    && node -e \
      'if (!/@sha256:[0-9a-f]{64}$/.test(process.argv[1])) process.exit(1)' \
      "${BASE_IMAGE}"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bubblewrap \
        ca-certificates \
        git \
        python3 \
        ripgrep \
        socat \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN test "$(npm view "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" dist.integrity)" = "${CLAUDE_CODE_INTEGRITY}" \
    && test "$(npm view "@anthropic-ai/sandbox-runtime@${SANDBOX_RUNTIME_VERSION}" dist.integrity)" = "${SANDBOX_RUNTIME_INTEGRITY}" \
    && npm install --global \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
        "@anthropic-ai/sandbox-runtime@${SANDBOX_RUNTIME_VERSION}" \
    && npm cache clean --force

RUN groupadd --gid 10001 reviewer \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin reviewer \
    && install -d -o reviewer -g reviewer -m 0700 \
        /home/reviewer/.claude \
        /review-input \
        /review-run \
        /workspace/source \
        /opt/review-engine

COPY --chmod=0555 run-review.sh /usr/local/bin/run-review.sh

USER reviewer:reviewer
WORKDIR /workspace/source

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/run-review.sh"]
