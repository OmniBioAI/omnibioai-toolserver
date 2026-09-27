# syntax=docker/dockerfile:1
# omnibioai-toolserver/Dockerfile.new
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.source=https://github.com/man4ish/omnibioai

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
# HIPAA-V2-019: requirements.txt pins omnibioai-iam-client (private repo)
# as a git+https dependency, so pip needs git plus a GitHub credential.
# Same pattern as omnibioai-model-registry/Dockerfile: a BuildKit secret
# mount (never an ARG, which BuildKit echoes into build logs) read through
# an ephemeral GIT_ASKPASS helper, so git never builds a credentialed URL
# that a failed clone could print. Without this the image could not be
# built at all, which is how the delegated-auth fix (a4cdf8a) never
# reached the running deployment.
RUN --mount=type=secret,id=github_token \
    set -eu; \
    printf '%s\n' \
      '#!/bin/sh' \
      'case "$1" in' \
      '  *Username*) printf "%s\n" "x-access-token" ;;' \
      '  *Password*) cat /run/secrets/github_token ;;' \
      '  *) exit 1 ;;' \
      'esac' > /tmp/git-askpass; \
    chmod 700 /tmp/git-askpass; \
    trap 'rm -f /tmp/git-askpass' EXIT; \
    GIT_ASKPASS=/tmp/git-askpass GIT_TERMINAL_PROMPT=0 \
      pip install --no-cache-dir -r requirements.txt

COPY toolserver/ ./toolserver/
COPY toolserver_app.py .
COPY scripts/ ./scripts/

EXPOSE 9090
CMD ["uvicorn", "toolserver_app:create_app", "--factory", "--host", "0.0.0.0", "--port", "9090"]