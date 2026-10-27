#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Start or stop the model gateway (gateway.py) that run_one.sh's sandboxed agents call instead of
# the model API.
#
# usage: gateway.sh start <name>   needs API_KEY, API_URL and FLYDSL_REVIEW_SANDBOX_IMAGE (any image
#                                  with python3); API_USER is optional. Prints the
#                                  FLYDSL_REVIEW_SANDBOX_NETWORK and ANTHROPIC_BASE_URL lines
#                                  run_one.sh needs, in KEY=value form.
#        gateway.sh stop <name>    prints the gateway's log, then removes the gateway and its networks.
#
# The agents join only the network <name>. It is internal, and its bridge has no host address, so
# from it they reach neither the host nor the internet. The gateway also joins <name>-egress for the
# way out, holds the key, and listens only on its <name> address, which the host cannot route to.
set -euo pipefail

SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-}"
NAME="${2:?usage: gateway.sh start|stop <name>}"
GW="$NAME-gateway"

case "$ACTION" in
  start)
    : "${API_KEY:?}" "${API_URL:?}" "${FLYDSL_REVIEW_SANDBOX_IMAGE:?}"
    docker network create --internal -o com.docker.network.bridge.inhibit_ipv4=true "$NAME" >/dev/null
    docker network create "$NAME-egress" >/dev/null
    # `-e VAR` without a value copies VAR from this environment, so the key is on no command line.
    docker run -d --rm --name "$GW" --network "$NAME" --user 65534:65534 --read-only --cap-drop ALL \
      --security-opt no-new-privileges --cpus 1 --memory 256m --pids-limit 64 -e API_KEY -e API_URL -e API_USER \
      -v "$SKILL/gateway.py:/gateway.py:ro" --entrypoint python3 "$FLYDSL_REVIEW_SANDBOX_IMAGE" /gateway.py \
      >/dev/null
    # gateway.py binds the address it has now, before the egress network is attached.
    for _ in $(seq 60); do
      docker logs "$GW" 2>/dev/null | grep -q '^listening on ' && break
      sleep 1
    done
    if ! docker logs "$GW" 2>/dev/null | grep -q '^listening on '; then
      echo "gateway.sh: the gateway did not start" >&2
      docker logs "$GW" >&2 || true
      exit 1
    fi
    docker network connect "$NAME-egress" "$GW"
    echo "FLYDSL_REVIEW_SANDBOX_NETWORK=$NAME"
    echo "ANTHROPIC_BASE_URL=http://$GW:8080"
    ;;
  stop)
    # Request lines and upstream error bodies: the only record of why model calls failed.
    if docker inspect "$GW" >/dev/null 2>&1; then
      docker logs "$GW" 2>&1 | tail -n 50 || true
    fi
    docker rm -f "$GW" >/dev/null 2>&1 || true
    docker network rm "$NAME" "$NAME-egress" >/dev/null 2>&1 || true
    ;;
  *)
    echo "usage: gateway.sh start|stop <name>" >&2
    exit 2
    ;;
esac
