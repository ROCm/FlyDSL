#!/usr/bin/env bash
# ci_dashboard_local.sh — cross-check FlyDSL on a local gfx950 / MI350X box.
#
# Runs scripts/run_benchmark.sh on this machine, parses the output with the *same*
# parser CI uses (ci-dashboard/ingest/parse_bench.py), tags the records
# source=local-gfx950, and writes ci-dashboard/data/local.json. With --push it
# publishes local.json to the dashboard data branch so the "Local gfx950" tab can
# show your numbers beside CI's.
#
# Run from a built FlyDSL tree (see docs/installation.rst). Examples:
#   bash scripts/ci_dashboard_local.sh                       # bench all, write local.json
#   bash scripts/ci_dashboard_local.sh --only softmax,layernorm,rmsnorm,gemm
#   bash scripts/ci_dashboard_local.sh --push                # also push to the data branch
#   bash scripts/ci_dashboard_local.sh --push --remote origin --branch ci-dashboard-data
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONLY=""; PUSH=0; REMOTE="origin"; BRANCH="ci-dashboard-data"; ARCH="gfx950"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)   ONLY="$2"; shift 2 ;;
    --push)   PUSH=1; shift ;;
    --remote) REMOTE="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --arch)   ARCH="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo local)"
HOST="$(hostname)"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

echo "[local] arch=$ARCH commit=${COMMIT:0:8} host=$HOST"
echo "[local] running scripts/run_benchmark.sh ${ONLY:+--only $ONLY} ..."
BENCH_ARGS=(); [[ -n "$ONLY" ]] && BENCH_ARGS=(--only "$ONLY")
bash scripts/run_benchmark.sh "${BENCH_ARGS[@]}" 2>&1 | tee "$LOG"

OUT="$REPO_ROOT/ci-dashboard/data/local.json"
echo "[local] parsing -> $OUT"
python3 ci-dashboard/ingest/parse_bench.py "$LOG" \
  --runner "local-${HOST}" --arch "$ARCH" --source local-gfx950 \
  --commit "$COMMIT" --ts "$TS" --out /tmp/local_records.json

python3 - "$OUT" "$TS" <<'PY'
import json, os, sys
out, ts = sys.argv[1], sys.argv[2]
recs = json.load(open("/tmp/local_records.json"))
# merge with any existing local.json (newest per runner+kernel+metric wins)
prev = []
if os.path.exists(out):
    try: prev = json.load(open(out)).get("records", [])
    except Exception: pass
key = lambda r: (r["runner"], r["op"], r["shape"], r["dtype"], r["metric"])
merged = {key(r): r for r in prev}
for r in recs: merged[key(r)] = r
json.dump({"schema": 1, "updated": ts, "repo": "local", "records": list(merged.values())},
          open(out, "w"), separators=(",", ":"))
print(f"[local] wrote {len(recs)} fresh records ({len(merged)} total) -> {out}")
PY

if [[ "$PUSH" == "1" ]]; then
  echo "[local] publishing local.json to ${REMOTE}/${BRANCH}"
  URL="$(git remote get-url "$REMOTE")"
  PUB="$(mktemp -d)"
  ( cd "$PUB"
    git init -q
    git config user.name "$(git -C "$REPO_ROOT" config user.name || echo flydsl-local)"
    git config user.email "$(git -C "$REPO_ROOT" config user.email || echo local@flydsl)"
    git remote add origin "$URL"
    if git fetch --depth=1 origin "$BRANCH" 2>/dev/null; then git checkout -q "$BRANCH";
    else git checkout -q --orphan "$BRANCH"; git rm -rfq . 2>/dev/null || true; fi
    cp "$OUT" ./local.json
    git add local.json
    git commit -qm "ci-dashboard: local gfx950 cross-check ${TS} [skip ci]" || { echo "[local] no change"; exit 0; }
    git push -q origin "HEAD:${BRANCH}" && echo "[local] pushed to ${BRANCH}" ) \
    || echo "[local] push skipped/failed (no write access to ${REMOTE}? local.json is saved at ${OUT})"
  rm -rf "$PUB"
fi
echo "[local] done."
