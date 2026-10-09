#!/bin/bash
# Configure git credentials for private LLVM repos, then run build_llvm.sh.
# Usage: bash scripts/setup_llvm_credentials.sh [build_llvm.sh args...]
#
# Prompts for username and password once, caches them for the session,
# then invokes build_llvm.sh with all forwarded arguments.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== LLVM private repo credentials ==="
read -rp "Username: " LLVM_USER
read -rsp "Password/PAT: " LLVM_PASS
echo

# Read the repository URL from build info JSON using the resolved source.
LLVM_SOURCE="${LLVM_SOURCE:-custom-old}"
for arg in "$@"; do
    if [[ "$arg" == "--source" ]]; then _next=1
    elif [[ "${_next:-}" == "1" ]]; then LLVM_SOURCE="$arg"; unset _next; fi
done

REPO_URL=$(python3 -c "
import json
d = json.load(open('${SCRIPT_DIR}/../thirdparty/llvm-build-info.json'))
entry = d.get('${LLVM_SOURCE}', {})
print(entry.get('repository', ''))
")

if [[ -z "$REPO_URL" ]]; then
    echo "No repository URL found for source '${LLVM_SOURCE}'" >&2
    exit 1
fi

# Inject credentials into the URL: https://user:pass@github.com/...
LLVM_REMOTE=$(echo "$REPO_URL" | sed "s|https://|https://${LLVM_USER}:${LLVM_PASS}@|")
export LLVM_REMOTE

echo "Authenticated remote configured for ${LLVM_SOURCE}."
echo "Running: bash scripts/build_llvm.sh $*"
echo

exec bash "${SCRIPT_DIR}/build_llvm.sh" "$@"
