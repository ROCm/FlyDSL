#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Create, rebase and capture patches in thirdparty/llvm-extensions/.
#
# A patch must be a diff on top of its predecessors. They are replayed and
# committed to a throwaway baseline first; diffing without it gives a cumulative
# patch that fails to replay with a misleading "patch does not apply".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLVM_SRC_DIR="${LLVM_SRC_DIR:-$(cd "${REPO_ROOT}/.." && pwd)/llvm-project}"
EXT_DIR="${REPO_ROOT}/thirdparty/llvm-extensions"
BUILD_SCRIPT="${SCRIPT_DIR}/build_llvm.sh"
STATE="${LLVM_SRC_DIR}/.flydsl-extension-in-progress"
BASELINE_SUBJECT="flydsl: patch baseline (throwaway)"

usage() {
    cat >&2 <<'USAGE'
usage: llvm_extension.sh <slug>              # start authoring a new extension
       llvm_extension.sh --rebase <name>     # rebase an existing patch onto the base
       llvm_extension.sh --finish            # capture edits into the patch started above

Typical flow:
    bash scripts/llvm_extension.sh my-fix    # prepares llvm-project, prints next steps
    <edit or add files under llvm-project/>
    bash scripts/llvm_extension.sh --finish
    <add the extension to LLVM_EXTENSIONS in scripts/build_llvm.sh>
USAGE
    exit 2
}

[ $# -ge 1 ] || usage

# "<required|extension> <file>" per line, in apply order.
list_patches() {
    bash "${BUILD_SCRIPT}" --list-patches
}

# The commit build_llvm.sh would build: FLYDSL_LLVM_REF (etc.) if set, else the pin.
base_commit() {
    local ref
    ref="${FLYDSL_LLVM_REF:-${LLVM_REF:-${LLVM_COMMIT:-$(python3 -c "import json; print(json.load(open('${REPO_ROOT}/thirdparty/llvm-build-info.json'))['upstream']['llvm_hash'])")}}}"
    if ! git -C "${LLVM_SRC_DIR}" rev-parse -q --verify "${ref}^{commit}"; then
        echo "Error: ${ref} is not in ${LLVM_SRC_DIR}." >&2
        echo "       Run scripts/build_llvm.sh with the same FLYDSL_LLVM_REF (a commit SHA) first." >&2
        exit 1
    fi
}

# --finish stages everything the author created, so keep build outputs and the
# state file out of it. info/exclude is local to the checkout, not an LLVM edit.
ignore_build_outputs() {
    local exclude p
    exclude="$(git rev-parse --git-path info/exclude)"
    mkdir -p "$(dirname "${exclude}")"
    for p in /mlir_install/ /mlir_install.tgz /.flydsl-extension-in-progress; do
        grep -qxF "${p}" "${exclude}" 2>/dev/null || echo "${p}" >> "${exclude}"
    done
}

# Restore the checkout if a run dies before the baseline is ready.
ROLLBACK_TO=""
rollback() {
    [ -n "${ROLLBACK_TO}" ] || return 0
    echo "Interrupted; restoring ${LLVM_SRC_DIR} to ${ROLLBACK_TO}" >&2
    git -C "${LLVM_SRC_DIR}" reset -q --hard "${ROLLBACK_TO}" || true
    rm -f "${STATE}"
}
trap rollback EXIT INT TERM

# Reset to the base, replay the series up to (not including) $2, and commit that
# as the baseline. $1 is recorded as the patch being authored.
prepare_tree() {
    local target="$1" stop_at="${2:-}" base patches kind name
    base="$(base_commit)"
    patches="$(list_patches)"
    cd "${LLVM_SRC_DIR}"
    if [ -f "${STATE}" ]; then
        echo "Error: a patch is already in progress: $(cat "${STATE}")" >&2
        echo "       Finish it with --finish, or discard it with:" >&2
        echo "         rm ${STATE} && git -C ${LLVM_SRC_DIR} reset --hard ${base}" >&2
        exit 1
    fi
    ignore_build_outputs
    ROLLBACK_TO="${base}"
    git reset -q --hard "${base}"   # moves HEAD, unlike `git checkout -- .`
    echo "Replaying onto ${base} ..."
    while read -r kind name; do
        [ -n "${name}" ] || continue
        [ "${name}" = "${stop_at}" ] && break
        echo "  ${name} (${kind})"
        git apply --index "${EXT_DIR}/${name}"
    done <<< "${patches}"
    git -c user.email=flydsl@localhost -c user.name=flydsl \
        commit -q -m "${BASELINE_SUBJECT}" --allow-empty
    # STATE before disarming the trap, so a failure always leaves one or the other.
    echo "${target}" > "${STATE}"
    ROLLBACK_TO=""
}

case "$1" in
  --finish)
    [ -f "${STATE}" ] || { echo "No patch in progress. Run llvm_extension.sh <slug> first." >&2; exit 1; }
    target="$(cat "${STATE}")"
    cd "${LLVM_SRC_DIR}"
    if [ "$(git log -1 --format=%s 2>/dev/null)" != "${BASELINE_SUBJECT}" ]; then
        echo "Error: the baseline commit is gone from ${LLVM_SRC_DIR}; capturing now" >&2
        echo "       would write a cumulative patch that does not replay." >&2
        echo "       Start over: rm ${STATE} && bash scripts/llvm_extension.sh ${target%.patch}" >&2
        exit 1
    fi
    git add -A   # new files too; a plain `git diff` would drop them
    if git diff --cached --quiet; then
        echo "Error: no changes in ${LLVM_SRC_DIR}; nothing to capture." >&2
        exit 1
    fi
    git diff --cached > "${EXT_DIR}/${target}"
    git reset -q --hard HEAD~1   # back to the base the baseline sits on
    rm -f "${STATE}"
    echo "Wrote ${EXT_DIR}/${target}"
    if list_patches | awk '{print $2}' | grep -qxF "${target}"; then
        echo "Already listed in scripts/build_llvm.sh."
    else
        echo ""
        echo "NEXT: add it to the LLVM_EXTENSIONS array in scripts/build_llvm.sh,"
        echo "      with a comment saying what it does and its upstream status:"
        echo ""
        echo "          ${target}"
        echo ""
        echo "The build fails until you do -- an unlisted patch is never applied."
    fi
    ;;
  --rebase)
    [ $# -eq 2 ] || usage
    target="$2"
    [ -f "${EXT_DIR}/${target}" ] || { echo "No such patch: ${target}" >&2; exit 1; }
    prepare_tree "${target}" "${target}"
    echo ""
    echo "Tree is at the base with the preceding patches applied."
    echo "Re-apply ${target} by hand (it did not apply cleanly), then run:"
    echo "  bash scripts/llvm_extension.sh --finish"
    ;;
  -h|--help)
    usage
    ;;
  *)
    target="$1.patch"
    if [ -e "${EXT_DIR}/${target}" ]; then
        echo "Error: ${EXT_DIR}/${target} already exists." >&2
        echo "       To modify it, use: llvm_extension.sh --rebase ${target}" >&2
        exit 1
    fi
    prepare_tree "${target}"
    echo ""
    echo "Tree is at the base with all patches applied."
    echo "Edit or add files under ${LLVM_SRC_DIR}, then run:"
    echo "  bash scripts/llvm_extension.sh --finish"
    echo "to capture them as ${target}."
    ;;
esac
