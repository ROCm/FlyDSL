#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Create, rebase and capture LLVM extensions in thirdparty/llvm-extensions/.
#
# This exists because the obvious recipe is wrong. An extension must be a diff
# against the tree with the preceding ones applied, but those are left as
# uncommitted work-tree edits (HEAD stays on the pin), so a plain `git diff`
# emits a CUMULATIVE diff that re-applies the earlier hunks. Replaying such a
# series fails with "patch does not apply" -- which reads as staleness and sends
# the author looking in the wrong place. Committing the preceding extensions to
# a throwaway commit first is what makes `git diff` incremental.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLVM_SRC_DIR="${LLVM_SRC_DIR:-$(cd "${REPO_ROOT}/.." && pwd)/llvm-project}"
EXT_DIR="${REPO_ROOT}/thirdparty/llvm-extensions"
BUILD_SCRIPT="${SCRIPT_DIR}/build_llvm.sh"

usage() {
    cat >&2 <<'USAGE'
usage: llvm_extension.sh <slug>              # start authoring a new extension
       llvm_extension.sh --rebase <name>     # rebase an existing extension onto the current pin
       llvm_extension.sh --finish            # capture edits into the extension started above

Typical flow:
    bash scripts/llvm_extension.sh my-fix    # prepares llvm-project, prints next steps
    <edit files under llvm-project/>
    bash scripts/llvm_extension.sh --finish
    <add the extension to LLVM_EXTENSIONS in scripts/build_llvm.sh>
USAGE
    exit 2
}

[ $# -ge 1 ] || usage

STATE="${LLVM_SRC_DIR}/.flydsl-extension-in-progress"

# The apply order lives in the arrays in build_llvm.sh, so that is what is read
# here -- a second list would be a second source of truth.
read_required() {
    local body
    body="$(sed -n '/^REQUIRED_PATCHES=(/,/^)/p' "${BUILD_SCRIPT}")"
    if [ -z "${body}" ]; then
        echo "Error: could not find the REQUIRED_PATCHES array in ${BUILD_SCRIPT}" >&2
        exit 1
    fi
    local -a REQUIRED_PATCHES=()
    eval "${body}"
    [ ${#REQUIRED_PATCHES[@]} -eq 0 ] || printf '%s\n' "${REQUIRED_PATCHES[@]}"
}

#
# Bash parses the array rather than sed: a sed version disagrees with the shell
# on quoted entries, a one-line array and an indented closing paren, and a
# parser that silently differs from what the build actually applies is worse
# than no parser. Only the array assignment is evaluated, not the script.
read_extensions() {
    local body
    body="$(sed -n '/^LLVM_EXTENSIONS=(/,/^)/p' "${BUILD_SCRIPT}")"
    if [ -z "${body}" ]; then
        echo "Error: could not find the LLVM_EXTENSIONS array in ${BUILD_SCRIPT}" >&2
        exit 1
    fi
    local -a LLVM_EXTENSIONS=()
    eval "${body}"
    [ ${#LLVM_EXTENSIONS[@]} -eq 0 ] || printf '%s\n' "${LLVM_EXTENSIONS[@]}"
}

pin() {
    # The extended pin: extensions are diffs against the commit they are meant
    # to apply to, which is not necessarily the baseline the default build uses.
    python3 -c "
import json, sys
d = json.load(open('${REPO_ROOT}/thirdparty/llvm-build-info.json'))
e = d.get('extended') or d.get('upstream')
if not e:
    sys.exit(\"llvm-build-info.json has neither an 'extended' nor an 'upstream' pin\")
print(e['llvm_hash'])"
}

# An interrupted run must not leave the throwaway commit and a half-reset tree
# behind: --finish would then diff against the wrong baseline. The trap fires on
# error and on Ctrl-C, and is cleared once the tree is in its intended state.
ROLLBACK_TO=""
rollback() {
    [ -n "${ROLLBACK_TO}" ] || return 0
    echo "Interrupted; restoring ${LLVM_SRC_DIR} to ${ROLLBACK_TO}" >&2
    git -C "${LLVM_SRC_DIR}" reset -q --hard "${ROLLBACK_TO}" || true
    rm -f "${STATE}"
}
trap rollback EXIT INT TERM

# Reset to the pin and replay the series up to (but excluding) $1, then commit
# that state so a later `git diff` is incremental rather than cumulative.
# $1 = the extension being authored (written to STATE); $2 = stop replaying before it.
prepare_tree() {
    local target="$1"
    local stop_at="${2:-}"
    local llvm_pin
    llvm_pin="$(pin)"
    cd "${LLVM_SRC_DIR}"
    # Starting over discards whatever the tree held; refuse when that would
    # throw away an unfinished extension rather than doing it silently.
    if [ -f "${STATE}" ]; then
        echo "Error: an extension is already in progress: $(cat "${STATE}")" >&2
        echo "       Finish it with --finish, or discard it with:" >&2
        echo "         rm ${STATE} && git -C ${LLVM_SRC_DIR} reset --hard ${llvm_pin}" >&2
        exit 1
    fi
    ROLLBACK_TO="${llvm_pin}"
    # Move HEAD, not just the working tree: `git checkout -- .` restores paths
    # from the index and leaves HEAD wherever it was, so a leftover baseline
    # commit would become the parent of the next one and every extension authored
    # here would be a diff against the wrong base.
    git reset -q --hard "${llvm_pin}"
    echo "Replaying onto ${llvm_pin} ..."
    while IFS= read -r name; do
        echo "  ${name}"
        git apply "${EXT_DIR}/${name}"
    done < <(read_required)
    while IFS= read -r name; do
        [ "${name}" = "${stop_at}" ] && break
        echo "  ${name}"
        git apply "${EXT_DIR}/${name}"
    done < <(read_extensions)
    # Throwaway commit: this is the whole point of the script.
    git add -A
    git -c user.email=flydsl@localhost -c user.name=flydsl \
        commit -q -m "flydsl: patch baseline (throwaway)" --allow-empty
    # Record the in-progress extension before disarming the trap. Clearing
    # ROLLBACK_TO any earlier would leave a window where a failure neither
    # rolls the tree back nor leaves a STATE file for the guard to find.
    echo "${target}" > "${STATE}"
    ROLLBACK_TO=""
}

case "$1" in
  --finish)
    [ -f "${STATE}" ] || { echo "No extension in progress. Run llvm_extension.sh <slug> first." >&2; exit 1; }
    target="$(cat "${STATE}")"
    cd "${LLVM_SRC_DIR}"
    # Without the baseline commit, `git diff` is against the pin and yields a
    # CUMULATIVE patch that re-applies the preceding extensions' hunks -- the exact
    # failure this script exists to prevent, and it would pass review silently.
    if [ "$(git log -1 --format=%s 2>/dev/null)" != "flydsl: patch baseline (throwaway)" ]; then
        echo "Error: the extension baseline commit is gone from ${LLVM_SRC_DIR}." >&2
        echo "       Capturing now would produce a cumulative patch that does not replay." >&2
        echo "       Start over: rm ${STATE} && bash scripts/llvm_extension.sh ${target%.patch}" >&2
        exit 1
    fi
    if git diff --quiet; then
        echo "Error: no changes in ${LLVM_SRC_DIR}; nothing to capture." >&2
        exit 1
    fi
    git diff > "${EXT_DIR}/${target}"
    # Return the checkout to the pin so the next build_llvm.sh run is correct.
    git reset -q --hard "$(pin)"
    rm -f "${STATE}"
    echo "Wrote ${EXT_DIR}/${target}"
    if read_extensions | grep -qxF "${target}"; then
        echo "Already listed in LLVM_EXTENSIONS."
    else
        echo ""
        echo "NEXT: add it to the LLVM_EXTENSIONS array in scripts/build_llvm.sh,"
        echo "      with a comment saying what it does and its upstream status:"
        echo ""
        echo "          ${target}"
        echo ""
        echo "The build fails until you do -- an unlisted extension is never applied."
    fi
    ;;
  --rebase)
    [ $# -eq 2 ] || usage
    target="$2"
    [ -f "${EXT_DIR}/${target}" ] || { echo "No such extension: ${target}" >&2; exit 1; }
    prepare_tree "${target}" "${target}"
    echo ""
    echo "Tree is at the pin with the preceding extensions applied."
    echo "Re-apply ${target} by hand (it did not apply cleanly), then run:"
    echo "  bash scripts/llvm_extension.sh --finish"
    ;;
  -h|--help)
    usage
    ;;
  *)
    slug="$1"
    target="${slug}.patch"
    if [ -e "${EXT_DIR}/${target}" ]; then
        echo "Error: ${EXT_DIR}/${target} already exists." >&2
        echo "       To modify it, use: llvm_extension.sh --rebase ${target}" >&2
        exit 1
    fi
    prepare_tree "${target}"
    echo ""
    echo "Tree is at the pin with all extensions applied."
    echo "Edit files under ${LLVM_SRC_DIR}, then run:"
    echo "  bash scripts/llvm_extension.sh --finish"
    echo "to capture them as ${target}."
    ;;
esac
