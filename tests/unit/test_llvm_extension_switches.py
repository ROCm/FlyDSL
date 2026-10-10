# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""scripts/check_llvm_extensions.py: every extension needs a run-time switch."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_llvm_extensions", _REPO_ROOT / "scripts" / "check_llvm_extensions.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_ACCEPTED = {
    "cl_opt": """\
+static cl::opt<bool> EnableFlyThing(
+    "fly-thing", cl::Hidden, cl::init(false));
 void f() {
-  old();
+  if (EnableFlyThing) neu(); else old();
""",
    "cl_opt_wrapped": """\
+static cl::opt<bool>
+    FlyWiden("fly-widen", cl::init(false));
 void g() {
+  if (!FlyWiden) return;
""",
    "getenv": """\
 void h() {
+  if (std::getenv("FLYDSL_ENABLE_Z") == nullptr)
+    return legacy();
""",
}

_REJECTED = {
    "no_switch": """\
 void k() {
-  old();
+  neu();
""",
    # Declared, but nothing branches on it.
    "unused_switch": """\
+static cl::opt<bool> DumpStats("fly-dump-stats", cl::init(false));
 void m() {
-  old();
+  neu();
""",
    # Branches on a switch the extension did not add.
    "switch_in_context": """\
 static cl::opt<bool> Existing("other", cl::init(false));
 void n() {
-  old();
+  if (Existing) neu();
""",
}


@pytest.mark.parametrize("diff", _ACCEPTED.values(), ids=_ACCEPTED.keys())
def test_accepted(diff):
    assert _mod.has_switch("--- a/x.cpp\n+++ b/x.cpp\n" + diff)


@pytest.mark.parametrize("diff", _REJECTED.values(), ids=_REJECTED.keys())
def test_rejected(diff):
    assert not _mod.has_switch("--- a/x.cpp\n+++ b/x.cpp\n" + diff)


def test_repo_passes_and_exempts_only_required_patches():
    ext_dir = _REPO_ROOT / "thirdparty" / "llvm-extensions"
    required = _mod.required_patches()

    assert required and required <= {p.name for p in ext_dir.glob("*.patch")}
    assert _mod.main() == 0
