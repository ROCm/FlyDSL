# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The switch requirement for LLVM extensions.

Each case here is a spelling the checker has to get right. The two rejection
cases are the point: an extension with no switch, and one that declares a
switch and then changes behavior unconditionally anyway.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER = _REPO_ROOT / "scripts" / "check_llvm_extensions.py"

_spec = importlib.util.spec_from_file_location("check_llvm_extensions", _CHECKER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_cl_opt_on_one_line_is_a_switch():
    assert _mod.has_switch("""\
--- a/x.cpp
+++ b/x.cpp
+static cl::opt<bool> EnableFlyThing(
+    "fly-thing", cl::Hidden, cl::init(false),
+    cl::desc("FlyDSL: enable the thing"));
+
 void f() {
-  old();
+  if (EnableFlyThing) { neu(); } else { old(); }
 }
""")


def test_cl_opt_wrapped_onto_the_next_line_is_a_switch():
    assert _mod.has_switch("""\
--- a/y.cpp
+++ b/y.cpp
+static cl::opt<bool>
+    FlyWiden("fly-widen", cl::desc("FlyDSL: widen"), cl::init(false));
+
 void g() {
+  if (!FlyWiden) return;
 }
""")


def test_getenv_guard_is_a_switch():
    assert _mod.has_switch("""\
--- a/z.cpp
+++ b/z.cpp
 void h() {
+  if (std::getenv("FLYDSL_ENABLE_Z") == nullptr)
+    return legacy();
   neu();
 }
""")


def test_unconditional_change_is_rejected():
    assert not _mod.has_switch("""\
--- a/w.cpp
+++ b/w.cpp
 void k() {
-  old();
+  neu();
 }
""")


def test_declared_but_unused_switch_is_rejected():
    """A switch nothing branches on does not make the change disablable."""
    assert not _mod.has_switch("""\
--- a/v.cpp
+++ b/v.cpp
+static cl::opt<bool> DumpStats("fly-dump-stats", cl::init(false));
+
 void m() {
-  old();
+  neu();
 }
""")


def test_a_switch_in_context_does_not_count():
    """The extension must ADD the switch, not reuse one already in the file.

    The added line does branch on ``Existing``, so only the fact that its
    declaration is context -- not an added line -- makes this a rejection.
    """
    assert not _mod.has_switch("""\
--- a/u.cpp
+++ b/u.cpp
 static cl::opt<bool> Existing("other", cl::init(false));
 void n() {
-  old();
+  if (Existing) neu();
 }
""")


def test_every_shipped_extension_satisfies_the_rule():
    """Anything not grandfathered must carry a switch."""
    ext_dir = _REPO_ROOT / "thirdparty" / "llvm-extensions"
    for ext in sorted(ext_dir.glob("*.patch")):
        if ext.name in _mod.GRANDFATHERED:
            continue
        assert _mod.has_switch(ext.read_text(encoding="utf-8")), f"{ext.name} has no switch"


def test_grandfathered_list_still_matches_reality():
    """A grandfathered entry that is gone should leave the list, not linger."""
    ext_dir = _REPO_ROOT / "thirdparty" / "llvm-extensions"
    on_disk = {p.name for p in ext_dir.glob("*.patch")}
    assert _mod.GRANDFATHERED <= on_disk
