#!/usr/bin/env python3
"""Flag legacy FlyDSL spellings that the maintainers reliably ask to be replaced.

Distilled from 479 human review comments across all 865 FlyDSL PRs: 46 of them are the
same objection -- "the current spelling exists, use it". The pairs below are taken from
those comments, not invented. Like the D9 scanner this is a candidate list, not a verdict.

usage: scan_legacy_spelling.py <owner/repo> <PR> | --diff <file>
"""
import re, subprocess, sys

RULES = [
 ("raw ir.* / ArithValue",  r"(?<![\w.])ir\.[A-Za-z_]|\b_mlir\.|ArithValue|\barith\.unwrap\b|\bas_ir_value\b",
  "use the internal fx types (fx.Float32 / fx.Int32, expr/numeric.py) instead of raw ir.* / _mlir.* values",
  "coderfeli #202 #250 #300 #326 #426 #850"),
 ("scf.* control flow",     r"(?<![\w.])scf\.(If|For|While|Yield)",
  "use ordinary Python if/for inside the kernel; scf.* is the lowering detail",
  "coderfeli #33 #433"),
 ("buffer_ops.*",           r"(?<![\w.])buffer_ops\.",
  "use fx.copy / a copy atom; buffer_ops is the pre-layout-API spelling",
  "coderfeli #404 #416 #894 #1032"),
 ("SmemAllocator",          r"\bSmemAllocator\b",
  "SmemAllocator is deprecated -- use SharedAllocator",
  "sjfeng1999 #549 #567"),
 ("make_ptr for retyping",  r"\bmake_ptr\b",
  "to change a pointer's type use recast_iter, not make_ptr",
  "sjfeng1999 #288 #745"),
]
KERNEL_EXT = (".py",)

def get_diff(argv):
    if argv[0] == "--diff":
        return open(argv[1]).read()
    return subprocess.run(["gh","pr","diff",argv[1],"--repo",argv[0]],
                          capture_output=True,text=True).stdout

def scan(diff):
    cur, hits = None, []
    for line in diff.splitlines():
        if line.startswith("+++ b/"): cur = line[6:]
        if not line.startswith("+") or line.startswith("+++") or not cur: continue
        if not cur.endswith(KERNEL_EXT): continue
        code = line[1:]
        if code.strip().startswith("#"): continue
        for name, pat, advice, prov in RULES:
            if re.search(pat, code):
                hits.append((cur, name, code.strip()[:100], advice, prov))
    return hits

def main():
    hits = scan(get_diff(sys.argv[1:]))
    if not hits:
        print("no legacy spellings on added lines"); return
    seen = set()
    print(f"== legacy spellings on added lines: {len(hits)} ==")
    for f, name, code, advice, prov in hits:
        if (f, name) in seen: continue
        seen.add((f, name))
        n = sum(1 for h in hits if h[0] == f and h[1] == name)
        print(f"  {f}\n      {name} x{n} -- {advice}\n      e.g. {code}\n      maintainers: {prov}")

if __name__ == "__main__":
    main()
