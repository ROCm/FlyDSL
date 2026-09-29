"""Conservative task/DAG audit, including CTA program order and TP8 waits.

This checks the proposed schedule, not GPU memory ordering or numerical accuracy.
Readiness fan-in vertices avoid expanding every producer/consumer pair.
Input readiness conservatively waits all projection tasks, even where a head
needs fewer rows. Barriers with publication before polling use separate events.
"""
import ast
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class DAG:
    def __init__(self):
        self.ids = {}
        self.names = []
        self.out = []
        self.inc = []
        self.last = {}

    def vertex(self, name):
        if name not in self.ids:
            self.ids[name] = len(self.names)
            self.names.append(name)
            self.out.append(set())
            self.inc.append(set())
        return self.ids[name]

    def edge(self, before, after):
        a, b = self.vertex(before), self.vertex(after)
        self.out[a].add(b)
        self.inc[b].add(a)

    def event(self, rank, cta, name, waits=(), publishes=()):
        key = (rank, cta)
        self.vertex(name)
        if key in self.last:
            self.edge(self.last[key], name)
        self.last[key] = name
        for wait in waits:
            self.edge(wait, name)
        for publish in publishes:
            self.edge(name, publish)
        return name

    def check(self):
        indegree = [len(x) for x in self.inc]
        ready = deque(i for i, n in enumerate(indegree) if n == 0)
        count = 0
        while ready:
            i = ready.popleft()
            count += 1
            for j in self.out[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    ready.append(j)
        assert count == len(self.names), [self.names[i] for i, n in enumerate(indegree) if n][:30]

    def ancestors(self, name):
        seen = set()
        todo = [self.vertex(name)]
        while todo:
            i = todo.pop()
            for j in self.inc[i]:
                if j not in seen:
                    seen.add(j)
                    todo.append(j)
        return {self.names[i] for i in seen}


def audit(group, name=None):
    name = name or f"token_g{group}"
    source = ROOT / f"{name}/kernels/kimi_k3_monokernel/kernel.py"
    tree = ast.parse(source.read_text())
    constants = {n.targets[0].id: n.value.value for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and isinstance(n.value, ast.Constant)}
    assert [constants[k] for k in ("_TOKEN_GROUP", "_FRONT_CTAS", "_BACK_CTAS", "_BLOCKS")] == [group, 36, 220, 256]
    # Guard the source expressions that the schedule enumerator interprets.
    code = source.read_text()
    for expression in (
        "for token_group in range(samples // _TOKEN_GROUP):",
        "post_task = (backend_id + _BACK_CTAS - _OUTPUT_TASKS) % _BACK_CTAS",
        "projection_task = (backend_id + _BACK_CTAS - (_OUTPUT_TASKS + _TOKEN_GROUP * _ATTN_RES_CTAS)) % _BACK_CTAS",
        "selector_task = (backend_id + 1) % _BACK_CTAS",
        "shared_sample = (backend_id + 1 + _TOKEN_GROUP) % _BACK_CTAS",
        "norm_sample = (backend_id + 1) % _BACK_CTAS",
    ):
        assert expression in code, expression
    d = DAG()
    cover = Counter()
    for rank in range(8):
        for cta in range(256):
            def event(kind, *indices, waits=(), publishes=()):
                return d.event(rank, cta, (kind, rank, *indices), waits, publishes)

            # Pre-AttnRes: all chunk statistics must be published before
            # each chunk may finish normalization and produce hidden rows.
            if cta < 16:
                sample, chunk = divmod(cta, 4)
                event("pre_stats", sample, chunk, publishes=[("pre_stats_ready", rank, sample)])
                event("pre", sample, chunk, waits=[("pre_stats_ready", rank, sample)],
                      publishes=[("pre_ready", rank)])
            if cta < 200:
                event("input", cta, waits=[("pre_ready", rank)], publishes=[("input_ready", rank)])
                cover["input", rank, cta] += 1
            if cta < 12:
                for sample in range(4):
                    event("conv", sample, cta, waits=[("input_ready", rank)],
                          publishes=[("conv_ready", rank, sample, cta)])
                    cover["conv", rank, sample, cta] += 1
            elif cta < 36:
                head, split = divmod(cta - 12, 2)
                for sample in range(4):
                    event("state_publish", sample, head, split,
                          waits=[("conv_ready", rank, sample, head)],
                          publishes=[("state_ready", rank, sample, head)])
                    event("kda_norm", sample, head, split,
                          waits=[("state_ready", rank, sample, head)],
                          publishes=[("norm_ready", rank, sample)])
                    cover["recurrence", rank, sample, head, split] += 1
            else:
                bid = cta - 36
                for g in range(4 // group):
                    samples = list(range(g * group, (g + 1) * group))
                    if bid < 112:
                        event("output_push", g, bid,
                              waits=[("norm_ready", rank, s) for s in samples],
                              publishes=[("output_tp", g, bid)])
                        event("output", g, bid, waits=[("output_tp", g, bid)],
                              publishes=[("attention_ready", rank, s) for s in samples])
                        for s in samples:
                            cover["output", rank, s, bid] += 1
                    post = (bid + 220 - 112) % 220
                    if post < group * 4:
                        s, chunk = g * group + post // 4, post % 4
                        event("post_stats", s, chunk, waits=[("attention_ready", rank, s)],
                              publishes=[("post_stats_ready", rank, s)])
                        event("post", s, chunk, waits=[("post_stats_ready", rank, s)],
                              publishes=[("moe_ready", rank, s)])
                        cover["post", rank, s, chunk] += 1
                    proj = (bid + 220 - (112 + group * 4)) % 220
                    if proj < 163:
                        kind = "router" if proj < 56 else "latent" if proj < 131 else "shared_gu"
                        event("projection", g, proj,
                              waits=[("moe_ready", rank, s) for s in samples],
                              publishes=[(kind + "_ready", rank, s) for s in samples])
                        for s in samples:
                            cover["projection", rank, s, proj] += 1
                    if (bid + 1) % 220 == 0:
                        event("select", g,
                              waits=[("router_ready", rank, s) for s in samples],
                              publishes=[("selection_ready", rank, s) for s in samples])
                        for s in samples:
                            cover["select", rank, s] += 1
                    shared = (bid + 1 + group) % 220
                    if shared < group:
                        s = g * group + shared
                        event("shared_mid", s, waits=[("shared_gu_ready", rank, s)],
                              publishes=[("shared_mid_ready", rank, s)])
                        cover["shared_mid", rank, s] += 1
                    for up in range(bid, group * 384, 220):
                        s, route, tile = g * group + up // 384, (up // 24) % 16, up % 24
                        event("up", s, route, tile,
                              waits=[("selection_ready", rank, s), ("latent_ready", rank, s)],
                              publishes=[("up_ready", rank, s)])
                        cover["up", rank, s, route, tile] += 1
                    for down in range(bid, group * 224, 220):
                        s, tile = g * group + down // 224, down % 224
                        event("down_push", s, tile, waits=[("up_ready", rank, s)],
                              publishes=[("down_tp", s, tile)])
                        event("down", s, tile, waits=[("down_tp", s, tile)],
                              publishes=[("down_ready", rank, s)])
                        cover["down", rank, s, tile] += 1
                    norm = (bid + 1) % 220
                    if norm < group:
                        s = g * group + norm
                        event("routed_norm", s, waits=[("down_ready", rank, s)],
                              publishes=[("routed_inv", rank, s)])
                        cover["routed_norm", rank, s] += 1
                    for tail in range(bid, group * 448, 220):
                        s, tile = g * group + tail // 448, tail % 448
                        event("tail_push", s, tile,
                              waits=[("routed_inv", rank, s), ("shared_mid_ready", rank, s)],
                              publishes=[("tail_tp", s, tile)])
                        event("tail", s, tile, waits=[("tail_tp", s, tile)],
                              publishes=[("final_ready", rank, s)])
                        cover["tail", rank, s, tile] += 1
    # Each publish/wait fan-in must have at least one actual producer.
    for i, name in enumerate(d.names):
        if "ready" in name[0] or name[0] in ("output_tp", "down_tp", "tail_tp", "routed_inv"):
            assert d.inc[i], ("unproduced", name)
    expected = dict(input=200 * 8, conv=4 * 12 * 8, recurrence=4 * 24 * 8,
                    output=4 * 112 * 8, post=4 * 4 * 8, projection=4 * 163 * 8,
                    select=4 * 8, shared_mid=4 * 8, up=4 * 384 * 8,
                    down=4 * 224 * 8, routed_norm=4 * 8, tail=4 * 448 * 8)
    assert set(cover.values()) == {1}, "Duplicate task ownership"
    counts = Counter(key[0] for key in cover)
    assert dict(counts) == expected, (counts, expected)
    d.check()
    ancestors = d.ancestors(("final_ready", 0, 0))
    kda_tokens = sorted({name[2] for name in ancestors if name[0] == "kda_norm"})
    assert kda_tokens == list(range(group)), kda_tokens
    return dict(variant=name, group=group, tasks=dict(counts), nodes=len(d.names),
                edges=sum(map(len, d.out)), acyclic=True,
                token0_final_requires_kda_tokens=kda_tokens,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())


if __name__ == "__main__":
    names = [(g, f'token_g{g}') for g in (1, 2, 4)]
    names += [(1 if 'g1' in n else 2, n) for n in
              ['token_g1_scoped', 'token_g2_scoped', 'token_g1_sd_scoped', 'token_g1_sd_pf_scoped']]
    result = dict(variants=[audit(g, name) for g, name in names],
                  scope="S4 MTP full MonoKernel TP8; conservative static schedule model",
                  limitations=["Not a proof of compiler lowering or GPU memory ordering",
                               "No GPU correctness/performance measured by this audit"])
    (ROOT / "dependency_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
