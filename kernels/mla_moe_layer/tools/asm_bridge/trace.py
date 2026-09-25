"""Diagnostic timestamps at verified S=1/2/4 Proto1 producer/consumer boundaries."""

# Absolute PCs belong to the symbol recorded by extract(), not another protocol.
POINTS = {
    0x10E00: "entry",
    0x11B34: "q_a_publish",
    0x11BA4: "q_b_wait",
    0x11BD0: "q_b_ready",
    0x12124: "q_nope_publish",
    0x12288: "uk_wait",
    0x122C4: "uk_ready",
    0x124F0: "q_lat_publish",
    0x132F4: "attn_wait",
    0x13428: "attn_ready",
    0x1407C: "attn_publish",
    0x14108: "merge_wait",
    0x1411C: "merge_ready",
    0x142FC: "merge_publish",
    0x1482C: "uv_wait",
    0x14874: "uv_ready",
    0x14D40: "uv_publish",
    0x14FF4: "o_wait",
    0x1503C: "o_ready",
    0x15C20: "attn_reduce_publish",
    0x15D18: "moe_wait",
    0x15D78: "moe_ready",
    0x16C98: "shared_mid_publish",
    0x17828: "router_publish",
    0x17D2C: "router_wait",
    0x17D90: "router_ready",
    0x18BFC: "routed_mid_publish",
    0x18F0C: "down_wait",
    0x18F80: "down_ready",
    0x196BC: "down_computed",
}


def points(samples):
    if samples == 1:
        return POINTS
    addresses = {
        2: (
            0x23900,
            0x247EC,
            0x24880,
            0x248D8,
            0x24E90,
            0x25030,
            0x2509C,
            0x252F8,
            0x261B8,
            0x262EC,
            0x26F48,
            0x26FD4,
            0x26FE8,
            0x271C8,
            0x27700,
            0x27790,
            0x27CC0,
            0x27FA0,
            0x28034,
            0x28CB0,
            0x28DB4,
            0x28E14,
            0x29D30,
            0x2A8C8,
            0x2ADD8,
            0x2AE4C,
            0x2BDF4,
            0x2C1A4,
            0x2C218,
            0x2CB70,
        ),
        4: (
            0x37A00,
            0x38BB4,
            0x38C84,
            0x38D3C,
            0x39424,
            0x39604,
            0x396D0,
            0x39940,
            0x3A830,
            0x3A964,
            0x3B5B4,
            0x3B640,
            0x3B654,
            0x3B834,
            0x3BD90,
            0x3BEB0,
            0x3C46C,
            0x3C748,
            0x3C86C,
            0x3D56C,
            0x3D674,
            0x3D6D4,
            0x3E60C,
            0x3EDD4,
            0x3F2E0,
            0x3F354,
            0x4050C,
            0x408A0,
            0x40914,
            0x41708,
        ),
    }[samples]
    return dict(zip(addresses, POINTS.values(), strict=True))


def instrument(body, vgprs, samples):
    """Use dedicated registers and restore EXEC/SCC around every timestamp.

    Waits drain outstanding memory operations: these timings diagnose ordering
    and stalls, and must never be reported as uninstrumented performance.
    """
    marks = points(samples)
    for pc in marks:
        if f".Ltile_{pc:x}:" not in body:
            raise ValueError("Unexpected Proto1 specialization for the trace map")
    prefix = f"""s_mul_i32 s94, s2, {len(marks) * 8}
v_cmp_eq_u32_e64 s[96:97], 0, v0
"""
    for slot, pc in enumerate(marks):
        stamp = f"""
s_cselect_b32 s95, 1, 0
s_memrealtime s[92:93]
s_waitcnt lgkmcnt(0)
s_mov_b64 s[98:99], exec
s_mov_b64 exec, s[96:97]
v_mov_b32_e32 v{vgprs}, s94
v_mov_b32_e32 v{vgprs + 2}, s92
v_mov_b32_e32 v{vgprs + 3}, s93
global_store_dwordx2 v{vgprs}, v[{vgprs + 2}:{vgprs + 3}], s[90:91] offset:{slot * 8}
s_waitcnt vmcnt(0)
s_mov_b64 exec, s[98:99]
s_cmp_lg_u32 s95, 0
"""
        label = f".Ltile_{pc:x}:"
        # Backedges target the load label; record wait entry only once rather
        # than overwriting the start timestamp on every polling iteration.
        body = body.replace(label, stamp + label if marks[pc].endswith("_wait") else label + stamp)
    return prefix + body
