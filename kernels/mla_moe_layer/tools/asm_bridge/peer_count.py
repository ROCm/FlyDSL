"""Adapt the eight-peer Proto1 collective to a smaller fixed-shard group."""


def adapt(body, metadata, samples, npes):
    if npes not in (2, 4):
        raise ValueError("Only 2/4-peer adaptations are needed")
    # Keep the native seven-remote-slot stride. Zero absent slots before the
    # collective, and mask both sends and receives by (remote_slot % 7).
    # The original ordered eight-term sum then adds zero for absent ranks.
    points = {
        1: ((0x15668, 0x17820, 0x15690, 10), (0x196C0, 7168, 0x196F4, 8)),
        2: ((0x286C8, 0x17C40, 0x286EC, 43), (0x2CB74, 6144, 0x2CB9C, 43)),
        4: ((0x3CF78, 0x18480, 0x3CFA0, 8), (0x4170C, 4096, 0x41740, 6)),
    }[samples]
    v = metadata[".vgpr_count"]
    for entry, local, mask_pc, slot in points:
        label = f".Ltile_{entry:x}:"
        if label not in body:
            raise ValueError("Unrecognized collective boundary")
        zero = f"""
s_mov_b32 s90, {samples * 84}
v_cmp_gt_u32_e64 s[92:93], s90, v{v + 4}
s_and_saveexec_b64 s[90:91], s[92:93]
v_lshlrev_b32_e32 v{v}, 2, v{v + 4}
v_add_u32_e32 v{v}, {local + samples * 48}, v{v}
v_mov_b32_e32 v{v + 1}, 0
ds_write_b32 v{v}, v{v + 1}
s_or_b64 exec, exec, s[90:91]
s_waitcnt lgkmcnt(0)
s_barrier
"""
        body = body.replace(label, label + zero)
        mask = f".Ltile_{mask_pc:x}:\nv_cmp_gt_u32_e32 vcc, {samples * 7}, v{slot}"
        if mask not in body:
            raise ValueError("Unrecognized peer dispatch")
        extra = f"\ns_mov_b64 s[90:91], vcc\nv_mov_b32_e32 v{v}, v{slot}\n"
        for _ in range(samples - 1):
            extra += f"""v_cmp_gt_u32_e64 s[92:93], 7, v{v}
v_subrev_u32_e32 v{v + 1}, 7, v{v}
v_cndmask_b32_e64 v{v}, v{v + 1}, v{v}, s[92:93]
"""
        extra += f"v_cmp_gt_u32_e32 vcc, {npes - 1}, v{v}\ns_and_b64 vcc, vcc, s[90:91]\n"
        body = body.replace(mask, mask + extra)
    body = f"v_mov_b32_e32 v{v + 4}, v0\n" + body
    return body, dict(metadata, **{".vgpr_count": v + 5})
