"""Checked substitutions for the released gfx950 W8A8 Proto1 specializations."""

import re


def replace_dispatch(body, samples):
    """Lift CTA-role and thread-coordinate arithmetic into FlyDSL.

    Keep argument prefetch and epoch arithmetic in their original order. The
    three role values arrive in s90..92, outside the native live register set,
    and move to native registers after argument loads have completed. Some
    argument loads overlap the eventual role registers in S=2/4.
    """
    cutoff, tag_pcs, scalar, vector = {
        1: (0xD4, {0x94, 0x9C, 0xA0, 0xA8, 0xB0}, (80, 34, 81), (34, 35, 49, 48)),
        2: (0xB8, {0x44, 0x4C, 0x50, 0x58, 0x70}, (71, 76, 77), (34, 35, 51, 50)),
        4: (0xA8, {0x44, 0x4C, 0x50, 0x58, 0x68}, (83, 88, 89), (40, 41, 59, 58)),
    }[samples]
    pairs = re.findall(r"(\.Ltile_([0-9a-f]+):)\n([^\n]+)", body)
    base = int(pairs[0][1], 16)
    kept, removed = [], []
    for label, address, inst in pairs:
        offset = int(address, 16) - base
        if offset == cutoff:
            kept.extend(f"s_mov_b32 s{dst}, s{src}" for dst, src in zip(scalar, (90, 91, 92)))
        keep = (
            offset >= cutoff
            or offset in tag_pcs
            or inst.startswith(("s_load_", "s_waitcnt"))
            or inst == "s_ashr_i32 s3, s2, 31"
        )
        # v67 is already derived from the lifted quad coordinate in S=4.
        if samples == 4 and offset == 0xA4:
            keep = True
        kept.append(label)
        if keep:
            kept.append(inst)
        else:
            removed.append((offset, inst))
    if not any("s_cselect_b32 s" + str(scalar[2]) in inst for _, inst in removed):
        raise ValueError("Unexpected TileRT prologue; cannot replace dispatch")
    return "\n".join(kept), vector, removed
