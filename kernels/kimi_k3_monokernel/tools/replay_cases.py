"""State-slot cases shared by the strict replay tool and its CPU checks."""


def replay_slot_stride(seq: int) -> int:
    return max(7, seq + 3)


def replay_state_chains(batch: int, seq: int, mtp: bool) -> list[list[int]]:
    stride = replay_slot_stride(seq)
    if mtp:
        permuted = [5, 2, 6, 1, 4, 0, 3] + list(range(7, stride))
        patterns = [
            list(range(seq + 1)),
            permuted[:seq + 1],
            [0 if t == 0 else (-1 if t == min(2, seq) else t) for t in range(seq + 1)],
            [-1] * (seq + 1),
            [0] * (seq + 1),
            [t % 2 for t in range(seq + 1)],
        ]
        return [
            [b * stride + slot if slot >= 0 else -1 for b in range(batch) for slot in pattern]
            for pattern in patterns
        ]
    return [
        [b * stride for b in range(batch)],
        [b * stride + 5 for b in reversed(range(batch))],
        [b * stride + 2 if b % 2 else -1 for b in range(batch)],
        [-1] * batch,
        [b * stride + 6 for b in range(batch)],
    ]
