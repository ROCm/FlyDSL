"""Explicit independent state-chain dimensions, shared by host and golden."""


def resolve_sequence_shape(samples: int, mtp: bool, seq_len: int | None = None):
    if type(samples) is not int or not 1 <= samples <= 32:
        raise ValueError("samples must be in [1, 32]")
    seq_len = (samples if mtp else 1) if seq_len is None else seq_len
    if type(seq_len) is not int or not 1 <= seq_len <= 4:
        raise ValueError("seq_len must be in [1, 4]")
    if not mtp and seq_len != 1:
        raise ValueError("seq_len > 1 requires ordered MTP snapshots")
    if samples % seq_len or not 1 <= samples // seq_len <= 8:
        raise ValueError("samples must equal batch_size * seq_len with batch_size in [1, 8]")
    return samples // seq_len, seq_len
