"""CPU coverage for Kimi's eight-token speculative-decode interface."""

from dataclasses import asdict
import json
from pathlib import Path
import unittest

from kernels.kimi_k3_monokernel.compile_config import KimiK3CompileConfig
from kernels.kimi_k3_monokernel.shapes import resolve_sequence_shape
from kernels.kimi_k3_monokernel.tools.replay_cases import replay_slot_stride, replay_state_chains


class KimiK3ShapeTests(unittest.TestCase):
    def test_all_batch_sequence_pairs(self):
        for batch in range(1, 9):
            for seq in range(1, 9):
                with self.subTest(batch=batch, seq=seq):
                    self.assertEqual(resolve_sequence_shape(batch * seq, seq > 1, seq), (batch, seq))
                    config = KimiK3CompileConfig().resolve(batch * seq, seq > 1, seq)
                    self.assertEqual((config.batch, config.seq), (batch, seq))
                    self.assertEqual(config.samples % config.staged_samples, 0)
                    if seq > 4:
                        self.assertEqual(config.path, "general")
                        self.assertEqual(config.pre_attn_res, "parallel4")
                        self.assertEqual(config.packed_scale_high_rows, batch * seq > 16)

    def test_existing_specializations_match_validated_snapshot(self):
        repo = Path(__file__).resolve().parents[2]
        rows = json.loads((repo / "experiments/kimi_one_kernel/opt254/compiled_shapes.json").read_text())
        for row in rows:
            batch, seq = row["batch"], row["seq"]
            with self.subTest(batch=batch, seq=seq):
                config = KimiK3CompileConfig().resolve(batch * seq, seq > 1, seq)
                self.assertEqual(asdict(config), row["specialization"])

    def test_implicit_eight_token_mtp(self):
        self.assertEqual(resolve_sequence_shape(8, True), (1, 8))
        self.assertEqual(resolve_sequence_shape(8, False), (8, 1))
        self.assertEqual(resolve_sequence_shape(64, True, 8), (8, 8))

    def test_invalid_shapes_are_rejected(self):
        for args in [(0, True, 8), (65, True, 8), (72, True, 8), (9, True, 9),
                     (9, False, 1), (8, False, 8), (10, True, 8), (True, False, 1), (8, True, True)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_sequence_shape(*args)
        with self.assertRaises(ValueError):
            KimiK3CompileConfig(path="small_batch").resolve(8, True, 8)

    def test_replay_cases_cover_complete_independent_chains(self):
        for batch in range(1, 9):
            for seq in range(2, 9):
                stride = replay_slot_stride(seq)
                chains = replay_state_chains(batch, seq, True)
                self.assertEqual(len(chains), 6)
                for case, chain in enumerate(chains):
                    with self.subTest(batch=batch, seq=seq, case=case):
                        self.assertEqual(len(chain), batch * (seq + 1))
                        for b in range(batch):
                            slots = chain[b * (seq + 1):(b + 1) * (seq + 1)]
                            self.assertTrue(all(slot == -1 or b * stride <= slot < (b + 1) * stride
                                                for slot in slots))
                            if case in (0, 1):
                                self.assertEqual(len(set(slots)), seq + 1)
                self.assertEqual(chains[3], [-1] * (batch * (seq + 1)))

    def test_existing_replay_patterns_are_preserved(self):
        self.assertEqual(replay_slot_stride(4), 7)
        self.assertEqual(replay_state_chains(1, 4, True), [
            [0, 1, 2, 3, 4], [5, 2, 6, 1, 4], [0, 1, -1, 3, 4],
            [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0],
        ])


if __name__ == "__main__":
    unittest.main()
