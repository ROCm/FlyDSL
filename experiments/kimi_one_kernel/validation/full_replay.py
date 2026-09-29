"""Untimed full-MonoKernel replay checks, including MTP snapshot/alias semantics."""
import torch
import torch.distributed as dist
from kernels.kimi_k3_monokernel.kernel import monokernel_layout
from kernels.monokernel.config import EPS, MoeMode, moe_format
from kernels.monokernel.formats import quant_dequant_mxfp8
from kernels.monokernel.reference import (
    golden_kimi_k3_kda_attention, kimi_attn_res, route, dequant_expert, bf, situ,
)


def relative(got, expected):
    return float((got.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def allreduce(value, npes):
    parts = [torch.empty_like(value.cpu()) for _ in range(npes)]
    dist.all_gather(parts, value.cpu().contiguous())
    total = parts[0].float()
    for part in parts[1:]:
        total.add_(part.float())
    return total.to(torch.bfloat16).to(value.device)


def ranks_equal(value, npes):
    parts = [torch.empty_like(value.cpu()) for _ in range(npes)]
    dist.all_gather(parts, value.cpu().contiguous())
    return all(torch.equal(parts[0], p) for p in parts[1:])


def production_moe_reference(weights, hidden, npes):
    """FP32 arithmetic with the full kernel's documented BF16 stage boundaries.

    In particular gate/up are rounded before SiTU, as in kernel.py. The generic
    golden's unrounded UG is retained in the original tool's diagnostic metrics.
    """
    c, t = weights.config, weights.t
    x = quant_dequant_mxfp8(hidden).to(torch.bfloat16)
    latent = bf(x.float() @ t['w_latent_down'].float().T)
    scores = torch.sigmoid((hidden @ t['w_r'].T).float())
    ids, probs, mids = [], [], []
    partial = torch.zeros(hidden.shape[0], c.routed_hidden, device=hidden.device)
    fmt = moe_format(MoeMode.A16W4)
    for s in range(hidden.shape[0]):
        selected, prob = route(scores[s], t['bias'], c)
        ids.append(selected.to(torch.int32))
        probs.append(prob)
        sample_mid = []
        for expert, weight in zip(selected.tolist(), prob.tolist()):
            ug = bf(dequant_expert(t['w_ug'][expert], t['s_ug'][expert], fmt.weight) @ latent[s].float())
            mid = bf(situ(ug, beta=c.situ_beta, linear_beta=c.situ_linear_beta))
            sample_mid.append(mid)
            down = dequant_expert(t['w_dn'][expert], t['s_dn'][expert], fmt.weight) @ mid
            partial[s] += weight * down
        mids.append(torch.stack(sample_mid))
    routed = allreduce(bf(partial), npes)
    norm = bf(routed.float() * torch.rsqrt(routed.float().square().mean(-1, keepdim=True) + EPS)
              * t['g_latent'].float())
    shared_ug = bf(x.float() @ t['w_shared_ug'].float().T)
    shared_mid = bf(situ(shared_ug, beta=c.situ_beta, linear_beta=c.situ_linear_beta))
    shared = bf(shared_mid.float() @ t['w_shared_dn'].float().T)
    latent_up = bf(norm.float() @ t['w_latent_up'].float().T)
    shard = t['w_latent_up'].shape[0]
    shared[:, weights.rank * shard:(weights.rank + 1) * shard] = bf(
        shared[:, weights.rank * shard:(weights.rank + 1) * shard].float() + latent_up.float())
    return dict(sel=torch.stack(ids), prob=torch.stack(probs), mid=torch.stack(mids),
                routed_reduced=routed, moe_delta=allreduce(shared, npes))


def run(layer, reference_weights, prefix, blocks, indices, conv, recurrent, output, args):
    assert not args.staged and not args.attention_only and args.mtp and args.samples == 4
    config, t = reference_weights.config, reference_weights.t
    assert args.layer_idx % config.attn_res_block_size != 0, "This replay suite targets layer 1"
    layout = monokernel_layout(4, fuse_attn_res=True, fuse_moe=True, mtp=True)
    scratch = layer.attention.monokernel_scratch

    def raw(name, count):
        off = layout[name]
        return scratch[off:off + 2 * count].view(torch.bfloat16)

    def tagged(name, count, dtype=torch.float32):
        off = layout[name]
        return scratch[off:off + 8 * count].view(torch.int32).view(count, 2)[:, 0].contiguous().view(dtype)

    # Capture exactly one complete layer: repeated slot aliases are stateful,
    # so replaying sixteen layers would require sixteen reference advances.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.graph(graph):
        layer.forward(prefix, blocks, indices, conv, recurrent, x_out=output,
                      epoch_layer=0, advance=False)
        layer.advance_step()
    records = []
    chains = ([0, 1, 2, 3, 4], [5, 2, 6, 1, 4], [0, 1, -1, 3, 4],
              [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0])
    previous_blocks = (args.layer_idx + config.attn_res_block_size - 1) // config.attn_res_block_size
    for case, chain in enumerate(chains):
        indices.copy_(torch.tensor(chain, device=prefix.device, dtype=torch.int32))
        destinations = {chain[i + 1] for i in range(4) if chain[i] >= 0 and chain[i + 1] >= 0}
        untouched = [s for s in range(conv.shape[0]) if s not in destinations]
        for replay in range(3):
            gen = torch.Generator(device=prefix.device).manual_seed(args.seed + 4000 + case * 32 + replay)
            prefix.copy_(torch.randn(prefix.shape, device=prefix.device, dtype=prefix.dtype, generator=gen)
                         * (0.5 + replay * 0.25))
            blocks.copy_(torch.randn(blocks.shape, device=blocks.device, dtype=blocks.dtype, generator=gen))
            initial_blocks = blocks.clone()
            initial_conv = torch.randn(conv.shape, device=conv.device, dtype=conv.dtype, generator=gen)
            initial_state = torch.randn(recurrent.shape, device=recurrent.device, dtype=recurrent.dtype,
                                        generator=gen) * (0.01 + 0.01 * replay)
            conv.copy_(initial_conv)
            recurrent.copy_(initial_state)
            expected_conv, expected_state = initial_conv.clone(), initial_state.clone()
            graph.replay()
            torch.cuda.synchronize()
            pre, _ = kimi_attn_res(prefix, None, initial_blocks, t['g_self_res'], t['w_self_res'],
                                  t['g_in'], previous_blocks, -1)
            # Keep end-to-end diagnostics; stage checks use the actual BF16
            # pre-AttnRes output to avoid attributing upstream rounding to KDA.
            e2e_conv, e2e_state = initial_conv.clone(), initial_state.clone()
            e2e_attention = golden_kimi_k3_kda_attention(reference_weights, pre, indices,
                e2e_conv, e2e_state, lambda x: allreduce(x, args.npes), mtp=True)
            attention = golden_kimi_k3_kda_attention(reference_weights, layer.pre_attn, indices,
                expected_conv, expected_state, lambda x: allreduce(x, args.npes), mtp=True)
            expected_moe, expected_prefix = kimi_attn_res(prefix, layer.attention_delta, initial_blocks,
                t['g_mlp_res'], t['w_mlp_res'], t['g_post'], previous_blocks)
            moe = production_moe_reference(reference_weights, layer.moe_input, args.npes)
            fused_ids = tagged('selection_id', 4 * config.top_k, torch.int32).view(4, config.top_k)
            fused_prob = tagged('selection_weight', 4 * config.top_k).view(4, config.top_k)
            fused_mid = raw('expert_mid', 4 * config.top_k * config.inter).view(4, config.top_k, config.inter)
            routed = raw('routed', 4 * config.routed_hidden).view(4, config.routed_hidden)
            routed_inv = tagged('routed_inv', 4).view(4, 1)
            expected_inv = torch.rsqrt(routed.float().square().mean(-1, keepdim=True) + EPS)
            expected_output = (layer.updated_prefix.float() + moe['moe_delta'].float()).to(torch.bfloat16)
            record = dict(chain=list(chain), replay=replay,
                e2e_attention_rel_l2=relative(layer.attention_delta, e2e_attention['output']),
                e2e_recurrent_snapshot_max_rel_l2=max(relative(recurrent[s], e2e_state[s]) for s in range(conv.shape[0])),
                pre_attn_rel_l2=relative(layer.pre_attn, pre),
                attention_rel_l2=relative(layer.attention_delta, attention['output']),
                moe_input_rel_l2=relative(layer.moe_input, expected_moe),
                updated_prefix_rel_l2=relative(layer.updated_prefix, expected_prefix),
                conv_snapshot_max_rel_l2=max(relative(conv[s], expected_conv[s]) for s in range(conv.shape[0])),
                recurrent_snapshot_max_rel_l2=max(relative(recurrent[s], expected_state[s]) for s in range(conv.shape[0])),
                untouched_conv_equal=torch.equal(conv[untouched], initial_conv[untouched]),
                untouched_recurrent_equal=torch.equal(recurrent[untouched], initial_state[untouched]),
                residual_equal=torch.equal(layer.updated_prefix, (prefix.float() + layer.attention_delta.float()).to(torch.bfloat16)),
                blocks_equal=torch.equal(blocks, initial_blocks),
                selection_equal=torch.equal(fused_ids, moe['sel']),
                selection_weight_rel_l2=relative(fused_prob, moe['prob']),
                expert_mid_rel_l2=relative(fused_mid, moe['mid']),
                routed_rel_l2=relative(routed, moe['routed_reduced']),
                routed_inv_rel_l2=relative(routed_inv, expected_inv),
                output_rel_l2=relative(output, expected_output),
                routed_rank_equal=ranks_equal(routed, args.npes),
                output_rank_equal=ranks_equal(output, args.npes),
                finite=bool(torch.isfinite(output).all() and torch.isfinite(conv).all() and torch.isfinite(recurrent).all()))
            # Repeat the same input through a new epoch tag; output and every
            # stored state snapshot must agree exactly, including aliased slots.
            saved = (output.clone(), conv.clone(), recurrent.clone(), fused_ids.clone())
            conv.copy_(initial_conv)
            recurrent.copy_(initial_state)
            blocks.copy_(initial_blocks)
            graph.replay()
            torch.cuda.synchronize()
            record['new_epoch_equal'] = (torch.equal(output, saved[0]) and torch.equal(conv, saved[1])
                and torch.equal(recurrent, saved[2]) and torch.equal(
                    tagged('selection_id', 4 * config.top_k, torch.int32).view(4, config.top_k), saved[3]))
            records.append(record)
    limits = dict(pre_attn_rel_l2=2e-3, attention_rel_l2=2e-3, moe_input_rel_l2=2e-3,
                  updated_prefix_rel_l2=2e-3, conv_snapshot_max_rel_l2=5e-4,
                  recurrent_snapshot_max_rel_l2=5e-4, selection_weight_rel_l2=2e-3,
                  expert_mid_rel_l2=2e-3, routed_rel_l2=2e-2, routed_inv_rel_l2=1e-3,
                  output_rel_l2=1e-2)
    ok = all(all(r[k] < limit for k, limit in limits.items())
             and all(v for v in r.values() if isinstance(v, bool)) for r in records)
    return dict(full_replay_correct=ok, full_replay_records=records, full_replay_limits=limits)
