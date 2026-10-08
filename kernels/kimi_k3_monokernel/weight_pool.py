"""Fixed K3 TP8 weight byte layout; pools are built before graph capture."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Region:
    name: str
    dtype: str
    numel: int
    offset: int
    nbytes: int

# These are output sizes of the existing packers, including256-row scale padding.
_DENSE_SPEC = (
    ('input', 'int16', 6400 * 7168),
    ('output', 'int16', 7168 * 1536),
    ('w_kda_fb', 'bfloat16', 1536 * 128),
    ('w_kda_conv', 'bfloat16', 4608 * 4),
    ('kda_a_log', 'float32', 12),
    ('kda_dt_bias', 'bfloat16', 12 * 128),
    ('g_kda_out', 'bfloat16', 128),
    ('w_r', 'int16', 896 * 7168),
    ('bias', 'bfloat16', 896),
    ('w_latent_down', 'uint8', 3584 * 7168),
    ('s_latent_down', 'uint8', 3584 * (7168 // 32)),
    ('w_shared_ug', 'uint8', 1536 * 7168),
    ('s_shared_ug', 'uint8', 1536 * (7168 // 32)),
    ('w_shared_dn', 'uint8', 7168 * 768),
    ('s_shared_dn', 'uint8', 7168 * (768 // 32)),
    ('g_latent', 'bfloat16', 3584),
    ('w_latent_up', 'uint8', 896 * 3584),
    ('s_latent_up', 'uint8', 1024 * (3584 // 32)),
)
_EXPERT_SPEC = (
    ('w_ug', 'uint8', 896 * 768 * (3584 // 2)),
    ('s_ug', 'uint8', 896 * 768 * (3584 // 32)),
    ('w_dn', 'uint8', 896 * 3584 * (384 // 2)),
    ('s_dn', 'uint8', 896 * 3584 * (384 // 32)),
)

def _layout(spec):
    result=[];offset=0
    for name,dtype,numel in spec:
        offset=(offset+255)//256*256
        nbytes=numel*{'int16':2,'bfloat16':2,'float32':4,'uint8':1}[dtype]
        result.append(Region(name,dtype,numel,offset,nbytes));offset+=nbytes
    total=(offset+255)//256*256
    assert total<2**31 and result[0].offset==0
    return tuple(result),total

DENSE_REGIONS,DENSE_BYTES=_layout(_DENSE_SPEC)
EXPERT_REGIONS,EXPERT_BYTES=_layout(_EXPERT_SPEC)
DENSE_OFFSETS={x.name:x.offset for x in DENSE_REGIONS}
EXPERT_OFFSETS={x.name:x.offset for x in EXPERT_REGIONS}

def pack_weight_pool(regions,total,tensors):
    import torch
    first=tensors[regions[0].name]
    for region in regions:
        value=tensors[region.name]
        if value.dtype!=getattr(torch,region.dtype) or value.numel()!=region.numel or not value.is_contiguous() or value.device!=first.device:
            raise ValueError(f'weight pool contract failed: {region.name}, {value.dtype}, {tuple(value.shape)}, {value.device}')
    pool=torch.zeros(total,dtype=torch.uint8,device=first.device)
    for region in regions:
        raw=tensors[region.name].view(torch.uint8).reshape(-1)
        pool[region.offset:region.offset+region.nbytes].copy_(raw)
    return pool

def make_weight_pools(packed_input,packed_output,packed,tensors,*,include_experts):
    # Raw checkpoint tensors are never replaced or modified; private packed copies
    # also remain intact. Only the dedicated pool pointer changes the launch ABI value.
    weights={**tensors,**packed,'input':packed_input,'output':packed_output}
    result={'dense':pack_weight_pool(DENSE_REGIONS,DENSE_BYTES,weights)}
    if include_experts:
        result['expert']=pack_weight_pool(EXPERT_REGIONS,EXPERT_BYTES,weights)
    return result
