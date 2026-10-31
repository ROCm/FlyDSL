# Standalone exercise of cdna5.tdm_partition: feed each case's LDS(s) / coord(g)
# layouts + the atom's nv, print the partitioned ps/pg, compare to expected.
import re, torch, traceback
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.primitive import (SwizzleType, static, make_composed_layout,
                                    composition, make_tile)
fly_rocdl = fx.rocdl.cdna5.fly_rocdl
L = fx.make_layout

def box2d(nv):                      # any 2D factor of nv, both dims<=128 (shape irrelevant to result)
    g = 1
    for d in range(1, int(nv**0.5)+1):
        if nv % d == 0 and d <= 128 and nv//d <= 128: g = d
    return (nv//g, g)

def norm(x):                        # canonical compact string, "_" and "Layout<>"/"ComposedLayout<>" stripped
    s = str(x)
    s = re.sub(r'^(Composed)?Layout<(.*)>$', r'\2', s)
    return s.replace('_', '').replace('Sw<', 'S<').replace(' ', '')

# name -> (nv, s_layout_factory, g_layout_factory, expected_ps, expected_pg)
CASES = [
 ("gemm_mk",     4096, lambda:L(((64,64),3),((1,64),4096)),           lambda:L(((64,64),8),((1,64),4096)),
                       "((4096,1),3):((1,0),4096)",                    "((4096,1),8):((1,0),4096)"),
 ("gemm_mk_iter", 128, lambda:L(((64,64),3),((1,64),4096)),           lambda:L(((64,64),8),((1,64),4096)),
                       "((128,32),3):((1,128),4096)",                  "((128,32),8):((1,128),4096)"),
 ("nested_tile",   64, lambda:L((((8,8),(8,4)),2),(((1,8),(64,512)),2048)), lambda:L((((8,8),(8,4)),6),(((1,8),(64,512)),2048)),
                       "((64,32),2):((1,64),2048)",                    "((64,32),6):((1,64),2048)"),
 ("box3d_2rest",  128, lambda:L(((16,8,4),3,2),((1,16,128),512,1536)),lambda:L(((16,8,4),5,3),((1,16,128),512,2560)),
                       "((128,4),3,2):((1,128),512,1536)",             "((128,4),5,3):((1,128),512,2560)"),
 ("box4d",         32, lambda:L(((8,4,4,4),3),((1,8,32,128),512)),    lambda:L(((8,4,4,4),5),((1,8,32,128),512)),
                       "((32,16),3):((1,32),512)",                     "((32,16),5):((1,32),512)"),
 ("kcontig_s_vs_mcontig_g",64, lambda:L(((16,32),4),((1,16),512)),    lambda:L(((16,32),7),((32,1),512)),
                       "((64,8),4):((1,64),512)",                      "(((16,4),8),7):(((32,1),4),512)"),
 ("diff_nest_mode0",64, lambda:L(((32,16),2),((1,32),512)),           lambda:L((((8,4),16),5),(((1,8),32),512)),
                       "((64,8),2):((1,64),512)",                      "((64,8),5):((1,64),512)"),
 ("g_extra_rest", 256, lambda:L(((32,16),(2,3)),((1,32),(512,1024))), lambda:L(((32,16),4,3,2),((1,32),512,2048,6144)),
                       "((256,2),(2,3)):((1,256),(512,1024))",         "((256,2),4,3,2):((1,256),512,2048,6144)"),
 ("holes",          8, lambda:L(((8,8),2),((1,16),128)),              lambda:L(((8,8),5),((1,8),64)),
                       "((8,8),2):((1,16),128)",                       "((8,8),5):((1,8),64)"),
 ("dense_s_holes_g",16, lambda:L(((8,8),2),((1,8),64)),               lambda:L(((8,8),5),((1,16),128)),
                       "((16,4),2):((1,16),64)",                       "(((8,2),4),5):(((1,16),32),128)"),
 ("swizzle_mk",   256, lambda:make_composed_layout(static(SwizzleType.get(3,3,3)),L(((64,64),2),((1,64),4096))),
                       lambda:L(((64,64),8),((1,64),4096)),
                       "S<3,3,3>o0o((256,16),2):((1,256),4096)",       "((256,16),8):((1,256),4096)"),
]

@flyc.kernel
def run(A: fx.Tensor):
    buf = fx.SharedAllocator().allocate(fx.Array[fx.Float16, 8192]).peek()
    for name, nv, sf, gf, eps, epg in CASES:
        a, b = box2d(nv)
        atom, _ = fx.rocdl.cdna5.make_tiled_tdm_atom(fx.rocdl.TensorLoad(), A, L((a,b),(b,1)), (a,b))
        s = fx.make_view(buf.ptr, sf()); g = fx.make_view(buf.ptr, gf())
        try:
            ps, pg = fx.rocdl.cdna5.tdm_partition(atom, 0, L(1,1), s, g)
            path = "direct"
        except Exception as e:
            # composed-layout LDS tile: tdm_partition_layout rejects it; derive layout_V
            # from the plain base, then apply by-mode (exactly what tdm_partition does).
            base = fx.make_view(buf.ptr, L(((64,64),2),((1,64),4096)))
            lV = static(fly_rocdl.tdm_partition_layout(atom.type, base.type, g.type, 1))
            ps = composition(s, make_tile(lV, None)); pg = composition(g, make_tile(lV, None))
            path = "2-step (layout API blocks composed .type)"
        okp = norm(ps.layout) == norm(eps); okg = norm(pg.layout) == norm(epg)
        print(f"[{'PASS' if okp and okg else 'FAIL'}] {name:24s} nv={nv:<5d} ({path})")
        print(f"        s  {norm(s.layout)}")
        print(f"        g  {norm(g.layout)}")
        print(f"        ps {norm(ps.layout)}   {'==' if okp else '!= EXP '+eps} ")
        print(f"        pg {norm(pg.layout)}   {'==' if okg else '!= EXP '+epg} ")

@flyc.jit
def j(A: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    run(A).launch(grid=(1,1,1), block=(32,1,1), stream=stream)
A = torch.zeros(128,128, dtype=torch.float16)
try: j(A)
except Exception: traceback.print_exc()
