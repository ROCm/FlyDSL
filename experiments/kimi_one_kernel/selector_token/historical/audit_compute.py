"""Verify unchanged compute bodies independently of schedule/marker edits."""
import ast
import json
from pathlib import Path
from textwrap import dedent

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'kimi_full_token_pipeline_20260929'
KERNEL=Path('kernels/kimi_k3_monokernel/kernel.py')
base=(ROOT/'full_tail4_proj4_ug32_events'/KERNEL).read_text()

class StripMarkers(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name in {'stamp','mark','wave_mark'}:return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in {'stamp','mark','wave_mark'}:return None
        return self.generic_visit(node)
    def visit_If(self,node):
        node=self.generic_visit(node)
        if not node.body and not node.orelse:return None
        return node

def normalized(code):
    return ast.dump(StripMarkers().visit(ast.parse(code)),include_attributes=False)
assert normalized(base)==normalized((ROOT/'best_fine_timeline'/KERNEL).read_text())

def while_body(code,target):
    tree=ast.parse(code)
    return next(n for n in ast.walk(tree) if isinstance(n,ast.While) and
                isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id==target).body
def dumps(nodes):return [ast.dump(n,include_attributes=False) for n in nodes]

checks=[]
for tile in [16,32]:
    name=f'moe_g2_ug{tile}_events';code=(ROOT/name/KERNEL).read_text()
    original=base if tile==32 else (OLD/'full_tail4_services_proj4_v2_events'/KERNEL).read_text()
    # Only sample-number decoding changes inside either while loop.
    assert dumps(while_body(code,'up_task')[1:])==dumps(while_body(original,'up_task')[1:])
    assert dumps(while_body(code,'down_task')[1:])==dumps(while_body(base,'down_task')[1:])
    marker='            # One CTA per sample collapses'
    assert code[code.index(marker):]==base[base.index(marker):]
    prefix='            # Stage 6:'
    new_prefix='            # Only UG/down are grouped'
    assert code[:code.index(new_prefix)]==base[:base.index(prefix)]
    checks.append(dict(variant=name,ug_body_only_sample_index_changed=True,
                       down_body_only_sample_index_changed=True,front_norm_tail_launch_unchanged=True))
out=dict(fine_instrumentation_compute_ast_identical=True,pipeline_variants=checks)
(ROOT/'compute_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
