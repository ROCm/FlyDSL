#ifndef FLY_TRANSFORMS_LAYOUT_LOWERING
#define FLY_TRANSFORMS_LAYOUT_LOWERING

include "mlir/IR/PatternBase.td"
include "flydsl/Dialect/Fly/IR/FlyOps.td"

def : Pat<(Fly_GetLeafOp Fly_Layout:$layout, I32Attr:$leaf_idx), 
          (Fly_MakeLayoutOp  
            (Fly_GetLeafOp (Fly_GetShapeOp $layout), $leaf_idx),
            (Fly_GetLeafOp (Fly_GetStrideOp $layout), $leaf_idx)
        )>;

def : Pat<(Fly_GetShapeOp Fly_MemRef:$memref), 
          (Fly_GetShapeOp (Fly_GetLayoutOp $memref))>;
def : Pat<(Fly_GetStrideOp Fly_MemRef:$memref), 
          (Fly_GetStrideOp (Fly_GetLayoutOp $memref))>;


def : Pat<(Fly_SliceOp Fly_Layout:$layout, Fly_IntTuple:$coord), 
          (Fly_MakeLayoutOp  
            (Fly_SliceOp (Fly_GetShapeOp $layout), $coord),
            (Fly_SliceOp (Fly_GetStrideOp $layout), $coord)
        )>;
def : Pat<(Fly_SliceOp Fly_MemRef:$memref, Fly_IntTuple:$coord), 
          (Fly_MakeViewOp  
            (Fly_AddOffsetOp (Fly_GetIterOp $memref),
                              (Fly_Crd2IdxOp $coord, (Fly_GetLayoutOp $memref))),
            (Fly_SliceOp (Fly_GetLayoutOp $memref), $coord)
        )>;

def : Pat<(Fly_SizeOp Fly_Layout:$layout), 
          (Fly_SizeOp (Fly_GetShapeOp $layout))>;
def : Pat<(Fly_SizeOp Fly_MemRef:$memref), 
          (Fly_SizeOp (Fly_GetLayoutOp $memref))>;

def : Pat<(Fly_SelectOp Fly_Layout:$layout, DenseI32ArrayAttr:$indices),
          (Fly_MakeLayoutOp
            (Fly_SelectOp (Fly_GetShapeOp $layout), $indices),
            (Fly_SelectOp (Fly_GetStrideOp $layout), $indices)
        )>;

def : Pat<(Fly_GroupOp Fly_Layout:$layout, I32Attr:$begin, I32Attr:$end),
          (Fly_MakeLayoutOp
            (Fly_GroupOp (Fly_GetShapeOp $layout), $begin, $end),
            (Fly_GroupOp (Fly_GetStrideOp $layout), $begin, $end)
        )>;

def : Pat<(Fly_LogicalDivideOp Fly_ComposedLayout:$layout, AnyTypeOf<[Fly_Layout, Fly_Tile]>:$divisor),
          (Fly_MakeComposedLayoutOp
            (Fly_ComposedGetInnerOp $layout),
            (Fly_ComposedGetOffsetOp $layout),
            (Fly_LogicalDivideOp (Fly_ComposedGetOuterOp $layout), $divisor)
        )>;
        
        
def : Pat<(Fly_LogicalDivideOp Fly_MemRef:$mem, AnyTypeOf<[Fly_Layout, Fly_Tile]>:$divisor),
          (Fly_MakeViewOp
            (Fly_GetIterOp $mem),
            (Fly_LogicalDivideOp (Fly_GetLayoutOp $mem), $divisor)
        )>;
        
def : Pat<(Fly_ZippedDivideOp Fly_MemRef:$mem, AnyTypeOf<[Fly_Layout, Fly_Tile]>:$divisor),
          (Fly_MakeViewOp
            (Fly_GetIterOp $mem),
            (Fly_ZippedDivideOp (Fly_GetLayoutOp $mem), $divisor)
        )>;

#endif // FLY_TRANSFORMS_LAYOUT_LOWERING
