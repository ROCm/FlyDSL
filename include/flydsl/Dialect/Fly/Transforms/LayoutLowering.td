#ifndef FLY_TRANSFORMS_LAYOUT_LOWERING
#define FLY_TRANSFORMS_LAYOUT_LOWERING

include "mlir/IR/PatternBase.td"
include "flydsl/Dialect/Fly/IR/FlyOps.td"

// GetShapeOp patterns
def : Pat<(Fly_GetShapeOp Fly_ComposedLayout:$layout), 
          (Fly_GetShapeOp (Fly_ComposedGetOuterOp $layout))>;
def : Pat<(Fly_GetShapeOp Fly_MemRef:$memref), 
          (Fly_GetShapeOp (Fly_GetLayoutOp $memref))>;
def : Pat<(Fly_GetShapeOp Fly_CoordTensor:$tensor), 
          (Fly_GetShapeOp (Fly_GetLayoutOp $tensor))>;
    
// GetStrideOp patterns
def : Pat<(Fly_GetStrideOp Fly_ComposedLayout:$layout), 
          (Fly_GetStrideOp (Fly_ComposedGetOuterOp $layout))>;
def : Pat<(Fly_GetStrideOp Fly_MemRef:$memref), 
          (Fly_GetStrideOp (Fly_GetLayoutOp $memref))>;
def : Pat<(Fly_GetStrideOp Fly_CoordTensor:$tensor), 
          (Fly_GetStrideOp (Fly_GetLayoutOp $tensor))>;

// SizeOp patterns
def : Pat<(Fly_SizeOp Fly_IntTuple:$int_tuple),
          (Fly_IntTupleProductOp $int_tuple)>;
def : Pat<(Fly_SizeOp Fly_Layout:$layout), 
          (Fly_IntTupleProductOp (Fly_GetShapeOp $layout))>;
def : Pat<(Fly_SizeOp Fly_ComposedLayout:$layout), 
          (Fly_SizeOp (Fly_ComposedGetOuterOp $layout))>;
def : Pat<(Fly_SizeOp Fly_MemRef:$memref), 
          (Fly_SizeOp (Fly_GetLayoutOp $memref))>;
def : Pat<(Fly_SizeOp Fly_CoordTensor:$tensor), 
          (Fly_SizeOp (Fly_GetLayoutOp $tensor))>;

// CoprofileOp patterns
def : Pat<(Fly_CoprofileOp Fly_ComposedLayout:$layout),
          (Fly_CoprofileOp (Fly_ComposedGetOuterOp $layout))>;
def : Pat<(Fly_CoprofileOp Fly_MemRef:$memref),
          (Fly_CoprofileOp (Fly_GetLayoutOp $memref))>;
def : Pat<(Fly_CoprofileOp Fly_CoordTensor:$tensor),
          (Fly_CoprofileOp (Fly_GetLayoutOp $tensor))>;

// CoshapeOp patterns
def : Pat<(Fly_CoshapeOp Fly_ComposedLayout:$layout),
          (Fly_CoshapeOp (Fly_ComposedGetOuterOp $layout))>;
def : Pat<(Fly_CoshapeOp Fly_MemRef:$memref),
          (Fly_CoshapeOp (Fly_GetLayoutOp $memref))>;
def : Pat<(Fly_CoshapeOp Fly_CoordTensor:$tensor),
          (Fly_CoshapeOp (Fly_GetLayoutOp $tensor))>;

// CosizeOp patterns
def : Pat<(Fly_CosizeOp Fly_ComposedLayout:$layout),
          (Fly_CosizeOp (Fly_ComposedGetOuterOp $layout))>;
def : Pat<(Fly_CosizeOp Fly_MemRef:$memref),
          (Fly_CosizeOp (Fly_GetLayoutOp $memref))>;
def : Pat<(Fly_CosizeOp Fly_CoordTensor:$tensor),
          (Fly_CosizeOp (Fly_GetLayoutOp $tensor))>;

// CoalesceOp patterns
def : Pat<(Fly_CoalesceOp Fly_ComposedLayout:$layout, $attr),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_CoalesceOp (Fly_ComposedGetOuterOp $layout), $attr))>;
def : Pat<(Fly_CoalesceOp Fly_MemRef:$memref, $attr),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_CoalesceOp (Fly_GetLayoutOp $memref), $attr))>;
def : Pat<(Fly_CoalesceOp Fly_CoordTensor:$tensor, $attr),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_CoalesceOp (Fly_GetLayoutOp $tensor), $attr))>;

// CompositionOp patterns
def : Pat<(Fly_CompositionOp Fly_ComposedLayout:$layout, $inner),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_CompositionOp (Fly_ComposedGetOuterOp $layout), $inner))>;
def : Pat<(Fly_CompositionOp Fly_MemRef:$memref, $inner),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_CompositionOp (Fly_GetLayoutOp $memref), $inner))>;
def : Pat<(Fly_CompositionOp Fly_CoordTensor:$tensor, $inner),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_CompositionOp (Fly_GetLayoutOp $tensor), $inner))>;

// ComplementOp patterns
def : Pat<(Fly_ComplementOp Fly_ComposedLayout:$layout, $codom),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_ComplementOp (Fly_ComposedGetOuterOp $layout), $codom))>;

// RightInverseOp patterns
def : Pat<(Fly_RightInverseOp Fly_ComposedLayout:$layout),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_RightInverseOp (Fly_ComposedGetOuterOp $layout)))>;

// LeftInverseOp patterns
def : Pat<(Fly_LeftInverseOp Fly_ComposedLayout:$layout),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_LeftInverseOp (Fly_ComposedGetOuterOp $layout)))>;

// LogicalDivideOp patterns
def : Pat<(Fly_LogicalDivideOp Fly_ComposedLayout:$layout, $divisor),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_LogicalDivideOp (Fly_ComposedGetOuterOp $layout), $divisor))>;
def : Pat<(Fly_LogicalDivideOp Fly_MemRef:$memref, $divisor),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_LogicalDivideOp (Fly_GetLayoutOp $memref), $divisor))>;
def : Pat<(Fly_LogicalDivideOp Fly_CoordTensor:$tensor, $divisor),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_LogicalDivideOp (Fly_GetLayoutOp $tensor), $divisor))>;

// ZippedDivideOp patterns
def : Pat<(Fly_ZippedDivideOp Fly_ComposedLayout:$layout, $divisor),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_ZippedDivideOp (Fly_ComposedGetOuterOp $layout), $divisor))>;
def : Pat<(Fly_ZippedDivideOp Fly_MemRef:$memref, $divisor),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_ZippedDivideOp (Fly_GetLayoutOp $memref), $divisor))>;
def : Pat<(Fly_ZippedDivideOp Fly_CoordTensor:$tensor, $divisor),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_ZippedDivideOp (Fly_GetLayoutOp $tensor), $divisor))>;

// TiledDivideOp patterns
def : Pat<(Fly_TiledDivideOp Fly_ComposedLayout:$layout, $divisor),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_TiledDivideOp (Fly_ComposedGetOuterOp $layout), $divisor))>;
def : Pat<(Fly_TiledDivideOp Fly_MemRef:$memref, $divisor),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_TiledDivideOp (Fly_GetLayoutOp $memref), $divisor))>;
def : Pat<(Fly_TiledDivideOp Fly_CoordTensor:$tensor, $divisor),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_TiledDivideOp (Fly_GetLayoutOp $tensor), $divisor))>;

// FlatDivideOp patterns
def : Pat<(Fly_FlatDivideOp Fly_ComposedLayout:$layout, $divisor),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_FlatDivideOp (Fly_ComposedGetOuterOp $layout), $divisor))>;
def : Pat<(Fly_FlatDivideOp Fly_MemRef:$memref, $divisor),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_FlatDivideOp (Fly_GetLayoutOp $memref), $divisor))>;
def : Pat<(Fly_FlatDivideOp Fly_CoordTensor:$tensor, $divisor),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_FlatDivideOp (Fly_GetLayoutOp $tensor), $divisor))>;

// LogicalProductOp patterns
def : Pat<(Fly_LogicalProductOp Fly_ComposedLayout:$layout, $tile),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_LogicalProductOp (Fly_ComposedGetOuterOp $layout), $tile))>;

// ZippedProductOp patterns
def : Pat<(Fly_ZippedProductOp Fly_ComposedLayout:$layout, $tile),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_ZippedProductOp (Fly_ComposedGetOuterOp $layout), $tile))>;

// TiledProductOp patterns
def : Pat<(Fly_TiledProductOp Fly_ComposedLayout:$layout, $tile),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_TiledProductOp (Fly_ComposedGetOuterOp $layout), $tile))>;

// FlatProductOp patterns
def : Pat<(Fly_FlatProductOp Fly_ComposedLayout:$layout, $tile),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_FlatProductOp (Fly_ComposedGetOuterOp $layout), $tile))>;

// BlockedProductOp patterns
def : Pat<(Fly_BlockedProductOp Fly_ComposedLayout:$layout, $tile),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_BlockedProductOp (Fly_ComposedGetOuterOp $layout), $tile))>;

// RakedProductOp patterns
def : Pat<(Fly_RakedProductOp Fly_ComposedLayout:$layout, $tile),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_RakedProductOp (Fly_ComposedGetOuterOp $layout), $tile))>;

// RecastLayoutOp patterns
def : Pat<(Fly_RecastLayoutOp $new_bits, $old_bits, Fly_ComposedLayout:$layout),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_RecastLayoutOp $new_bits, $old_bits, (Fly_ComposedGetOuterOp $layout)))>;

// TileToShapeOp patterns
def : Pat<(Fly_TileToShapeOp Fly_ComposedLayout:$block, $trg_shape, $ord_shape),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $block),
                                    (Fly_ComposedGetOffsetOp $block),
                                    (Fly_TileToShapeOp (Fly_ComposedGetOuterOp $block), $trg_shape, $ord_shape))>;

#endif // FLY_TRANSFORMS_LAYOUT_LOWERING
