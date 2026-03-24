// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_TRANSFORMS_INT_TUPLE_LOWERING
#define FLY_TRANSFORMS_INT_TUPLE_LOWERING

include "mlir/IR/PatternBase.td"
include "flydsl/Dialect/Fly/IR/FlyOps.td"

// GetOp patterns
def : Pat<(Fly_GetOp Fly_Layout:$layout, $mode), 
          (Fly_MakeLayoutOp (Fly_GetOp (Fly_GetShapeOp $layout), $mode), 
                            (Fly_GetOp (Fly_GetStrideOp $layout), $mode))>;
def : Pat<(Fly_GetOp Fly_ComposedLayout:$layout, $mode),
          (Fly_MakeComposedLayoutOp
            (Fly_ComposedGetInnerOp $layout),
            (Fly_ComposedGetOffsetOp $layout),
            (Fly_GetOp (Fly_ComposedGetOuterOp $layout), $mode)
        )>;
def : Pat<(Fly_GetOp Fly_MemRef:$memref, $mode),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_GetOp (Fly_GetLayoutOp $memref), $mode))>;
def : Pat<(Fly_GetOp Fly_CoordTensor:$tensor, $mode),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_GetOp (Fly_GetLayoutOp $tensor), $mode))>;

// TakeOp patterns
def : Pat<(Fly_TakeOp Fly_Layout:$layout, $begin, $end), 
          (Fly_MakeLayoutOp (Fly_TakeOp (Fly_GetShapeOp $layout), $begin, $end), 
                            (Fly_TakeOp (Fly_GetStrideOp $layout), $begin, $end))>;
def : Pat<(Fly_TakeOp Fly_ComposedLayout:$layout, $begin, $end),
          (Fly_MakeComposedLayoutOp
            (Fly_ComposedGetInnerOp $layout),
            (Fly_ComposedGetOffsetOp $layout),
            (Fly_TakeOp (Fly_ComposedGetOuterOp $layout), $begin, $end)
        )>;
def : Pat<(Fly_TakeOp Fly_MemRef:$memref, $begin, $end),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_TakeOp (Fly_GetLayoutOp $memref), $begin, $end))>;
def : Pat<(Fly_TakeOp Fly_CoordTensor:$tensor, $begin, $end),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_TakeOp (Fly_GetLayoutOp $tensor), $begin, $end))>;

// SelectOp patterns
def : Pat<(Fly_SelectOp Fly_Layout:$layout, $indices), 
          (Fly_MakeLayoutOp (Fly_SelectOp (Fly_GetShapeOp $layout), $indices), 
                            (Fly_SelectOp (Fly_GetStrideOp $layout), $indices))>;
def : Pat<(Fly_SelectOp Fly_ComposedLayout:$layout, $indices),
          (Fly_MakeComposedLayoutOp
            (Fly_ComposedGetInnerOp $layout),
            (Fly_ComposedGetOffsetOp $layout),
            (Fly_SelectOp (Fly_ComposedGetOuterOp $layout), $indices)
        )>;
def : Pat<(Fly_SelectOp Fly_MemRef:$memref, $indices),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_SelectOp (Fly_GetLayoutOp $memref), $indices))>;
def : Pat<(Fly_SelectOp Fly_CoordTensor:$tensor, $indices),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_SelectOp (Fly_GetLayoutOp $tensor), $indices))>;

// GroupOp patterns
def : Pat<(Fly_GroupOp Fly_Layout:$layout, $begin, $end), 
          (Fly_MakeLayoutOp (Fly_GroupOp (Fly_GetShapeOp $layout), $begin, $end), 
                            (Fly_GroupOp (Fly_GetStrideOp $layout), $begin, $end))>;
def : Pat<(Fly_GroupOp Fly_ComposedLayout:$layout, $begin, $end),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_GroupOp (Fly_ComposedGetOuterOp $layout), $begin, $end))>;
def : Pat<(Fly_GroupOp Fly_MemRef:$memref, $begin, $end),
          (Fly_MakeViewOp (Fly_GetIterOp $memref), (Fly_GroupOp (Fly_GetLayoutOp $memref), $begin, $end))>;
def : Pat<(Fly_GroupOp Fly_CoordTensor:$tensor, $begin, $end),
          (Fly_MakeViewOp (Fly_GetIterOp $tensor), (Fly_GroupOp (Fly_GetLayoutOp $tensor), $begin, $end))>;

// AppendOp patterns
def : Pat<(Fly_AppendOp Fly_Layout:$layout, Fly_Layout:$elem, $n),
          (Fly_MakeLayoutOp (Fly_AppendOp (Fly_GetShapeOp $layout), (Fly_GetShapeOp $elem), $n),
                            (Fly_AppendOp (Fly_GetStrideOp $layout), (Fly_GetStrideOp $elem), $n))>;
def : Pat<(Fly_AppendOp Fly_ComposedLayout:$layout, Fly_Layout:$elem, $n),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_AppendOp (Fly_ComposedGetOuterOp $layout), $elem, $n))>;
        
// PrependOp patterns
def : Pat<(Fly_PrependOp Fly_Layout:$layout, Fly_Layout:$elem, $n),
          (Fly_MakeLayoutOp (Fly_PrependOp (Fly_GetShapeOp $layout), (Fly_GetShapeOp $elem), $n),
                            (Fly_PrependOp (Fly_GetStrideOp $layout), (Fly_GetStrideOp $elem), $n))>;
def : Pat<(Fly_PrependOp Fly_ComposedLayout:$layout, Fly_Layout:$elem, $n),
          (Fly_MakeComposedLayoutOp (Fly_ComposedGetInnerOp $layout),
                                    (Fly_ComposedGetOffsetOp $layout),
                                    (Fly_PrependOp (Fly_ComposedGetOuterOp $layout), $elem, $n))>;

// SliceOp patterns
def : Pat<(Fly_SliceOp Fly_Layout:$layout, $coord), 
          (Fly_MakeLayoutOp (Fly_SliceOp (Fly_GetShapeOp $layout), $coord),
                            (Fly_SliceOp (Fly_GetStrideOp $layout), $coord))>;
def : Pat<(Fly_SliceOp Fly_ComposedLayout:$layout, $coord),
          (Fly_MakeComposedLayoutOp
            (Fly_ComposedGetInnerOp $layout),
            (Fly_IntTupleAddOp (Fly_Crd2IdxOp $coord, (Fly_ComposedGetOuterOp $layout)), 
                               (Fly_ComposedGetOffsetOp $layout)),
            (Fly_SliceOp (Fly_ComposedGetOuterOp $layout), $coord))>;

def : Pat<(Fly_SliceOp Fly_MemRef:$memref, $coord), 
          (Fly_MakeViewOp  
            (Fly_AddOffsetOp (Fly_GetIterOp $memref),
                             (Fly_Crd2IdxOp $coord, (Fly_GetLayoutOp $memref))),
            (Fly_SliceOp (Fly_GetLayoutOp $memref), $coord)),
          [(Fly_SimpleLayoutMemRef $memref)]>;
def : Pat<(Fly_SliceOp Fly_CoordTensor:$tensor, $coord), 
          (Fly_MakeViewOp  
            (Fly_AddOffsetOp (Fly_GetIterOp $tensor),
                             (Fly_Crd2IdxOp $coord, (Fly_GetLayoutOp $tensor))),
            (Fly_SliceOp (Fly_GetLayoutOp $tensor), $coord)),
          [(Fly_SimpleLayoutCoordTensor $tensor)]>;

def : Pat<(Fly_SliceOp Fly_MemRef:$memref, $coord), 
          (Fly_MakeViewOp  
            (Fly_GetIterOp $memref),
            (Fly_SliceOp (Fly_GetLayoutOp $memref), $coord)),
          [(Fly_ComposedLayoutMemRef $memref)]>;
def : Pat<(Fly_SliceOp Fly_CoordTensor:$tensor, $coord), 
          (Fly_MakeViewOp  
            (Fly_GetIterOp $tensor),
            (Fly_SliceOp (Fly_GetLayoutOp $tensor), $coord)),
          [(Fly_ComposedLayoutCoordTensor $tensor)]>;
          
// DiceOp patterns
def : Pat<(Fly_DiceOp Fly_Layout:$layout, $coord), 
          (Fly_MakeLayoutOp (Fly_DiceOp (Fly_GetShapeOp $layout), $coord),
                            (Fly_DiceOp (Fly_GetStrideOp $layout), $coord))>;

#endif // FLY_TRANSFORMS_INT_TUPLE_LOWERING
