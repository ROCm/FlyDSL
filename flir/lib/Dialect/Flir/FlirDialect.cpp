//===- FlirDialect.cpp - Flir Dialect Implementation --------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/APInt.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flir/FlirDialect.h"
#include "flir/FlirOps.h" // For generated op classes.

using namespace mlir;
using namespace mlir::flir;

#include "flir/FlirDialect.cpp.inc"

namespace {

void printIntTuplePattern(::mlir::AsmPrinter &os, Attribute attr) {
  if (!attr)
    return;
  if (auto d = llvm::dyn_cast<DyncI32Attr>(attr)) {
    if (d.getDivisibility() == 1)
      os << "?";
    else
      os << "?{div=" << d.getDivisibility() << "}";
  } else if (auto constIntAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    os << constIntAttr.getValue();
  } else if (llvm::isa<UnderscoreAttr>(attr)) {
    os << "*";
  } else if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
    os << "(";
    bool first = true;
    for (auto element : arrayAttr.getValue()) {
      if (!first)
        os << ",";
      printIntTuplePattern(os, element);
      first = false;
    }
    os << ")";
  } else {
    // Fallback (shouldn't happen for well-formed patterns).
    os.printAttribute(attr);
  }
}

struct ParseIntTuplePattern {
  int dyncElemIdx = 0;

  ParseResult operator()(::mlir::AsmParser &odsParser, ::mlir::Attribute &attr) {
    // Tuple: '(' elem (',' elem)* ')'
    if (odsParser.parseOptionalLParen().succeeded()) {
      SmallVector<Attribute> elements;
      do {
        Attribute element;
        if ((*this)(odsParser, element))
          return ::mlir::failure();
        elements.push_back(element);
      } while (odsParser.parseOptionalComma().succeeded());
      if (odsParser.parseRParen())
        return ::mlir::failure();
      attr = ArrayAttr::get(odsParser.getContext(), elements);
      return ParseResult::success();
    }

    // Dynamic leaf: '?'
    if (odsParser.parseOptionalQuestion().succeeded()) {
      int32_t divisibility = 1;
      if (odsParser.parseOptionalLBrace().succeeded()) {
        if (odsParser.parseKeyword("div") || odsParser.parseEqual() ||
            odsParser.parseInteger(divisibility) || odsParser.parseRBrace()) {
          return ::mlir::failure();
        }
      }
      attr = DyncI32Attr::get(odsParser.getContext(), dyncElemIdx++, divisibility);
      return ParseResult::success();
    }

    // Wildcard leaf: '*'
    if (odsParser.parseOptionalStar().succeeded()) {
      attr = UnderscoreAttr::get(odsParser.getContext());
      return ParseResult::success();
    }

    // Constant integer leaf.
    int64_t value = 0;
    if (odsParser.parseDecimalInteger(value))
      return ::mlir::failure();
    attr = IntegerAttr::get(IntegerType::get(odsParser.getContext(), 32),
                            APInt(32, static_cast<uint64_t>(value), /*isSigned=*/true));
    return ::mlir::success();
  }
};

ParseResult parseIntTuplePattern(::mlir::AsmParser &odsParser, ::mlir::Attribute &attr,
                                 int /*unused*/ = 0) {
  return ParseIntTuplePattern{}(odsParser, attr);
}

} // namespace

#define GET_TYPEDEF_CLASSES
#include "flir/FlirTypeDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "flir/FlirAttrDefs.cpp.inc"

void FlirDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flir/FlirTypeDefs.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "flir/FlirAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "flir/FlirOps.cpp.inc"
      >();
}

