//===- EmitOpTest.cpp - The EmitOp category -----------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// EmitOp differs from Runtime only in how a leaf operation produces its value:
// Runtime computes it in C++, EmitOp emits arith/scf IR.  The algebra above the
// leaves is shared and branches only on leaf properties, so checking every leaf
// operation over all kinds of operands checks EmitOp as a whole.  Runtime is
// the reference.  A case is outside the domain
// and skipped when its Runtime result is an error or its operands are ones the
// operation leaves undefined (ceil_div and shape_div need a >= 0, b > 0; see
// Category.def).  Each case checks that
//
//   1. the EmitOp result has the Runtime result's property (static or dynamic,
//      width, divisibility, basis modes);
//   2. a static result emits no IR and has the Runtime value;
//   3. a dynamic result, lowered to LLVM and executed, has the Runtime value;
//   4. `divsi exact` is emitted only when the operand properties prove the
//      division exact, since an inexact one is poison that execution can miss.
//
// The EmitOpCategory cases check what only EmitOp has: binding a Trait profile
// to SSA values and reading them back, the lowered LLVM struct of a profile,
// the pointer offset callback, and the IR that folds and proven exact
// divisions leave.
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

#include "flydsl/Core/EmitOp.hpp"
#include "flydsl/Core/Symbol.hpp"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using namespace core_test;

namespace {

/// A leaf operation written once and instantiated for both categories.
struct LeafOp {
  const char *name;
  bool unary;
  Leaf<Runtime> (*runtime)(Leaf<Runtime>, Leaf<Runtime>);
  Leaf<EmitOp> (*emit)(Leaf<EmitOp>, Leaf<EmitOp>);
};

#define BINARY(NAME, EXPR)                                                                         \
  LeafOp {                                                                                         \
    NAME, false, [](Leaf<Runtime> a, Leaf<Runtime> b) -> Leaf<Runtime> { return EXPR; },           \
        [](Leaf<EmitOp> a, Leaf<EmitOp> b) -> Leaf<EmitOp> { return EXPR; }                        \
  }
#define UNARY(NAME, EXPR)                                                                          \
  LeafOp {                                                                                         \
    NAME, true, [](Leaf<Runtime> a, Leaf<Runtime>) -> Leaf<Runtime> { return EXPR; },              \
        [](Leaf<EmitOp> a, Leaf<EmitOp>) -> Leaf<EmitOp> { return EXPR; }                          \
  }

// clang-format off
const LeafOp ops[] = {
    BINARY("+", a + b),
    BINARY("-", a - b),
    BINARY("*", a * b),
    BINARY("/", a / b),
    BINARY("%", a % b),
    BINARY("min", min(a, b)),
    BINARY("max", max(a, b)),
    BINARY("gcd", gcd(a, b)),
    BINARY("ceil_div", ceil_div(a, b)),
    BINARY("shape_div", shape_div(a, b)),
    BINARY("safe_div", safe_div(a, b)),
    BINARY("exact_div", exact_div(a, b)),
    BINARY("&", a & b),
    BINARY("|", a | b),
    BINARY("^", a ^ b),
    BINARY("<<", a << b),
    BINARY(">>", a >> b),
    BINARY("lshr", lshr(a, b)),
    BINARY("<", a < b),
    BINARY("eq", eq(a, b)),
    UNARY("neg", -a),
    UNARY("abs", abs(a)),
    UNARY("signum", signum(a)),
};
// clang-format on

#undef BINARY
#undef UNARY

/// Operands: static values, dynamic values of several widths and
/// divisibilities, a static ratio and basis leaves.
std::vector<Leaf<Runtime>> operands() {
  return {
      Leaf<Runtime>::getSInt(0),
      Leaf<Runtime>::getSInt(1),
      Leaf<Runtime>::getSInt(-1),
      Leaf<Runtime>::getSInt(3),
      Leaf<Runtime>::getSInt(4),
      Leaf<Runtime>::getSInt(-8),
      Leaf<Runtime>::getDyn(-7),
      Leaf<Runtime>::getDyn(0),
      Leaf<Runtime>::getDyn(13),
      Leaf<Runtime>::getDyn(-8, LogWidth::I32, 4),
      Leaf<Runtime>::getDyn(24, LogWidth::I32, 8),
      Leaf<Runtime>::getDyn(-6, LogWidth::I64, 2),
      Leaf<Runtime>::getDyn(-3, LogWidth::I8),
      Leaf<Runtime>::getRatio(3, 2),
      Leaf<Runtime>::getBasis(Leaf<Runtime>::getSInt(2), {1}),
      Leaf<Runtime>::getBasis(Leaf<Runtime>::getDyn(6, LogWidth::I32, 2), {0}),
  };
}

/// Whether `op` is defined for the values `a` and `b` hold.
bool defined(llvm::StringRef op, Leaf<Runtime> a, Leaf<Runtime> b) {
  if (op == "ceil_div" || op == "shape_div")
    return a.value >= 0 && b.value > 0;
  if (op == "lshr")
    return a.value >= 0 && b.value >= 0;
  return true;
}

/// Whether the operand properties alone prove `a / b` exact.
bool provenExact(llvm::StringRef op, Leaf<Runtime> a, Leaf<Runtime> b) {
  if (op == "exact_div")
    return true;
  auto x = Leaf<Trait>(a).scale(), y = Leaf<Trait>(b).scale();
  auto divides = [](Leaf<Trait> value, int64_t divisor) {
    return value.isDInt() && divisor != 0 && value.divisibility() % std::llabs(divisor) == 0;
  };
  if (y.isSInt() && divides(x, y.staticValue()))
    return true;
  // ratio * dynamic divides the dynamic operand by the denominator first.
  return op == "*" &&
         ((x.isRatio() && divides(y, x.den())) || (y.isRatio() && divides(x, y.den())));
}

class LeafHarness {
public:
  LeafHarness()
      : context(registry()), builder(&context), loc(builder.getUnknownLoc()),
        module(mlir::ModuleOp::create(loc)), ctx(builder, loc) {
    context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect,
                        mlir::LLVM::LLVMDialect>();
  }

  /// Emits `op(a, b)` and records the Runtime value its execution must give.
  void check(const LeafOp &op, Leaf<Runtime> a, Leaf<Runtime> b) {
    auto label = op.unary ? std::string(op.name) + " " + toString(a)
                          : toString(a) + " " + op.name + " " + toString(b);
    if (!defined(op.name, a, b)) {
      ++skipped;
      return;
    }
    auto expected = op.runtime(a, b);
    if (expected.isError()) {
      ++skipped;
      return;
    }
    if (!expected.isSound()) {
      ADD_FAILURE() << label << ": the Runtime reference " << toString(expected)
                    << " holds a value its property excludes";
      return;
    }
    auto name = "case" + std::to_string(cases++);
    builder.setInsertionPointToEnd(module->getBody());
    auto i64 = builder.getI64Type();
    auto function =
        mlir::func::FuncOp::create(builder, loc, name, builder.getFunctionType({i64, i64}, {i64}));
    function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    auto &body = *function.addEntryBlock();
    builder.setInsertionPointToEnd(&body);
    auto lhs = bind(a, body.getArgument(0));
    auto rhs = bind(b, body.getArgument(1));
    auto boundOps = body.getOperations().size();
    auto actual = op.emit(lhs, rhs);

    if (actual.isError()) {
      ADD_FAILURE() << label << ": " << toString(Leaf<Trait>(actual));
      function.erase();
      return;
    }
    EXPECT_EQ(toString(Leaf<Trait>(actual)), toString(Leaf<Trait>(expected))) << label;
    auto exact = false;
    function.walk([&](mlir::arith::DivSIOp div) { exact |= div.getIsExact(); });
    EXPECT_TRUE(!exact || provenExact(op.name, a, b)) << label << ": unproven divsi exact";

    if (actual.isStatic()) {
      EXPECT_EQ(body.getOperations().size(), boundOps) << label << ": static result emitted IR";
      // A static ratio has no integer value; its property was compared above.
      if (Leaf<Trait>(actual).scale().isSInt())
        EXPECT_EQ(Leaf<Trait>(actual).scale().staticValue(), expected.value) << label;
      function.erase();
      ++folded;
      return;
    }
    auto value = actual.materialize(ctx);
    if (mlir::failed(value)) {
      ADD_FAILURE() << label << ": cannot materialize the result";
      function.erase();
      return;
    }
    // An i1 predicate zero-extends; wider Core integers sign-extend.
    mlir::func::ReturnOp::create(
        builder, loc,
        mlir::convertScalarToDtype(builder, loc, *value, builder.getI64Type(),
                                   /*isUnsignedCast=*/value->getType().isInteger(1)));
    pending.push_back({name, label, a.value, b.value, expected.value});
  }

  /// Lowers the module to LLVM, then runs every emitted case.
  void execute() {
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    auto pm = mlir::PassManager(&context);
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
    auto engine = mlir::ExecutionEngine::create(*module);
    ASSERT_TRUE(bool(engine)) << llvm::toString(engine.takeError());
    for (const auto &run : pending) {
      auto result = int64_t{0};
      auto error = (*engine)->invoke(run.function, run.a, run.b,
                                     mlir::ExecutionEngine::Result<int64_t>(result));
      ASSERT_FALSE(bool(error)) << run.label << ": " << llvm::toString(std::move(error));
      EXPECT_EQ(result, run.expected) << run.label;
    }
  }

  int cases = 0, folded = 0, skipped = 0;
  size_t executed() const { return pending.size(); }

private:
  struct Pending {
    std::string function, label;
    int64_t a, b, expected;
  };

  static mlir::DialectRegistry registry() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    auto registry = mlir::DialectRegistry();
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    return registry;
  }

  /// The EmitOp operand for `operand`: a dynamic scale reads its value from
  /// the i64 argument, truncated to the operand width.
  Leaf<EmitOp> bind(Leaf<Runtime> operand, mlir::Value argument) {
    auto property = Leaf<Trait>(operand);
    if (!property.scale().isDInt())
      return Leaf<EmitOp>(property);
    auto value = mlir::convertScalarToDtype(builder, loc, argument,
                                            builder.getIntegerType(property.scale().bitWidth()),
                                            /*isUnsignedCast=*/false);
    return Leaf<EmitOp>(property, value, &ctx);
  }

  mlir::MLIRContext context;
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  EmitContext ctx;
  std::vector<Pending> pending;
};

} // namespace

TEST(LeafEmitOpParity, MatchesRuntime) {
  auto harness = LeafHarness();
  auto values = operands();
  for (const auto &op : ops)
    for (auto a : values)
      for (auto b : op.unary ? std::vector<Leaf<Runtime>>{Leaf<Runtime>::getZero()} : values)
        harness.check(op, a, b);
  harness.execute();
  llvm::outs() << harness.cases << " cases: " << harness.executed() << " executed, "
               << harness.folded << " static; " << harness.skipped << " outside the domain\n";
}

namespace {

class EmitOpCategory : public ::testing::Test {
protected:
  mlir::MLIRContext mlirContext;
  mlir::OpBuilder builder{&mlirContext};
  mlir::Location loc = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  mlir::Block *entry = nullptr;
  EmitContext ctx{builder, loc};

  void SetUp() override {
    mlirContext.loadDialect<mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                            mlir::LLVM::LLVMDialect, mlir::func::FuncDialect>();
    mlirContext.allowUnregisteredDialects();
    module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module->getBody());
    auto types = llvm::SmallVector<mlir::Type>{
        builder.getI8Type(),
        builder.getI16Type(),
        builder.getI32Type(),
        builder.getI64Type(),
        builder.getI64Type(),
        builder.getI1Type(),
        mlir::LLVM::LLVMPointerType::get(&mlirContext, 3),
        mlir::OpaqueType::get(builder.getStringAttr("test"), "layout")};
    function = mlir::func::FuncOp::create(builder, loc, "test", builder.getFunctionType(types, {}));
    entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    mlir::func::ReturnOp::create(builder, loc);
    builder.setInsertionPointToStart(entry);
  }
  Leaf<EmitOp> arg(unsigned i) { return Leaf<EmitOp>::getDyn(entry->getArgument(i), ctx); }
  void check(Leaf<EmitOp> value, Leaf<Trait> expected) {
    ASSERT_FALSE(value.isError()) << toString(value);
    EXPECT_TRUE(value.isSound());
    EXPECT_EQ(to_trait(value), expected);
    if (value.isStatic())
      EXPECT_FALSE(value.value());
    auto materialized = value.materialize(ctx);
    ASSERT_TRUE(mlir::succeeded(materialized));
    EXPECT_TRUE((*materialized).getType().isInteger(value.bitWidth()));
  }
  void TearDown() override { EXPECT_TRUE(mlir::succeeded(mlir::verify(*module))); }
};

TEST_F(EmitOpCategory, RoundTripsOpaqueMlirTypeAcrossCategories) {
  auto context = mlir::MLIRContext();
  auto input = mlir::IntegerType::get(&context, 32);
  auto pointer = Pointer<Trait>(OpaqueValue::get(input), 32, 16);
  auto runtime = Pointer<Runtime>(pointer.elementType(), 32, 16, nullptr);
  auto symbol = Pointer<Symbol>(runtime.elementType(), 32, 16, "opaque_mlir_round_trip");
  auto emit = Pointer<EmitOp>(symbol.elementType(), 32, 16, mlir::Value{});
  auto result = Pointer<Trait>(emit.elementType(), 32, 16);

  ASSERT_TRUE(result.elementType().is<mlir::IntegerType>());
  EXPECT_EQ(result.elementType().get<mlir::IntegerType>(), input);
  EXPECT_EQ(result.elementType(), OpaqueValue::get(input));
  EXPECT_EQ(hash_value(result.elementType()), hash_value(OpaqueValue::get(input)));
}

TEST_F(EmitOpCategory, StaticFoldsDiscardSSAAndMaterializeAtInferredWidth) {
  auto x = arg(3);
  auto count = entry->getOperations().size();
  auto zero = x * Leaf<EmitOp>::getZero();
  EXPECT_EQ(entry->getOperations().size(), count);
  check(zero, Leaf<Trait>::getZero());
  EXPECT_EQ(zero.bitWidth(), 32);
  check(Leaf<EmitOp>::getSInt(INT32_MAX) + Leaf<EmitOp>::getOne(),
        Leaf<Trait>::getSInt(int64_t{INT32_MAX} + 1));
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
  auto ranged = x.withRange(2, 9);
  count = entry->getOperations().size();
  auto comparison = lt(ranged, Leaf<EmitOp>::getSInt(10));
  EXPECT_EQ(entry->getOperations().size(), count);
  check(comparison, Leaf<Trait>::getOne());
#endif
  check(x % Leaf<EmitOp>::getOne(), Leaf<Trait>::getZero());
  // A static property ignores an unrelated runtime payload; it cannot leak
  // through a fold as a stale i64 value.
  check(Leaf<EmitOp>(Leaf<Trait>::getSInt(5), x.value(), &ctx), Leaf<Trait>::getSInt(5));
}

TEST_F(EmitOpCategory, IdentityShortcutsCastToTheTypeResult) {
  auto x = arg(0);
  auto one = x * Leaf<EmitOp>::getOne();
  check(one, to_trait(x) * Leaf<Trait>::getOne());
  EXPECT_TRUE(one.value().getDefiningOp<mlir::arith::ExtSIOp>());
  auto sum = x + Leaf<EmitOp>::getZero();
  check(sum, to_trait(x) + Leaf<Trait>::getZero());
  EXPECT_TRUE(sum.value().getDefiningOp<mlir::arith::ExtSIOp>());
  check(x | Leaf<EmitOp>::getZero(), to_trait(x) | Leaf<Trait>::getZero());
  auto y = arg(3);
  check(y / Leaf<EmitOp>::getOne(), to_trait(y) / Leaf<Trait>::getOne());
  EXPECT_EQ((y / Leaf<EmitOp>::getOne()).value(), y.value());
}

TEST_F(EmitOpCategory, ShiftsAndBooleanArithmeticUseTheTypeResult) {
  // A shift takes the wider operand type: an i64 amount is not truncated.
  auto shifted = Leaf<EmitOp>::getSInt(2) << arg(3);
  check(shifted, Leaf<Trait>::getSInt(2) << to_trait(arg(3)));
  EXPECT_EQ(shifted.bitWidth(), 64);
  // Arithmetic computes i1 at i32, as the DSL widens Boolean.
  auto flag = arg(5);
  auto sum = flag + flag;
  check(sum, to_trait(flag) + to_trait(flag));
  EXPECT_EQ(sum.bitWidth(), 32);
  EXPECT_TRUE(sum.value().getDefiningOp<mlir::arith::AddIOp>());
  EXPECT_TRUE(llvm::none_of(entry->getOperations(), [](mlir::Operation &op) {
    return mlir::isa<mlir::arith::TruncIOp>(op);
  }));
}

TEST_F(EmitOpCategory, ProvenExactDivisionSkipsRounding) {
  // ?{div=64} / 32 is exact, so ceil_div and shape_div need no rounding fixup.
  auto x = Leaf<EmitOp>::getDyn(entry->getArgument(2), ctx, 64);
  auto thirtyTwo = Leaf<EmitOp>::getSInt(32);
  for (auto quotient :
       {ceil_div(x, thirtyTwo), shape_div(x, thirtyTwo), x / thirtyTwo, safe_div(x, thirtyTwo)}) {
    check(quotient, Leaf<Trait>::getDInt(LogWidth::I32, 2));
    auto division = quotient.value().getDefiningOp<mlir::arith::DivSIOp>();
    ASSERT_TRUE(division);
    EXPECT_TRUE(division.getIsExact());
  }

  auto fortyEight = Leaf<EmitOp>::getSInt(48);
  auto rounded = ceil_div(x, fortyEight);
  check(rounded, Leaf<Trait>::getDInt(LogWidth::I32));
  EXPECT_TRUE(rounded.value().getDefiningOp<mlir::arith::CeilDivSIOp>());
  auto truncated = (x / fortyEight).value().getDefiningOp<mlir::arith::DivSIOp>();
  ASSERT_TRUE(truncated);
  EXPECT_FALSE(truncated.getIsExact());
  // A dynamic operand makes safe_div a plain quotient, exact only when proven.
  auto safe = safe_div(x, fortyEight);
  check(safe, Leaf<Trait>::getDInt(LogWidth::I32));
  auto safeDivision = safe.value().getDefiningOp<mlir::arith::DivSIOp>();
  ASSERT_TRUE(safeDivision);
  EXPECT_FALSE(safeDivision.getIsExact());
}

TEST_F(EmitOpCategory, ComparisonsUseOperandWidthAndProduceTheCoreResultWidth) {
  for (unsigned i : {0u, 1u, 2u, 3u, 5u}) {
    auto x = arg(i);
    auto result = lt(x, Leaf<EmitOp>::getSInt(7));
    check(result, lt(to_trait(x), Leaf<Trait>::getSInt(7)));
    auto ext = result.value().getDefiningOp<mlir::arith::ExtUIOp>();
    ASSERT_TRUE(ext);
    EXPECT_TRUE(ext.getIn().getDefiningOp<mlir::arith::CmpIOp>());
  }
  check(lt(arg(5), arg(5)), lt(to_trait(arg(5)), to_trait(arg(5))));
  check(signum(arg(3)), signum(to_trait(arg(3))));
  check(abs(arg(0)), abs(to_trait(arg(0))));
}

TEST_F(EmitOpCategory, AllScalarOperationsAndBasisPreserveProjection) {
  auto a = arg(2), b = arg(3);
  a = a.withRange(-16, 16);
  EXPECT_EQ(a.value(), arg(2).value());
  EXPECT_EQ(a.context(), &ctx);
#define CHECK_EMIT(OP) check(a OP b, to_trait(a) OP to_trait(b))
  CHECK_EMIT(+);
  CHECK_EMIT(-);
  CHECK_EMIT(*);
  CHECK_EMIT(/);
  CHECK_EMIT(%);
  CHECK_EMIT(&);
  CHECK_EMIT(|);
  CHECK_EMIT(^);
  CHECK_EMIT(<<);
  CHECK_EMIT(>>);
#undef CHECK_EMIT
  check(min(a, b), min(to_trait(a), to_trait(b)));
  check(max(a, b), max(to_trait(a), to_trait(b)));
  check(gcd(a, b), gcd(to_trait(a), to_trait(b)));
  check(ceil_div(a, b), ceil_div(to_trait(a), to_trait(b)));
  check(shape_div(a, b), shape_div(to_trait(a), to_trait(b)));
  auto divisible = Leaf<EmitOp>::getDyn(a.value(), ctx, 8);
  check(safe_div(divisible, Leaf<EmitOp>::getSInt(4)),
        safe_div(to_trait(divisible), Leaf<Trait>::getSInt(4)));
  check(exact_div(divisible, Leaf<EmitOp>::getSInt(2)),
        exact_div(to_trait(divisible), Leaf<Trait>::getSInt(2)));
  auto basis = Leaf<EmitOp>::getBasis(a, {1, 0});
  auto result = basis * Leaf<EmitOp>::getSInt(2);
  ASSERT_TRUE(result.isBasis());
  EXPECT_EQ(to_trait(result), to_trait(basis) * Leaf<Trait>::getSInt(2));
  EXPECT_TRUE(result.value().getType().isInteger(result.scale().bitWidth()));
  EXPECT_EQ(to_trait(result.scale()), to_trait(result).scale());
  check(a << Leaf<EmitOp>::getSInt(-2), to_trait(a) << Leaf<Trait>::getSInt(-2));
}

TEST_F(EmitOpCategory, RejectsMissingAndMismatchedBindingsWithoutEmission) {
  auto count = entry->getOperations().size();
  EXPECT_TRUE(Leaf<EmitOp>(Leaf<Trait>::getDInt(LogWidth::I64), arg(0).value(), &ctx).isError());
  EXPECT_TRUE(Leaf<EmitOp>::getDInt().isError());
  EXPECT_TRUE(Leaf<EmitOp>::getDyn({}, ctx).isError());
  EXPECT_TRUE((arg(3) / Leaf<EmitOp>::getZero()).isError());
  auto other = EmitContext(builder, loc);
  auto x = Leaf<EmitOp>::getDyn(arg(3).value(), other);
  EXPECT_TRUE((x + arg(3)).isError());
  EXPECT_EQ(entry->getOperations().size(), count);
  EXPECT_TRUE(mlir::failed(Leaf<EmitOp>::getRatio(1, 2).materialize(ctx)));
}

TEST_F(EmitOpCategory, BindChecksScalarSignature) {
  auto profile = *Layout<Trait>::fromString("(?,4):(1,?i64)");
  auto bound = bind(profile, EmitBindings{{arg(2).value(), arg(3).value()}, {}}, ctx);
  ASSERT_TRUE(mlir::succeeded(bound));
  EXPECT_EQ(to_trait(*bound), profile);
  auto values = get_bindings(*bound, ctx);
  ASSERT_TRUE(mlir::succeeded(values));
  ASSERT_EQ(values->scalars.size(), 2u);
  EXPECT_TRUE(values->scalars[0].getType().isInteger(32));
  EXPECT_TRUE(values->scalars[1].getType().isInteger(64));
  EXPECT_EQ(get_binding_types(profile, ctx),
            (llvm::SmallVector<mlir::Type, 8>{builder.getI32Type(), builder.getI64Type()}));
  EXPECT_TRUE(mlir::failed(bind(profile, EmitBindings{}, ctx)));
  auto wrong = *values;
  std::swap(wrong.scalars[0], wrong.scalars[1]);
  EXPECT_TRUE(mlir::failed(bind(profile, wrong, ctx)));
  wrong = *values;
  wrong.scalars.push_back(arg(2).value());
  EXPECT_TRUE(mlir::failed(bind(profile, wrong, ctx)));
}

TEST_F(EmitOpCategory, AlgebraCommutesWithProjectionAndBindingsRoundTrip) {
  auto profile = *Layout<Trait>::fromString("(?,4):(1,?i64)");
  auto bound = bind(profile, EmitBindings{{arg(2).value(), arg(3).value()}, {}}, ctx);
  ASSERT_TRUE(mlir::succeeded(bound));
  auto expected = coalesce(profile);
  auto result = coalesce(*bound);
  EXPECT_EQ(to_trait(result), expected);
  auto values = get_bindings(result, ctx);
  ASSERT_TRUE(mlir::succeeded(values));
  auto rebound = bind(to_trait(result), *values, ctx);
  ASSERT_TRUE(mlir::succeeded(rebound));
  EXPECT_EQ(*rebound, result);
  auto coordinate = make_tuple(IntTuple<EmitOp>::fromLeaf(arg(2)), IntTuple<EmitOp>::getSInt(2));
  EXPECT_EQ(to_trait(crd2idx(coordinate, *bound)), crd2idx(to_trait(coordinate), profile));
}

TEST_F(EmitOpCategory, PointerArithmeticAndTensorBindingKeepOpaqueAddressSpace) {
  auto space = OpaqueValue::get(builder.getStringAttr("shared"));
  auto pointerType = Pointer<Trait>(OpaqueValue::get(builder.getF32Type()), 32, 16, space);
  auto pointer = Pointer<EmitOp>(pointerType, entry->getArgument(6), ctx);
  auto result = add_offset(pointer, arg(2));
  ASSERT_TRUE(result.isValid());
  EXPECT_EQ(result.addressSpace(), space);
  EXPECT_EQ(result.value().getType(), pointer.value().getType());
  EXPECT_EQ(to_trait(result), add_offset(pointerType, to_trait(arg(2))));
  auto unchanged = add_offset(pointer, Leaf<EmitOp>::getZero());
  EXPECT_EQ(unchanged.value(), pointer.value());
  EXPECT_EQ(to_trait(unchanged), to_trait(pointer));
  // Equality includes the SSA address, as for leaves.
  EXPECT_EQ(unchanged, pointer);
  auto rebased = Pointer<EmitOp>(pointerType, result.value(), ctx);
  EXPECT_NE(rebased, Pointer<EmitOp>(pointerType, pointer.value(), ctx));
  auto profile = MemRef<Trait>(pointerType, *Layout<Trait>::fromString("?:1"));
  auto bound = bind(profile, EmitBindings{{arg(2).value()}, {pointer.value()}}, ctx);
  ASSERT_TRUE(mlir::succeeded(bound));
  EXPECT_EQ(to_trait(*bound).data(), profile.data());
  EXPECT_EQ(to_trait(*bound).layout(), profile.layout());
  auto values = get_bindings(*bound, ctx);
  ASSERT_TRUE(mlir::succeeded(values));
  EXPECT_EQ(values->pointers.front(), pointer.value());
  EXPECT_EQ(get_binding_types(profile, ctx),
            (llvm::SmallVector<mlir::Type, 8>{builder.getI32Type()}));
  // A pointer of another context is rejected, as a scalar of another type is.
  auto foreignContext = mlir::MLIRContext();
  foreignContext.loadDialect<mlir::LLVM::LLVMDialect>();
  foreignContext.allowUnregisteredDialects();
  auto foreignState = mlir::OperationState(mlir::UnknownLoc::get(&foreignContext), "test.ptr");
  foreignState.addTypes(mlir::LLVM::LLVMPointerType::get(&foreignContext, 3));
  auto *foreign = mlir::Operation::create(foreignState);
  EXPECT_TRUE(
      mlir::failed(bind(profile, EmitBindings{{arg(2).value()}, {foreign->getResult(0)}}, ctx)));
  foreign->destroy();
}

TEST_F(EmitOpCategory, PointerByteBoolStorageSurvivesBindingAndOffset) {
  auto profile = Pointer<Trait>(OpaqueValue::get(builder.getI1Type()), 1, 16, int64_t{128},
                                Swizzle<Trait>{}, OpaqueValue::getType<void>(), 8);
  auto bound = bind(profile, EmitBindings{{}, {entry->getArgument(6)}}, ctx);
  ASSERT_TRUE(mlir::succeeded(bound));
  EXPECT_EQ(to_trait(*bound), profile);
  auto shifted = add_offset(*bound, arg(2));
  ASSERT_TRUE(shifted.isValid());
  EXPECT_EQ(to_trait(shifted).elementBitWidth(), 1);
  EXPECT_EQ(to_trait(shifted).storageElementBitWidth(), 8);
  EXPECT_EQ(to_trait(shifted), add_offset(profile, to_trait(arg(2))));
  auto values = get_bindings(shifted, ctx);
  ASSERT_TRUE(mlir::succeeded(values));
  auto rebound = bind(to_trait(shifted), *values, ctx);
  ASSERT_TRUE(mlir::succeeded(rebound));
  EXPECT_EQ(to_trait(*rebound), to_trait(shifted));
}

TEST_F(EmitOpCategory, ComposedTraversalKeepsOuterOffsetInnerAndDynamicBasis) {
  auto outer = *Layout<Trait>::fromString("?:1");
  auto offset = *IntTuple<Trait>::fromString("?i64");
  auto inner = *Layout<Trait>::fromString("4:?i16");
  auto profile = make_composed_layout(inner, offset, outer);
  auto bindings = EmitBindings{{arg(2).value(), arg(3).value(), arg(1).value()}, {}};
  auto bound = bind(profile, bindings, ctx);
  ASSERT_TRUE(mlir::succeeded(bound));
  EXPECT_EQ(to_trait(*bound), profile);
  auto collected = get_bindings(*bound, ctx);
  ASSERT_TRUE(mlir::succeeded(collected));
  EXPECT_EQ(collected->scalars, bindings.scalars);
  auto basis = *IntTuple<Trait>::fromString("(4,?i64E1)");
  auto boundBasis = bind(basis, EmitBindings{{arg(3).value()}, {}}, ctx);
  ASSERT_TRUE(mlir::succeeded(boundBasis));
  EXPECT_EQ(to_trait(*boundBasis), basis);
}

TEST_F(EmitOpCategory, AtomDescriptorsAndTiledObjectsRoundTripWithoutSSAState) {
  auto copy = to_trait(make_tiled_copy(universalCopy(128, 16),
                                       make_layout(make_tuple(32_c, 8_c), make_tuple(8_c, 1_c)),
                                       make_layout(make_tuple(1_c, 8_c))));
  auto emitCopy = bind(copy, EmitBindings{}, ctx);
  ASSERT_TRUE(mlir::succeeded(emitCopy));
  // The printed form omits the opaque operation and internal type, so check them separately.
  EXPECT_EQ(toString(to_trait(*emitCopy)), toString(copy));
  EXPECT_EQ(emitCopy->copyAtom().copyOperation(), copy.copyAtom().copyOperation());
  EXPECT_EQ(emitCopy->copyAtom().copyInternalType(), copy.copyAtom().copyInternalType());
  auto mma =
      to_trait(make_tiled_mma(MMAAtom_CDNA3Mfma16x16x16F16(), make_layout(make_tuple(2_c, 2_c))));
  auto emitMma = bind(mma, EmitBindings{}, ctx);
  ASSERT_TRUE(mlir::succeeded(emitMma));
  EXPECT_EQ(toString(to_trait(*emitMma)), toString(mma));
  EXPECT_EQ(emitMma->mmaAtom().mmaOperation(), mma.mmaAtom().mmaOperation());
  auto slice = emitCopy->get_slice(IntTuple<EmitOp>::fromLeaf(arg(2)));
  EXPECT_EQ(to_trait(slice).thr_idx(), IntTuple<Trait>::fromLeaf(to_trait(arg(2))));
}

/// Hooks whose pointer offset is a test callback.
struct OffsetCallbackHooks : EmitHooks {
  std::function<mlir::FailureOr<mlir::Value>(EmitContext &, const Pointer<EmitOp> &,
                                             IntTupleRef<EmitOp>, const Pointer<Trait> &)>
      offset;

  mlir::FailureOr<mlir::Value> addOffset(EmitContext &c, const Pointer<EmitOp> &pointer,
                                         IntTupleRef<EmitOp> by,
                                         const Pointer<Trait> &result) const override {
    return offset(c, pointer, by, result);
  }
};

TEST_F(EmitOpCategory, PointerCallbackPreservesOffsetProfileAndAllowsDialectTypeChanges) {
  unsigned calls = 0;
  auto hooks = OffsetCallbackHooks();
  ctx.hooks = &hooks;
  hooks.offset = [&](EmitContext &c, const Pointer<EmitOp> &pointer, IntTupleRef<EmitOp> offset,
                     const Pointer<Trait> &resultProfile) -> mlir::FailureOr<mlir::Value> {
    ++calls;
    EXPECT_EQ(pointer.elementBitWidth(), 4);
    EXPECT_EQ(to_trait(offset), IntTuple<Trait>::fromLeaf(to_trait(arg(2))));
    EXPECT_EQ(resultProfile, add_offset(to_trait(pointer), to_trait(offset)));
    auto state = mlir::OperationState(c.location, "test.add_offset");
    state.addOperands({pointer.value(), offset.getLeaf().value()});
    state.addTypes(c.builder.getI64Type());
    return c.builder.create(state)->getResult(0);
  };
  auto property = Pointer<Trait>(OpaqueValue::getType<int>(), 4, 16,
                                 OpaqueValue::get(builder.getStringAttr("buffer-desc")));
  auto pointer = Pointer<EmitOp>(property, entry->getArgument(7), ctx);
  auto shifted = add_offset(pointer, arg(2));
  ASSERT_TRUE(shifted.isValid());
  EXPECT_EQ(calls, 1u);
  EXPECT_EQ(shifted.addressSpace(), property.addressSpace());
  EXPECT_TRUE(shifted.value().getType().isInteger(64));
  EXPECT_EQ(to_trait(shifted), add_offset(property, to_trait(arg(2))));
  auto handler = mlir::ScopedDiagnosticHandler(&mlirContext,
                                               [](mlir::Diagnostic &) { return mlir::success(); });
  hooks.offset = [&](EmitContext &, const Pointer<EmitOp> &, IntTupleRef<EmitOp>,
                     const Pointer<Trait> &) -> mlir::FailureOr<mlir::Value> {
    return mlir::failure();
  };
  EXPECT_FALSE(add_offset(pointer, arg(2)).isValid());
  ctx.hooks = nullptr;
}

TEST_F(EmitOpCategory, LoweredConversionNestsComponentsAndDropsStaticValues) {
  auto profile = *Layout<Trait>::fromString("(4,?i64):(1,?i16)");
  auto type = get_llvm_type(profile, mlirContext);
  ASSERT_TRUE(type);
  EXPECT_TRUE(type.isPacked());
  ASSERT_EQ(type.getBody().size(), 2u);
  auto shape = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type.getBody()[0]);
  auto stride = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type.getBody()[1]);
  ASSERT_TRUE(shape && stride);
  ASSERT_EQ(shape.getBody().size(), 1u);
  EXPECT_TRUE(shape.getBody()[0].isInteger(64));
  ASSERT_EQ(stride.getBody().size(), 1u);
  EXPECT_TRUE(stride.getBody()[0].isInteger(16));
  auto source = mlir::OperationState(loc, "test.call");
  source.addTypes(type);
  auto value = builder.create(source)->getResult(0);
  auto restored = bind_llvm_value(profile, value, ctx);
  ASSERT_TRUE(mlir::succeeded(restored));
  EXPECT_EQ(to_trait(*restored), profile);
  auto rebuilt = build_llvm_value(*restored, ctx);
  ASSERT_TRUE(mlir::succeeded(rebuilt));
  EXPECT_EQ((*rebuilt).getType(), type);
  auto count = entry->getOperations().size();
  EXPECT_TRUE(mlir::failed(bind_llvm_value(profile, arg(3).value(), ctx)));
  EXPECT_EQ(entry->getOperations().size(), count);
  auto empty = bind(*IntTuple<Trait>::fromString("(4,8)"), EmitBindings{}, ctx);
  ASSERT_TRUE(mlir::succeeded(empty));
  EXPECT_FALSE(get_llvm_type(to_trait(*empty), mlirContext));
  auto packedEmpty = build_llvm_value(*empty, ctx);
  ASSERT_TRUE(mlir::succeeded(packedEmpty));
  EXPECT_FALSE(*packedEmpty);
  EXPECT_TRUE(mlir::succeeded(bind_llvm_value(to_trait(*empty), mlir::Value(), ctx)));
}

TEST_F(EmitOpCategory, LoweredTileFlattensModesIntoScalarsAndLayoutStructs) {
  auto layout = *Layout<Trait>::fromString("(?,4):(1,?)");
  auto profile = Tile<Trait>::getTuple(
      {Tile<Trait>(Leaf<Trait>::getDInt(LogWidth::I32, 1)),
       Tile<Trait>::getTuple({Tile<Trait>(layout), Tile<Trait>(Leaf<Trait>::getSInt(2))})});
  auto type = get_llvm_type(profile, mlirContext);
  ASSERT_TRUE(type);
  ASSERT_EQ(type.getBody().size(), 2u);
  EXPECT_TRUE(type.getBody()[0].isInteger(32));
  EXPECT_EQ(type.getBody()[1], get_llvm_type(layout, mlirContext));
  auto bindings = EmitBindings{};
  for (int64_t scalar : {1, 2, 3})
    bindings.scalars.push_back(mlir::arith::ConstantIntOp::create(builder, loc, scalar, 32));
  auto bound = bind(profile, bindings, ctx);
  ASSERT_TRUE(mlir::succeeded(bound));
  auto packed = build_llvm_value(*bound, ctx);
  ASSERT_TRUE(mlir::succeeded(packed));
  EXPECT_EQ((*packed).getType(), type);
  auto restored = bind_llvm_value(profile, *packed, ctx);
  ASSERT_TRUE(mlir::succeeded(restored));
  EXPECT_EQ(to_trait(*restored), profile);
}

TEST_F(EmitOpCategory, LoweredFieldsSkipStaticComponents) {
  auto layout = *Layout<Trait>::fromString("(?,4):(1,?)");
  auto layoutType = get_llvm_type(layout, mlirContext);
  for (auto tensor : {CoordTensor<Trait>(IntTuple<Trait>::getDInt(LogWidth::I32, 1), layout),
                      make_coord_tensor(layout)}) {
    auto expected = llvm::SmallVector<mlir::Type>{};
    if (auto baseType = get_llvm_type(tensor.data(), mlirContext))
      expected.push_back(baseType);
    expected.push_back(layoutType);
    auto type = get_llvm_type(tensor, mlirContext);
    ASSERT_TRUE(type);
    EXPECT_EQ(type.getBody(), llvm::ArrayRef<mlir::Type>(expected));
  }
}

TEST_F(EmitOpCategory, ScalarTileAndExplicitZeroBasisBindingKeepIdentity) {
  auto zeroBasis = Leaf<Trait>::getBasis(Leaf<Trait>::getOne(), {1});
  zeroBasis.payload64_ = 0;
  auto zeroProfile = IntTuple<Trait>::fromLeaf(zeroBasis);
  auto zero = bind(zeroProfile, EmitBindings{}, ctx);
  ASSERT_TRUE(mlir::succeeded(zero));
  EXPECT_EQ(to_trait(*zero), zeroProfile);
  auto profile = Tile<Trait>(Leaf<Trait>::getDInt(LogWidth::I64, 8));
  auto bindings = EmitBindings{};
  bindings.scalars.push_back(arg(3).value());
  // Use a matching i64 SSA independent of fixture argument ordering.
  bindings.scalars[0] = mlir::arith::ConstantIntOp::create(builder, loc, 16, 64);
  auto bound = bind(profile, bindings, ctx);
  ASSERT_TRUE(mlir::succeeded(bound));
  EXPECT_EQ(to_trait(*bound), profile);
  auto packed = build_llvm_value(*bound, ctx);
  ASSERT_TRUE(mlir::succeeded(packed));
  auto restored = bind_llvm_value(profile, *packed, ctx);
  ASSERT_TRUE(mlir::succeeded(restored));
  EXPECT_EQ(to_trait(*restored), profile);
}

TEST_F(EmitOpCategory, NestedComposedInnerBindingPreservesTheTree) {
  auto layout = *Layout<Trait>::fromString("8:1");
  auto inner = ComposedLayout<Trait>(Swizzle<Trait>(1, 1, 1),
                                     IntTuple<Trait>::getDInt(LogWidth::I64, 2), layout);
  auto profile = ComposedLayout<Trait>(inner, IntTuple<Trait>::getSInt(3), layout);
  EXPECT_FALSE(profile.isStatic());
  EXPECT_TRUE(std::holds_alternative<ComposedLayout<Trait>>(profile.inner()));
  auto bindings = EmitBindings{};
  bindings.scalars.push_back(mlir::arith::ConstantIntOp::create(builder, loc, 2, 64));
  auto bound = bind(profile, bindings, ctx);
  ASSERT_TRUE(mlir::succeeded(bound));
  EXPECT_EQ(to_trait(*bound), profile);
  EXPECT_EQ(*ComposedLayout<Trait>::fromString(toString(profile)), profile);
  auto rightAssociated =
      ComposedLayout<Trait>(Swizzle<Trait>(1, 1, 1), IntTuple<Trait>::getDInt(LogWidth::I64, 2),
                            ComposedLayout<Trait>(layout, IntTuple<Trait>::getSInt(3), layout));
  auto coord = IntTuple<EmitOp>::getSInt(1);
  auto rightAssociatedBound = bind(rightAssociated, bindings, ctx);
  ASSERT_TRUE(mlir::succeeded(rightAssociatedBound));
  EXPECT_EQ(to_trait(crd2idx(coord, *bound)), to_trait(crd2idx(coord, *rightAssociatedBound)));
}

TEST_F(EmitOpCategory, TupleValueEqualityRetainsDynamicBindings) {
  auto lhs = make_tuple(IntTuple<EmitOp>::fromLeaf(arg(3)), IntTuple<EmitOp>::getSInt(7));
  auto rhs = make_tuple(IntTuple<EmitOp>::fromLeaf(arg(4)), IntTuple<EmitOp>::getSInt(7));
  auto result = eq(lhs, rhs);
  ASSERT_TRUE(result.isLeaf());
  EXPECT_EQ(to_trait(result), eq(to_trait(lhs), to_trait(rhs)));
  ASSERT_TRUE(result.getLeaf().isSound());
  EXPECT_TRUE(mlir::succeeded(result.getLeaf().materialize(ctx)));
  EXPECT_TRUE(eq(lhs, make_tuple(IntTuple<EmitOp>::fromLeaf(arg(4)), IntTuple<EmitOp>::getSInt(8)))
                  .isZero());
}

TEST_F(EmitOpCategory, ComparisonPreservesUnsupportedDynamicRatioError) {
  auto ratio = Leaf<EmitOp>(Leaf<Trait>::getRatio(1, 2));
  auto result = eq(ratio, arg(3));
  EXPECT_TRUE(result.isError());
  EXPECT_EQ(result.errorCode(), ErrorCode::DynamicRatioUnsupported);
}

/// The ops of `T` the entry block holds.
template <class T> size_t countOps(mlir::Block *block) {
  size_t count = 0;
  block->walk([&](T) { ++count; });
  return count;
}

// The bits a swizzle moves are masked first, so it shifts them right logically.
TEST_F(EmitOpCategory, SwizzleShiftsTheMaskedBitsLogically) {
  auto value = IntTuple<EmitOp>::fromLeaf(arg(2));
  auto swizzled = swizzle_apply(Swizzle<EmitOp>(3, 3, 3), value);
  ASSERT_TRUE(swizzled.isLeaf());
  check(swizzled.getLeaf(), swizzle_apply(Swizzle<Trait>(3, 3, 3), to_trait(value)).getLeaf());
  EXPECT_EQ(countOps<mlir::arith::ShRUIOp>(entry), 1u);
  EXPECT_EQ(countOps<mlir::arith::ShRSIOp>(entry), 0u);
}

TEST_F(EmitOpCategory, FastDivModDividesByItsHandleAndDegradesElsewhere) {
  auto n = Leaf<EmitOp>::getDyn(
      mlir::arith::TruncIOp::create(builder, loc, builder.getI32Type(), arg(3).value()), ctx);
  auto divisor = arg(2).withFastDivMod();
  ASSERT_TRUE(divisor.hasFastDivMod());
  EXPECT_TRUE(divisor.isSound());
  EXPECT_EQ(to_trait(divisor), to_trait(arg(2)).withFastDivMod());
  EXPECT_EQ(divisor.value().getType(), emit_op::getFastDivModType(mlirContext, 32));
  EXPECT_EQ(divisor.withFastDivMod(), divisor);
  // The magic numbers are computed once, by withFastDivMod.
  auto magic = countOps<mlir::LLVM::CountLeadingZerosOp>(entry);
  EXPECT_EQ(magic, 1u);

  // Every division by the divisor uses its handle; none rounds by divsi.
  for (auto [result, expected] :
       {std::pair{n / divisor, to_trait(n) / to_trait(divisor)},
        std::pair{n % divisor, to_trait(n) % to_trait(divisor)},
        std::pair{ceil_div(n, divisor), ceil_div(to_trait(n), to_trait(divisor))},
        std::pair{shape_div(n, divisor), shape_div(to_trait(n), to_trait(divisor))},
        std::pair{safe_div(n, divisor), safe_div(to_trait(n), to_trait(divisor))},
        std::pair{Leaf<EmitOp>::getSInt(100) / divisor,
                  Leaf<Trait>::getSInt(100) / to_trait(divisor)}})
    check(result, expected);
  EXPECT_EQ(countOps<mlir::arith::MulUIExtendedOp>(entry), 6u);
  EXPECT_EQ(countOps<mlir::arith::DivSIOp>(entry) + countOps<mlir::arith::RemSIOp>(entry) +
                countOps<mlir::arith::CeilDivSIOp>(entry),
            0u);
  EXPECT_TRUE((n / divisor).value().getDefiningOp<mlir::arith::ShRUIOp>());
  EXPECT_TRUE((n % divisor).value().getDefiningOp<mlir::arith::SubIOp>());

  // An i32 dividend divides by an i64 divisor's handle; an i64 one cannot use
  // an i32 handle and falls back to divsi.
  auto wide = arg(4).withFastDivMod();
  check(n / wide, to_trait(n) / to_trait(wide));
  EXPECT_TRUE((n / wide).value().getDefiningOp<mlir::arith::ShRUIOp>());
  check(arg(3) / divisor, to_trait(arg(3)) / to_trait(divisor));
  EXPECT_TRUE((arg(3) / divisor).value().getDefiningOp<mlir::arith::DivSIOp>());

  // Anywhere else a divisor is its divisor, and the result is a plain integer.
  auto tripled = divisor * Leaf<EmitOp>::getSInt(3);
  check(tripled, to_trait(divisor) * Leaf<Trait>::getSInt(3));
  EXPECT_FALSE(tripled.hasFastDivMod());
  auto product = tripled.value().getDefiningOp<mlir::arith::MulIOp>();
  ASSERT_TRUE(product);
  EXPECT_TRUE(product.getLhs().getDefiningOp<mlir::LLVM::ExtractValueOp>());
  check(divisor / Leaf<EmitOp>::getSInt(4), to_trait(divisor) / Leaf<Trait>::getSInt(4));
  check(n + divisor, to_trait(n) + to_trait(divisor));
  check(lt(n, divisor), lt(to_trait(n), to_trait(divisor)));

  // An identity keeps the handle; a divisor is positive.
  for (auto same :
       {divisor * Leaf<EmitOp>::getOne(), divisor / Leaf<EmitOp>::getOne(),
        divisor | Leaf<EmitOp>::getZero(), Leaf<EmitOp>::getOne() * divisor, abs(divisor)}) {
    EXPECT_TRUE(same.hasFastDivMod());
    EXPECT_EQ(same, divisor);
  }
  check(signum(divisor), Leaf<Trait>::getOne());

  // A basis scale is the divisor itself.
  auto basis = Leaf<EmitOp>::getBasis(divisor, {1});
  ASSERT_TRUE(basis.isBasis());
  EXPECT_TRUE(basis.isSound());
  EXPECT_FALSE(basis.scale().hasFastDivMod());
  EXPECT_TRUE(basis.value().getType().isInteger(32));
  auto scaled = Leaf<EmitOp>::getBasis(Leaf<EmitOp>::getOne(), {0}) * divisor;
  EXPECT_TRUE(scaled.isSound());
  EXPECT_FALSE(scaled.scale().hasFastDivMod());
  EXPECT_EQ(countOps<mlir::LLVM::CountLeadingZerosOp>(entry), magic + 2);
}

TEST_F(EmitOpCategory, FastDivModHandleIsOneScalarOfTheLoweredStruct) {
  auto profile = *IntTuple<Trait>::fromString("(?{fdm},4,(?,?i64{div=2 fdm}))");
  auto type = get_llvm_type(profile, mlirContext);
  ASSERT_TRUE(type);
  auto handle32 = emit_op::getFastDivModType(mlirContext, 32);
  auto handle64 = emit_op::getFastDivModType(mlirContext, 64);
  EXPECT_FALSE(handle32.isPacked());
  EXPECT_EQ(type.getBody(), llvm::ArrayRef<mlir::Type>({handle32, builder.getI32Type(), handle64}));
  auto source = mlir::OperationState(loc, "test.call");
  source.addTypes(type);
  auto aggregate = builder.create(source)->getResult(0);
  auto scalars = collect_llvm_scalars(profile, aggregate, ctx);
  ASSERT_TRUE(mlir::succeeded(scalars));
  ASSERT_EQ(scalars->size(), 3u);
  EXPECT_EQ((*scalars)[0].getType(), handle32);
  EXPECT_EQ((*scalars)[2].getType(), handle64);
  // Another profile's value is rejected before any extraction.
  auto ops = entry->getOperations().size();
  EXPECT_TRUE(
      mlir::failed(collect_llvm_scalars(*IntTuple<Trait>::fromString("(?,?)"), aggregate, ctx)));
  EXPECT_EQ(entry->getOperations().size(), ops);
  auto bound = bind_llvm_value(profile, aggregate, ctx);
  ASSERT_TRUE(mlir::succeeded(bound));
  EXPECT_EQ(to_trait(*bound), profile);
  auto rebuilt = build_llvm_value(*bound, ctx);
  ASSERT_TRUE(mlir::succeeded(rebuilt));
  EXPECT_EQ((*rebuilt).getType(), type);
  // A handle binds a fast-divmod leaf only; an integer does not.
  EXPECT_TRUE(Leaf<EmitOp>(to_trait(arg(2)).withFastDivMod(), arg(2).value(), &ctx).isError());
  EXPECT_TRUE(Leaf<EmitOp>(to_trait(arg(2)), (*scalars)[0], &ctx).isError());
}

TEST_F(EmitOpCategory, EmitHooksBindAndEmitFastDivModHandles) {
  /// Hooks whose handle is an opaque type operated on by test ops.
  struct TestHooks : EmitHooks {
    mlir::Type getFastDivModType(EmitContext &c, unsigned bitWidth) const override {
      return mlir::OpaqueType::get(c.builder.getStringAttr("test"),
                                   "handle" + std::to_string(bitWidth));
    }
    mlir::Value makeFastDivMod(EmitContext &c, mlir::Value divisor) const override {
      return create(c, "test.make", divisor, {getFastDivModType(c, 32)});
    }
    mlir::Value getFastDivModDivisor(EmitContext &c, mlir::Value handle) const override {
      return create(c, "test.divisor", handle, {c.builder.getI32Type()});
    }
    std::pair<mlir::Value, mlir::Value> fastDivMod(EmitContext &c, mlir::Value dividend,
                                                   mlir::Value handle) const override {
      auto state = mlir::OperationState(c.location, "test.divmod");
      state.addOperands({dividend, handle});
      state.addTypes({dividend.getType(), dividend.getType()});
      auto *op = c.builder.create(state);
      return {op->getResult(0), op->getResult(1)};
    }
    static mlir::Value create(EmitContext &c, llvm::StringRef name, mlir::Value operand,
                              llvm::ArrayRef<mlir::Type> types) {
      auto state = mlir::OperationState(c.location, name);
      state.addOperands(operand);
      state.addTypes(types);
      return c.builder.create(state)->getResult(0);
    }
  };
  auto hooks = TestHooks();
  ctx.hooks = &hooks;
  auto divisor = arg(2).withFastDivMod();
  ASSERT_TRUE(divisor.hasFastDivMod());
  EXPECT_EQ(divisor.value().getDefiningOp()->getName().getStringRef(), "test.make");
  auto quotient = arg(2) / divisor;
  EXPECT_EQ(quotient.value().getDefiningOp()->getName().getStringRef(), "test.divmod");
  auto sum = arg(2) + divisor;
  auto add = sum.value().getDefiningOp<mlir::arith::AddIOp>();
  ASSERT_TRUE(add);
  EXPECT_EQ(add.getRhs().getDefiningOp()->getName().getStringRef(), "test.divisor");
  // Binding checks the handle type of the hooks.
  auto profile = *IntTuple<Trait>::fromString("?{fdm}");
  EXPECT_EQ(get_binding_types(profile, ctx),
            (llvm::SmallVector<mlir::Type, 8>{hooks.getFastDivModType(ctx, 32)}));
  EXPECT_TRUE(mlir::succeeded(bind(profile, EmitBindings{{divisor.value()}, {}}, ctx)));
  EXPECT_TRUE(mlir::failed(bind(profile, EmitBindings{{arg(2).value()}, {}}, ctx)));
  ctx.hooks = nullptr;
}

TEST_F(EmitOpCategory, FastDivModMaterializesAStaticDivisor) {
  auto seven = Leaf<EmitOp>::getSInt(7);
  EXPECT_EQ(seven.withFastDivMod().errorCode(), ErrorCode::MissingEmitContext);
  EXPECT_EQ(Leaf<EmitOp>::getZero().withFastDivMod(&ctx).errorCode(),
            ErrorCode::FastDivModRequiresPositiveInt);
  auto divisor = seven.withFastDivMod(&ctx);
  ASSERT_TRUE(divisor.hasFastDivMod());
  EXPECT_TRUE(divisor.isSound());
  EXPECT_EQ(divisor.context(), &ctx);
  EXPECT_EQ(to_trait(divisor), Leaf<Trait>::getSInt(7).withFastDivMod());
  auto handle = divisor.value().getDefiningOp<mlir::LLVM::InsertValueOp>();
  ASSERT_TRUE(handle);
  // The handle is built of the materialized constant.
  auto fields = handle.getContainer()
                    .getDefiningOp<mlir::LLVM::InsertValueOp>()
                    .getContainer()
                    .getDefiningOp<mlir::LLVM::InsertValueOp>();
  ASSERT_TRUE(fields);
  EXPECT_EQ(mlir::getConstantIntValue(fields.getValue()), 7);
  check(arg(2) / divisor, to_trait(arg(2)) / to_trait(divisor));
}

} // namespace

namespace {

/// Divisors that stress the magic numbers of a `bits`-bit divisor: one, powers
/// of two and their neighbours, the largest divisor, and a fixed sample.
std::vector<int64_t> fastDivModDivisors(unsigned bits) {
  auto top = int64_t((uint64_t{1} << (bits - 1)) - 1);
  auto divisors =
      std::vector<int64_t>{1, 2, 3, 5, 7, 10, 255, 1000, 65535, 65536, 65537, top - 1, top};
  for (unsigned k = 2; k + 1 < bits; k += 3) {
    auto power = int64_t{1} << k;
    divisors.insert(divisors.end(), {power - 1, power, power + 1});
  }
  auto state = uint64_t{0x9e3779b97f4a7c15u};
  for (int i = 0; i < 40; ++i) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    divisors.push_back(int64_t(state % uint64_t(top)) + 1);
    divisors.push_back(int64_t(state % 4096) + 1);
  }
  return divisors;
}

/// Dividends in [0, 2^(bits-1)) around the multiples of `d` and the bounds.
std::vector<int64_t> fastDivModDividends(int64_t d, unsigned bits) {
  auto top = int64_t((uint64_t{1} << (bits - 1)) - 1);
  auto dividends = std::vector<int64_t>{
      0, 1, d - 1, d, top, top - 1, top - top % d, top - top % d - 1, int64_t(uint64_t(top) / 3)};
  if (d < top)
    dividends.push_back(d + 1);
  if (d <= top / 2)
    dividends.insert(dividends.end(), {2 * d - 1, 2 * d});
  std::vector<int64_t> valid;
  for (auto n : dividends)
    if (n >= 0 && n <= top)
      valid.push_back(n);
  return valid;
}

} // namespace

/// Hooks that make an i32 handle in FP64, as a GPU backend does.
struct FloatMultiplierHooks : EmitHooks {
  mlir::Value makeFastDivMod(EmitContext &ctx, mlir::Value divisor) const override {
    if (divisor.getType().isInteger(32))
      return emit_op::makeFastDivModF64(ctx, divisor);
    return EmitHooks::makeFastDivMod(ctx, divisor);
  }
};

// The lowered handle executed on the host, with the default (integer division)
// and the FP64 multiplier: every division by a fast-divmod divisor gives the
// integer quotient and remainder.
TEST(LeafEmitOpFastDivMod, MatchesIntegerDivision) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto registry = mlir::DialectRegistry();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  auto context = mlir::MLIRContext(registry);
  context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect,
                      mlir::LLVM::LLVMDialect>();
  auto builder = mlir::OpBuilder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(loc));
  auto ctx = EmitContext(builder, loc);

  struct Case {
    const char *name;
    Leaf<EmitOp> (*emit)(Leaf<EmitOp>, Leaf<EmitOp>);
    int64_t (*reference)(int64_t, int64_t);
  };
  const Case cases[] = {
      {"div", [](Leaf<EmitOp> a, Leaf<EmitOp> b) { return a / b; },
       [](int64_t n, int64_t d) { return n / d; }},
      {"mod", [](Leaf<EmitOp> a, Leaf<EmitOp> b) { return a % b; },
       [](int64_t n, int64_t d) { return n % d; }},
      {"ceil", [](Leaf<EmitOp> a, Leaf<EmitOp> b) { return ceil_div(a, b); },
       [](int64_t n, int64_t d) { return n / d + (n % d != 0); }},
  };
  struct Variant {
    std::string name;
    unsigned bits;
    const EmitHooks *hooks;
  };
  auto floatMultiplier = FloatMultiplierHooks();
  const Variant variants[] = {
      {"int32", 32, nullptr}, {"int64", 64, nullptr}, {"f64_32", 32, &floatMultiplier}};
  auto i64 = builder.getI64Type();
  for (const auto &variant : variants) {
    ctx.hooks = variant.hooks;
    auto bits = variant.bits;
    for (const auto &c : cases) {
      builder.setInsertionPointToEnd(module->getBody());
      auto name = std::string(c.name) + "_" + variant.name;
      auto function = mlir::func::FuncOp::create(builder, loc, name,
                                                 builder.getFunctionType({i64, i64}, {i64}));
      function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
      auto &body = *function.addEntryBlock();
      builder.setInsertionPointToEnd(&body);
      auto narrow = [&](mlir::Value value) {
        return mlir::convertScalarToDtype(builder, loc, value, builder.getIntegerType(bits),
                                          /*isUnsignedCast=*/false);
      };
      auto n = Leaf<EmitOp>::getDyn(narrow(body.getArgument(0)), ctx);
      auto d = Leaf<EmitOp>::getDyn(narrow(body.getArgument(1)), ctx).withFastDivMod();
      ASSERT_TRUE(d.hasFastDivMod()) << name;
      auto value = c.emit(n, d).materialize(ctx);
      ASSERT_TRUE(mlir::succeeded(value)) << name;
      EXPECT_EQ(countOps<mlir::arith::DivSIOp>(&body) + countOps<mlir::arith::RemSIOp>(&body) +
                    countOps<mlir::arith::CeilDivSIOp>(&body),
                0u)
          << name;
      mlir::func::ReturnOp::create(builder, loc,
                                   mlir::convertScalarToDtype(builder, loc, *value, i64, false));
    }
  }
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  auto pm = mlir::PassManager(&context);
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));
  auto engine = mlir::ExecutionEngine::create(*module);
  ASSERT_TRUE(bool(engine)) << llvm::toString(engine.takeError());

  size_t checked = 0;
  for (const auto &variant : variants)
    for (auto d : fastDivModDivisors(variant.bits))
      for (auto n : fastDivModDividends(d, variant.bits))
        for (const auto &c : cases) {
          auto name = std::string(c.name) + "_" + variant.name;
          auto result = int64_t{0};
          auto error =
              (*engine)->invoke(name, n, d, mlir::ExecutionEngine::Result<int64_t>(result));
          ASSERT_FALSE(bool(error)) << name << ": " << llvm::toString(std::move(error));
          EXPECT_EQ(result, c.reference(n, d)) << name << "(" << n << ", " << d << ")";
          ++checked;
        }
  llvm::outs() << checked << " fast divisions checked\n";
}
