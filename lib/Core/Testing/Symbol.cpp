// SPDX-FileCopyrightText: Copyright (c) 2026 FlyDSL Project Contributors
// SPDX-License-Identifier: Apache-2.0

#include "flydsl/Core/Symbol.hpp"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <atomic>
#include <string>

namespace mlir::fly::core {
namespace symbol {

//===----------------------------------------------------------------------===//
// Symbol expressions
//===----------------------------------------------------------------------===//

/// Operators match `Leaf<Trait>` arithmetic. `Const`, `Sym` and `Fresh` are
/// the terminals: a static leaf stores no expression, but a dynamic result may
/// embed a static operand as `Const`.
enum class SymbolOp : uint8_t {
  Const,
  Sym,
  Fresh,
  Add,
  Sub,
  Mul,
  Neg,
  Div,
  Ceil,
  Mod,
  Exact,
  Safe,
  Shape,
  BitAnd,
  BitOr,
  BitXor,
  Not,
  Shl,
  Shr,
  LShr,
  Lt,
  Eq,
  Ne,
  Min,
  Max,
  Gcd,
  Abs,
  Signum,
};

/// An immutable expression node. Operands are shared by reference, so an
/// expression is a DAG whose subexpressions are stored once however many
/// expressions use them. Equality is structural; `hash` caches the structural
/// hash so that most unequal expressions are told apart without a walk.
struct SymbolNode : llvm::ThreadSafeRefCountedBase<SymbolNode> {
  SymbolNode(SymbolOp op, int64_t payload, std::string name, SymbolExprRef lhs, SymbolExprRef rhs)
      : op(op), payload(payload), name(std::move(name)), lhs(std::move(lhs)), rhs(std::move(rhs)) {
    hash = llvm::hash_combine(static_cast<uint8_t>(op), payload, this->name, hashOf(this->lhs),
                              hashOf(this->rhs));
  }

  static llvm::hash_code hashOf(const SymbolExprRef &expr) {
    return expr ? expr->hash : llvm::hash_code(0);
  }

  SymbolOp op;
  /// The value of a `Const` or the id of a `Fresh` symbol.
  int64_t payload;
  /// The name of a `Sym`.
  std::string name;
  SymbolExprRef lhs, rhs;
  llvm::hash_code hash;
};

struct SymbolAccess {
  static Leaf<Symbol> leaf(LeafProperty property, SymbolExprRef expr) {
    return Leaf<Symbol>(property, std::move(expr));
  }
  static const SymbolExprRef &expr(const Leaf<Symbol> &leaf) { return leaf.expr_; }
  static Pointer<Symbol> pointer(PointerProperty property, SymbolExprRef address) {
    return Pointer<Symbol>(std::move(property), std::move(address));
  }
  static const SymbolExprRef &address(const Pointer<Symbol> &pointer) { return pointer.address_; }
};

} // namespace symbol
} // namespace mlir::fly::core

namespace llvm {

void IntrusiveRefCntPtrInfo<const mlir::fly::core::symbol::SymbolNode>::retain(
    const mlir::fly::core::symbol::SymbolNode *node) {
  node->Retain();
}

void IntrusiveRefCntPtrInfo<const mlir::fly::core::symbol::SymbolNode>::release(
    const mlir::fly::core::symbol::SymbolNode *node) {
  node->Release();
}

} // namespace llvm

namespace mlir::fly::core {
namespace symbol {
namespace {

SymbolExprRef makeNode(SymbolOp op, int64_t payload, std::string name = {},
                       SymbolExprRef lhs = nullptr, SymbolExprRef rhs = nullptr) {
  return SymbolExprRef(
      new SymbolNode(op, payload, std::move(name), std::move(lhs), std::move(rhs)));
}

SymbolExprRef getConst(int64_t value) { return makeNode(SymbolOp::Const, value); }

SymbolExprRef getSym(llvm::StringRef name) { return makeNode(SymbolOp::Sym, 0, name.str()); }

SymbolExprRef getFresh() {
  static std::atomic<int64_t> next{0};
  return makeNode(SymbolOp::Fresh, next.fetch_add(1, std::memory_order_relaxed));
}

bool isConst(const SymbolExprRef &node, int64_t value) {
  return node && node->op == SymbolOp::Const && node->payload == value;
}

/// `op` applied to `a` (and `b`), folding the identities that return an
/// operand unchanged. The result shares the nodes of its operands.
SymbolExprRef apply(SymbolOp op, SymbolExprRef a, SymbolExprRef b = nullptr) {
  if (op == SymbolOp::Neg || op == SymbolOp::Not || op == SymbolOp::Abs || op == SymbolOp::Signum) {
    if (!a)
      return nullptr;
    if (op == SymbolOp::Neg && isConst(a, 0))
      return a;
    if (op == SymbolOp::Abs) {
      if (a->op == SymbolOp::Abs)
        return a;
      if (a->op == SymbolOp::Neg)
        return makeNode(SymbolOp::Abs, 0, {}, a->lhs);
    }
    if (op == SymbolOp::Signum && a->op == SymbolOp::Signum)
      return a;
    return makeNode(op, 0, {}, std::move(a));
  }

  if (!a || !b)
    return nullptr;
  switch (op) {
  case SymbolOp::Add:
    if (isConst(a, 0))
      return b;
    if (isConst(b, 0))
      return a;
    break;
  case SymbolOp::Sub:
    if (isConst(b, 0))
      return a;
    break;
  case SymbolOp::Mul:
    if (isConst(a, 1))
      return b;
    if (isConst(b, 1))
      return a;
    break;
  case SymbolOp::Div:
  case SymbolOp::Ceil:
  case SymbolOp::Exact:
  case SymbolOp::Safe:
  case SymbolOp::Shape:
    if (isConst(b, 1))
      return a;
    break;
  case SymbolOp::BitOr:
  case SymbolOp::BitXor:
    if (isConst(a, 0))
      return b;
    if (isConst(b, 0))
      return a;
    break;
  case SymbolOp::Shl:
  case SymbolOp::Shr:
  case SymbolOp::LShr:
    if (isConst(b, 0))
      return a;
    break;
  default:
    break;
  }
  return makeNode(op, 0, {}, std::move(a), std::move(b));
}

bool equal(const SymbolNode *a, const SymbolNode *b) {
  if (a == b)
    return true;
  if (!a || !b || a->hash != b->hash || a->op != b->op || a->payload != b->payload ||
      a->name != b->name)
    return false;
  return equal(a->lhs.get(), b->lhs.get()) && equal(a->rhs.get(), b->rhs.get());
}

int precedence(SymbolOp op) {
  switch (op) {
  case SymbolOp::Neg:
  case SymbolOp::Not:
    return 4;
  case SymbolOp::Mul:
  case SymbolOp::Div:
  case SymbolOp::Mod:
  case SymbolOp::Shl:
  case SymbolOp::Shr:
    return 3;
  case SymbolOp::Add:
  case SymbolOp::Sub:
  case SymbolOp::BitAnd:
  case SymbolOp::BitOr:
  case SymbolOp::BitXor:
    return 2;
  case SymbolOp::Lt:
  case SymbolOp::Eq:
  case SymbolOp::Ne:
    return 1;
  default:
    return 0;
  }
}

void printExpr(const SymbolNode *node, llvm::raw_ostream &os, int parentPrec);

void printCall(llvm::StringRef name, const SymbolNode *node, llvm::raw_ostream &os) {
  os << name << '(';
  printExpr(node->lhs.get(), os, 0);
  os << ',';
  printExpr(node->rhs.get(), os, 0);
  os << ')';
}

void printExpr(const SymbolNode *node, llvm::raw_ostream &os, int parentPrec) {
  if (!node) {
    os << "?";
    return;
  }
  auto op = node->op;
  auto prec = precedence(op);
  auto paren = prec != 0 && prec < parentPrec;
  if (paren)
    os << '(';
  auto lhs = node->lhs.get(), rhs = node->rhs.get();
  switch (op) {
  case SymbolOp::Const:
    os << node->payload;
    break;
  case SymbolOp::Sym:
    os << node->name;
    break;
  case SymbolOp::Fresh:
    os << 's' << node->payload;
    break;
  case SymbolOp::Neg:
    os << '-';
    printExpr(lhs, os, prec);
    break;
  case SymbolOp::Not:
    os << '~';
    printExpr(lhs, os, prec);
    break;
  case SymbolOp::Abs:
    os << "abs(";
    printExpr(lhs, os, 0);
    os << ')';
    break;
  case SymbolOp::Signum:
    os << "signum(";
    printExpr(lhs, os, 0);
    os << ')';
    break;
  case SymbolOp::Add:
    printExpr(lhs, os, prec);
    os << '+';
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Sub:
    printExpr(lhs, os, prec);
    os << '-';
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Mul:
    printExpr(lhs, os, prec);
    os << '*';
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Div:
    printExpr(lhs, os, prec);
    os << '/';
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Mod:
    printExpr(lhs, os, prec);
    os << '%';
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::BitAnd:
    printExpr(lhs, os, prec);
    os << '&';
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::BitOr:
    printExpr(lhs, os, prec);
    os << '|';
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::BitXor:
    printExpr(lhs, os, prec);
    os << '^';
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Shl:
    printExpr(lhs, os, prec);
    os << "<<";
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Shr:
    printExpr(lhs, os, prec);
    os << ">>";
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Lt:
    printExpr(lhs, os, prec + 1);
    os << '<';
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Eq:
    printExpr(lhs, os, prec + 1);
    os << "==";
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Ne:
    printExpr(lhs, os, prec + 1);
    os << "!=";
    printExpr(rhs, os, prec + 1);
    break;
  case SymbolOp::Ceil:
    printCall("ceil_div", node, os);
    break;
  case SymbolOp::Exact:
    printCall("exact_div", node, os);
    break;
  case SymbolOp::Safe:
    printCall("safe_div", node, os);
    break;
  case SymbolOp::Shape:
    printCall("shape_div", node, os);
    break;
  case SymbolOp::LShr:
    printCall("lshr", node, os);
    break;
  case SymbolOp::Min:
    printCall("min", node, os);
    break;
  case SymbolOp::Max:
    printCall("max", node, os);
    break;
  case SymbolOp::Gcd:
    printCall("gcd", node, os);
    break;
  }
  if (paren)
    os << ')';
}

/// Parses the expression part of a printed leaf; null on malformed text.
class SymbolExprParser {
public:
  explicit SymbolExprParser(llvm::StringRef text) : text_(text) {}

  SymbolExprRef parse() {
    auto result = parseComparison();
    return result && text_.empty() ? result : nullptr;
  }

private:
  char peek() const { return text_.empty() ? '\0' : text_.front(); }

  bool consume(char value) {
    if (text_.empty() || text_.front() != value)
      return false;
    text_ = text_.drop_front();
    return true;
  }

  bool consume(llvm::StringRef value) { return text_.consume_front(value); }

  SymbolExprRef parseComparison() {
    auto lhs = parseAdd();
    if (!lhs)
      return nullptr;
    SymbolOp op;
    if (consume("=="))
      op = SymbolOp::Eq;
    else if (consume("!="))
      op = SymbolOp::Ne;
    else if (consume('<'))
      op = SymbolOp::Lt;
    else
      return lhs;
    auto rhs = parseAdd();
    return rhs ? apply(op, lhs, rhs) : nullptr;
  }

  SymbolExprRef parseAdd() {
    auto lhs = parseMul();
    while (lhs) {
      SymbolOp op;
      if (consume('+'))
        op = SymbolOp::Add;
      else if (consume('-'))
        op = SymbolOp::Sub;
      else if (consume('&'))
        op = SymbolOp::BitAnd;
      else if (consume('|'))
        op = SymbolOp::BitOr;
      else if (consume('^'))
        op = SymbolOp::BitXor;
      else
        break;
      auto rhs = parseMul();
      if (!rhs)
        return nullptr;
      lhs = apply(op, lhs, rhs);
    }
    return lhs;
  }

  SymbolExprRef parseMul() {
    auto lhs = parseUnary();
    while (lhs) {
      SymbolOp op;
      if (consume("<<"))
        op = SymbolOp::Shl;
      else if (consume(">>"))
        op = SymbolOp::Shr;
      else if (consume('*'))
        op = SymbolOp::Mul;
      else if (consume('/'))
        op = SymbolOp::Div;
      else if (consume('%'))
        op = SymbolOp::Mod;
      else
        break;
      auto rhs = parseUnary();
      if (!rhs)
        return nullptr;
      lhs = apply(op, lhs, rhs);
    }
    return lhs;
  }

  SymbolExprRef parseUnary() {
    if (text_.size() > 1 && text_[0] == '-' && llvm::isDigit(text_[1]))
      return parsePrimary();
    if (consume('-')) {
      auto value = parseUnary();
      return value ? apply(SymbolOp::Neg, value) : nullptr;
    }
    if (consume('~')) {
      auto value = parseUnary();
      return value ? apply(SymbolOp::Not, value) : nullptr;
    }
    return parsePrimary();
  }

  SymbolExprRef parsePrimary() {
    if (consume('(')) {
      auto value = parseComparison();
      return value && consume(')') ? value : nullptr;
    }
    if (peek() == '-' || (peek() >= '0' && peek() <= '9')) {
      auto value = detail::parseInt64(text_);
      return value ? getConst(*value) : nullptr;
    }

    auto name = text_.take_while([](char value) { return llvm::isAlnum(value) || value == '_'; });
    if (name.empty())
      return nullptr;
    text_ = text_.drop_front(name.size());
    if (!consume('('))
      return getSym(name);

    auto lhs = parseComparison();
    if (!lhs)
      return nullptr;
    if (consume(')')) {
      auto op = llvm::StringSwitch<std::optional<SymbolOp>>(name)
                    .Case("abs", SymbolOp::Abs)
                    .Case("signum", SymbolOp::Signum)
                    .Default(std::nullopt);
      return op ? apply(*op, lhs) : nullptr;
    }
    if (!consume(','))
      return nullptr;
    auto rhs = parseComparison();
    if (!rhs || !consume(')'))
      return nullptr;
    auto op = llvm::StringSwitch<std::optional<SymbolOp>>(name)
                  .Case("ceil_div", SymbolOp::Ceil)
                  .Case("exact_div", SymbolOp::Exact)
                  .Case("safe_div", SymbolOp::Safe)
                  .Case("shape_div", SymbolOp::Shape)
                  .Case("lshr", SymbolOp::LShr)
                  .Case("min", SymbolOp::Min)
                  .Case("max", SymbolOp::Max)
                  .Case("gcd", SymbolOp::Gcd)
                  .Default(std::nullopt);
    return op ? apply(*op, lhs, rhs) : nullptr;
  }

  llvm::StringRef text_;
};

/// The expression of `leaf` as an operand: its own, or a constant for a static
/// integer.
SymbolExprRef embed(const Leaf<Symbol> &leaf) {
  if (const auto &expr = SymbolAccess::expr(leaf))
    return expr;
  if (leaf.isSInt())
    return getConst(leaf.staticValue());
  return nullptr;
}

Leaf<Symbol> binary(Leaf<Trait> type, SymbolOp op, const Leaf<Symbol> &a, const Leaf<Symbol> &b) {
  if (type.isStatic() || type.isError())
    return SymbolAccess::leaf(type, nullptr);
  auto lhs = a.scale(), rhs = b.scale();
  if (op == SymbolOp::Mul && (lhs.isRatio() || rhs.isRatio())) {
    auto ratio = lhs.isRatio() ? lhs : rhs;
    auto value = lhs.isRatio() ? rhs : lhs;
    auto quotient = apply(SymbolOp::Safe, embed(value), getConst(ratio.den()));
    return SymbolAccess::leaf(type, apply(SymbolOp::Mul, quotient, getConst(ratio.num())));
  }
  return SymbolAccess::leaf(type, apply(op, embed(a), embed(b)));
}

Leaf<Symbol> unary(Leaf<Trait> type, SymbolOp op, const Leaf<Symbol> &a) {
  if (type.isStatic())
    return SymbolAccess::leaf(type, nullptr);
  return SymbolAccess::leaf(type, apply(op, embed(a)));
}

} // namespace
} // namespace symbol

//===----------------------------------------------------------------------===//
// Leaf<Symbol>
//===----------------------------------------------------------------------===//

Leaf<Symbol> Leaf<Symbol>::getDInt(LogWidth logWidth, int32_t div) {
  return Leaf(Leaf<Trait>::getDInt(logWidth, div), symbol::getFresh());
}

Leaf<Symbol> Leaf<Symbol>::getSym(llvm::StringRef name, LogWidth logWidth, int32_t div) {
  return Leaf(Leaf<Trait>::getDInt(logWidth, div), symbol::getSym(name));
}

Leaf<Symbol> Leaf<Symbol>::getDyn(llvm::StringRef name, LogWidth logWidth, int32_t div,
                                  std::optional<std::pair<int32_t, int32_t>> range,
                                  bool fastDivMod) {
  auto t = Leaf<Trait>::getDInt(logWidth, div);
  if (range)
    t = t.withRange(range->first, range->second);
  if (fastDivMod && !t.isError())
    t = t.withFastDivMod();
  return Leaf(t, t.isError() ? nullptr : symbol::getSym(name));
}

Leaf<Symbol> Leaf<Symbol>::withRange(int32_t lo, int32_t hi) const {
  if constexpr (!FLYDSL_CORE_ENABLE_RANGE_INFERENCE)
    return *this;
  return Leaf(Leaf<Trait>(*this).withRange(lo, hi), expr_);
}

Leaf<Symbol> Leaf<Symbol>::withFastDivMod() const {
  auto type = Leaf<Trait>(*this).withFastDivMod();
  // A static divisor becomes dynamic: its expression is the constant.
  if (isSInt() && !type.isError())
    return Leaf(type, symbol::getConst(staticValue()));
  return Leaf(type, expr_);
}

bool Leaf<Symbol>::isSound() const { return isStatic() ? !expr_ : bool(expr_); }

std::optional<Leaf<Symbol>> Leaf<Symbol>::fromString(llvm::StringRef text) {
  if (auto type = Leaf<Trait>::fromString(text))
    return Leaf(*type, nullptr);
  auto suffix = detail::parseLeafSuffix(text);
  if (!suffix || suffix->value.empty())
    return std::nullopt;

  auto expr = symbol::SymbolExprParser(suffix->value).parse();
  // The suffixes after an expression type it as they type a `?`.
  auto type = Leaf<Trait>::fromString("?" + text.drop_front(suffix->value.size()).str());
  if (!expr || !type)
    return std::nullopt;
  return Leaf(*type, std::move(expr));
}

bool operator==(const Leaf<Symbol> &a, const Leaf<Symbol> &b) {
  return static_cast<const LeafProperty &>(a) == static_cast<const LeafProperty &>(b) &&
         symbol::equal(a.expr_.get(), b.expr_.get());
}

//===----------------------------------------------------------------------===//
// Leaf operations
//===----------------------------------------------------------------------===//

#define FLYDSL_CORE_LEAF_BINARY(NAME, OP)                                                          \
  Leaf<Symbol> NAME(Leaf<Symbol> a, Leaf<Symbol> b) {                                              \
    return symbol::binary(NAME(Leaf<Trait>(a), Leaf<Trait>(b)), symbol::SymbolOp::OP, a, b);       \
  }
#define FLYDSL_CORE_LEAF_UNARY(NAME, OP)                                                           \
  Leaf<Symbol> NAME(Leaf<Symbol> a) {                                                              \
    return symbol::unary(NAME(Leaf<Trait>(a)), symbol::SymbolOp::OP, a);                           \
  }
#include "flydsl/Core/Category.def"

void print(Leaf<Symbol> l, llvm::raw_ostream &os) {
  const auto &expr = symbol::SymbolAccess::expr(l);
  if (l.isStatic() || !expr) {
    print(Leaf<Trait>(l), os);
    return;
  }

  symbol::printExpr(expr.get(), os, 0);
  detail::printDynIntSuffix(Leaf<Trait>(l).scale(), os);
  detail::printModes(l, os);
}

//===----------------------------------------------------------------------===//
// Pointer<Symbol>
//===----------------------------------------------------------------------===//

Pointer<Symbol>::Pointer(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment,
                         llvm::StringRef base, Swizzle<Symbol> swizzle, int64_t bitAlignment,
                         OpaqueValue addressSpace, int32_t storageElementBitWidth)
    : PointerProperty(std::move(elemType), elemBitWidth, byteAlignment,
                      bitAlignment == 0 ? int64_t{byteAlignment} * 8 : bitAlignment, swizzle,
                      std::move(addressSpace), storageElementBitWidth),
      address_(symbol::getSym(base)) {}

Leaf<Symbol> Pointer<Symbol>::value() const {
  return symbol::SymbolAccess::leaf(Leaf<Trait>::getDInt(LogWidth::I64), address_);
}

bool operator==(const Pointer<Symbol> &lhs, const Pointer<Symbol> &rhs) {
  return static_cast<const PointerProperty &>(lhs) == static_cast<const PointerProperty &>(rhs) &&
         symbol::equal(symbol::SymbolAccess::address(lhs).get(),
                       symbol::SymbolAccess::address(rhs).get());
}

Pointer<Symbol> add_offset(const Pointer<Symbol> &pointer, IntTupleRef<Symbol> offset) {
  FLYDSL_CORE_ASSERT(offset.isLeaf() && offset.isInt());
  auto expr = symbol::embed(offset.getLeaf());
  FLYDSL_CORE_ASSERT(expr);
  auto address =
      symbol::apply(symbol::SymbolOp::Add, symbol::SymbolAccess::address(pointer), std::move(expr));
  auto bitAlignment = detail::offsetBitAlignment(pointer, offset);
  return symbol::SymbolAccess::pointer(
      PointerProperty(pointer.elementType(), pointer.elementBitWidth(),
                      static_cast<int32_t>((bitAlignment + 7) / 8), bitAlignment, pointer.swizzle(),
                      pointer.addressSpace(), pointer.storageElementBitWidth()),
      std::move(address));
}

} // namespace mlir::fly::core
