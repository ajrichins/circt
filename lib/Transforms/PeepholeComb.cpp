#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;

namespace {

// class OptName : public RewritePattern {
// public:
//   OptName(MLIRContext *context)
//       : RewritePattern(comb::Operation::getOperationName(), 1, context) {}

//   void initialize() {
//     setDebugName("OptName");
//     addDebugLabels("MyRewritePass");
//   }

//   LogicalResult matchAndRewrite(Operation *op,
//                                 PatternRewriter &rewriter) const override {
//     return failure();
//   }
// };

// opt 28 newvar3 + (v0 - newvar3) => v0 handles add commutativity
class Opt28 : public RewritePattern {
public:
  Opt28(MLIRContext *context)
      : RewritePattern(comb::AddOp::getOperationName(), 1, context) {}

  void initialize() {
    setDebugName("Opt28");
    addDebugLabels("MyRewritePass");
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // need to make more variadic and communitative a little, but focus on that
    // later
    if (auto addOp = dyn_cast<comb::AddOp>(op)) {
      if (auto operand0_op = addOp->getOperand(0).getDefiningOp()) {
        if (auto subOp = dyn_cast<comb::SubOp>(operand0_op)) {
          if (subOp->getOperand(1) == addOp->getOperand(1)) {
            // Perform the rewrite: replace the AddOp with its second operand.
            rewriter.replaceOp(op, subOp->getOperand(0));
            return success();
          }
        }
      }
      if (auto operand1_op = addOp->getOperand(1).getDefiningOp()) {
        if (auto subOp = dyn_cast<comb::SubOp>(operand1_op)) {
          if (subOp->getOperand(1) == addOp->getOperand(0)) {
            // Perform the rewrite: replace the AddOp with its first operand.
            rewriter.replaceOp(op, subOp->getOperand(0));
            return success();
          }
        }
      }
    }
    return failure();
  }
};

class Opt54 : public RewritePattern {
public:
  Opt54(MLIRContext *context)
      : RewritePattern(comb::AddOp::getOperationName(), 1, context) {}

  void initialize() {
    setDebugName("Opt54");
    addDebugLabels("MyRewritePass");
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (auto addOp = dyn_cast<comb::AddOp>(op)) {
      if (auto operand0_op = addOp->getOperand(0).getDefiningOp()) {
        if (auto andOp = dyn_cast<comb::AndOp>(operand0_op)) {
          // canonicalize pass puts folded constant as the last operand
          if (andOp->getOperands()) {
            // Perform the rewrite: replace the AddOp with its second operand.
            rewriter.replaceOp(op, subOp->getOperand(0));
            return success();
          }
        }
      }
    }
    return failure();
  }
};

void collectMyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<Opt28, Opt54>(ctx);
  // patterns.addWithLabel<Opt28>("MyRewritePatterns", ctx);
}

class MyPatternRewriter : public PatternRewriter {
public:
  MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.
};

/// Apply the custom driver to `op`.
// need to m,odify to be recursive for regions and add to 'worklist'?
void applyMyPatternDriver(Operation *op,
                          const FrozenRewritePatternSet &patterns) {
  // Initialize the custom PatternRewriter.
  MyPatternRewriter rewriter(op->getContext());

  // Create the applicator and apply our cost model.
  PatternApplicator applicator(patterns);
  applicator.applyCostModel([](const Pattern &pattern) {
    // Apply a default cost model.
    // Note: This is just for demonstration, if the default cost model is truly
    //       desired `applicator.applyDefaultCostModel()` should be used
    //       instead.
    return pattern.getBenefit();
  });

  // Try to match and apply a pattern. need to figure out how the logical result
  // stuff should propigate
  LogicalResult result = applicator.matchAndRewrite(op, rewriter);
  if (failed(result)) {

  } else {
  }
}

struct PeepholeCombPass : public PeepholeCombPassBase<PeepholeCombPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    collectMyPatterns(patterns, ctx);

    // applyMyPatternDriver(getOperation(), std::move(patterns));
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt::createPeepholeCombPass() {
  return std::make_unique<PeepholeCombPass>();
}