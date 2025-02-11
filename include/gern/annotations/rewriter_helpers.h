#pragma once

#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include <map>
#include <set>

namespace gern {

template<typename T, typename F>
inline T replaceVariables(T annot,
                          const std::map<Variable, F> &rw_vars) {
    struct rewriteVar : public Rewriter {
        rewriteVar(std::map<Variable, F> rw_vars)
            : rw_vars(rw_vars) {
        }
        using Rewriter::rewrite;

        void visit(const VariableNode *op) {
            if (rw_vars.contains(op)) {
                expr = rw_vars.at(op);
            } else {
                expr = op;
            }
        }
        std::map<Variable, F> rw_vars;
    };
    rewriteVar rw{rw_vars};
    return to<T>(rw.rewrite(annot));
}

template<typename T>
inline T replaceADTs(T annot,
                     const std::map<AbstractDataTypePtr, AbstractDataTypePtr> &rw_ds) {
    struct rewriteDS : public Rewriter {
        rewriteDS(std::map<AbstractDataTypePtr, AbstractDataTypePtr> rw_ds)
            : rw_ds(rw_ds) {
        }
        using Rewriter::rewrite;

        void visit(const ADTMemberNode *op) {
            if (rw_ds.contains(op->ds)) {
                expr = ADTMember(rw_ds.at(op->ds), op->member, op->const_expr);
            } else {
                expr = op;
            }
        }
        void visit(const SubsetNode *op) {
            // Rewrite all the fields.
            std::vector<Expr> rw_expr;
            for (size_t i = 0; i < op->mdFields.size(); i++) {
                rw_expr.push_back(this->rewrite(op->mdFields[i]));
            }
            // Construct the new subset object.
            if (rw_ds.contains(op->data)) {
                stmt = SubsetObj(rw_ds[op->data], rw_expr);
            } else {
                stmt = SubsetObj(op->data, rw_expr);
            }
        }
        std::map<AbstractDataTypePtr, AbstractDataTypePtr> rw_ds;
    };
    rewriteDS rw{rw_ds};
    return to<T>(rw.rewrite(annot));
}

template<typename T>
inline std::set<Grid::Dim> getDims(T annot) {
    std::set<Grid::Dim> dims;
    match(annot, std::function<void(const GridDimNode *)>(
                     [&](const GridDimNode *op) { dims.insert(op->dim); }));
    return dims;
}

inline Annotation refreshVariables(Annotation annot, std::map<Variable, Variable> &new_vars) {
    // Only refresh output side variables.
    auto output_var_vec = annot.getPattern().getProducesField();
    auto interval_vars = annot.getIntervalVariables();
    std::set<Variable> output_var_set{output_var_vec.begin(), output_var_vec.end()};
    std::set<Variable> old_vars = getVariables(annot);
    for (const auto &v : old_vars) {
        new_vars[v] = Variable(getUniqueName(v.getName()), v.getDatatype(), v.isConstExpr());
    }
    return replaceVariables(annot, new_vars);
}
template<typename T>
inline T replaceDim(T annot, const std::map<Grid::Dim, Expr> &rw_dims) {
    struct rewriteDS : public Rewriter {
        rewriteDS(const std::map<Grid::Dim, Expr> &rw_dims)
            : rw_dims(rw_dims) {
        }
        using Rewriter::rewrite;

        void visit(const GridDimNode *op) {
            if (rw_dims.contains(op->dim)) {
                expr = rw_dims.at(op->dim);
            } else {
                expr = op;
            }
        }
        const std::map<Grid::Dim, Expr> &rw_dims;
    };
    rewriteDS rw{rw_dims};
    return to<T>(rw.rewrite(annot));
}

/**
 * @brief isConstExpr returns whether an expression is a
 *        constant expression (can be evaluated at program
 *        compile time).
 *
 * @return true
 * @return false
 */
template<typename T>
inline bool isConstExpr(T e) {
    bool is_const_expr = true;
    match(e,
          std::function<void(const VariableNode *)>(
              [&](const VariableNode *op) {
                  if (!op->const_expr) {
                      is_const_expr = false;
                  }
              }),
          std::function<void(const ADTMemberNode *op)>(
              [&](const ADTMemberNode *op) {
                  if (!op->const_expr) {
                      is_const_expr = false;
                  }
              }));
    return is_const_expr;
}

}  // namespace gern