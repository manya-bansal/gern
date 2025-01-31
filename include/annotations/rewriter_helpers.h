#pragma once

#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include <map>

namespace gern {

template<typename T>
inline T replaceVariables(T annot,
                          const std::map<Variable, Variable> &rw_vars) {
    struct rewriteVar : public Rewriter {
        rewriteVar(std::map<Variable, Variable> rw_vars)
            : rw_vars(rw_vars) {
        }
        using Rewriter::rewrite;

        void visit(const VariableNode *op) {
            if (rw_vars.find(op) != rw_vars.end()) {
                expr = rw_vars[op];
            } else {
                expr = op;
            }
        }
        std::map<Variable, Variable> rw_vars;
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
                expr = ADTMember(rw_ds.at(op->ds), op->member);
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

}  // namespace gern