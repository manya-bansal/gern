#pragma once

#include "annotations/data_dependency_language.h"
#include "annotations/lang_nodes.h"
#include "annotations/visitor.h"
#include <map>
#include <set>

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
            if (rw_vars.contains(op)) {
                expr = rw_vars.at(op);
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

inline Annotation refreshVariables(Annotation annot) {
    // Only refresh output side variables.
    auto output_var_vec = annot.getPattern().getProducesField();
    auto interval_vars = annot.getIntervalVariables();
    std::set<Variable> output_var_set{output_var_vec.begin(), output_var_vec.end()};
    std::set<Variable> old_vars = getVariables(annot);
    std::map<Variable, Variable> fresh_names;
    for (const auto &v : old_vars) {
        std::cout << v << std::endl;
        if (output_var_set.contains(v) && !interval_vars.contains(v)) {
            std::cout << "Inside" << v << std::endl;
            // Otherwise, generate a new name.
            fresh_names[v] = getUniqueName("_gern_" + v.getName());
        }
    }
    return replaceVariables(annot, fresh_names);
}

}  // namespace gern