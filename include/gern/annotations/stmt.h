#pragma once

#include "annotations/abstract_nodes.h"
#include "annotations/expr.h"
#include "annotations/grid.h"
#include "annotations/std_less_specialization.h"
#include "utils/error.h"
#include "utils/name_generator.h"
#include "utils/uncopyable.h"
#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace gern {

class Annotation;

struct SubsetNode;
struct SubsetObjManyNode;
struct ProducesNode;
struct ConsumesNode;
struct ConsumesForNode;
struct AllocatesNode;
struct ComputesForNode;
struct ComputesNode;
struct PatternNode;

class Stmt : public util::IntrusivePtr<const StmtNode> {
public:
    Stmt()
        : util::IntrusivePtr<const StmtNode>(nullptr) {
    }
    Stmt(const StmtNode *n)
        : util::IntrusivePtr<const StmtNode>(n) {
    }

    std::set<Variable> getDefinedVariables() const;
    std::set<Variable> getIntervalVariables() const;
    std::map<Variable, Variable> getConsumesIntervalAndStepVars() const;
    std::map<Variable, Variable> getComputesIntervalAndStepVars() const;
    std::map<Expr, std::tuple<Variable, Expr, Variable>> getTileableFields() const;
    std::map<Expr, std::tuple<Variable, Expr, Variable>> getReducableFields() const;
    void accept(StmtVisitorStrict *v) const;
    std::string str() const;
};

std::ostream &operator<<(std::ostream &os, const Stmt &);

DEFINE_BINARY_CLASS(Assign, Stmt)

class SubsetObj : public Stmt {
public:
    SubsetObj() = default;
    SubsetObj(const SubsetNode *);
    SubsetObj(AbstractDataTypePtr data,
              std::vector<Expr> mdFields);
    std::vector<Expr> getFields() const;
    AbstractDataTypePtr getDS() const;
    typedef SubsetNode Node;
};

class Produces : public Stmt {
public:
    explicit Produces(const ProducesNode *);
    // Factory method to produce make a produces node.
    static Produces Subset(AbstractDataTypePtr, std::vector<Variable>);
    SubsetObj getSubset() const;
    std::vector<Variable> getFieldsAsVars() const;
    typedef ProducesNode Node;
};

struct ConsumesNode;
class ConsumeMany;

class Consumes : public Stmt {
public:
    explicit Consumes(const ConsumesNode *);
    // Factory method to produce make a consumes node.
    static Consumes Subset(AbstractDataTypePtr, std::vector<Expr>);
    static Consumes Subsets(ConsumeMany);
    Consumes(SubsetObj s);
    typedef ConsumesNode Node;
};

class ConsumeMany : public Consumes {
public:
    ConsumeMany(const ConsumesNode *s)
        : Consumes(s){};
};

class SubsetObjMany : public ConsumeMany {
public:
    SubsetObjMany(const SubsetObjManyNode *);
    SubsetObjMany(const std::vector<SubsetObj> &subsets);
    SubsetObjMany(SubsetObj s)
        : SubsetObjMany(std::vector<SubsetObj>{s}) {
    }
    typedef SubsetObjManyNode Node;
};

// This ensures that a consumes node will only ever contain a for loop
// or a list of subsets. In this way, we can leverage the cpp type checker to
// ensures that only legal patterns are written down.
ConsumeMany Reducible(Assign start, Expr parameter, Variable step, ConsumeMany body,
                      bool parallel = false);
ConsumeMany Reducible(Assign start, Expr parameter, Variable step, std::vector<SubsetObj> body,
                      bool parallel = false);
ConsumeMany Reducible(Assign start, Expr parameter, Variable step, SubsetObj body,
                      bool parallel = false);

class Allocates : public Stmt {
public:
    Allocates()
        : Stmt() {
    }
    explicit Allocates(const AllocatesNode *);
    Allocates(Expr reg, Expr smem = Expr());
    typedef AllocatesNode Node;
};

class Pattern;
class Annotation : public Stmt {
public:
    Annotation() = default;
    Annotation(const AnnotationNode *);
    Annotation(Pattern, std::set<Grid::Unit>, std::vector<Constraint>);
    Pattern getPattern() const;
    std::vector<Constraint> getConstraints() const;

    Annotation assumes(std::vector<Constraint>) const;  // requires is already used as a keyword :(

    template<typename First, typename... Remaining>
    Annotation assumes(First first, Remaining... remaining) const {
        static_assert(std::is_base_of_v<Constraint, First>,
                      "All arguments must be children of Constraint");
        static_assert((std::is_base_of_v<Constraint, Remaining> && ...),
                      "All arguments must be children of Constraint");
        std::vector<Constraint> constraints{first, remaining...};
        return this->assumes(constraints);
    }

    std::set<Grid::Unit> getOccupiedUnits() const;
    typedef AnnotationNode Node;
};

class Pattern : public Stmt {
public:
    Pattern()
        : Stmt() {
    }
    explicit Pattern(const PatternNode *);
    Annotation occupies(std::set<Grid::Unit>) const;

    Annotation assumes(std::vector<Constraint>) const;
    /**
     * @brief assumes adds constraints to the pattern, and
     *        converts it to an annotation.
     *
     * @return Annotation
     */
    template<typename First, typename... Remaining>
    Annotation assumes(First first, Remaining... remaining) const {
        static_assert(std::is_base_of_v<Constraint, First>,
                      "All arguments must be children of Constraint");
        static_assert((std::is_base_of_v<Constraint, Remaining> && ...),
                      "All arguments must be children of Constraint");
        std::vector<Constraint> constraints{first, remaining...};
        return this->assumes(constraints);
    }

    std::vector<SubsetObj> getInputs() const;
    std::vector<SubsetObj> getAllADTs() const;
    std::vector<Variable> getProducesField() const;
    std::vector<Expr> getRequirement(AbstractDataTypePtr) const;
    SubsetObj getOutput() const;
    SubsetObj getCorrespondingSubset(AbstractDataTypePtr) const;
    typedef PatternNode Node;
};

class Computes : public Pattern {
public:
    explicit Computes(const ComputesNode *);
    Computes(Produces p, Consumes c, Allocates a = Allocates());
    typedef ComputesNode Node;
};

Annotation annotate(Pattern);
Annotation resetUnit(Annotation, std::set<Grid::Unit>);
// This ensures that a computes node will only ever contain a for loop
// or a (Produces, Consumes) node. In this way, we can leverage the cpp type
// checker to ensures that only legal patterns are written down.
Pattern Tileable(Assign start, Expr parameter, Variable step, Pattern body,
                 bool parallel = false);
// FunctionSignature so that users do need an explicit compute initialization.
Pattern Tileable(Assign start, Expr parameter, Variable step,
                 Produces produces, Consumes consumes,
                 bool parallel = false);
}  // namespace gern