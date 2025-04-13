#pragma once

#include <cassert>
#include <functional>
#include <iostream>

namespace gern {

class Expr;
class Constraint;
class Stmt;
class Consumes;
class ConsumeMany;
class Allocates;
class Computes;
class Pattern;
class Annotation;

// Expr nodes.
struct LiteralNode;
struct AddNode;
struct MulNode;
struct SubNode;
struct DivNode;
struct ModNode;
struct VariableNode;
struct ADTMemberNode;
// Constraint nodes.
struct EqNode;
struct NeqNode;
struct LeqNode;
struct GeqNode;
struct LessNode;
struct GreaterNode;
struct AndNode;
struct OrNode;
struct GridDimNode;
// Stmt nodes.
struct AssignNode;
struct SubsetNode;
struct SubsetObjManyNode;
struct ProducesNode;
struct ConsumesNode;
struct ConsumesForNode;
struct AllocatesNode;
struct ComputesForNode;
struct ComputesNode;
struct PatternNode;
struct AnnotationNode;

class ExprVisitorStrict {
public:
    virtual ~ExprVisitorStrict() = default;

    virtual void visit(Expr);
    virtual void visit(const LiteralNode *) = 0;
    virtual void visit(const AddNode *) = 0;
    virtual void visit(const SubNode *) = 0;
    virtual void visit(const MulNode *) = 0;
    virtual void visit(const DivNode *) = 0;
    virtual void visit(const ModNode *) = 0;
    virtual void visit(const VariableNode *) = 0;
    virtual void visit(const ADTMemberNode *) = 0;
    virtual void visit(const GridDimNode *) = 0;
};

class ConstraintVisitorStrict {
public:
    virtual ~ConstraintVisitorStrict() = default;

    virtual void visit(Constraint);
    virtual void visit(const EqNode *) = 0;
    virtual void visit(const NeqNode *) = 0;
    virtual void visit(const LeqNode *) = 0;
    virtual void visit(const GeqNode *) = 0;
    virtual void visit(const LessNode *) = 0;
    virtual void visit(const GreaterNode *) = 0;
    virtual void visit(const OrNode *) = 0;
    virtual void visit(const AndNode *) = 0;
};

class StmtVisitorStrict {
public:
    virtual ~StmtVisitorStrict() = default;

    virtual void visit(Stmt);
    virtual void visit(const AssignNode *) = 0;
    virtual void visit(const SubsetNode *) = 0;
    virtual void visit(const SubsetObjManyNode *) = 0;
    virtual void visit(const ProducesNode *) = 0;
    virtual void visit(const ConsumesNode *) = 0;
    virtual void visit(const ConsumesForNode *) = 0;
    virtual void visit(const AllocatesNode *) = 0;
    virtual void visit(const ComputesForNode *) = 0;
    virtual void visit(const ComputesNode *) = 0;
    virtual void visit(const PatternNode *) = 0;
    virtual void visit(const AnnotationNode *) = 0;
};

class AnnotVisitorStrict : public ExprVisitorStrict,
                           public ConstraintVisitorStrict,
                           public StmtVisitorStrict {
public:
    using ConstraintVisitorStrict::visit;
    using ExprVisitorStrict::visit;
    using StmtVisitorStrict::visit;
};

class Printer : public AnnotVisitorStrict {
public:
    using AnnotVisitorStrict::visit;

    Printer(std::ostream &os, int ident = 0)
        : os(os), ident(ident) {
    }

    void visit(Consumes c);
    void visit(ConsumeMany many);
    void visit(Allocates a);

    void visit(const LiteralNode *);
    void visit(const AddNode *);
    void visit(const SubNode *);
    void visit(const MulNode *);
    void visit(const DivNode *);
    void visit(const ModNode *);
    void visit(const VariableNode *);
    void visit(const ADTMemberNode *);

    void visit(const EqNode *);
    void visit(const NeqNode *);
    void visit(const LeqNode *);
    void visit(const GeqNode *);
    void visit(const LessNode *);
    void visit(const GreaterNode *);
    void visit(const OrNode *);
    void visit(const AndNode *);
    void visit(const GridDimNode *);

    void visit(const AssignNode *);
    void visit(const SubsetNode *);
    void visit(const SubsetObjManyNode *);
    void visit(const ProducesNode *);
    void visit(const ConsumesNode *);
    void visit(const AllocatesNode *);
    void visit(const ComputesForNode *);
    void visit(const ConsumesForNode *);
    void visit(const ComputesNode *);
    void visit(const PatternNode *);
    void visit(const AnnotationNode *);

private:
    std::ostream &os;
    int ident;
};

class AnnotVisitor : public AnnotVisitorStrict {

public:
    using AnnotVisitorStrict::visit;

    virtual ~AnnotVisitor() = default;
    void visit(const LiteralNode *);
    void visit(const AddNode *);
    void visit(const SubNode *);
    void visit(const MulNode *);
    void visit(const DivNode *);
    void visit(const ModNode *);
    void visit(const VariableNode *);
    void visit(const ADTMemberNode *);

    void visit(const EqNode *);
    void visit(const NeqNode *);
    void visit(const LeqNode *);
    void visit(const GeqNode *);
    void visit(const LessNode *);
    void visit(const GreaterNode *);
    void visit(const OrNode *);
    void visit(const AndNode *);
    void visit(const GridDimNode *);

    void visit(const AssignNode *);
    void visit(const SubsetNode *);
    void visit(const SubsetObjManyNode *);
    void visit(const ProducesNode *);
    void visit(const ConsumesNode *);
    void visit(const AllocatesNode *);
    void visit(const ComputesForNode *);
    void visit(const ConsumesForNode *);
    void visit(const ComputesNode *);
    void visit(const PatternNode *);
    void visit(const AnnotationNode *);
};

}  // namespace gern