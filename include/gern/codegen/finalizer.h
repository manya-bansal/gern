#include "codegen/lower_visitor.h"
#include "utils/scoped_set.h"

namespace gern {

class Finalizer : public LowerIRVisitor {
public:
    Finalizer(LowerIR ir)
        : ir(ir) {
    }

    LowerIR finalize();
    using LowerIRVisitor::visit;
    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const BlankNode *);
    void visit(const DefNode *);
    void visit(const AssertNode *);
    void visit(const BlockNode *);
    void visit(const GridDeclNode *);
    void visit(const SharedMemoryDeclNode *);
    void visit(const OpaqueCall *);

    util::ScopedSet<AbstractDataTypePtr> getToFree() const;

private:
    LowerIR ir;
    LowerIR final_ir;
    util::ScopedSet<AbstractDataTypePtr> to_free;
    util::ScopedSet<const InsertNode *> to_insert;
};

class Scoper : public LowerIRVisitor {
public:
    Scoper(LowerIR ir)
        : ir(ir) {
    }

    void construct();

    using LowerIRVisitor::visit;

    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const BlankNode *);
    void visit(const DefNode *);
    void visit(const AssertNode *);
    void visit(const BlockNode *);
    void visit(const GridDeclNode *);
    void visit(const SharedMemoryDeclNode *);
    void visit(const OpaqueCall *);

private:
    LowerIR ir;

    int32_t cur_scope = 0;
    int32_t min_scope(int32_t a, int32_t b) const;
    int32_t get_scope_var(Variable v) const;

    int32_t get_scope(Expr e) const;
    int32_t get_scope(std::vector<Argument> args) const;

    std::map<AbstractDataTypePtr, int32_t> adt_scope;
    std::map<Variable, int32_t> var_scope;

    std::vector<Variable> var_stack;
};

}  // namespace gern
