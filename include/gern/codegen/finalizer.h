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
    util::ScopedSet<LowerIR> to_insert;
};

class ADTReplacer : public LowerIRVisitor {
public:
    ADTReplacer(LowerIR ir,
                std::map<AbstractDataTypePtr, AbstractDataTypePtr> rewrites)
        : ir(ir), rewrites(rewrites) {
    }

    LowerIR replace();
    using LowerIRVisitor::visit;
    void visit(const AllocateNode *);
    void visit(const FreeNode *);
    void visit(const InsertNode *);
    void visit(const QueryNode *);
    void visit(const BlankNode *);
    void visit(const ComputeNode *);
    void visit(const IntervalNode *);
    void visit(const DefNode *);
    void visit(const AssertNode *);
    void visit(const BlockNode *);
    void visit(const GridDeclNode *);
    void visit(const SharedMemoryDeclNode *);
    void visit(const OpaqueCall *);

private:
    LowerIR ir;
    LowerIR final_ir;
    AbstractDataTypePtr get_adt(const AbstractDataTypePtr &adt) const;
    const std::map<AbstractDataTypePtr, AbstractDataTypePtr> rewrites;
};

class ADTReuser : public LowerIRVisitor {
public:
    ADTReuser(LowerIR ir)
        : ir(ir) {
    }

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

    LowerIR construct();
    using LowerIRVisitor::visit;

private:
    LowerIR ir;
    int32_t cur_lno = 0;
    /*
    * live_range tracks the first and last line number that a subset is used.
    * two subsets, subset1 and subset2, can be reused if the last use of subset1
    * is before the first use of subset2.
    */
    std::map<AbstractDataTypePtr, std::tuple<int, int>> live_range;
    std::map<AbstractDataTypePtr, FunctionCall> allocate_calls;

    void update_live_range(AbstractDataTypePtr adt);
    void update_live_range(Expr e);
};

class Scoper : public LowerIRVisitor {
public:
    Scoper(LowerIR ir)
        : ir(ir) {
    }

    LowerIR construct();

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
    int32_t get_scope(Expr e) const;
    int32_t get_scope(std::vector<Argument> args) const;

    LowerIR ir;
    int32_t cur_scope = 0;
    // The scope an ADT is at.
    std::map<AbstractDataTypePtr, int32_t> adt_scope;
    // The scope a variable is at.
    std::map<Variable, int32_t> var_scope;
    // Maps scope to statements to that are at that scope.
    std::map<int32_t, std::vector<LowerIR>> new_statements;
};

}  // namespace gern
