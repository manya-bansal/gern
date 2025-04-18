#include "codegen/codegen_printer.h"
#include "utils/error.h"

namespace gern {
namespace codegen {

static int codegen_indent = 0;

void CGPrinter::print(CGStmt s) {
    if (!s.defined()) {
        return;
    }
    s.accept(this);
}

void CGPrinter::visit(const Literal *op) {
    switch (op->type.getKind()) {
    case Datatype::Bool:
        os << op->getVal<bool>();
        break;
    case Datatype::UInt8:
        os << op->getVal<uint8_t>();
        break;
    case Datatype::UInt16:
        os << op->getVal<uint16_t>();
        break;
    case Datatype::UInt32:
        os << op->getVal<uint32_t>();
        break;
    case Datatype::UInt64:
        os << op->getVal<uint64_t>();
        break;
    case Datatype::Int8:
        os << op->getVal<int8_t>();
        break;
    case Datatype::Int16:
        os << op->getVal<int16_t>();
        break;
    case Datatype::Int32:
        os << op->getVal<int32_t>();
        break;
    case Datatype::Int64:
        os << op->getVal<int64_t>();
        break;
    case Datatype::Float32:
        os << op->getVal<float>();
        break;
    case Datatype::Float64:
        os << op->getVal<double>();
        break;
    case Datatype::Complex64:
        os << op->getVal<std::complex<float>>();
        break;
    case Datatype::Complex128:
        os << op->getVal<std::complex<double>>();
        break;
    case Datatype::Undefined:
        break;
    }
}

void CGPrinter::visit(const Scope *op) {
    codegen_indent++;
    os << "{" << std::endl;
    if (op->stmt.defined()) {
        doIdent();
        op->stmt.accept(this);
    }
    os << "}" << std::endl;
    codegen_indent--;
}

void CGPrinter::visit(const Type *op) {
    os << op->type_name;
}

void CGPrinter::doIdent() {
    for (int i = 0; i < codegen_indent; i++) {
        os << " ";
    }
}

void CGPrinter::visit(const VarDecl *op) {
    if (op->properties.is_const)
        os << "const ";
    os << op->type;

    for (int i = 0; i < op->properties.num_ref; i++) {
        os << "&";
    }

    for (int i = 0; i < op->properties.num_ptr; i++) {
        os << "*";
    }

    os << " " << op->var_name;
}

void CGPrinter::visit(const DeclFunc *op) {

    doIdent();
    int num_template_args = op->template_args.size();
    if (num_template_args > 0) {
        os << "template<";
        for (int i = 0; i < num_template_args; i++) {
            os << op->template_args[i];
            os << ((i != num_template_args - 1) ? ", " : "");
        }
        os << ">"
           << "\n";
    }

    doIdent();
    if (op->device) {
        os << "__global__ ";
    }
    os << op->return_type << " " << op->name << "(";
    if (op->args.size() != 0) {
        for (size_t i = 0; i < op->args.size() - 1; i++) {
            os << op->args[i] << ", ";
        }
        os << op->args[op->args.size() - 1];
    }
    os << ")";
    os << op->body << std::endl;
}

void CGPrinter::visit(const Block *op) {
    for (auto stmt : op->stmts) {
        doIdent();
        os << stmt << std::endl;
    }
}

void CGPrinter::visit(const EscapeCGExpr *op) {
    os << op->code;
}

void CGPrinter::visit(const EscapeCGStmt *op) {
    os << op->code;
}

void CGPrinter::visit(const VarAssign *op) {
    os << op->lhs << " = " << op->rhs << ";";
}

void CGPrinter::visit(const Var *op) {
    os << op->name;
}

void CGPrinter::visit(const Call *op) {
    os << op->name;

    int num_template_args = op->template_args.size();
    if (num_template_args > 0) {
        os << "<";
        for (int i = 0; i < num_template_args; i++) {
            os << op->template_args[i];
            os << ((i != num_template_args - 1) ? ", " : "");
        }
        os << ">";
    }

    if (op->arg.size() == 0) {
        os << "()";
    } else {
        os << "(";
        for (size_t i = 0; i < op->arg.size() - 1; i++) {
            os << op->arg[i] << ", ";
        }
        os << op->arg[op->arg.size() - 1] << ")";
    }
}

void CGPrinter::visit(const Cast *op) {
    os << "(" << op->cast_type << ")" << op->to_cast;
}

void CGPrinter::visit(const Deref *op) {
    os << "*(" << op->expr << ")";
}

void CGPrinter::visit(const SpecializedFunction *op) {
    os << op->name;
    if (op->template_args.size() > 0) {
        os << "<";
        for (size_t i = 0; i < op->template_args.size(); i++) {
            os << op->template_args[i];
            os << ((i != op->template_args.size() - 1) ? ", " : "");
        }
        os << ">";
    }
}

void CGPrinter::visit(const VoidCall *op) {
    doIdent();
    os << op->func << ";";
}

void CGPrinter::visit(const KernelLaunch *op) {
    os << op->name;
    int num_template_args = op->template_args.size();
    if (num_template_args > 0) {
        os << "<";
        for (int i = 0; i < num_template_args; i++) {
            os << op->template_args[i];
            os << ((i != num_template_args - 1) ? ", " : "");
        }
        os << ">";
    }
    os << "<<<" << op->grid << ", " << op->block;
    // Add the shared memory size if defined.
    if (op->smem_size.defined()) {
        os << ", " << op->smem_size;
    }
    os << ">>>";

    if (op->args.size() == 0) {
        os << "()";
    } else {
        os << "(";
        for (size_t i = 0; i < op->args.size() - 1; i++) {
            os << op->args[i] << ", ";
        }
        os << op->args[op->args.size() - 1] << ");";
    }
}

void CGPrinter::visit(const BlankLine *) {
    os << std::endl;
}

void CGPrinter::visit(const For *op) {
    auto step_ptr = to<VarAssign>(op->step);
    if (op->parallel) {
        os << "cilk_scope {" << std::endl;
        codegen_indent++;
        doIdent();
        os << "cilk_";
    }
    os << "for(" << op->start << op->cond << "; " << step_ptr->lhs;

    switch (step_ptr->op) {
    case AddEQ:
        os << "+=";
        break;
    case SubEQ:
        os << "-=";
        break;
    case MulEQ:
        os << "*=";
        break;
    case EQ:
        os << "=";
        break;
    default:
        throw error::InternalError("Unreachable");
        break;
    }

    os << step_ptr->rhs << ")";
    os << op->body;

    if (op->parallel) {
        doIdent();
        os << "}" << std::endl;
        codegen_indent--;
    }
}

void CGPrinter::visit(const Lt *op) {
    os << "(" << op->a << " < " << op->b << ")";
}
void CGPrinter::visit(const Gt *op) {
    os << "(" << op->a << " > " << op->b << ")";
}
void CGPrinter::visit(const Eq *op) {
    os << "(" << op->a << " == " << op->b << ")";
}
void CGPrinter::visit(const Neq *op) {
    os << "(" << op->a << " != " << op->b << ")";
}
void CGPrinter::visit(const Gte *op) {
    os << "(" << op->a << " >= " << op->b << ")";
}
void CGPrinter::visit(const Lte *op) {
    os << "(" << op->a << " <= " << op->b << ")";
}
void CGPrinter::visit(const And *op) {
    os << "(" << op->a << " && " << op->b << ")";
}
void CGPrinter::visit(const Or *op) {
    os << "(" << op->a << " || " << op->b << ")";
}
void CGPrinter::visit(const Add *op) {
    os << "(" << op->a << " + " << op->b << ")";
}
void CGPrinter::visit(const Mul *op) {
    os << "(" << op->a << " * " << op->b << ")";
}
void CGPrinter::visit(const Div *op) {
    os << "(" << op->a << " / " << op->b << ")";
}
void CGPrinter::visit(const Mod *op) {
    os << "(" << op->a << " % " << op->b << ")";
}
void CGPrinter::visit(const Sub *op) {
    os << "(" << op->a << " - " << op->b << ")";
}

void CGPrinter::visit(const MetaData *op) {
    os << op->var << "." << op->field;
}

}  // namespace codegen
}  // namespace gern