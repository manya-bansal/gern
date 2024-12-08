#ifndef GERN_CODEGEN_H
#define GERN_CODEGEN_H

#include "codegen/codegen_ir.h"
#include "compose/pipeline.h"

namespace gern {
namespace codegen {

class CodeGenerator {
public:
    CodeGenerator() = default;
    CGStmt generate_code(const Pipeline &);

private:
};

}  // namespace codegen
}  // namespace gern

#endif