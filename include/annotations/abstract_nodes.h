#ifndef ANNOT_ABSTRACT_NODES
#define ANNOT_ABSTRACT_NODES

#include "annotations/datatypes.h"
#include "utils/name_generator.h"
#include "utils/uncopyable.h"

namespace gern {

class ExprVisitorStrict;
class ConstraintVisitorStrict;
class ConsumesVisitorStrict;
class StmtVisitorStrict;

class ExprNode : public util::Manageable<ExprNode>, public util::Uncopyable {
public:
    ExprNode() = default;
    ExprNode(Datatype type)
        : datatype(type) {};
    virtual ~ExprNode() = default;

    virtual void accept(ExprVisitorStrict *) const = 0;
    Datatype getDatatype() const {
        return datatype;
    }

private:
    Datatype datatype;
};

class ConstraintNode : public util::Manageable<ConstraintNode>,
                       public util::Uncopyable {
public:
    ConstraintNode() = default;
    virtual ~ConstraintNode() = default;
    virtual void accept(ConstraintVisitorStrict *) const = 0;
};

class StmtNode : public util::Manageable<StmtNode>, public util::Uncopyable {
public:
    StmtNode() = default;
    virtual ~StmtNode() = default;
    virtual void accept(StmtVisitorStrict *) const = 0;
};

class AbstractDataType {
public:
    AbstractDataType() = default;

    AbstractDataType(const std::string &name, const std::string &type)
        : name(name), type(type) {
    }
    virtual ~AbstractDataType() = default;

    virtual std::string getName() const {
        return name;
    }

    virtual std::string getType() const {
        return type;
    }

private:
    std::string name;
    std::string type;
};

typedef std::shared_ptr<const AbstractDataType> AbstractDataTypePtr;

std::ostream &operator<<(std::ostream &os, const AbstractDataType &ads);

}  // namespace gern

#endif