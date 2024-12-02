#ifndef GERN_ARGUMENTS_H
#define GERN_ARGUMENTS_H

#include "utils/uncopyable.h"

namespace gern {

class ArgumentNode : public util::Manageable<ArgumentNode>,
                     public util::Uncopyable {
public:

};

}  // namespace gern

#endif