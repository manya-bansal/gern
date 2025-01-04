#pragma once

namespace gern {

class Grid {
public:
    enum Dim {
        BLOCK_X,
        BLOCK_Y,
        BLOCK_Z,
        THREAD_X,
        THREADY_Y,
        THREAD_Z,
    };
};

}  // namespace gern