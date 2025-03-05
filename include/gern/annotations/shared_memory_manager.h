#include "compose/compose.h"

namespace gern {

namespace grid {

/**
 * @brief A shared memory manager consumes a function that is 
 *         called to initialize the shared memory dynamic allocator.
 * 
 * @param init The function that is called to initialize the shared memory dynamic allocator.
 */
class SharedMemoryManager {
public:
    SharedMemoryManager() = default;
    SharedMemoryManager(FunctionCall init,
                        std::vector<std::string> headers = {})  // Header where the interface is defined.
        : init(init), headers(headers) {
    }

    FunctionCall getInit() const {
        return init;
    }

    std::vector<std::string> getHeaders() const {
        return headers;
    }

    bool isInitialized() const {
        return init.name != "int";
    }

private:
    // To test whether the function was ever initialized.
    // Little hacky, but no function can ever be called int....
    FunctionCall init{"int",
                      {},
                      {},
                      Parameter(),
                      LaunchArguments(),
                      LaunchArguments(),
                      DEVICE};
    std::vector<std::string> headers;
};

}  // namespace grid
}  // namespace gern