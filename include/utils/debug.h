#pragma once

#include <iostream>

#ifndef GERN_DEBUG_BUILD
#define DEBUG(x)
#else
#define DEBUG(x)                     \
    do {                             \
        std::cout << x << std::endl; \
    } while (0)
#endif