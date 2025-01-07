#pragma once

#include <iostream>

#ifndef GERN_DEBUG_BUILD
#define DEBUG(x)
#else
#define DEBUG(x)                     \
    do {                             \
        std::cout << x << std::cout; \
    } while (0)
#endif