#ifndef GERN_DEBUG_H
#define GERN_DEBUG_H

#include <iostream>

#ifndef GERN_DEBUG_BUILD
#define DEBUG(x)
#else
#define DEBUG(x)                                                               \
  do {                                                                         \
    std::cout << x << std::endl;                                               \
  } while (0)
#endif

#endif