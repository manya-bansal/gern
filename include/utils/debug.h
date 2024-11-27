#ifndef GERN_DEBUG_H
#define GERN_DEBUG_H

#include <iostream>

#ifndef _DEBUG
#define DEBUG(x)
#else
#define DEBUG(x)                                                               \
  do {                                                                         \
    std::cerr << x << std::endl;                                               \
  } while (0)
#endif

#endif