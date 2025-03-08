#include "cassert"
#include "cpu-array.h"




void function_3(gern::impl::ArrayCPU& input, gern::impl::ArrayCPU& output){
  
   gern::impl::add_1(input, output);

}


extern "C"
{
   void hook_function_3(void** args){
    gern::impl::ArrayCPU& input = *((gern::impl::ArrayCPU*)args[0]);
  gern::impl::ArrayCPU& output = *((gern::impl::ArrayCPU*)args[1]);
      function_3(input, output);

}

}

