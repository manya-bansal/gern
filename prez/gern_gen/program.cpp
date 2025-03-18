#include "cassert"
#include "cpu-array.h"

void function_9(library::impl::ArrayCPU &input, library::impl::ArrayCPU &output,
                int64_t t, int64_t t2) {
  for (int64_t _gern_x_2_6_8 = 0; (_gern_x_2_6_8 < output.size);
       _gern_x_2_6_8 = (_gern_x_2_6_8 + t)) {

    for (int64_t _gern_x_2_6 = 0; (_gern_x_2_6 < t);
         _gern_x_2_6 = (_gern_x_2_6 + t2)) {

      int64_t _gern_x_2 = (_gern_x_2_6 + _gern_x_2_6_8);
      int64_t _gern_step_1 = t2;

      int64_t _gern_x_4 = _gern_x_2;
      int64_t _gern_step_3 = _gern_step_1;
      auto _query_output_10 = output.query(_gern_x_2, _gern_step_1);

      library::impl::ArrayCPU temp =
          library::impl::ArrayCPU::allocate(_gern_x_4, _gern_step_3);

      auto _query_input_11 = input.query(_gern_x_4, _gern_step_3);

      library::impl::add_1(_query_input_11, temp);

      library::impl::add_1(temp, _query_output_10);

      temp.destroy();
    }
  }
}

extern "C" {
void hook_function_9(void **args) {
  library::impl::ArrayCPU &input = *((library::impl::ArrayCPU *)args[0]);
  library::impl::ArrayCPU &output = *((library::impl::ArrayCPU *)args[1]);
  int64_t t = *((int64_t *)args[2]);
  int64_t t2 = *((int64_t *)args[3]);
  function_9(input, output, t, t2);
}
}
