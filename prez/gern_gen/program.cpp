#include "cassert"
#include "cpu-array.h"

void function_7(library::impl::ArrayCPU &input, library::impl::ArrayCPU &output,
                int64_t t) {
  int64_t _gern_step_3_5 = output.size;
  int64_t _gern_x_4_6 = 0;

  int64_t _gern_x_2 = _gern_x_4_6;
  int64_t _gern_step_1 = _gern_step_3_5;
  library::impl::ArrayCPU temp =
      library::impl::ArrayCPU::allocate(_gern_x_2, _gern_step_1);

  library::impl::add_1(input, temp);

  for (_gern_x_4_6 = 0; (_gern_x_4_6 < output.size);
       _gern_x_4_6 = (_gern_x_4_6 + t)) {

    int64_t _gern_x_4 = _gern_x_4_6;
    int64_t _gern_step_3 = t;

    auto _query_output_8 = output.query(_gern_x_4, _gern_step_3);

    auto _query_temp_9 = temp.query(_gern_x_4, _gern_step_3);

    library::impl::add_1(_query_temp_9, _query_output_8);
  }

  temp.destroy();
}

extern "C" {
void hook_function_7(void **args) {
  library::impl::ArrayCPU &input = *((library::impl::ArrayCPU *)args[0]);
  library::impl::ArrayCPU &output = *((library::impl::ArrayCPU *)args[1]);
  int64_t t = *((int64_t *)args[2]);
  function_7(input, output, t);
}
}
