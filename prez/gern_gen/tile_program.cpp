#include "cassert"
#include "cpu-array.h"




void function_11(gern::impl::ArrayCPU& input, gern::impl::ArrayCPU& output, int64_t t, int64_t t2){
   int64_t _gern_step_7_9 = output.size;
 int64_t _gern_x_8_10 = 0;

 
 int64_t _gern_x_2_4_6 = _gern_x_8_10;
 int64_t _gern_step_1_3_5 = _gern_step_7_9;
    gern::impl::ArrayCPU temp = gern::impl::ArrayCPU::allocate(_gern_x_2_4_6, _gern_step_1_3_5);

 for(_gern_x_2_4_6 = 0;(_gern_x_2_4_6 < temp.size); _gern_x_2_4_6=(_gern_x_2_4_6 + t)){
    

  for(int64_t _gern_x_2_4 = 0;(_gern_x_2_4 < t); _gern_x_2_4=(_gern_x_2_4 + t2)){
      

         int64_t _gern_x_2 = (_gern_x_2_4 + _gern_x_2_4_6);
   int64_t _gern_step_1 = t2;

   
         auto _query_temp_12 = temp.query(_gern_x_2, _gern_step_1);

         
      auto _query_input_13 = input.query(_gern_x_2, _gern_step_1);

         gern::impl::add_1(_query_input_13, _query_temp_12);





}

}

 for(_gern_x_8_10 = 0;(_gern_x_8_10 < output.size); _gern_x_8_10=(_gern_x_8_10 + t)){
    

      int64_t _gern_x_8 = _gern_x_8_10;
  int64_t _gern_step_7 = t;

  
      auto _query_output_14 = output.query(_gern_x_8, _gern_step_7);

      
    auto _query_temp_15 = temp.query(_gern_x_8, _gern_step_7);

      gern::impl::add_1(_query_temp_15, _query_output_14);





}

  temp.destroy();


}


extern "C"
{
   void hook_function_11(void** args){
    gern::impl::ArrayCPU& input = *((gern::impl::ArrayCPU*)args[0]);
  gern::impl::ArrayCPU& output = *((gern::impl::ArrayCPU*)args[1]);
  int64_t t = *((int64_t*)args[2]);
  int64_t t2 = *((int64_t*)args[3]);
      function_11(input, output, t, t2);

}

}

