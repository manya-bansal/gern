# Gern Matrix Multiply App

The kernels in this repository map to the following kernels in
https://siboehm.com/articles/22/CUDA-MMM:

- `kernel_1.cu` : Kernel 1
- `kernel_2.cu` : Kernel 2
- `kernel_3.cu` : Kernel 3, 4, 5


Note that Kernel 7 & 8 are not in the blog (from the blog):

>   I skipped kernels 7 and 8, which I wrote while figuring out how to best eliminate 
    shared memory bank conflicts. They eliminate the conflicts but were overall still 
    slower, so I won’t cover them here.

and, Kernel 9 is just auto-tuning.

