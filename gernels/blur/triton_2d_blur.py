import torch
import triton
import triton.language as tl
import os
from itertools import product

# Blur X computed row wise
@triton.jit
def triton_blur_x(x_ptr, y_ptr, x_row_stride,  x_col_stride, 
           y_row_stride, y_col_stride, rows, cols, 
           BLOCK_SIZE:tl.constexpr):
    row_idx = tl.program_id(0) 
    row_start =  x_ptr + row_idx * x_row_stride
    col_enteries = tl.arange(0, BLOCK_SIZE)
    mask = col_enteries < cols
    row_ptr = row_start + col_enteries
    row_1 = tl.load(row_ptr, mask=mask)
    row_ptr_2 = row_start + x_col_stride + col_enteries
    row_2 = tl.load(row_ptr_2, mask=mask)
    row_ptr_3 = row_start + 2*x_col_stride+ col_enteries
    row_3 = tl.load(row_ptr_3, mask=mask)
    out = (row_1 + row_2 + row_3)/3
    out_loc = y_ptr + row_idx * y_row_stride
    col_enteries_out = tl.arange(0, BLOCK_SIZE)
    out_ptr = out_loc +  col_enteries_out
    tl.store(out_ptr, out, mask=mask)

def blur_x(x: torch.Tensor):
    rows, cols = x.shape
    y = torch.zeros(rows, cols-2, device='cuda')
    BLOCK_SIZE = triton.next_power_of_2(cols-2)
    assert y.is_cuda and x.is_cuda
    triton_blur_x[(rows, 1, 1)](x, y, x.stride(0), x.stride(1),
                                y.stride(0), y.stride(1), rows, cols-2, 
                                BLOCK_SIZE=BLOCK_SIZE)
    return y

def simple_blur_x(x: torch.Tensor):
    rows, cols = x.shape
    y = torch.zeros(rows, cols - 2)
    for i in range(rows):
        for j in range(cols - 2):
            y[i][j] = (x[i][j] + x[i][j+1] + x[i][j+2])/3
    return y

# Blur  Y computed row wise
@triton.jit
def triton_blur_y(x_ptr, y_ptr, x_row_stride,  x_col_stride, 
           y_row_stride, y_col_stride, rows, cols, 
           BLOCK_SIZE:tl.constexpr):
    row_idx = tl.program_id(0) 
    row_start =  x_ptr + row_idx * x_row_stride
    col_enteries = tl.arange(0, BLOCK_SIZE)
    mask = col_enteries < cols
    row_ptr = row_start + col_enteries
    row_1 = tl.load(row_ptr, mask=mask)
    row_ptr_2 = row_start + x_row_stride + col_enteries
    row_2 = tl.load(row_ptr_2, mask=mask)
    row_ptr_3 = row_start + 2*x_row_stride+ col_enteries
    row_3 = tl.load(row_ptr_3, mask=mask)
    out = (row_1 + row_2 + row_3)/3
    out_loc = y_ptr + row_idx * y_row_stride
    col_enteries_out = tl.arange(0, BLOCK_SIZE)
    out_ptr = out_loc +  col_enteries_out
    tl.store(out_ptr, out, mask=mask)

def blur_y(x: torch.Tensor):
    rows, cols = x.shape
    y = torch.zeros(rows-2, cols, device='cuda')
    BLOCK_SIZE = triton.next_power_of_2(cols)
    assert y.is_cuda and x.is_cuda
    triton_blur_y[(rows-2, 1, 1)](x, y, x.stride(0), x.stride(1),
                                y.stride(0), y.stride(1), rows-2, cols, 
                                BLOCK_SIZE=BLOCK_SIZE)
    return y

def simple_blur_y(x: torch.Tensor):
    rows, cols = x.shape
    y = torch.zeros(rows-2, cols)
    for i in range(rows-2):
        for j in range(cols):
            y[i][j] = (x[i][j] + x[i+1][j] + x[i+2][j])/3
    return y



def test_both_blurs():
    test_x = torch.rand(10, 10)
    simple_y = simple_blur_y(test_x)
    test_x_cuda = test_x.to(device='cuda')
    cuda_y = blur_y(test_x_cuda)
    test_y_local = cuda_y.to(device='cpu')
    assert torch.allclose(simple_y, test_y_local) 

    test_x = torch.rand(10, 10)
    simple_y = simple_blur_x(test_x)
    test_x_cuda = test_x.to(device='cuda')
    cuda_y = blur_x(test_x_cuda)
    test_y_local = cuda_y.to(device='cpu')
    assert torch.allclose(simple_y, test_y_local) 

# Blur X --> Blur Y fused computed row wise
@triton.jit
def triton_blur_fused(x_ptr, y_ptr, x_row_stride,  x_col_stride, 
           y_row_stride, y_col_stride, rows, cols, 
           BLOCK_SIZE:tl.constexpr):
    row_idx = tl.program_id(0) 
    row_start =  x_ptr + row_idx * x_row_stride
    col_enteries = tl.arange(0, BLOCK_SIZE)
    mask = col_enteries < cols
    row_ptr = row_start + col_enteries
    row_1 = tl.load(row_ptr, mask=mask)
    row_ptr_2 = row_start + x_col_stride + col_enteries
    row_2 = tl.load(row_ptr_2, mask=mask)
    row_ptr_3 = row_start + 2*x_col_stride+ col_enteries
    row_3 = tl.load(row_ptr_3, mask=mask)
    blur_y_1 = (row_1 + row_2 + row_3)/3


    row_start += x_row_stride 
    row_ptr = row_start + col_enteries
    row_1 = tl.load(row_ptr, mask=mask)
    row_ptr_2 = row_start + x_col_stride + col_enteries
    row_2 = tl.load(row_ptr_2, mask=mask)
    row_ptr_3 = row_start + 2*x_col_stride+ col_enteries
    row_3 = tl.load(row_ptr_3, mask=mask)
    blur_y_2 = (row_1 + row_2 + row_3)/3

    row_start += x_row_stride 
    row_ptr = row_start + col_enteries
    row_1 = tl.load(row_ptr, mask=mask)
    row_ptr_2 = row_start + x_col_stride + col_enteries
    row_2 = tl.load(row_ptr_2, mask=mask)
    row_ptr_3 = row_start + 2*x_col_stride+ col_enteries
    row_3 = tl.load(row_ptr_3, mask=mask)
    blur_y_3 = (row_1 + row_2 + row_3)/3

    out = (blur_y_1 + blur_y_2 + blur_y_3) / 3
    out_loc = y_ptr + row_idx * y_row_stride
    col_enteries_out = tl.arange(0, BLOCK_SIZE)
    out_ptr = out_loc +  col_enteries_out
    tl.store(out_ptr, out, mask=mask)

def blur_fused(x: torch.Tensor):
    rows, cols = x.shape
    y = torch.zeros(rows-2, cols-2, device='cuda')
    BLOCK_SIZE = triton.next_power_of_2(cols-2)
    assert y.is_cuda and x.is_cuda
    triton_blur_fused[(rows-2, 1, 1)](x, y, x.stride(0), x.stride(1),
                                y.stride(0), y.stride(1), rows-2, cols-2, 
                                BLOCK_SIZE=BLOCK_SIZE)
    return y

# Blur X --> Blur Y tiled & fused computed row wise
@triton.jit
def triton_blur_fused_tiled(x_ptr, y_ptr, x_row_stride,  x_col_stride, 
           y_row_stride, y_col_stride, rows, cols, BLOCK_SIZE:tl.constexpr):
    row_idx = tl.program_id(0) 
    col_idx = tl.program_id(1)
    row_start =  x_ptr + row_idx * x_row_stride + (col_idx * BLOCK_SIZE) * x_col_stride
    col_enteries = tl.arange(0, BLOCK_SIZE)
    mask = col_enteries < (cols - col_idx*BLOCK_SIZE)
    row_ptr = row_start + col_enteries
    row_1 = tl.load(row_ptr, mask=mask)
    row_ptr_2 = row_start + x_col_stride + col_enteries
    row_2 = tl.load(row_ptr_2, mask=mask)
    row_ptr_3 = row_start + 2*x_col_stride+ col_enteries
    row_3 = tl.load(row_ptr_3, mask=mask)
    blur_y_1 = (row_1 + row_2 + row_3)/3


    row_start += x_row_stride 
    row_ptr = row_start + col_enteries
    row_1 = tl.load(row_ptr, mask=mask)
    row_ptr_2 = row_start + x_col_stride + col_enteries
    row_2 = tl.load(row_ptr_2, mask=mask)
    row_ptr_3 = row_start + 2*x_col_stride+ col_enteries
    row_3 = tl.load(row_ptr_3, mask=mask)
    blur_y_2 = (row_1 + row_2 + row_3)/3

    row_start += x_row_stride 
    row_ptr = row_start + col_enteries
    row_1 = tl.load(row_ptr, mask=mask)
    row_ptr_2 = row_start + x_col_stride + col_enteries
    row_2 = tl.load(row_ptr_2, mask=mask)
    row_ptr_3 = row_start + 2*x_col_stride+ col_enteries
    row_3 = tl.load(row_ptr_3, mask=mask)
    blur_y_3 = (row_1 + row_2 + row_3)/3

    out = (blur_y_1 + blur_y_2 + blur_y_3) / 3
    out_loc = y_ptr + row_idx * y_row_stride + (col_idx* BLOCK_SIZE) * y_col_stride
    col_enteries_out = tl.arange(0, BLOCK_SIZE)
    out_ptr = out_loc +  col_enteries_out
    tl.store(out_ptr, out, mask=mask)


def blur_fused_tiled(x: torch.Tensor):
    rows, cols = x.shape
    rows_y = rows - 2
    cols_y = cols - 2
    y = torch.zeros(rows_y, cols_y, device='cuda')
    assert x.is_cuda
    grid = lambda meta : (rows_y, triton.cdiv(cols_y, meta['BLOCK_SIZE']),)
    triton_blur_fused_tiled[grid](x, y, x.stride(0), x.stride(1),
                                y.stride(0), y.stride(1), rows-2, cols-2, 
                                BLOCK_SIZE=4096*2)
    return y

def blur_fused_parrallelized(x: torch.Tensor, block_dim=1024):
    rows, cols = x.shape
    rows_y = rows - 2
    cols_y = cols - 2
    y = torch.zeros(rows_y, cols_y, device='cuda')
    assert x.is_cuda
    #Parrallelize across rows
    streams = []
    num_launches = rows_y//block_dim
    leftover = rows_y % block_dim
    for b in range(num_launches):
       s = torch.cuda.Stream()
       streams.append(s)
       i = b*block_dim
       with torch.cuda.stream(s):
            x_block = x[i:i+block_dim+2]
            y_block = y[i:i+block_dim]
            grid = lambda meta : (block_dim, triton.cdiv(cols_y, meta['BLOCK_SIZE']),)
            triton_blur_fused_tiled[grid](x_block, y_block, x_block.stride(0), x_block.stride(1),
                                y_block.stride(0), y_block.stride(1), block_dim-2, cols-2, 
                                BLOCK_SIZE=4096*2)
    if (leftover):
        i = num_launches*block_dim
        s = torch.cuda.Stream()
        streams.append(s)
        x_block = x[i:i+leftover+2]
        y_block = y[i:i+leftover]
        with torch.cuda.stream(s):
            grid = lambda meta : (leftover, triton.cdiv(cols_y, meta['BLOCK_SIZE']),)
            triton_blur_fused_tiled[grid](x_block, y_block, x_block.stride(0), x_block.stride(1),
                                y_block.stride(0), y_block.stride(1), leftover, cols-2, 
                                BLOCK_SIZE=4096*2)
    # wait for all streams to finish
    for s in streams:
        s.synchronize()

    return y



@triton.jit
def query_matrix(M, i, j, stride_i, stride_j, len_i, len_j, 
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    offsets_mi = i + tl.arange(0, BLOCK_I)
    offsets_mj = j + tl.arange(0, BLOCK_J)
    m_ptr = M + offsets_mi[:,None]*stride_i + offsets_mj[None, :] * stride_j
    m_mask = (offsets_mi[:, None] < len_i) & (offsets_mj[None, :] < len_j)
    return tl.load(m_ptr, mask=m_mask, other=0.0)

@triton.jit
def insert_matrix(M, data, i, j, stride_i, stride_j, len_i, len_j, 
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    offsets_mi = i + tl.arange(0, BLOCK_I)
    offsets_mj = j + tl.arange(0, BLOCK_J)
    m_ptr = M + offsets_mi[:,None]*stride_i + offsets_mj[None, :] * stride_j
    m_mask = (offsets_mi[:, None] < len_i) & (offsets_mj[None, :] < len_j)
    tl.store(m_ptr, data, mask=m_mask)

def get_autotune_config():
    block_size = [1024, 512, 256, 128, 64, 16]
    num_warps = [4, 8]
    num_stages = [2, 4, 8]

    fields = ['BLOCK_SIZE_M', 'BLOCK_SIZE_N']
    configs = []

    combinations = product(block_size, repeat=len(fields))
    block_configs = []
    for combination in combinations:
        config = {field: value for field, value in zip(fields, combination)}
        block_configs.append(config)

    for block_config in block_configs:
        for num_warp in num_warps:
            for num_stage in num_stages: 
                block_config['NUM_STAGES'] = num_stage
                configs.append(triton.Config(block_config, num_warps=num_warp))


    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=["rows", "cols"] 
)
@triton.jit
def triton_blur_fused_block(x_ptr, y_ptr, x_row_stride,  x_col_stride, 
                            y_row_stride, y_col_stride, rows, cols, 
                            BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                            NUM_STAGES: tl.constexpr):

    row_idx = tl.program_id(axis=0) * BLOCK_SIZE_M 
    col_idx = tl.program_id(axis=1) * BLOCK_SIZE_N
 
    blur_0_0 = query_matrix(x_ptr, row_idx, col_idx, x_row_stride, x_col_stride,
                            rows, cols, 1, BLOCK_SIZE_N) 
    blur_0_1 = query_matrix(x_ptr, row_idx, col_idx+1, x_row_stride, x_col_stride,
                            rows, cols, 1, BLOCK_SIZE_N) 
    blur_0_2 = query_matrix(x_ptr, row_idx, col_idx+2, x_row_stride, x_col_stride,
                            rows, cols, 1, BLOCK_SIZE_N) 
    blur_0 = (blur_0_0 + blur_0_1 + blur_0_2)/3

    blur_1_0 = query_matrix(x_ptr, row_idx+1, col_idx, x_row_stride, x_col_stride,
                            rows, cols, 1, BLOCK_SIZE_N) 
    blur_1_1 = query_matrix(x_ptr, row_idx+1, col_idx+1, x_row_stride, x_col_stride,
                            rows, cols, 1, BLOCK_SIZE_N) 
    blur_1_2 = query_matrix(x_ptr, row_idx+1, col_idx+2, x_row_stride, x_col_stride,
                            rows, cols, 1, BLOCK_SIZE_N) 
    blur_1 = (blur_1_0 + blur_1_1 + blur_1_2)/3

    for i in tl.range(BLOCK_SIZE_M, num_stages=NUM_STAGES): 
        blur_2_0 = query_matrix(x_ptr, row_idx+i+2, col_idx, x_row_stride, x_col_stride,
                            rows, cols, 1, BLOCK_SIZE_N) 
        blur_2_1 = query_matrix(x_ptr, row_idx+i+2, col_idx+1, x_row_stride, x_col_stride,
                            rows, cols, 1, BLOCK_SIZE_N) 
        blur_2_2 = query_matrix(x_ptr, row_idx+i+2, col_idx+2, x_row_stride, x_col_stride,
                            rows, cols, 1, BLOCK_SIZE_N) 
        blur_2 = (blur_2_0 + blur_2_1 + blur_2_2) / 3

        blur_x = (blur_0 + blur_1 + blur_2) / 3 
        insert_matrix(y_ptr, blur_x, row_idx + i, col_idx, y_row_stride, y_col_stride, rows - 2, cols-2,
                      1, BLOCK_SIZE_N)

        blur_0 = blur_1
        blur_1 = blur_2
         
def blur_fused_blocks(x: torch.Tensor):
    rows, cols = x.shape
    rows_y = rows-2
    cols_y = cols-2 
    y = torch.zeros(rows_y, cols_y, device='cuda')
    assert x.is_cuda
    grid = lambda meta : (triton.cdiv(rows_y, meta['BLOCK_SIZE_M']), 
                          triton.cdiv(cols_y, meta['BLOCK_SIZE_N']))
    triton_blur_fused_block[grid](x, y, x.stride(0), x.stride(1),
                                y.stride(0), y.stride(1), rows, cols, 
    #                            BLOCK_SIZE_M=256,BLOCK_SIZE_N=256, NUM_STAGES=4
                                )
    
    #print(test.asm['ttir'])
    #print(test.asm['ttir'])
    # print("TTGIR", test.asm['ttgir'])
    return y

def blur_unfused(x: torch.Tensor):
    b_x = blur_x(x)
    b_y = blur_y(b_x)
    return b_y

def test_fused_blur():
    test_x = torch.rand(12, 12)
    b_x = simple_blur_x(test_x)
    b_y = simple_blur_y(b_x)

    test_x_cuda = test_x.to(device='cuda')
    test_y_cuda = blur_fused(test_x_cuda)
    test_y = test_y_cuda.to(device='cpu')

    tiled_y_cuda = blur_fused_tiled(test_x_cuda)
    tiled_y_cuda = tiled_y_cuda.to(device='cpu')
    assert torch.allclose(b_y, tiled_y_cuda)

    tiled_y_cuda = blur_fused_parrallelized(test_x_cuda, block_dim=4)
    tiled_y_cuda = tiled_y_cuda.to(device='cpu')

    blocks_y = blur_fused_blocks(test_x_cuda)
    blocks_y_cuda = blocks_y.to(device='cpu')

    assert torch.allclose(b_y, tiled_y_cuda)
    assert torch.allclose(b_y, blocks_y_cuda)
    assert torch.allclose(b_y, test_y)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        #Triton only works for powers of 2!
        x_vals=[128 * i for i in range(2, 42, 2)],
        x_log=True,
        line_arg='system',
        #line_vals=['fused', 'unfused', 'tiled', 'parrallelized', 'blocks'], 
        #line_names=['Fused', 'Unfused', 'Tiled', 'Parrallelized', 'Blocks'],
        line_vals=['blocks'], 
        line_names=['Blocks'],
        ylabel='FLOPS/s',
        plot_name='blur2d.data',
        args={},
    )
)
def benchmark(size, system):
    print(size)
    x = torch.rand(size, size, device='cuda')
    x_local = torch.rand(size, size) 
    quantiles = [0.5, 0.2, 0.8]
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if system == 'fused':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: blur_fused(x), quantiles=quantiles)
    if system == 'unfused':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: blur_unfused(x), quantiles=quantiles)
    if system == 'tiled':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: blur_fused_tiled(x), quantiles=quantiles)
    if system == 'parrallelized':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: blur_fused_parrallelized(x, block_dim=4096), quantiles=quantiles)
    if system == 'blocks':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: blur_fused_blocks(x), quantiles=quantiles)
    perf = lambda ms : 6*(size-2)*size / (ms * 1e-3) 
    gbps = lambda ms : 2*(size-2)*(size-2)*4 / (ms * 1e-3) 
    print(gbps(ms))
    return gbps(ms)

current_path = os.getcwd()
benchmark.run(print_data=True, save_path=f'{current_path}/data/')


# x = torch.rand(18000, 18000, device='cuda')
# blur_fused_blocks(x)
#blur_unfused(x)
#blur_fused(x)
#for i in range(10):
#    blur_fused(x)
#    blur_unfused(x)
#test_fused_blur();