import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE:tl.constexpr):

    pid=tl.program_id(axis=0)

    offsets=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)

    mask=offsets<n_elements

    x=tl.load(x_ptr+offsets,mask=mask)          #这里offsets是list，x_ptr和列表中每个元素依次相加，得到的结果将再返回一个列表.此处的加法就是C++中的指针位移操作
    y=tl.load(y_ptr+offsets,mask=mask)          #load就会遍历返回的列表中的每一个指针然后取出对应的元素，最后返回给x或者y，两者也都是一个列表

    output=x+y

    tl.store(output_ptr+offsets,output,mask=mask)


def add(x:torch.Tensor,y:torch.Tensor):

    output=torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements=output.numel()

    grid=lambda meta:(tl.cdiv(n_elements,meta['BLOCK_SIZE']),)

    add_kernel[grid](x,y,output,n_elements,BLOCK_SIZE=1024)

    return output

