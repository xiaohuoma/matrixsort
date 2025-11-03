
#ifndef _H_PREFIXSUM
#define _H_PREFIXSUM

#include <stdio.h>
#include "hunyuangraph_GPU_memory.h"

// GPU memory
size_t max_memory_GPU = 0;
size_t now_memory_GPU = 0;
size_t freeMem, totalMem, usableMem;

void GPU_Memory()
{
	// 获取设备属性
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	// 检测当前程序执行前显存已使用的大小
	cudaMemGetInfo(&freeMem, &totalMem);
	usableMem = totalMem - freeMem;

	printf("Total GPU Memory:                   %zu B %zu KB %zu MB %zu GB\n", totalMem, totalMem / 1024,totalMem / 1024 / 1024,totalMem / 1024 / 1024 / 1024);
    printf("Usable GPU Memory before execution: %zu B %zu KB %zu MB %zu GB\n", freeMem, freeMem / 1024,freeMem / 1024 / 1024,freeMem / 1024 / 1024 / 1024);
}

void malloc_GPU(size_t space)
{
    if(now_memory_GPU + space > freeMem)
    {
        printf("-----don't have enough GPU memory   %zu+%zu=%zu>%zu\n",now_memory_GPU,space,now_memory_GPU + space, freeMem);
        return ;
    }
    now_memory_GPU += space;
    if(now_memory_GPU > max_memory_GPU)
        max_memory_GPU = now_memory_GPU;
    printf("-----now used GPU memory            %zu B\n",now_memory_GPU);
}

void free_GPU(size_t space)
{
    if(now_memory_GPU - space < 0)
    {
        printf("-----too much GPU memory is freed   %zu-%zu=%zu<0\n",now_memory_GPU,space,now_memory_GPU - space);
        return ;
    }
    now_memory_GPU -= space;
    if(now_memory_GPU > max_memory_GPU)
        max_memory_GPU = now_memory_GPU;
    printf("-----now used GPU memory            %zu B\n",now_memory_GPU);
}

__device__ int prefixsum_warp(int val, int lane)
{
	int temp = __shfl_up_sync(0xffffffff, val, 1);
	if(lane >= 1)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 2);
	if(lane >= 2)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 4);
	if(lane >= 4)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 8);
	if(lane >= 8)
		val += temp;
	temp = __shfl_up_sync(0xffffffff, val, 16);
	if(lane >= 16)
		val += temp;

	return val;
}

__device__ int prefixsum_block(int val)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    extern __shared__ int warp_sum[];

    val = prefixsum_warp(val,lane);
    __syncthreads();

    if(lane == 31)
        warp_sum[warp_id] = val;
    __syncthreads();

    if(warp_id == 0)
        warp_sum[lane] = prefixsum_warp(warp_sum[lane],lane);
    __syncthreads();

    if(warp_id > 0)
        val += warp_sum[warp_id - 1];

    return val;
}

__global__ void firstprefixsum(int *num, int *firstresult, int length, int blocknum)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;

    int val;
    if(ii < length)
        val = num[ii];
    else 
        val = 0;

    val = prefixsum_block(val);
    __syncthreads();

    if(ii < length)
        num[ii] = val;
    if(threadIdx.x == blockDim.x - 1)
        firstresult[blockIdx.x] = val;
}

__global__ void prefixsum_communication_template(int *endresult, int *secondresult, int length)
{
	int ii  = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < length)
	{
		int block_num = blockIdx.x;
		if (block_num != 0)
            endresult[ii] += secondresult[block_num - 1];
	}
}

__global__ void print_num(int *num, int length)
{
    for(int i = 0;i < 10 && i < length;i++)
        printf("%10d",num[i]);
    printf("\n");
}

/*
num:            1,1,1,1,1,...1(260)
num_out:        1,2,3,4,5,...,256,1,2,3,4(260)
firstresult:    256,4
secondresult:   256,260
*/
void prefixsum(int *num, int *num_out, int length, size_t blocksize, bool is_right)
{
    size_t blocknum  = (length + blocksize - 1) / blocksize;
    size_t sharesize = 32 * sizeof(int);

    if(blocknum == 0)
        return ;

    int *firstresult;
    // malloc_GPU(blocknum * sizeof(int));
    // printf("malloc firstresult length=%d\n",length);
    // cudaMalloc((void**)&firstresult,blocknum * sizeof(int));
    if(is_right) 
        firstresult = (int *)rmalloc_with_check(sizeof(int) * blocknum,"prefixsum: firstresult");
    else
        firstresult = (int *)lmalloc_with_check(sizeof(int) * blocknum,"prefixsum: firstresult");

    firstprefixsum<<<blocknum,blocksize,sharesize>>>(num,firstresult,length,blocknum);

	if(blocknum == 1)
	{
		// *num_out = endresult;
        // free_GPU(blocknum * sizeof(int));
        // printf("free firstresult length=%d\n",length);
        // cudaFree(firstresult);
        if(is_right)
            rfree_with_check((void *)firstresult, sizeof(int) * blocknum,"prefixsum: firstresult");
        else 
            lfree_with_check((void *)firstresult, sizeof(int) * blocknum,"prefixsum: firstresult");
        return ;
	}

	// do the second scan after the first scan
	int *secondresult = firstresult;
	prefixsum(firstresult,firstresult,blocknum,blocksize,is_right);

	// copy current result
    prefixsum_communication_template<<<blocknum, blocksize>>>(num, secondresult, length);
    cudaDeviceSynchronize();

    // free_GPU(blocknum * sizeof(int));
    // printf("free firstresult length=%d\n",length);
    // cudaFree(firstresult);
    if(is_right)
        rfree_with_check((void *)firstresult, sizeof(int) * blocknum,"prefixsum: firstresult");
    else 
        lfree_with_check((void *)firstresult, sizeof(int) * blocknum,"prefixsum: firstresult");
}

#endif