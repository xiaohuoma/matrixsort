#ifndef _H_GPU_MEMORY
#define _H_GPU_MEMORY

/*****************************		Attention about cudaMalloc!!!				*****************************
**********由于多级算法的对内存的要求与栈的性质完美的重合，因此本算法中涉及到的显存管理是以栈为基础的	   **********
**********由于算法中使用到了一些临时数组，仅使用单端栈的话会导致显存的浪费以及显存碎片化，			      **********
**********		因此本算法使用了双端栈的概念，左端存放需要保留的图的信息，右端存放临时数组				  **********
**********若利用本算法特定的显存管理方式，请注意“先申请后释放，后申请先释放”的特性						 **********
*/
// 左端栈先压入了cu_bn,cu_bt,cu_g,cu_csr,cu_que(cuMetis_10 yes/now no)
// 左端栈存放的图的信息从左至右依次为vwgt,xadj,adjncy,adjwgt,cmap,where,bnd,pwgts,tpwgts,maxwgt,minwgt,bndnum
// 粗化阶段右端栈存放的图的信息从右至左依次为match,txadj.tadjncy,tadjwgt,temp_scan

#include "hunyuangraph_define.h"
#include "hunyuangraph_timer.h"

/*pointer*/
int GPU_Memory_Pool = 1;	//	Hunyuan's GPU memory pool is enabled by default
char *deviceMemory;
char *front_pointer;
char *back_pointer;
char *lmove_pointer;
char *rmove_pointer;
char *tmove_pointer;
size_t used_by_me_now = 0;
size_t used_by_me_max = 0;
size_t lused = 0;
size_t rused = 0;
int prefixsum_blocksize = 256;
int is_mandatory_space = 0;
int lmandatory_space_pointer;
int rmandatory_space_pointer;

void Init_GPU_Memory(size_t remainingMem)
{
	front_pointer = deviceMemory;
	back_pointer = (char *)deviceMemory + remainingMem;
	lmove_pointer = front_pointer;
	rmove_pointer = back_pointer;
}

void Malloc_GPU_Memory(size_t nvtxs, size_t nedges)
{
    // 获取设备属性
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // 检测当前程序执行前显存已使用的大小
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    // size_t usableMem = totalMem - freeMem;

    // 计算剩余的部分显存大小
    // size_t remainingMem = freeMem - nedges * 3 * sizeof(int);
	size_t remainingMem = freeMem / 5 * 4;

    // 对齐缓存行
    if (remainingMem % hunyuangraph_GPU_cacheline != 0)
        remainingMem = remainingMem - remainingMem % hunyuangraph_GPU_cacheline;

    // 分配显存空间
    cudaError_t err = cudaMalloc(&deviceMemory, remainingMem);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }

    // 初始化内存指针
    Init_GPU_Memory(remainingMem);

    printf("Total GPU Memory:                   %zuB %zuKB %zuMB %zuGB\n", totalMem, totalMem / 1024, totalMem / 1024 / 1024, totalMem / 1024 / 1024 / 1024);
    printf("Usable GPU Memory before execution: %zuB %zuKB %zuMB %zuGB\n", freeMem, freeMem / 1024, freeMem / 1024 / 1024, freeMem / 1024 / 1024 / 1024);
    printf("Malloc GPU Memory:                  %zuB %zuKB %zuMB %zuGB\n", remainingMem, remainingMem / 1024, remainingMem / 1024 / 1024, remainingMem / 1024 / 1024 / 1024);
    printf("front_pointer=  %p\n", front_pointer);
    printf("back_pointer=   %p\n", back_pointer);
}

void Free_GPU_Memory()
{
	printf("lmove_pointer=  %p\n", lmove_pointer);
	printf("rmove_pointer=  %p\n", rmove_pointer);
	if (lmove_pointer == front_pointer)
		printf("The left stack has been freed\n");
	else
		printf("error ------------ The left stack hasn't been freed\n");
	if (rmove_pointer == back_pointer)
		printf("The right stack has been freed\n");
	else
		printf("error ------------ The right stack hasn't been freed\n");
	printf("Max memory used of GPU: %10ldB %10ldKB %10ldMB %10ldGB\n", used_by_me_max, used_by_me_max / 1024,
		   used_by_me_max / 1024 / 1024, used_by_me_max / 1024 / 1024 / 1024);
	cudaFree(deviceMemory);
}

// 申请和释放空间时参数为所需字节数
// l..为左端栈操作，r..为右端栈操作
void *lmalloc_with_check(size_t size, char *infor)
{
	if (size < 0)
		printf("error: %s is lmalloc_with_check( %d < 0)\n", infor, size);
	
	void *malloc_address = NULL;

	size_t used_size;
	if (size % hunyuangraph_GPU_cacheline != 0)
		used_size = size + hunyuangraph_GPU_cacheline - size % hunyuangraph_GPU_cacheline;
	else
		used_size = size;

	tmove_pointer = lmove_pointer + used_size;
	lused += used_size;
	// printf("lmove_pointer=%p\n",lmove_pointer);
	// printf("rmove_pointer=%p\n",rmove_pointer);
	// printf("size=         %d\n",size);
	// printf("tmove_pointer=%p\n",tmove_pointer);

	if (tmove_pointer > rmove_pointer)
	{
		printf("error ------------ don't have enough GPU memory %s, now uesd memory %zuB\n", infor, used_by_me_now);
		exit(0);
	}
	else
	{
		malloc_address = lmove_pointer;
		lmove_pointer = tmove_pointer;
		used_by_me_now += used_size;
		used_by_me_max = hunyuangraph_max(used_by_me_max, used_by_me_now);
	}

	// printf("lmalloc_with_check:%s\n",infor);
	// printf("used memory=    %10ld\n",used_by_me_now);
	// printf("malloc_address= %p\n",malloc_address);
	
#ifdef MEMORY_CHECK
	printf("lmalloc lmove_pointer=%p size=%10d used_size=%10d %s\n", lmove_pointer, size, used_size, infor);
#endif

	// printf("rmove_pointer=  %p lmalloc\n",rmove_pointer);
	// printf("available space %zuKB %zuMB %zuGB\n",(rmove_pointer - lmove_pointer) / 1024,(rmove_pointer - lmove_pointer) / 1024 / 1024,(rmove_pointer - lmove_pointer) / 1024 / 1024 / 1024);
	// printf("\n");

	return malloc_address;
}

void *rmalloc_with_check(size_t size, char *infor)
{
	if (size < 0)
		printf("error: %s is rmalloc_with_check( %d < 0)\n", infor, size);

	void *malloc_address = NULL;

	size_t used_size;
	if (size % hunyuangraph_GPU_cacheline != 0)
		used_size = size + hunyuangraph_GPU_cacheline - size % hunyuangraph_GPU_cacheline;
	else
		used_size = size;

	tmove_pointer = rmove_pointer - used_size;
	rused += used_size;
	// printf("lmove_pointer=%p\n",lmove_pointer);
	// printf("rmove_pointer=%p\n",rmove_pointer);
	// printf("size=         %d\n",size);
	// printf("tmove_pointer=%p\n",tmove_pointer);

	if (tmove_pointer < lmove_pointer)
	{
		printf("error ------------ don't have enough GPU memory %s, now uesd memory %zuB\n", infor, used_by_me_now);
		exit(0);
	}
	else
	{
		malloc_address = tmove_pointer;
		rmove_pointer = tmove_pointer;
		used_by_me_now += used_size;
		used_by_me_max = hunyuangraph_max(used_by_me_max, used_by_me_now);
	}

	// printf("rmalloc_with_check:%s\n",infor);
	// printf("used memory=    %10ld\n",used_by_me_now);
	// printf("malloc_address= %p\n",malloc_address);
	// printf("lmove_pointer=  %p\n",lmove_pointer);

#ifdef MEMORY_CHECK
	printf("rmalloc rmove_pointer=%p size=%10d used_size=%10d %s\n", rmove_pointer, size, used_size, infor);
#endif

	// printf("available space %zuKB %zuMB %zuGB\n",(rmove_pointer - lmove_pointer) / 1024,(rmove_pointer - lmove_pointer) / 1024 / 1024,(rmove_pointer - lmove_pointer) / 1024 / 1024 / 1024);
	// printf("\n");

	return malloc_address;
}

void *lfree_with_check(void *malloc_address, size_t size, char *infor)
{
	if (size < 0)
		printf("lfree_with_check( size < 0)\n");
	
	size_t used_size;
	if (size % hunyuangraph_GPU_cacheline != 0)
		used_size = size + hunyuangraph_GPU_cacheline - size % hunyuangraph_GPU_cacheline;
	else
		used_size = size;

	tmove_pointer = lmove_pointer - used_size;
	lused -= used_size;

	// printf("lmove_pointer=%p\n",lmove_pointer);
	// printf("rmove_pointer=%p\n",rmove_pointer);
	// printf("size=         %d\n",size);
	// printf("tmove_pointer=%p\n",tmove_pointer);

	if (tmove_pointer < front_pointer)
	{
		printf("error ------------ don't lmalloc enough GPU memory %s, now uesd memory %zuB\n", infor, used_by_me_now);
		exit(0);
	}
	else
	{
		lmove_pointer = tmove_pointer;
		used_by_me_now -= used_size;
		used_by_me_max = hunyuangraph_max(used_by_me_max, used_by_me_now);
	}

	// printf("lfree_with_check:%s\n",infor);
	// printf("used memory=    %10ld\n",used_by_me_now);
	// printf("lmove_pointer= %p\n",lmove_pointer);

#ifdef MEMORY_CHECK
	printf("lfree   lmove_pointer=%p size=%10d used_size=%10d %s\n", lmove_pointer, size, used_size, infor);
#endif
	
	// printf("rmove_pointer= %p\n",rmove_pointer);
	// printf("available space %zuKB %zuMB %zuGB\n",(rmove_pointer - lmove_pointer) / 1024,(rmove_pointer - lmove_pointer) / 1024 / 1024,(rmove_pointer - lmove_pointer) / 1024 / 1024 / 1024);
	// printf("\n");

	malloc_address = NULL;

	return malloc_address;
}

void *rfree_with_check(void *malloc_address, size_t size, char *infor)
{
	if (size < 0)
		printf("rfree_with_check( size < 0)\n");

	size_t used_size;
	if (size % hunyuangraph_GPU_cacheline != 0)
		used_size = size + hunyuangraph_GPU_cacheline - size % hunyuangraph_GPU_cacheline;
	else
		used_size = size;

	tmove_pointer = rmove_pointer + used_size;
	rused -= used_size;

	// printf("lmove_pointer=%p\n",lmove_pointer);
	// printf("rmove_pointer=%p\n",rmove_pointer);
	// printf("size=         %d\n",size);
	// printf("tmove_pointer=%p\n",tmove_pointer);

	if (tmove_pointer > back_pointer)
	{
		printf("error ------------ don't rmalloc enough GPU memory %s, now uesd memory %zuB\n", infor, used_by_me_now);
		exit(0);
	}
	else
	{
		rmove_pointer = tmove_pointer;
		used_by_me_now -= used_size;
		used_by_me_max = hunyuangraph_max(used_by_me_max, used_by_me_now);
	}

	// printf("rfree_with_check:%s\n",infor);
	// printf("used memory=    %10ld\n",used_by_me_now);
	// printf("lmove_pointer=  %p\n",lmove_pointer);
	// printf("rmove_pointer=  %p\n",rmove_pointer);

#ifdef MEMORY_CHECK
	printf("rfree   rmove_pointer=%p size=%10d used_size=%10d %s\n", rmove_pointer, size, used_size, infor);
#endif
	
	// printf("available space %zuKB %zuMB %zuGB\n",(rmove_pointer - lmove_pointer) / 1024,(rmove_pointer - lmove_pointer) / 1024 / 1024,(rmove_pointer - lmove_pointer) / 1024 / 1024 / 1024);
	// printf("\n");
	
	malloc_address = NULL;
	
	return malloc_address;
}

//	Memory after X Byte is forcibly applied for
//	Note that this application is not protected
void *lmalloc_with_mandatory_space(size_t size, size_t mandatory_space, char *infor)
{
	if (size < 0)
		printf("error: %s is lmalloc_with_mandatory_space( %d < 0)\n", infor, size);
	
	if(mandatory_space < 0)
		printf("error: %s is lmalloc_with_mandatory_space: Spacing distance( %d < 0)\n", infor, size);
	else if(mandatory_space == 0)
		printf("error: %s is lmalloc_with_mandatory_space: Spacing distance( %d == 0)\n", infor, size);
	
	void *malloc_address = NULL;

	size_t used_size;
	if (size % hunyuangraph_GPU_cacheline != 0)
		used_size = size + hunyuangraph_GPU_cacheline - size % hunyuangraph_GPU_cacheline;
	else
		used_size = size;
	//	The default mandatory_space is an integer multiple of hunyuangraph_GPU_cacheline
	tmove_pointer = lmove_pointer + mandatory_space + used_size;

	if (tmove_pointer > rmove_pointer)
	{
		printf("error ------------ don't have enough GPU memory %s, now uesd memory %zuB\n", infor, used_by_me_now);
		exit(0);
	}
	else
	{
		malloc_address = lmove_pointer + mandatory_space;
	}

#ifdef MEMORY_CHECK
	printf("lmalloc_with_mandatory_space lmalloc_mandatory_pointer=%p size=%10d used_size=%10d %s\n", malloc_address, size, used_size, infor);
#endif

	return malloc_address;
}

void *record_lmove_pointer()
{
	// printf("lmove_pointer=%p used_by_me_now=%d lused=%d\n", lmove_pointer, used_by_me_now, lused);
#ifdef MEMORY_CHECK
	printf("record_lmove_pointer\n");
#endif

	return lmove_pointer;
}

void *record_rmove_pointer()
{
	// printf("rmove_pointer=%p used_by_me_now=%d rused=%d\n", rmove_pointer, used_by_me_now, rused);
#ifdef MEMORY_CHECK
	printf("record_rmove_pointer\n");
#endif

	return rmove_pointer;
}

void return_lmove_pointer(void *lpointer)
{
	int less = lmove_pointer - (char *)lpointer;
	// printf("less=%d lmove_pointer=%p lpointer=%p\n", less, lmove_pointer, lpointer);
	lmove_pointer = (char *)lpointer;
	used_by_me_now -= less;
	lused -= less;
	// printf("used_by_me_now=%d lused=%d\n", used_by_me_now, lused);

#ifdef MEMORY_CHECK
	printf("return_lmove_pointer\n");
#endif

}

void return_rmove_pointer(void *rpointer)
{
	int less = (char *)rpointer - rmove_pointer;
	// printf("less=%d rmove_pointer=%p rpointer=%p\n", less, rmove_pointer, rpointer);
	rmove_pointer = (char *)rpointer;
	used_by_me_now -= less;
	rused -= less;
	// printf("used_by_me_now=%d rused=%d\n", used_by_me_now, rused);

#ifdef MEMORY_CHECK
	printf("return_rmove_pointer\n");
#endif

}

// 左端栈先压入了cu_bn,cu_bt,cu_g,cu_csr,cu_que(cuMetis_10 yes/now no)
// 左端栈存放的图的信息从左至右依次为vwgt,xadj,adjncy,adjwgt,cmap,where,bnd,pwgts,tpwgts,maxwgt,minwgt,bndnum
// 粗化阶段右端栈存放的图的信息从右至左依次为match,txadj,tadjncy,tadjwgt,temp_scan

#endif