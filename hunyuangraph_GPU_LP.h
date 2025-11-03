#ifndef _H_GPU_LP
#define _H_GPU_LP

#include "hunyuangraph_struct.h"

__global__ void init_moveto(int *where, int *moveto, int nvtxs)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < nvtxs)
  {
    int t = where[ii];
    moveto[ii] = t;
  }
}

__global__ void update_moveto(int nvtxs, int nedges, int *xadj, int *adjncy, int *adjwgt, int *where, int *moveto, int *list, int *gainv)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < nvtxs)
  {
    int i, j, ewj, wj, id, ed, g, moveto_j;
    int me, other, begin, end;
    me    = where[ii];
    other = (me + 1) % 2;
    begin = xadj[ii];
    end   = xadj[ii + 1];
    id    = 0;
    ed    = 0; 
    
    for(i = begin;i < end;i++)
    {
      j   = adjncy[i];
      ewj = adjwgt[i];
      wj  = where[j];

      if(me == wj) id += ewj;
      else ed += ewj;
    }

    g = ed - id;
    gainv[ii] = g;

    // 怎么筛，筛完怎么记录，要不要记录gainv
      // 移动�?1，不移动�?0
    // if(g > -0.25 * id) printf("ii1=%d\n",ii), list[ii] = 1, moveto[ii] = other;
    if(g > -0.25 * id) list[ii] = 1, moveto[ii] = other;
    else list[ii] = 0;
    // printf("ii1=%d moveto=%d\n",ii,moveto[ii]);

    __syncthreads();  //什么同步？？？！！！！

    // 晒第二遍 怎么�?
    if(g > -0.25 * id)
    {
      g = 0;
      for(i = begin;i < end;i++)
      {
        j   = adjncy[i];
        ewj = adjwgt[i];
        wj  = where[j];
        moveto_j = moveto[j];

        if(list[j] == 1 && (gainv[j] > gainv[ii] || gainv[j] == gainv[ii] && j < ii))
          moveto_j = wj;
        if(moveto_j == other) g += ewj;
        else g -= ewj;
      }

      // 需要吗�?
      // gainv[ii] = g;

      // if(g > -0.25 * id) printf("ii2yes=%d\n",ii);
      // else printf("ii2=%d me=%d to=%d\n",ii,me,other), list[ii] = 0, moveto[ii] = me;
      if(g > -0.25 * id) ;
      else list[ii] = 0, moveto[ii] = me;
    }

    __syncthreads();

    // 移动顶点，上面的操作对每层图要进行一次还是多次，什么时候结束，怎么判断 
      //先只筛一�?
    if(list[ii] == 1) where[ii] = moveto[ii];

    // __syncthreads();

    // 如何使得分区平衡 

    /*for(i = begin;i < end;i++)
    {
      j   = adjncy[i];
      ewj = adjwgt[i];
      wj  = where[j];

      if(me == wj) id += ewj;
      else ed += ewj;
    }*/ // 已存放到gainv�?

  }
}

__global__ void filter_first(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *moveto, int *list, int *gainv)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, ewj, wj, id, ed, g;
		int me, other, begin, end;
		me    = where[ii];
		other = (me + 1) % 2;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		id    = 0;
		ed    = 0; 
		
		for(i = begin;i < end;i++)
		{
			j   = adjncy[i];
			ewj = adjwgt[i];
			wj  = where[j];

			if(me == wj) id += ewj;
			else ed += ewj;
		}

		g = ed - id;
		gainv[ii] = g;

		// 怎么筛，筛完怎么记录，要不要记录gainv
		// 移动�?1，不移动�?0
		if(g > -0.25 * id) list[ii] = 1, moveto[ii] = other;
		else list[ii] = 0;

	}
}

__global__ void filter_second(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *moveto, int *list, int *gainv)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, ewj, wj, l, g0, g, gj, moveto_j;
		int me, other, begin, end;
		me    = where[ii];
		other = (me + 1) % 2;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		l     = list[ii];
		g0    = gainv[ii];

		// 晒第二遍 怎么�?
		if(l)
		{
			g = 0;
			for(i = begin;i < end;i++)
			{
				j   = adjncy[i];
				ewj = adjwgt[i];
				wj  = where[j];
				gj  = gainv[j];
				moveto_j = moveto[j];

				if(list[j] == 1 && (gj > g0 || gj == g0 && j < ii))
					moveto_j = wj;
				if(moveto_j == other) g += ewj;
				else g -= ewj;
			}

			if(g > 0) where[ii] = other;
			else list[ii] = 0, moveto[ii] = me;
		}
	}
}

__global__ void filter_first_atomic(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *moveto, int *list, int *gainv)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, ewj, wj, id, ed, g;
		int me, other, begin, end;
		me    = where[ii];
		other = (me + 1) % 2;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		id    = 0;
		ed    = 0; 
		
		for(i = begin;i < end;i++)
		{
			j   = adjncy[i];
			ewj = adjwgt[i];
			wj  = where[j];

			if(me == wj) id += ewj;
			else ed += ewj;
		}

		g = ed - id;
		gainv[ii] = g;

		// 怎么筛，筛完怎么记录，要不要记录gainv
		// 移动�?1，不移动�?0
		if(g > -0.25 * id) list[ii] = 1, moveto[ii] = other;
		else list[ii] = 0;

	}
}

__global__ void filter_second_atomic(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *moveto, int *list, int *gainv)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, ewj, wj, l, g0, g, gj, moveto_j;
		int me, other, begin, end;
		me    = where[ii];
		other = (me + 1) % 2;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		l     = list[ii];
		g0    = gainv[ii];

		// 晒第二遍 怎么�?
		if(l)
		{
			g = 0;
			for(i = begin;i < end;i++)
			{
				j   = adjncy[i];
				ewj = adjwgt[i];
				wj  = where[j];
				gj  = gainv[j];
				moveto_j = moveto[j];

				if(list[j] == 1 && (gj > g0 || gj == g0 && j < ii))
					moveto_j = wj;
				if(moveto_j == other) g += ewj;
				else g -= ewj;
			}

			if(g > 0) where[ii] = other;
			else list[ii] = 0, moveto[ii] = me;
		}
	}
}

__global__ void filter_atomic(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, me, other, k, wk, ptr, begin, end, ed, id;
		// for(i = ii;i < nvtxs;i += 1024)
		// {
			ptr = ii;
			// if(ptr >= nvtxs) break;
				
			begin = xadj[ptr];
			end   = xadj[ptr + 1];
			me    = where[ptr];
			other = (me + 1) % 2;
			ed = 0;
			id = 0;

			for(j = begin;j < end;j++)
			{
				k  = adjncy[j];
				wk = where[k];

				if(wk == me) ed += adjwgt[j];
				else id += adjwgt[j];
			}

			// printf("me=%d other=%d\n",me,other);
			// if(me + other == 1 && me != other) ;
			// else printf("No\n");
			// if(ed > id) atomicExch(&where[ptr], other);
			if(ed <= id) 
			{
				int oldVal = atomicExch(&where[ptr], other);
			}
			// if(ed > id) 
			// {
			// 	if(me == 0) where[ptr] = 1;
			// 	else where[ptr] = 0;
			// }
		// }
	}
}

__global__ void filter_atomic_group(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

	// if(ii < nvtxs)
	// {
		int i, j, me, other, k, wk, ptr, begin, end, ed, id;
		for(i = ii;i < nvtxs;i += 8)
		{
			ptr = i;
			if(ptr >= nvtxs) break;
				
			begin = xadj[ptr];
			end   = xadj[ptr + 1];
			me    = where[ptr];
			other = (me + 1) % 2;
			ed = 0;
			id = 0;

			for(j = begin;j < end;j++)
			{
				k  = adjncy[j];
				wk = where[k];

				if(wk == me) ed += adjwgt[j];
				else id += adjwgt[j];
			}

			// if(me + other == 1 && me != other) ;
			// else printf("No\n");
			// if(ed > id) atomicExch(&where[ptr], me);
			if(ed <= id) 
			{
				int oldVal = atomicExch(&where[ptr], other);
			}

		}
	// }
}

__global__ void update_moveto_atomic(int nvtxs, int nedges, int *xadj, int *adjncy, int *adjwgt, int *where, int *moveto, int *list, int *gainv)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < nvtxs)
  {
    int i, j, ewj, wj, id, ed, g, moveto_j;
    int me, other, begin, end;
    me    = where[ii];
    other = (me + 1) % 2;
    begin = xadj[ii];
    end   = xadj[ii + 1];
    id    = 0;
    ed    = 0; 
    
    for(i = begin;i < end;i++)
    {
      j   = adjncy[i];
      ewj = adjwgt[i];
      wj  = where[j];

      if(me == wj) id += ewj;
      else ed += ewj;
    }

    g = ed - id;
    gainv[ii] = g;

    // 怎么筛，筛完怎么记录，要不要记录gainv
      // 移动�?1，不移动�?0
    // if(g > -0.25 * id) printf("ii1=%d\n",ii), list[ii] = 1, moveto[ii] = other;
    if(g > -0.25 * id) list[ii] = 1, moveto[ii] = other;
    else list[ii] = 0;
    // printf("ii1=%d moveto=%d\n",ii,moveto[ii]);

    __syncthreads();

    // 晒第二遍 怎么�?
    if(g > -0.25 * id)
    {
      g = 0;
      for(i = begin;i < end;i++)
      {
        j   = adjncy[i];
        ewj = adjwgt[i];
        wj  = where[j];
        moveto_j = moveto[j];

        if(list[j] > 1 && (gainv[j] > gainv[ii] || gainv[j] == gainv[ii] && j < ii))
          moveto_j = wj;
        if(moveto_j == other) g += ewj;
        else g -= ewj;
      }

      // 需要吗�?
      // gainv[ii] = g;

      // if(g > -0.25 * id) printf("ii2yes=%d\n",ii);
      // else printf("ii2=%d me=%d to=%d\n",ii,me,other), list[ii] = 0, moveto[ii] = me;
      if(g > -0.25 * id) ;
      else list[ii] = 0, moveto[ii] = me;
    }

    __syncthreads();

    // 移动顶点，上面的操作对每层图要进行一次还是多次，什么时候结束，怎么判断 
      //先只筛一�?
    if(list[ii] == 1) where[ii] = moveto[ii];

  }
}

__global__ void compute_pwgts(int nvtxs, int *vwgt, int *where, int *gainv)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(where[ii] == 0) gainv[ii] = vwgt[ii];
		else gainv[ii] = 0;
	}
}

__device__ void warpReduction6(volatile int *share_num, int tid ,int blocksize)
{
    if(blocksize >= 64) share_num[tid] += share_num[tid + 32];
    if(blocksize >= 32) share_num[tid] += share_num[tid + 16];
    if(blocksize >= 16) share_num[tid] += share_num[tid + 8];
    if(blocksize >= 8) share_num[tid] += share_num[tid + 4];
    if(blocksize >= 4) share_num[tid] += share_num[tid + 2];
    if(blocksize >= 2) share_num[tid] += share_num[tid + 1];
}

__global__ void reduction6(int *num, int length)
{
    int ii  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ int share_num[];

    if(ii + blockDim.x < length) share_num[tid] = num[ii] + num[ii + blockDim.x];
    else if(ii < length) share_num[tid] = num[ii];
    else share_num[tid] = 0;

     __syncthreads();

    if(ii < length)
    {
        if(blockDim.x >= 512) 
        {
            if(tid < 256) share_num[tid] += share_num[tid + 256];
            __syncthreads();
        }
        if(blockDim.x >= 256) 
        {
            if(tid < 128) share_num[tid] += share_num[tid + 128];
            __syncthreads();
        }
        if(blockDim.x >= 128) 
        {
            if(tid < 64) share_num[tid] += share_num[tid + 64];
            __syncthreads();
        }

        if(tid < 32) warpReduction6(share_num, tid, blockDim.x);

        if(tid == 0) num[blockIdx.x] = share_num[0];
    }
}

__global__ void compute_gainkp(int nvtxs, int nedges, int from, int *xadj, int *adjncy, int *adjwgt, int *where, kp_t *kp)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, ewj, wj, id, ed, g;
		int me, begin, end;
		me    = where[ii];
		begin = xadj[ii];
		end   = xadj[ii + 1];
		id    = 0;
		ed    = 0; 
		
		for(i = begin;i < end;i++)
		{
			j   = adjncy[i];
			ewj = adjwgt[i];
			wj  = where[j];

			if(me == wj) id += ewj;
			else ed += ewj;
		}

		g = ed - id;
		if(from == me) kp[ii].key = g;
		else kp[ii].key = -nedges;
		kp[ii].ptr = ii;
	}
}

struct compRule
{
	__host__ __device__ bool operator()(const kp_t &p1,const kp_t &p2)
	{
		if (p1.key != p2.key)
			return p1.key > p2.key;
		else
      return p1.ptr < p2.ptr;
	}
};

__global__ void compute_gainvkp(int nvtxs, int *vwgt, kp_t *kp, int *gainv)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < nvtxs)
  {
    int ptr = kp[ii].ptr;
    gainv[ii] = vwgt[ptr];
  }
}

__global__ void rebalancekp(int nvtxs, int num, int to, int *where, kp_t *kp, int *gainv)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < nvtxs)
  {
    int ptr, prefixsum;
    ptr = kp[ii].ptr;
    prefixsum = gainv[ii];
    if(prefixsum <= num) where[ptr] = to;
  }
}

__global__ void exam_csr(int nvtxs, int *xadj, int *adjncy)
{
	for (int i = 0; i <= nvtxs; i++)
		printf("%d ", xadj[i]);

	printf("\nadjncy:\n");
	for (int i = 0; i < nvtxs; i++)
	{
		for (int j = xadj[i]; j < xadj[i + 1]; j++)
			printf("%d ", adjncy[j]);
		printf("\n");
	}
}

__global__ void exam_where(int nvtxs, int *where)
{
	for(int i = 0;i < nvtxs;i++)
		printf("%d ",where[i]);
	printf("\n");
}

__global__ void exam_gain_gainptr(int nvtxs, int *gain, int *gain_ptr)
{
  printf("gain:");
  for(int i = 0;i < nvtxs;i++)
    printf("%d ",gain[i]);
  printf("\n");
  printf("gain_ptr:");
  for(int i = 0;i < nvtxs;i++)
    printf("%d ",gain_ptr[i]);
  printf("\n");
}

__global__ void exam_kp(int nvtxs, kp_t *kp)
{
  printf("gain:");
  for(int i = 0;i < nvtxs;i++)
    printf("%d ",kp[i].key);
  printf("\n");
  printf("gain_ptr:");
  for(int i = 0;i < nvtxs;i++)
    printf("%d ",kp[i].ptr);
  printf("\n");
}

int compute_edgecut(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where)
{
  int edgecut, me;
  edgecut = 0;
  for(int i = 0;i < nvtxs;i++)
  {
    me =  where[i];
    for(int j = xadj[i];j < xadj[i + 1];j++)
      if(where[adjncy[j]] != me) edgecut += adjwgt[j];
  }
  return edgecut / 2;
}

__global__ void calculate_edgecut(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *temp)
{
	/*int iii = threadIdx.x;
	int ii  = blockIdx.x * 4 + iii / 32;
	int tid = iii % 32;

	extern __shared__ int cache_d[128];
	cache_d[iii] = 0;

	if (ii < nvtxs)
	{
		int begin, end, me, i, j;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		me    = where[ii];

		for(i = begin + tid;i < end;i += 32)
		{
			j = adjncy[i];
			if(where[j] != me) cache_d[iii] += adjwgt[i];
		}
	}
	__syncthreads();

	// Block内求和
	if(iii < 64) cache_d[iii] += cache_d[iii + 64];
	__syncthreads();
	if(iii < 32) cache_d[iii] += cache_d[iii + 32];
	__syncthreads();
	// Warp内求和
	if(iii < 16) cache_d[iii] += cache_d[iii + 16];
	if(iii < 8) cache_d[iii] += cache_d[iii + 8];
	if(iii < 4) cache_d[iii] += cache_d[iii + 4];
	if(iii < 2) cache_d[iii] += cache_d[iii + 2];
	if(iii < 1) cache_d[iii] += cache_d[iii + 1];
	
	if(iii == 0) temp[blockIdx.x] = cache_d[0];*/

	int ii  = blockIdx.x;
	int iii = threadIdx.x;

	__shared__ int cache_d[32];
	cache_d[iii] = 0;

	int begin, end, me, i, j;
	begin = xadj[ii];
	end   = xadj[ii + 1];
	me    = where[ii];

	for(i = begin + iii;i < end;i += 32)
	{
		j = adjncy[i];
		if(where[j] != me) cache_d[iii] += adjwgt[i];
	}
	__syncthreads();

	// Warp内求和
	if(iii < 16) cache_d[iii] += cache_d[iii + 16];
	__syncthreads();
	if(iii < 8) cache_d[iii] += cache_d[iii + 8];
	__syncthreads();
	if(iii < 4) cache_d[iii] += cache_d[iii + 4];
	__syncthreads();
	if(iii < 2) cache_d[iii] += cache_d[iii + 2];
	__syncthreads();
	if(iii < 1) cache_d[iii] += cache_d[iii + 1];
	__syncthreads();
	
	if(iii == 0) temp[ii] = cache_d[0];
}

__global__ void heat(int nvtxs, int *h)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int t = h[ii];
	}
}

void jetLP(hunyuangraph_graph_t *graph)
{
  int nvtxs, from, to;
  int *move_to, *list, *gainv;
  kp_t *kp;

  nvtxs = graph->nvtxs;

  cudaDeviceSynchronize();
  gettimeofday(&begin_malloc_2way, NULL);
  cudaMalloc((void**)&move_to,sizeof(int) * nvtxs);
  cudaMalloc((void**)&list,sizeof(int) * nvtxs);
  cudaMalloc((void**)&gainv,sizeof(int) * nvtxs);
  cudaMalloc((void**)&kp,sizeof(kp_t) * nvtxs);
  cudaDeviceSynchronize();
  gettimeofday(&end_malloc_2way, NULL);
//   malloc_2way += (end_malloc_2way.tv_sec - begin_malloc_2way.tv_sec) * 1000 + (end_malloc_2way.tv_usec - begin_malloc_2way.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_initmoveto, NULL);
  init_moveto<<<(nvtxs + 127) / 128,128>>>(graph->cuda_where,move_to,nvtxs);
  cudaDeviceSynchronize();
  gettimeofday(&end_initmoveto, NULL);
  initmoveto += (end_initmoveto.tv_sec - begin_initmoveto.tv_sec) * 1000 + (end_initmoveto.tv_usec - begin_initmoveto.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_updatemoveto, NULL);
  for(int i = 0;i < 1;i++)
  {
  	// filter_first<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, move_to, list, gainv);
  	// filter_second<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, move_to, list, gainv);
	
	// filter_first_atomic<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, move_to, list, gainv);
  	// filter_second_atomic<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, move_to, list, gainv);
	filter_atomic<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where);
	// filter_atomic_group<<<(nvtxs / 8) / 32 + 1,32>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where);

	// update_moveto_atomic<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->nedges, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, move_to, list, gainv);
  }
  cudaDeviceSynchronize();
  gettimeofday(&end_updatemoveto, NULL);
  updatemoveto += (end_updatemoveto.tv_sec - begin_updatemoveto.tv_sec) * 1000 + (end_updatemoveto.tv_usec - begin_updatemoveto.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_computepwgts, NULL);
  compute_pwgts<<<(nvtxs + 127) / 128,128>>>(nvtxs,graph->cuda_vwgt,graph->cuda_where,gainv);
  cudaDeviceSynchronize();
  gettimeofday(&end_computepwgts, NULL);
  computepwgts += (end_computepwgts.tv_sec - begin_computepwgts.tv_sec) * 1000 + (end_computepwgts.tv_usec - begin_computepwgts.tv_usec) / 1000.0;

  cudaDeviceSynchronize();
  gettimeofday(&begin_thrustreduce, NULL);
  for(int l = nvtxs;l != 1;l = (l + 512 - 1) / 512)
        reduction6<<<(l + 512 - 1) / 512,256>>>(gainv,l);
  cudaMemcpy(&graph->pwgts[0], &gainv[0],sizeof(int),cudaMemcpyDeviceToHost);
//   graph->pwgts[0] = thrust::reduce(thrust::device, gainv, gainv + nvtxs);
  graph->pwgts[1] = graph->tvwgt[0] - graph->pwgts[0];
  cudaDeviceSynchronize();
  gettimeofday(&end_thrustreduce, NULL);
  thrustreduce += (end_thrustreduce.tv_sec - begin_thrustreduce.tv_sec) * 1000 + (end_thrustreduce.tv_usec - begin_thrustreduce.tv_usec) / 1000.0;

  if((graph->pwgts[0] >= graph->tvwgt[0] * 0.5 / 1.03 && graph->pwgts[0] <= graph->tvwgt[0] * 0.5 * 1.03) && (graph->pwgts[1] >= graph->tvwgt[0] * 0.5 / 1.03 && graph->pwgts[1] <= graph->tvwgt[0] * 0.5 * 1.03)) /*printf("balance\n")*/;
  else
  {
    // printf("1\n");

    if(graph->pwgts[0] > graph->pwgts[1]) from = 0;
    else from = 1;
    to = (from + 1) % 2;

    cudaDeviceSynchronize();
    gettimeofday(&begin_computegain, NULL);
    compute_gainkp<<<(nvtxs + 127) / 128,128>>>(nvtxs, graph->nedges, from, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, kp);
    cudaDeviceSynchronize();
    gettimeofday(&end_computegain, NULL);
    computegain += (end_computegain.tv_sec - begin_computegain.tv_sec) * 1000 + (end_computegain.tv_usec - begin_computegain.tv_usec) / 1000.0;

    cudaDeviceSynchronize();
    gettimeofday(&begin_thrustsort, NULL);
    thrust::sort(thrust::device, kp, kp + nvtxs, compRule());
    cudaDeviceSynchronize();
    gettimeofday(&end_thrustsort, NULL);
    thrustsort += (end_thrustsort.tv_sec - begin_thrustsort.tv_sec) * 1000 + (end_thrustsort.tv_usec - begin_thrustsort.tv_usec) / 1000.0;

    cudaDeviceSynchronize();
    gettimeofday(&begin_computegainv, NULL);
    compute_gainvkp<<<(nvtxs + 127) / 128,128>>>(nvtxs,graph->cuda_vwgt,kp,gainv);
    cudaDeviceSynchronize();
    gettimeofday(&end_computegainv, NULL);
    computegainv += (end_computegainv.tv_sec - begin_computegainv.tv_sec) * 1000 + (end_computegainv.tv_usec - begin_computegainv.tv_usec) / 1000.0;

    cudaDeviceSynchronize();
    gettimeofday(&begin_inclusive, NULL);
    thrust::inclusive_scan(thrust::device, gainv, gainv + nvtxs, gainv);
    cudaDeviceSynchronize();
    gettimeofday(&end_inclusive, NULL);
    inclusive += (end_inclusive.tv_sec - begin_inclusive.tv_sec) * 1000 + (end_inclusive.tv_usec - begin_inclusive.tv_usec) / 1000.0;

    //move
    cudaDeviceSynchronize();
    gettimeofday(&begin_rebalance, NULL);
    rebalancekp<<<(nvtxs + 127) / 128,128>>>(nvtxs,(graph->pwgts[from] - graph->pwgts[to]) / 2,to,graph->cuda_where,kp,gainv);
    cudaDeviceSynchronize();
    gettimeofday(&end_rebalance, NULL);
    re_balance += (end_rebalance.tv_sec - begin_rebalance.tv_sec) * 1000 + (end_rebalance.tv_usec - begin_rebalance.tv_usec) / 1000.0;

  }
  
}

__global__ void init_0(int nvtxs, int *num)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
		num[ii] = 0;
}

__global__ void set_bnd(int nvtxs, int *xadj, int *adjncy, int *where, int *moved, int *bnd)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(moved[ii] == 0)
		{
			int begin, end, me, flag, i, j;
			begin = xadj[ii];
			end   = xadj[ii + 1];
			me    = where[ii];
			flag  = 0;

			for(i = begin;i < end;i++)
			{
				j  = adjncy[i];
				if(where[j] != me)
				{
					flag = 1;
					break;
				}
			}

			if(flag == 1) bnd[ii] = 1;
			else bnd[ii] = 0;
		}
		else bnd[ii] = 0;
	}
}

__global__ void calculate_to(int nvtxs, int nparts, int *xadj, int *adjncy, int *adjwgt, int *where, int *bnd, int *moveto, int *gain)
{
	int ii  = blockIdx.x;
	int tid = threadIdx.x;

	if(bnd[ii] == 1)
	{
		extern __shared__ int cache_all[];
		int *cache_d   = cache_all;
		int *cache_ptr = cache_all + nparts;
		for (int i = tid; i < nparts; i += 32)
		{
			cache_d[i]   = 0;
			cache_ptr[i] = i;
		}
		__syncthreads();

		int begin, end, me, flag, i, j, id;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		me    = where[ii];
		flag  = 0;

		for(i = begin + tid;i < end;i += 32)
		{
			j  = adjncy[i];
			atomicAdd(&cache_d[where[j]],adjwgt[i]);
		}
		__syncthreads();

		id = cache_d[me];
		__syncthreads();

		// Warp内reduce选择最大度
			// 数据缩小至1个warp
		if(nparts > 32)
		{
			for(i = tid;i < nparts;i += 32)
			{
				if(i + 32 < nparts && cache_d[tid] < cache_d[i + 32])
				{
					cache_d[tid]   = cache_d[i + 32];
					cache_ptr[tid] = cache_ptr[tid + 32];
				}
			}
			__syncthreads();
		}

		if(tid + 16 < nparts && cache_d[tid] < cache_d[tid + 16])
		{
			cache_d[tid]   = cache_d[tid + 16];
			cache_ptr[tid] = cache_ptr[tid + 16];
		}
		__syncthreads();
		if(tid + 8 < nparts && cache_d[tid] < cache_d[tid + 8])
		{
			cache_d[tid]   = cache_d[tid + 8];
			cache_ptr[tid] = cache_ptr[tid + 8];
		}
		__syncthreads();
		if(tid + 4 < nparts && cache_d[tid] < cache_d[tid + 4])
		{
			cache_d[tid]   = cache_d[tid + 4];
			cache_ptr[tid] = cache_ptr[tid + 4];
		}
		__syncthreads();
		if(tid + 2 < nparts && cache_d[tid] < cache_d[tid + 2])
		{
			cache_d[tid]   = cache_d[tid + 2];
			cache_ptr[tid] = cache_ptr[tid + 2];
		}
		__syncthreads();
		if(tid + 1 < nparts && cache_d[tid] < cache_d[tid + 1])
		{
			cache_d[tid]   = cache_d[tid + 1];
			cache_ptr[tid] = cache_ptr[tid + 1];
		}

		if(tid == 0)
		{
			// printf("ii=%d cache=%d ptr=%d where=%d\n",ii,cache_d[0],cache_ptr[0],me);
			flag = cache_ptr[0];
			if(flag == me) moveto[ii] = -1;
			else
			{
				moveto[ii] = flag;
				gain[ii]   = cache_d[0] - id;
			}
		}
	}
	else
	{
		if(tid == 0)
			moveto[ii] = -1;
	}
}

__global__ void init_max(int nparts,int *max, int *maxptr)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nparts)
	{
		max[ii] = 0;
		maxptr[ii] = -1;
	}
}

__global__ void calculate_Max(int nvtxs, int nparts, int *max, int *maxptr, int *moveto, int *gain, int *moved)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	
	/*extern __shared__ int cache_d[];
	extern __shared__ int cache_ptr[];
	for (int i = threadIdx.x; i < nparts; i += 128)
	{
		cache_d[i]   = 0;
		cache_ptr[i] = -1;
	}
	__syncthreads();

	if(ii < nvtxs && moved[ii] == -1)
	{
		int t = moveto[ii];
		int val = gain[ii];
		if(t != -1)
		{
			atomicMax(&cache_d[t],val);
			__syncthreads();
			if(cache_d[t] == val) cache_ptr[t] = ii;
		}
		__syncthreads();

		printf("1\n");

			for (int i = 0; i < nparts; i++)
			{
				printf("ii=%d cache=%d cache_ptr=%d\n",ii,cache_d[i],cache_ptr[i]);
			}

		int ptr;
		for (int i = threadIdx.x; i < nparts; i += 128)
		{
			ptr = cache_ptr[i];
			if(ptr != -1)
			{
				atomicMax(&max[t],val);
				if(max[t] == val) maxptr[t] = ptr;
			}
		}
	}*/

	if(ii < nvtxs)
	{
		int t = moveto[ii];
		if(t != -1)
		{
			int val = gain[ii];
			if(val > 0)
			{
				atomicMax(&max[t],val);
				__syncthreads();
				if(max[t] == val) atomicExch(&maxptr[t],ii);
			}
		}
	}
}

__global__ void calculate_move(int nparts, int *maxptr, int *moveto, int *where, int *moved, int *pwgts, int *vwgt)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(ii < nparts)
	{
		int ptr    = maxptr[ii];
		if(ptr != -1)
		{
			int val = vwgt[ptr];
			atomicSub(&pwgts[where[ptr]],val);
			atomicAdd(&pwgts[ii],val);
			where[ptr] = ii;
			moved[ptr] = 1;
		}
	}
}

__global__ void calculate_overweight(int nparts, float temp, int tvwgt,int *pwgts, int *overwgt, int *overthin)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nparts)
	{
		int val;
		float max, min;
		min = temp / 1.03 * tvwgt;
		max = temp * 1.03 * tvwgt;
		val = pwgts[ii];

		if(val > max) overwgt[ii] = 1;
		else overwgt[ii] = 0;

		if(val < min) overthin[ii] = 1;
		else overthin[ii] = 0;
	}
}

__global__ void set_bnd_reba(int nvtxs, int *xadj, int *adjncy, int *where, int *overwgt, int* overthin, int *bnd)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int me = where[ii];
		if(overwgt[me] == 1)
		{
			int begin, end, flag, i, j, wj;
			begin = xadj[ii];
			end   = xadj[ii + 1];

			for(i = begin;i < end;i++)
			{
				j  = adjncy[i];
				wj = where[j];
				if(wj != me && overthin[wj] == 1)
				{
					flag = 1;
					break;
				}
			}

			if(flag == 1) bnd[ii] = 1;
			else bnd[ii] = 0;
		}
		else bnd[ii] = 0;
	}
}

__global__ void calculate_to_reba(int nvtxs, int nparts, int *xadj, int *adjncy, int *adjwgt, int *where, int *bnd, int *overthin, int *overwgt, int *moveto, int *gain)
{
	int ii  = blockIdx.x;
	int tid = threadIdx.x;

	if(bnd[ii] == 1)
	{
		extern __shared__ int cache_all[];
		int *cache_d   = cache_all;
		int *cache_ptr = cache_all + nparts;
		for (int i = tid; i < nparts; i += 32)
		{
			cache_d[i]   = 0;
			cache_ptr[i] = i;
		}
		__syncthreads();

		int begin, end, me, flag, i, j, wj;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		me    = where[ii];
		flag  = 0;

		for(i = begin + tid;i < end;i += 32)
		{
			j  = adjncy[i];
			wj = where[j];
			if(wj != me && overthin[wj] == 1) atomicAdd(&cache_d[wj],adjwgt[i]);
		}
		__syncthreads();


		/*if(tid == 0)
		{
			for(i = 0;i < nparts;i++)
			{
				printf("ii=%d nparts=%d cache=%d \n",ii,i,cache_d[i]);
			}
		}
		__syncthreads();*/

		// Warp内reduce选择最大度
			// 数据缩小至1个warp
		if(nparts > 32)
		{
			for(i = tid;i < nparts;i += 32)
			{
				if(i + 32 < nparts && cache_d[tid] < cache_d[i + 32])
				{
					cache_d[tid]   = cache_d[i + 32];
					cache_ptr[tid] = cache_ptr[tid + 32];
				}
			}
			__syncthreads();
		}

		if(tid + 16 < nparts && cache_d[tid] < cache_d[tid + 16])
		{
			cache_d[tid]   = cache_d[tid + 16];
			cache_ptr[tid] = cache_ptr[tid + 16];
		}
		__syncthreads();
		if(tid + 8 < nparts && cache_d[tid] < cache_d[tid + 8])
		{
			cache_d[tid]   = cache_d[tid + 8];
			cache_ptr[tid] = cache_ptr[tid + 8];
		}
		__syncthreads();
		if(tid + 4 < nparts && cache_d[tid] < cache_d[tid + 4])
		{
			cache_d[tid]   = cache_d[tid + 4];
			cache_ptr[tid] = cache_ptr[tid + 4];
		}
		__syncthreads();
		if(tid + 2 < nparts && cache_d[tid] < cache_d[tid + 2])
		{
			cache_d[tid]   = cache_d[tid + 2];
			cache_ptr[tid] = cache_ptr[tid + 2];
		}
		__syncthreads();
		if(tid + 1 < nparts && cache_d[tid] < cache_d[tid + 1])
		{
			cache_d[tid]   = cache_d[tid + 1];
			cache_ptr[tid] = cache_ptr[tid + 1];
		}

		if(tid == 0)
		{
			// printf("ii=%d cache=%d ptr=%d where=%d\n",ii,cache_d[0],cache_ptr[0],me);
			flag = cache_ptr[0];
			if(overthin[flag] == 0) moveto[ii] = -1;
			else
			{
				moveto[ii] = cache_ptr[0];
				gain[ii]   = cache_d[0];
			}
		}
	}
	else
	{
		if(tid == 0)
			moveto[ii] = -1;
	}
}

__global__ void calculate_Max_reba(int nvtxs, int nparts, int *max, int *maxptr, int *moveto, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(ii < nvtxs)
	{
		int t = moveto[ii];
		if(t != -1)
		{
			int val = gain[ii];
			atomicMax(&max[t],val);
			__syncthreads();
			if(max[t] == val) atomicExch(&maxptr[t],ii);
		}
	}
}

__global__ void calculate_move_reba(int nparts, int *maxptr, int *moveto, int *where, int *pwgts, int *vwgt)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(ii < nparts)
	{
		int ptr    = maxptr[ii];
		if(ptr != -1)
		{
			int val = vwgt[ptr];
			atomicSub(&pwgts[where[ptr]],val);
			atomicAdd(&pwgts[ii],val);
			where[ptr] = ii;
		}
	}
}

__global__ void exam_csrpart(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where)
{
	for(int i = 0;i < nvtxs;i++)
	{
		for(int j = xadj[i];j < xadj[i + 1];j++)
		{
			printf("%d ",adjncy[j]);
		}
		printf("\n");
		for(int j = xadj[i];j < xadj[i + 1];j++)
		{
			printf("%d ",adjwgt[j]);
		}
		printf("\n");
		for(int j = xadj[i];j < xadj[i + 1];j++)
		{
			printf("%d ",where[adjncy[j]]);
		}
		printf("\n%d ii=%d\n",where[i],i);
		printf("\n");
	}
}

__global__ void exam_gain(int nvtxs,int *where,int *moveto, int *gain)
{
	for(int i = 0;i < nvtxs;i++)
	{
		if(moveto[i] != -1) printf("i=%d from=%d gain[i]=%d moveto=%d\n",i,where[i],gain[i],moveto[i]);
	}
}

__global__ void exam_max(int nparts,int *cu_max, int *cu_maxptr)
{
	for(int i = 0;i < nparts;i++)
	{
		if(cu_maxptr[i] != -1) printf("nparts=%d cu_max=%d cu_maxptr=%d\n",i,cu_max[i],cu_maxptr[i]);
	}
}

__global__ void exam_over(int nparts, int *cu_overwgt, int *cu_overthin)
{
	for(int i = 0;i < nparts;i++)
	{
		printf("nparts=%d cu_overwgt=%d cu_overthin=%d\n",i,cu_overwgt[i],cu_overthin[i]);
	}
}

void hunyuangraph_k_refinement_me(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs, nedges, nparts, threshold;
	nvtxs  = graph->nvtxs;
	nedges = graph->nedges;
	nparts = hunyuangraph_admin->nparts;
	threshold = hunyuangraph_max(nvtxs / (6000 * (hunyuangraph_compute_log2(nparts))),8 * nparts);
	printf("threshold=%d\n",threshold);
	
	int *cu_bnd, *cu_moved, *cu_moveto, *cu_gain, *cu_max, *cu_maxptr, *cu_overwgt, *cu_overthin;

	cudaMalloc((void**)&cu_bnd,sizeof(int) * nvtxs);
	cudaMalloc((void**)&cu_moved,sizeof(int) * nvtxs);
	cudaMalloc((void**)&cu_moveto,sizeof(int) * nvtxs);
	cudaMalloc((void**)&cu_gain,sizeof(int) * nvtxs);
	cudaMalloc((void**)&cu_max,sizeof(int) * nparts);
	cudaMalloc((void**)&cu_maxptr,sizeof(int) * nparts);
	cudaMalloc((void**)&cu_overwgt,sizeof(int) * nparts);
	cudaMalloc((void**)&cu_overthin,sizeof(int) * nparts);

	// cudaDeviceSynchronize();
	// exam_csrpart<<<1,1>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,graph->cuda_where);
	// cudaDeviceSynchronize();

	// init_0<<<(nvtxs + 127) / 128,128>>>(nvtxs,cu_bnd);

	// 初始化移动记录数组
	init_0<<<(nvtxs + 127) / 128,128>>>(nvtxs,cu_moved);

	// 先移动
	int iter;
	for(iter = 0;iter < threshold;iter++)
	{
		// printf("iter=%d\n",iter);
		// 标记边界顶点
		set_bnd<<<(nvtxs + 127) / 128,128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_where,cu_moved,cu_bnd);

		// 计算顶点移动
		calculate_to<<<nvtxs,32,2 * nparts * sizeof(int)>>>(nvtxs,nparts,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,graph->cuda_where,cu_bnd,cu_moveto,cu_gain);

		// cudaDeviceSynchronize();
		// exam_gain<<<1,1>>>(nvtxs,graph->cuda_where,cu_moveto,cu_gain);
		// cudaDeviceSynchronize();

		init_max<<<(nparts + 31) / 32, 32>>>(nparts,cu_max,cu_maxptr);

		// 选择移动至相同分区的最佳顶点
		calculate_Max<<<(nvtxs + 127) / 128,128>>>(nvtxs,nparts,cu_max,cu_maxptr,cu_moveto,cu_gain,cu_moved);

		// cudaDeviceSynchronize();
		// exam_max<<<1,1>>>(nparts,cu_max,cu_maxptr);
		// cudaDeviceSynchronize();

		// 移动并记录至移动记录数组
		calculate_move<<<(nparts + 31) / 32,32>>>(nparts,cu_maxptr,cu_moveto,graph->cuda_where,cu_moved,graph->cuda_pwgts,graph->cuda_vwgt);
	}

	// cudaDeviceSynchronize();
	// exam_pwgts<<<1,1>>>(nparts,hunyuangraph_admin->tpwgts[0],graph->tvwgt[0],graph->cuda_pwgts);
	// cudaDeviceSynchronize();

	// 再平衡
	printf("rbalance\n");
	// 记录过轻或过重 (0:正常,1:过重/过轻)
	calculate_overweight<<<(nparts + 31) / 32, 32>>>(nparts,hunyuangraph_admin->tpwgts[0],graph->tvwgt[0],graph->cuda_pwgts,cu_overwgt,cu_overthin);
	int val_weight = thrust::reduce(thrust::device, cu_overwgt, cu_overwgt + nparts);
	int val_thin   = thrust::reduce(thrust::device, cu_overthin, cu_overthin + nparts);

	// cudaDeviceSynchronize();
	// exam_over<<<1,1>>>(nparts,cu_overwgt,cu_overthin);
	// cudaDeviceSynchronize();
	printf("val_weight=%d val_thin=%d\n",val_weight,val_thin);

	iter = 0;
	while(val_weight > 0 && val_thin > 0)
	{
		if(iter >= threshold) break;
		iter++;
		// 标记过重边界顶点
		set_bnd_reba<<<(nvtxs + 127) / 128,128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_where,cu_overwgt,cu_overthin,cu_bnd);

		// 计算顶点移动
		calculate_to_reba<<<nvtxs,32,2 * nparts * sizeof(int)>>>(nvtxs,nparts,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,\
			graph->cuda_where,cu_bnd,cu_overthin,cu_overwgt,cu_moveto,cu_gain);
		
		// cudaDeviceSynchronize();
		// exam_gain<<<1,1>>>(nvtxs,graph->cuda_where,cu_moveto,cu_gain);
		// cudaDeviceSynchronize();

		init_max<<<(nparts + 31) / 32, 32>>>(nparts,cu_max,cu_maxptr);

		// 选择移动至相同分区的最佳顶点
		calculate_Max_reba<<<(nvtxs + 127) / 128,128>>>(nvtxs,nparts,cu_max,cu_maxptr,cu_moveto,cu_gain);

		// cudaDeviceSynchronize();
		// exam_max<<<1,1>>>(nparts,cu_max,cu_maxptr);
		// cudaDeviceSynchronize();

		// 移动并记录至移动记录数组
		calculate_move<<<(nparts + 31) / 32,32>>>(nparts,cu_maxptr,cu_moveto,graph->cuda_where,cu_moved,graph->cuda_pwgts,graph->cuda_vwgt);

		// calculateSum<<<(nvtxs + 127) / 128,128,nparts * sizeof(int)>>>(nvtxs,nparts,graph->cuda_pwgts,graph->cuda_where,graph->cuda_vwgt);

		calculate_overweight<<<(nparts + 31) / 32, 32>>>(nparts,hunyuangraph_admin->tpwgts[0],graph->tvwgt[0],graph->cuda_pwgts,cu_overwgt,cu_overthin);
		val_weight = thrust::reduce(thrust::device, cu_overwgt, cu_overwgt + nparts);
		val_thin   = thrust::reduce(thrust::device, cu_overthin, cu_overthin + nparts);

		// cudaDeviceSynchronize();
		// exam_pwgts<<<1,1>>>(nparts,hunyuangraph_admin->tpwgts[0],graph->tvwgt[0],graph->cuda_pwgts);
		// cudaDeviceSynchronize();

		// cudaDeviceSynchronize();
		// exam_over<<<1,1>>>(nparts,cu_overwgt,cu_overthin);
		// cudaDeviceSynchronize();

		// printf("val_weight=%d val_thin=%d\n",val_weight,val_thin);
	}

	printf("iter=%d\n",iter);

	cudaFree(cu_bnd);
	cudaFree(cu_moved);
	cudaFree(cu_moveto);
	cudaFree(cu_gain);
	cudaFree(cu_max);
	cudaFree(cu_maxptr);
	cudaFree(cu_overwgt);
	cudaFree(cu_overthin);
}

#endif