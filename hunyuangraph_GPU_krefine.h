#ifndef _H_KREFINE
#define _H_KREFINE

#include "hunyuangraph_struct.h"

/*CUDA-get the max/min pwgts*/
__global__ void Sum_maxmin_pwgts(int *maxwgt, float *tpwgts, int tvwgt, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nparts)
	{
		float result = tpwgts[ii] * tvwgt;

		maxwgt[ii] = int(result * IMB);
	}
}

/*CUDA-init boundary vertex num*/
__global__ void initbndnum(int *bndnum)
{
  bndnum[0] = 0;
}

/*CUDA-find vertex where ed-id>0 */
__global__ void Find_real_bnd_info(int *cuda_real_bnd_num, int *cuda_real_bnd, int *where, \
	int *xadj, int *adjncy, int *adjwgt, int nvtxs, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs) // && moved[ii] == 0
	{
		int i, k, begin, end, me, other, from;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		me    = 0;
		other = 0;
		from  = where[ii];

		for(i = begin;i < end;i++)
		{
			k = adjncy[i];
			if(where[k] == from) me += adjwgt[i];
			else other += adjwgt[i];
		}
		if(other > me) cuda_real_bnd[atomicAdd(&cuda_real_bnd_num[0],1)] = ii;
	}
}

/**/
__global__ void init_bnd_info(int *bnd_info, int length)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < length)
    bnd_info[ii] = 0;
}

/*CUDA-find boundary vertex should ro which part*/
__global__ void find_kayparams(int *cuda_real_bnd_num, int *bnd_info, int *cuda_real_bnd, int *where, \
int *xadj, int *adjncy, int *adjwgt, int nparts, int *cuda_bn, int *to, int *gain)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii<cuda_real_bnd_num[0])
  {
    int pi, other, i, k, me_wgt, other_wgt;
    int start, end, begin, last;

    pi    = cuda_real_bnd[ii];
    begin = xadj[pi];
    last  = xadj[pi+1];
    start = nparts * ii;
    end   = nparts * (ii+1);
    other = where[pi];

    for(i = begin;i < last;i++)
    {
      k = adjncy[i];
      k = start + where[k];
      bnd_info[k] += adjwgt[i];
    }

    me_wgt = other_wgt = bnd_info[start + other];

    for(i=start;i<end;i++)
    {
      k = bnd_info[i];
      if(k > other_wgt)
      {
        other_wgt = k;
        other     = i - start;
      }
    }

    gain[ii]  = other_wgt - me_wgt;
    to[ii] = other;
    cuda_bn[ii] = pi;

  }
}

/*CUDA-init params*/
__global__ void initcucsr(int *cu_csr, int *bndnum)
{
  cu_csr[0] = 0;
  cu_csr[1] = bndnum[0];
}

/**/
__global__ void init_cu_que(int *cuda_que, int length)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < length)
    cuda_que[ii] = -1;
}

/*CUDA-get a csr array*/
__global__ void findcsr(int *to, int *cuda_que, int *bnd_num, int nparts)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii < nparts)
  {
    int i, t;
    int begin, end;
    begin = 2 * ii;
    end   = bnd_num[0];

    for(i = 0;i < end;i++)
    {
      if(ii == to[i])
      {
        cuda_que[begin] = i;
        break; 
      }
    }

    t = cuda_que[begin];

    if(t!=-1)
    {
      for(i = t;i < end;i++)
      {
        if(to[i] != ii)
        {
          cuda_que[begin + 1] = i - 1;
          break; 
        }
      }
    }

    t = 2 * to[end - 1] + 1;
    cuda_que[t] = end - 1;
  }
}

/*Find boundary vertex information*/
void hunyuangraph_findgraphbndinfo(hunyuangraph_admin_t *hunyuangraph_admin,hunyuangraph_graph_t *graph)
{
	int nvtxs  = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;
	int bnd_num; 

	initbndnum<<<1,1>>>(graph->cuda_bndnum);

	Find_real_bnd_info<<<nvtxs / 32 + 1,32>>>(graph->cuda_bndnum,graph->cuda_bnd,graph->cuda_where,\
		graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,nvtxs,nparts);
  
	cudaMemcpy(&bnd_num,graph->cuda_bndnum, sizeof(int), cudaMemcpyDeviceToHost);
  
	if(bnd_num > 0)
	{
		// cudaMalloc((void**)&graph->cuda_info, bnd_num * nparts * sizeof(int));
		if(GPU_Memory_Pool)
			graph->cuda_info = (int *)rmalloc_with_check(sizeof(int) * bnd_num * nparts,"info");
		else
			cudaMalloc((void **)&graph->cuda_info, sizeof(int) * bnd_num * nparts);

		init_bnd_info<<<bnd_num * nparts / 32 + 1,32>>>(graph->cuda_info, bnd_num * nparts);

		find_kayparams<<<bnd_num/32+1,32>>>(graph->cuda_bndnum,graph->cuda_info,graph->cuda_bnd,graph->cuda_where,\
			graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,nparts,graph->cuda_bn,graph->cuda_to,graph->cuda_gain);

		initcucsr<<<1,1>>>(graph->cuda_csr,graph->cuda_bndnum);

		bb_segsort(graph->cuda_to, graph->cuda_bn, bnd_num, graph->cuda_csr, 1);
		
		init_cu_que<<<2 * nparts / 32 + 1,32>>>(graph->cuda_que, 2 * nparts);
		
		findcsr<<<nparts/32+1,32>>>(graph->cuda_to,graph->cuda_que,graph->cuda_bndnum,nparts);
		
		if(GPU_Memory_Pool)
			rfree_with_check((void *)graph->cuda_info, sizeof(int) * bnd_num * nparts,"graph->cuda_info");	//	graph->cuda_info
		else
			cudaFree(graph->cuda_info);
	}

	graph->cpu_bndnum=(int *)malloc(sizeof(int));
	graph->cpu_bndnum[0]=bnd_num;
}

/*CUDA-move vertex*/
__global__ void Exnode_part1(int *cuda_que, int *pwgts, int *bnd, int *bndto, int *vwgt,\
  int *maxvwgt, int *where, int nparts)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii<nparts)
  {
    int i, me, to, vvwgt;
    int memax, tomax, mepwgts, topwgts;
    int begin, end, k;
    begin = cuda_que[2 * ii];
    end   = cuda_que[2 * ii + 1];
    if(begin != -1)
    {
      for(i = begin;i <= end;i++)
      {
        k     = bnd[i];
        vvwgt = vwgt[k];
        me    = where[k];
        to    = bndto[i];

        memax   = maxvwgt[me];
        tomax   = maxvwgt[to];
        mepwgts = pwgts[me];
        topwgts = pwgts[to];

        if(me <= to)
        {
          if((topwgts + vvwgt <= tomax) && (mepwgts - vvwgt <= memax))
          {
            atomicAdd(&pwgts[to],vvwgt);
            atomicSub(&pwgts[me],vvwgt);
            where[k] = to;
          }
        }
      }
    }
  }
}

/*CUDA-move vertex*/
__global__ void Exnode_part2(int *cuda_que, int *pwgts, int *bnd, int *bndto, int *vwgt,\
  int *maxvwgt, int *where, int nparts)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if(ii<nparts)
  {
    int i, me, to, vvwgt;
    int memax, tomax, mepwgts, topwgts;
    int begin, end, k;
    begin = cuda_que[2 * ii];
    end   = cuda_que[2 * ii + 1];
    if(begin != -1)
    {
      for(i = begin;i <= end;i++)
      {
        k     = bnd[i];
        vvwgt = vwgt[k];
        me    = where[k];
        to    = bndto[i];

        memax   = maxvwgt[me];
        tomax   = maxvwgt[to];
        mepwgts = pwgts[me];
        topwgts = pwgts[to];

        if(me > to)
        {
          if((topwgts + vvwgt <= tomax) && (mepwgts - vvwgt <= memax))
          {
            atomicAdd(&pwgts[to],vvwgt);
            atomicSub(&pwgts[me],vvwgt);
            where[k] = to;
          }
        }
      }
    }
  }
}

/*Graph multilevel uncoarsening algorithm*/
void hunyuangraph_k_refinement(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs  = graph->nvtxs;

	Sum_maxmin_pwgts<<<nparts / 32 + 1,32>>>(graph->cuda_maxwgt,graph->cuda_tpwgts,graph->tvwgt[0],nparts);

	for(int i = 0;i < 2;i++)
	{
		hunyuangraph_findgraphbndinfo(hunyuangraph_admin,graph);

		if(graph->cpu_bndnum[0] > 0)
		{
			Exnode_part1<<<nparts/32+1,32>>>(graph->cuda_que,graph->cuda_pwgts,graph->cuda_bn,graph->cuda_to,graph->cuda_vwgt,\
				graph->cuda_maxwgt,graph->cuda_where,nparts);

			Exnode_part2<<<nparts/32+1,32>>>(graph->cuda_que,graph->cuda_pwgts,graph->cuda_bn,graph->cuda_to,graph->cuda_vwgt,\
				graph->cuda_maxwgt,graph->cuda_where,nparts);
		}
		else
			break;
	}
}

__global__ void init_connection(int nedges, int *connection, int *connection_to)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nedges)
	{
		connection[ii]    = 0;
		connection_to[ii] = -1;
	}
}

__global__ void init_select(int nvtxs, char *select)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii >= nvtxs)
		return ;

	select[ii] = 0;
}

__global__ void init_moved(int nvtxs, int *moved)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(ii >= nvtxs)
		return ;
	
	moved[ii] = 0;
}

__global__ void init_ed_id(int nvtxs, int *ed, int *id)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		ed[ii] = 0;
		id[ii] = 0;
	}
}

__global__ void select_bnd_vertices_old(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *connection, int *connection_to, int *gain, int *to)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int k, begin, end, length, where_i, where_k, hash_addr;
		char flag_bnd;
		begin = xadj[ii];
		end   = xadj[ii + 1];
		length = end - begin;
		where_i = where[ii];
		flag_bnd = 0;

		for(int j = begin; j < end; j++)
		{
			k = adjncy[j];
			where_k = where[k];

			if(where_i != where_k)
				flag_bnd = 1;

			//	compute connection with other partitions for every vertex by hash table
			hash_addr = where_k & length;	//	hash value
			while(1)
			{
				int key_exist = connection_to[begin + hash_addr];
				if(key_exist == ii)
				{
					connection[begin + hash_addr] += adjwgt[j];
					break;
				}
				else if(key_exist == -1)
				{
					connection_to[begin + hash_addr] = where_k;
					connection[begin + hash_addr] += adjwgt[j];
				}
				else 
				{
					hash_addr = (hash_addr + 1) & length;
				}
			}

		}

	}
}

__global__ void select_bnd_vertices_warp(int nvtxs, int nparts, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *gain, int *to)
{
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + warp_id;

	extern __shared__ int connection_shared[];
	int *ptr_shared = connection_shared + warp_id * nparts;
	int *to_shared = ptr_shared + 4 * nparts;
	int *id = connection_shared + 8 * nparts;
	int *flag_bnd = id + 4;

	// if(ii == 5920 && lane_id == 0)
	// {
	// 	// printf("ii:%d lane_id:%d\n", ii, lane_id);
	// 	printf("ii:%d lane_id:%d id: %p, flag_bnd: %p, ptr_shared: %p, to_shared: %p\n", ii, lane_id, id, flag_bnd, ptr_shared, to_shared);
	// }

	for(int i = lane_id;i < nparts;i += 32)
	{
		ptr_shared[i] = 0;
		to_shared[i] = i;
		id[warp_id] = 0;
		flag_bnd[warp_id] = 0;
	}
	__syncwarp();
	// __syncthreads();

	if(ii < nvtxs)
	{
		int begin, end, where_i, where_k, j, k;
		begin   = xadj[ii];
		end     = xadj[ii + 1];
		where_i = where[ii];
		ptr_shared[where_i] = -1;

		for(j = begin + lane_id;j < end;j += 32)
		{
			k = adjncy[j];
			where_k = where[k];

			if(where_i != where_k)
			{
				// printf("flag_bnd\n");
				flag_bnd[warp_id] = 1;
				atomicAdd(&ptr_shared[where_k], adjwgt[j]);
			}
			else	
				atomicAdd(&id[warp_id], adjwgt[j]);
		}
		__syncwarp();
		// __syncthreads();

		//	bnd vertex
		if(flag_bnd[warp_id] == 1)
		{
			end = nparts / 2;
			while(end > 32)
			{
				for(j = lane_id;j < end;j += 32)
				{
					if(ptr_shared[j + end] > ptr_shared[j])
					{
						ptr_shared[j] = ptr_shared[j + end];
						to_shared[j] = to_shared[j + end];
					}
				}
				end >>= 1;
				__syncwarp();
				// __syncthreads();
			}
			// if(lane_id >= end)
			// {
			// 	ptr_shared[lane_id] = -1;
			// 	to_shared[lane_id] = -1;
			// }
			// // __syncwarp();
			// __syncthreads();
			// else
			// {
			// 	k = ptr_shared[lane_id];
			// 	// j = to_shared[lane_id];
			// 	j = lane_id;
			// }
			// if(ii == 5920 && lane_id == 0)
			// 	printf("i:8 ii: %d, where_i:%d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
			// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
			
			// __syncwarp();
			// __syncthreads();
			for(int i = end; i > 0; i >>= 1)
			{
				// if(lane_id == 0)
				// 	printf("ii: %d i: %d\n", ii, i);
				if(lane_id < i && ptr_shared[lane_id + i] > ptr_shared[lane_id])
				{
					ptr_shared[lane_id] = ptr_shared[lane_id + i];
					to_shared[lane_id] = to_shared[lane_id + i];
				}
				__syncwarp();
				// __syncthreads();
				// if(ii == 5920 && lane_id == 0)
				// 	printf("i:%d ii: %d, where_i:%d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", i, ii, where_i, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
				// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
				// __syncwarp();
				// __syncthreads();
			}
			
			// if(lane_id == 0)
			// {
			// 	printf("ii: %d, where_i:%d  %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
			// 	ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
			// }

			if(lane_id == 0)
			{
				j = to_shared[0];
				// if(j == -1)
				// {
				// 	printf("ii:%d warp_id:%d lane_id:%d id: %p, flag_bnd: %p, ptr_shared: %p, to_shared: %p\n", ii, warp_id, lane_id, id, flag_bnd, ptr_shared, to_shared);
	
				// 	printf("ii: %d, where_i:%d j:%d to:%d k: %d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, j, to_shared[j], k, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
				// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
				// }
				k = ptr_shared[j] - id[warp_id];
				// printf("ii: %d, where_i:%d k: %d id: %d\n", ii, where_i, k, id[warp_id]);
				// if(ii == 473678)
				// 	printf("ii: %d, where_i:%d j:%d to:%d k: %d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, j, to_shared[j], k, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
				// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
				// if(k >= -0.50 * id[warp_id] / sqrtf(begin - end))
				if(j != where_i && k >= -0.15 * id[warp_id])
				{
					to[ii] = j;
					gain[ii] = k;
					select[ii] = 1;
					// printf("ii: %d\n", ii);
				}
				// if(j == where_i)
				// 	printf("ii: %d, where_i: %d to_i: %d select: %d\n", ii, where_i, j, (int)select[ii]);
			}
		}
	}
}

__global__ void select_bnd_vertices_warp_bin(int num, int nparts, int bin, int *bin_offset, int *bin_idx, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *gain, int *to, int *moved)
{
	int blockwarp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int subwarp_id = blockIdx.x * 4 + blockwarp_id;

	extern __shared__ int connection_shared[];
	int *ptr_shared = connection_shared + blockwarp_id * nparts;
	int *to_shared = ptr_shared + 4 * nparts;
	int *id = connection_shared + 8 * nparts;
	int *flag_bnd = id + 4;

	// if(ii == 5920 && lane_id == 0)
	// {
	// 	// printf("ii:%d lane_id:%d\n", ii, lane_id);
	// 	printf("ii:%d lane_id:%d id: %p, flag_bnd: %p, ptr_shared: %p, to_shared: %p\n", ii, lane_id, id, flag_bnd, ptr_shared, to_shared);
	// }

	for(int i = lane_id;i < nparts;i += 32)
	{
		ptr_shared[i] = 0;
		to_shared[i] = i;
		id[blockwarp_id] = 0;
		flag_bnd[blockwarp_id] = 0;
	}
	__syncwarp();
	// __syncthreads();

	if(subwarp_id < num)
	{
		int bin_row_offset, vertex, begin, end, where_v, where_k, j, k;
		bin_row_offset = bin_offset[bin] + subwarp_id;
		vertex = bin_idx[bin_row_offset];
		if(moved[vertex] != 0)
			return ;
		
		begin  = xadj[vertex];
		end    = xadj[vertex + 1];
		where_v = where[vertex];
		ptr_shared[where_v] = -1;

		for(j = begin + lane_id;j < end;j += 32)
		{
			k = adjncy[j];
			where_k = where[k];

			if(where_v != where_k)
			{
				// printf("flag_bnd\n");
				flag_bnd[blockwarp_id] = 1;
				atomicAdd(&ptr_shared[where_k], adjwgt[j]);
			}
			else	
				atomicAdd(&id[blockwarp_id], adjwgt[j]);
		}
		__syncwarp();
		
		// __syncthreads();
		// if(blockIdx.x < 1)
		// {
		// 	for(int p = lane_id;p < nparts;p += 32)
		// 		printf("blockIdx.x=%d threadIdx.x=%d lane_id=%d ptr_shared=%d to_shared=%d id=%d length=%d\n", blockIdx.x, threadIdx.x, lane_id, ptr_shared[p], to_shared[p], id[blockwarp_id], xadj[vertex + 1] - xadj[vertex]);
		// }

		//	bnd vertex
		if(flag_bnd[blockwarp_id] == 1)
		{
			end = nparts / 2;
			while(end > 32)
			{
				for(j = lane_id;j < end;j += 32)
				{
					if(ptr_shared[j + end] > ptr_shared[j])
					{
						ptr_shared[j] = ptr_shared[j + end];
						to_shared[j] = to_shared[j + end];
					}
				}
				end >>= 1;
				__syncwarp();
				// __syncthreads();
			}
			// if(lane_id >= end)
			// {
			// 	ptr_shared[lane_id] = -1;
			// 	to_shared[lane_id] = -1;
			// }
			// // __syncwarp();
			// __syncthreads();
			// else
			// {
			// 	k = ptr_shared[lane_id];
			// 	// j = to_shared[lane_id];
			// 	j = lane_id;
			// }
			// if(ii == 5920 && lane_id == 0)
			// 	printf("i:8 ii: %d, where_i:%d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
			// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
			
			// __syncwarp();
			// __syncthreads();
			for(int i = end; i > 0; i >>= 1)
			{
				// if(lane_id == 0)
				// 	printf("ii: %d i: %d\n", ii, i);
				if(lane_id < i && ptr_shared[lane_id + i] > ptr_shared[lane_id])
				{
					ptr_shared[lane_id] = ptr_shared[lane_id + i];
					to_shared[lane_id] = to_shared[lane_id + i];
				}
				__syncwarp();
				// __syncthreads();
				// if(ii == 5920 && lane_id == 0)
				// 	printf("i:%d ii: %d, where_i:%d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", i, ii, where_i, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
				// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
				// __syncwarp();
				// __syncthreads();
			}
			
			// if(lane_id == 0)
			// {
			// 	printf("vertex: %d, where_i:%d  %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", vertex, where_i, ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
			// 	ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
			// }

			// __syncthreads();
			// if(blockIdx.x < 1 && lane_id == 0)
			// {
			// 	printf("blockIdx.x=%d threadIdx.x=%d lane_id=%d ptr_shared=%d to_shared=%d length=%d\n", blockIdx.x, threadIdx.x, lane_id, ptr_shared[lane_id], to_shared[lane_id], xadj[vertex + 1] - xadj[vertex]);
			// }

			if(lane_id == 0)
			{
				j = to_shared[0];
				// if(j == -1)
				// {
				// 	printf("ii:%d warp_id:%d lane_id:%d id: %p, flag_bnd: %p, ptr_shared: %p, to_shared: %p\n", ii, warp_id, lane_id, id, flag_bnd, ptr_shared, to_shared);
	
				// 	printf("vertex: %d, where_i:%d j:%d to:%d k: %d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", vertex, where_i, j, to_shared[j], k, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
				// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
				// }
				k = ptr_shared[j] - id[blockwarp_id];
				// printf("vertex: %d, where_i:%d k: %d id: %d\n", vertex, where_i, k, id[warp_id]);
				// if(ii == 473678)
				// 	printf("ii: %d, where_i:%d j:%d to:%d k: %d id: %d %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d\n", ii, where_i, j, to_shared[j], k, id[warp_id], ptr_shared[0], to_shared[0], ptr_shared[1], to_shared[1], ptr_shared[2], to_shared[2], \
				// 		ptr_shared[3], to_shared[3], ptr_shared[4], to_shared[4], ptr_shared[5], to_shared[5], ptr_shared[6], to_shared[6], ptr_shared[7], to_shared[7]);
				// if(k >= -0.50 * id[warp_id] / sqrtf(begin - end))
				if(j != where_v && k >= -0.15 * id[blockwarp_id])
				{
					to[vertex] = j;
					gain[vertex] = k;
					select[vertex] = 1;
					// printf("vertex: %d\n", vertex);
				}
				// if(j == where_i)
				// 	printf("ii: %d, where_i: %d to_i: %d select: %d\n", ii, where_i, j, (int)select[ii]);
			}
		}

		// __syncthreads();
		// if(blockIdx.x < 1 && lane_id == 0)
		// 	printf("blockIdx.x=%d threadIdx.x=%d vertex=%d flag_bnd=%d select=%d to=%d from=%d length=%d\n", blockIdx.x, threadIdx.x, vertex, flag_bnd[blockwarp_id], select[vertex], to[vertex], where_v,  xadj[vertex + 1] -  xadj[vertex]);

	}
}

__device__ int scan_max_ed_subwarp(int where_v, int id, int p_idx, int c_dgreen, int mask, int lane_id, int subwarp_size)
{
	int range = subwarp_size >> 1;

	#pragma unroll
	while(range > 0)
	{
		int tmp_id = __shfl_down_sync(mask, id, range, subwarp_size);
		int tmp_p_idx = __shfl_down_sync(mask, p_idx, range, subwarp_size);
		int tmp_c_dgreen = __shfl_down_sync(mask, c_dgreen, range, subwarp_size);
		if(lane_id < range)
		{
			bool valid = (tmp_p_idx != -1) & (tmp_p_idx != where_v); // 位运算替代逻辑判断
			
			if(valid)
			{
				bool current_invalid = (p_idx == -1) | (p_idx == where_v);
                bool new_value_better = tmp_c_dgreen > c_dgreen;

				if(current_invalid | new_value_better)
				{
					id = tmp_id;
					p_idx = tmp_p_idx;
					c_dgreen = tmp_c_dgreen;
				}
			}
		}
		
		range >>= 1;
	}

	return id;
}

template <int SUBWARP_SIZE>
__global__ void select_bnd_vertices_subwarp(int num, int nparts, int bin, int *bin_offset, int *bin_idx, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *gain, int *to, int *moved)
{
	int lane_id = threadIdx.x & (SUBWARP_SIZE - 1);
	int blockwarp_id = threadIdx.x / SUBWARP_SIZE;
	int subwarp_num = blockDim.x / SUBWARP_SIZE;
	int subwarp_id = blockIdx.x * subwarp_num + blockwarp_id;

	extern __shared__ int connection_shared[];
	int *part_idx = connection_shared + blockwarp_id * SUBWARP_SIZE;
	int *part_dgreen = part_idx + blockDim.x;
	int *id = connection_shared + blockDim.x * 2 + blockwarp_id;

	part_idx[lane_id] = -1;
	part_dgreen[lane_id] = 0;
	if(lane_id == 0)
		id[0] = 0;
	__syncthreads();
	// if(blockIdx.x < 1)
	// 	printf("blockIdx.x=%d threadIdx.x=%d part_id=%p part_dgreen=%p id=%p\n", blockIdx.x, threadIdx.x, part_idx, part_dgreen, id);

	if(subwarp_id < num)
	{
		int vertex, begin, end, register_to;
		int bin_row_offset = bin_offset[bin] + subwarp_id;
		vertex = bin_idx[bin_row_offset];
		if(moved[vertex] != 0)
			return ;
		
		begin  = xadj[vertex];
		end    = xadj[vertex + 1];

		int j, k, where_k, where_v, wgt_k;
		j = begin + lane_id;
		where_v = where[vertex];
		// if(j >= end)
		// 	return ;
		
		if(j < end)
		{
			k = adjncy[j];
			where_k = where[k];
			wgt_k   = adjwgt[j];

			// __syncthreads();
			// if(SUBWARP_SIZE == 2)
			// 	printf("blockIdx.x=%d threadIdx.x=%d k=%d where_k=%d wgt_k=%d length=%d\n", blockIdx.x, threadIdx.x, k, where_k, wgt_k, end - begin);

			//	hash table 
			int key = where_k;
			int hashadr = key & (SUBWARP_SIZE - 1);
			int tmp_l = 0;
			while(1)
			{
				int keyexist = part_idx[hashadr];
				if(keyexist == key)
				{
					atomicAdd(&part_dgreen[hashadr], wgt_k);
					break;
				}
				else if(keyexist == -1)
				{
					if(atomicCAS(&part_idx[hashadr], -1, key) == -1)
					{
						atomicAdd(&part_dgreen[hashadr], wgt_k);
						tmp_l++;
						break;
					}
				}
				else
				{
					hashadr = (hashadr + 1) & (SUBWARP_SIZE - 1);
				}

				// if(blockIdx.x < 1)
				// 	printf("blockIdx.x=%d threadIdx.x=%d hashadr=%d part_idx[hashadr]=%d\n", blockIdx.x, threadIdx.x, hashadr, part_idx[hashadr]);

			}
		}
		__syncwarp();
		// __syncthreads();
		// if(blockIdx.x < 1)
		// // if(SUBWARP_SIZE == 2)
		// 	printf("blockIdx.x=%d threadIdx.x=%d part_idx=%d part_dgreen=%d length=%d\n", blockIdx.x, threadIdx.x, part_idx[lane_id], part_dgreen[lane_id], end - begin);

		unsigned mask = (SUBWARP_SIZE == 32) ? 0xffffffff : ((1u << SUBWARP_SIZE) - 1);
		
		//	id
		// id[0] = (part_idx[lane_id] == where_v) ? part_dgreen[lane_id] : 0;
		if(part_idx[lane_id] == where_v)
		{
			id[0] = part_dgreen[lane_id];
		}
		__syncwarp();

		// __syncthreads();
		// if(blockIdx.x < 1)
		// // if(SUBWARP_SIZE == 2)
		// 	printf("blockIdx.x=%d threadIdx.x=%d id=%d from=%d length=%d\n", blockIdx.x, threadIdx.x, id[0], where_v, end - begin);

		//	max ed
		int p_idx = part_idx[lane_id];
		int c_dgreen = part_dgreen[lane_id];
		int ptr = scan_max_ed_subwarp(where_v, lane_id, p_idx, c_dgreen, mask, lane_id, SUBWARP_SIZE);
		// __syncthreads();
		// if(blockIdx.x < 1)
		// 	printf("blockIdx.x=%d threadIdx.x=%d ptr=%d from=%d length=%d\n", blockIdx.x, threadIdx.x, ptr, where_v, end - begin);
		if(lane_id == 0)
		{
			register_to = part_idx[ptr];

			// __shfl_sync(mask, register_to, 0, SUBWARP_SIZE);
			
			if(register_to != -1 && register_to != where_v)
			{
				// to[vertex] = register_to;
				// if(register_to != where_v)
				// {
					k = part_dgreen[ptr] - id[0];
					if(k >= -0.15 * id[0])
					{
						select[vertex] = 1;
						to[vertex] = register_to;
						gain[vertex] = k;
					}
				// }
			}
		}

		// __syncthreads();
		// if(blockIdx.x < 1 && lane_id == 0)
		// if(SUBWARP_SIZE == 2 && lane_id == 0)
		// 	printf("blockIdx.x=%d threadIdx.x=%d ptr=%d select=%d gain=%d to=%d from=%d length=%d\n", blockIdx.x, threadIdx.x, ptr, select[vertex], part_dgreen[ptr] - id[0], register_to, where_v, end - begin);
		
	}
}

__global__ void moving_vertices_interaction_SC25(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *gain, int *to)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && select[ii] == 1)
	{
		int begin, end, k, where_i, where_k, to_i, to_k, interaction_i, interaction_k, edgewgt;
		where_i = where[ii];
		to_i    = to[ii];
		if(where_i != to_i)
		{
			begin   = xadj[ii];
			end     = xadj[ii + 1];

			interaction_i = 0;
			for(int j = begin;j < end;j++)
			{
				k = adjncy[j];
				if(select[k] == 1 && where_k != to_k)
				{
					where_k = where[k];
					to_k = to[k];
					interaction_k = 0;
					edgewgt = adjwgt[j];
					if(where_k == where_i && to_k == to_i)
					{
						// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
						interaction_i += edgewgt;
						interaction_k += edgewgt;
					}
					else if(where_k == where_i && to_k != to_i)
					{
						// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
						interaction_i += edgewgt;
					}
					else if(where_k != where_i && to_k == to_i)
					{
						// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
						interaction_k += edgewgt;
					}
					else if(where_k != where_i && to_k != to_i)
					{
						if(where_k != where_i && where_k != to_i && to_k != where_i)
						{
							// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
							// inter_action[3] += graph->adjwgt[j];
						}
						else if(to_k == where_i && to_i != where_k)
						{
							// if(where_i != to_k || to_i != where_k)
							// 	printf("ATTENTION where_i=%"PRIDX" to_k=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX"\n", where_i, to_k, to_i, where_k);
							// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
							interaction_k -= edgewgt;
						}
						else if(where_k == to_i && to_k != where_i)
						{
							// if(where_i != to_k || to_i != where_k)
							// 	printf("ATTENTION where_i=%"PRIDX" to_k=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX"\n", where_i, to_k, to_i, where_k);
							// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
							interaction_i -= edgewgt;
						}
						else if(where_k == to_i && to_k == where_i)
						{
							// printf("i=%"PRIDX" k=%"PRIDX" where_i=%"PRIDX" to_i=%"PRIDX" where_k=%"PRIDX" to_k=%"PRIDX"\n", i, k, where_i, to_i, where_k, to_k);
							interaction_i -= edgewgt;
							interaction_k -= edgewgt;
						}
					}
					atomicAdd(&gain[k], interaction_k);
				}
			}
			atomicAdd(&gain[ii], interaction_i);
		}
	}
}

__global__ void update_select_SC25(int nvtxs, char *select, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && select[ii] == 1 && gain[ii] < 0)
	{
		select[ii] = 0;
	}
}

__global__ void execute_move(int nvtxs, int *vwgt, int *where, int *pwgts, char *select, int *to, int *moved)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && select[ii] == 1)
	{
		int t = to[ii];
		// int vwgt_i = vwgt[ii];
		int where_i = where[ii];
		moved[ii]++;
		// atomicAdd(&pwgts[t], vwgt_i);
		where[ii] = t;
		// atomicSub(&pwgts[where_i], vwgt_i);
	}
}

__global__ void select_balance_vertex(int num, int nparts, int bin, int *bin_offset, int *bin_idx, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int *where, int *pwgts, int *maxwgt, \
	char *select, int *gain, int *to, int *poverload, int *kway_bin)
{
	int blockwarp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int warp_id = blockIdx.x * 4 + blockwarp_id;

	extern __shared__ int connection_shared[];
	int *val_shared = connection_shared + blockwarp_id * nparts;

	for(int i = lane_id;i < nparts;i += 32)
		val_shared[i] = 0;
	__syncwarp();

	if(warp_id >= num)
		return ;
	
	int vertex = warp_id;
	int me = where[vertex];

	// if(lane_id == 0)
	// 	printf("vertex=%d me=%d poverload=%d return \n", vertex, me, poverload[me]);
	if(poverload[me] != OverLoaded)
		return ;

	if(vwgt[vertex] > maxwgt[me])
		return ;

	// if(lane_id == 0)
	// 	printf("vertex=%d me=%d poverload=%d\n", vertex, me, poverload[me]);
	
	int begin = xadj[vertex];
	int end   = xadj[vertex + 1];
	int id    = 0;
	
	for(int j = begin + lane_id;j < end;j += 32)
	{
		int vertex_k = adjncy[j];
		int where_k = where[vertex_k];

		// if(blockIdx.x == 0)
		// 	printf("vertex=%d me=%d k=%d where_k=%d wgt=%d\n", vertex, me, vertex_k, where_k, adjwgt[j]);

		if(where_k == me)
		{
			id += adjwgt[j];
			continue;
		}
		
		if(poverload[where_k] == OverLoaded)
			continue;
		
		atomicAdd(&val_shared[where_k], adjwgt[j]);
	}

	// if(blockIdx.x == 0)
	// 	printf("vertex=%d id=%d before scan\n", vertex, id);

	//	scan for id
	for(int i = 16;i > 0;i >>= 1)
	{
		int tmp_id = __shfl_down_sync(0xffffffff, id, i);
		if(lane_id < i)
			id += tmp_id;
	}
	id = __shfl_sync(0xffffffff, id, 0);

	// if(blockIdx.x == 0 && lane_id < nparts)
	// 	printf("vertex=%d id=%d\n", vertex, id);

	int loss_min = val_shared[lane_id] - id;
	int loss_to = lane_id;
	for(int j = lane_id + 32;j < nparts;j += 32)
	{
		int part_to = j;
		if(poverload[part_to] == OverLoaded)
			continue;
		
		if(loss_to == me || loss_min < val_shared[part_to])
		{
			loss_min = val_shared[part_to] - id;
			loss_to = part_to;
		}
	}

	if(poverload[me] == OverLoaded)
	// if(loss_to == me)
		loss_min = INT_MIN;

	// if(blockIdx.x == 0 && lane_id < nparts)
	// 	printf("vertex=%d id=%d loss_min=%d loss_to=%d\n", vertex, id, loss_min, loss_to);

	int range = min(32, nparts);
	range >>= 1;

	#pragma unroll
	while(range > 0)
	{
		int tmp_loss_min = __shfl_down_sync(0xffffffff, loss_min, range);
		int tmp_loss_to  = __shfl_down_sync(0xffffffff, loss_to, range);

		if(lane_id < range)
		{
			bool vaild = (tmp_loss_to != -1) & (tmp_loss_min > loss_min);
			if(vaild)
			{
				loss_min = tmp_loss_min;
				loss_to = tmp_loss_to;
			}
		}
		range >>= 1;
	}

	if(lane_id == 0)
	{
		// if(blockIdx.x == 0)
			// printf("vertex=%d me=%d id=%d loss_min=%d loss_to=%d\n", vertex, me, id, loss_min, loss_to);
		if(loss_to == -1 || loss_min == -id)
			return ;
		
		// printf("me=%d vertex=%d\n", me, vertex);
		// printf("vertex=%d me=%d id=%d loss_min=%d loss_to=%d\n", vertex, me, id, loss_min, loss_to);
		select[vertex] = 1;
		gain[vertex] = loss_min;
		to[vertex] = loss_to;
		atomicAdd(&kway_bin[me], 1);
	}
}

__global__ void select_balance_vertex_to(int num, int nparts, int bin, int *bin_offset, int *bin_idx, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int *where, int *pwgts, int *maxwgt, \
	char *select, int *gain, int *to, int *poverload, int *kway_bin)
{
	int blockwarp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int warp_id = blockIdx.x * 4 + blockwarp_id;

	extern __shared__ int connection_shared[];
	int *val_shared = connection_shared + blockwarp_id * nparts;

	for(int i = lane_id;i < nparts;i += 32)
		val_shared[i] = 0;
	__syncwarp();

	if(warp_id >= num)
		return ;
	
	int vertex = warp_id;
	int me = where[vertex];

	// if(lane_id == 0)
	// 	printf("vertex=%d me=%d poverload=%d return \n", vertex, me, poverload[me]);
	if(poverload[me] != OverLoaded)
		return ;

	if(vwgt[vertex] > maxwgt[me])
		return ;

	// if(lane_id == 0)
	// 	printf("vertex=%d me=%d vwgt=%d poverload=%d\n", vertex, me, vwgt[vertex], poverload[me]);
	
	int begin = xadj[vertex];
	int end   = xadj[vertex + 1];
	int id    = 0;
	
	for(int j = begin + lane_id;j < end;j += 32)
	{
		int vertex_k = adjncy[j];
		int where_k = where[vertex_k];

		// if(blockIdx.x == 0)
		// 	printf("vertex=%d me=%d k=%d where_k=%d wgt=%d\n", vertex, me, vertex_k, where_k, adjwgt[j]);

		if(where_k == me)
		{
			id += adjwgt[j];
			continue;
		}
		
		if(poverload[where_k] == OverLoaded)
			continue;
		
		atomicAdd(&val_shared[where_k], adjwgt[j]);
	}

	// if(blockIdx.x == 0)
	// 	printf("vertex=%d id=%d before scan\n", vertex, id);

	//	scan for id
	for(int i = 16;i > 0;i >>= 1)
	{
		int tmp_id = __shfl_down_sync(0xffffffff, id, i);
		if(lane_id < i)
			id += tmp_id;
	}
	id = __shfl_sync(0xffffffff, id, 0);

	// if(blockIdx.x == 0 && lane_id < nparts)
	// 	printf("vertex=%d id=%d\n", vertex, id);

	int loss_min = val_shared[lane_id] - id;
	int loss_to = lane_id;
	for(int j = lane_id + 32;j < nparts;j += 32)
	{
		int part_to = j;
		if(poverload[part_to] == OverLoaded)
			continue;
		
		if(loss_to == me || loss_min < val_shared[part_to])
		{
			loss_min = val_shared[part_to] - id;
			loss_to = part_to;
		}
	}

	if(poverload[me] == OverLoaded)
	{
		loss_min = INT_MIN;
		loss_to = -1;
	}
	// if(blockIdx.x == 0 && lane_id < nparts)
	// if(lane_id < nparts)
	// 	printf("vertex=%d id=%d loss_min=%d loss_to=%d\n", vertex, id, loss_min, loss_to);

	int range = min(32, nparts);
	range >>= 1;

	#pragma unroll
	while(range > 0)
	{
		int tmp_loss_min = __shfl_down_sync(0xffffffff, loss_min, range);
		int tmp_loss_to  = __shfl_down_sync(0xffffffff, loss_to, range);

		if(lane_id < range)
		{
			bool vaild = (tmp_loss_to != -1) & (tmp_loss_min > loss_min);
			if(vaild)
			{
				loss_min = tmp_loss_min;
				loss_to = tmp_loss_to;
			}
		}
		range >>= 1;
	}

	if(lane_id == 0)
	{
		// if(blockIdx.x == 0)
			// printf("vertex=%d me=%d id=%d loss_min=%d loss_to=%d before\n", vertex, me, id, loss_min, loss_to);
		if(loss_to == -1 || loss_min == -id)
			return ;
		
		// printf("me=%d vertex=%d\n", me, vertex);
		// printf("vertex=%d me=%d id=%d loss_min=%d loss_to=%d\n", vertex, me, id, loss_min, loss_to);
		select[vertex] = 1;
		gain[vertex] = loss_min;
		to[vertex] = loss_to;
		atomicAdd(&kway_bin[loss_to], 1);
	}
}

__global__ void set_kway_idx(int num, int nparts, int bin, int *bin_offset, int *bin_idx, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *gain, int *to, int *poverload, int *kway_size, int *kway_bin, int *kway_idx, int *kway_loss)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii >= num)
		return ;
	
	if(select[ii] == 1)
	{
		int me = where[ii];
		int offset = atomicAdd(&kway_size[me], 1);
		int ptr = kway_bin[me] + offset;

		kway_idx[ptr] = ii;
		kway_loss[ptr] = gain[ii];
	}
}

__global__ void set_kway_idx_to(int num, int nparts, int bin, int *bin_offset, int *bin_idx, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int *where, \
	char *select, int *gain, int *to, int *poverload, int *kway_size, int *kway_bin, int *kway_idx, int *kway_loss)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii >= num)
		return ;
	
	if(select[ii] == 1)
	{
		int me = where[ii];
		int to_v = to[ii];
		int offset = atomicAdd(&kway_size[to_v], 1);
		int ptr = kway_bin[to_v] + offset;

		kway_idx[ptr] = ii;
		kway_loss[ptr] = gain[ii];

		printf("ptr=%d idx=%d loss=%d\n", ptr, ii, gain[ii]);

	}
}
	
__global__ void exam_loss_sort(int num, int nparts, int bin, int *bin_offset, int *bin_idx, int *kway_loss)
{
	int ptr = bin_offset[bin];

	for(int offset = 0;offset < num;offset++)
	{
		int vertex = bin_idx[ptr + offset];
		int loss = kway_loss[ptr + offset];
		printf("loss=%10d vertex=%10d\n", loss, vertex);
	}
}

__global__ void count_move_num(int nparts, int *bin_offset, int *bin_idx, int *vwgt, int *pwgts, int *maxwgt, int *poverload, int *kway_size)
{
	int ii = blockIdx.x;
	int lane_id = threadIdx.x;

	int part_id = ii;
	if(poverload[part_id] != OverLoaded)
	{
		kway_size[part_id] = 0;
		return ;
	}
	if(lane_id != 0)
		return ;
	
	int part_wgt = pwgts[part_id];
	int max_allowed = maxwgt[part_id];
	int ptr = bin_offset[part_id];
	int end = bin_offset[part_id + 1];
	while(part_wgt > max_allowed && ptr < end)
	{
		int vertex = bin_idx[ptr];
		part_wgt -= vwgt[vertex];
		ptr++;
		// printf("part_id=%d vertex=%d part_wgt=%d max_allowed=%d ptr=%d\n", part_id, vertex, part_wgt, max_allowed, ptr);
	}
	printf("part_id=%d part_wgt=%d max_allowed=%d ptr=%d\n", part_id, part_wgt, max_allowed, ptr);
	kway_size[part_id] = ptr;
}

__global__ void execute_move_balance(int nvtxs, int nparts, int *bin, int *kway_size, int *bin_idx, int *vwgt, int *pwgts, int *maxwgt,int *where, int *to, int *poverload)
{
	int ii = blockIdx.x;
	int tid = threadIdx.x;

	int part_id = ii;
	if(poverload[part_id] != OverLoaded)
		return ;
	
	int begin = bin[part_id];
	int end = kway_size[part_id];
	for(int ptr = begin + tid;ptr < end;ptr += blockDim.x)
	{
		int vertex = bin_idx[ptr];
		// int from = where[vertex];
		int to_v = to[vertex];
		int wgt_v = vwgt[vertex];
		if(pwgts[to_v] + wgt_v <= maxwgt[to_v])
		{
			where[vertex] = to_v;
			// printf("ii=%d tid=%d vertex=%d begin=%d end=%d ptr=%d part_id=%d to=%d vwgt=%d\n", ii, tid, vertex, begin, end, ptr, part_id, to_v, wgt_v);
			atomicAdd(&pwgts[part_id], -wgt_v);
			atomicAdd(&pwgts[to_v], wgt_v);
		}
	}

}

__global__ void execute_move_balance_to(int nvtxs, int nparts, int *bin, int *kway_size, int *bin_idx, int *vwgt, int *pwgts, int *maxwgt,int *where, int *to, char *select, int *poverload)
{
	int ii = blockIdx.x;
	int tid = threadIdx.x;

	int part_id = ii;
	if(poverload[part_id] == OverLoaded)
		return ;

	if(tid != 0)
		return ;
	
	int begin = bin[part_id];
	int end   = bin[part_id + 1];

	for(int ptr = begin;ptr < end;ptr++)
	{
		int vertex = bin_idx[ptr];
		int from = where[vertex];
		// printf("vertex=%d vwgt=%d from=%d to=%d begin=%d end=%d ptr=%d\n", vertex, vwgt[vertex], from, part_id, begin, end, ptr);
		if(pwgts[from] > maxwgt[from])
		{
			int wgt_v = vwgt[vertex];
			if(pwgts[part_id] + wgt_v <= maxwgt[part_id])
			{
				select[vertex] = 1;
				where[vertex] = part_id;
				atomicSub(&pwgts[from], wgt_v);
				atomicAdd(&pwgts[part_id], wgt_v);
			}
		}
	}
}

__global__ void exam_where(int nvtxs, int *where)
{
	for(int i = 0;i < nvtxs && i < 100;i++)
		printf("%5d\n", where[i]);
}

		
__global__ void is_balance(int nvtxs, int nparts, int *pwgts, int *maxwgt, int *balance, int *poverload)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nparts)
	{
		// printf("ii:%d pwgts:%d maxwgt:%d\n", ii, pwgts[ii], maxwgt[ii]);
		bool flag = (pwgts[ii] > maxwgt[ii]);
		// balance[0] = flag ? 0 : OverLoaded;
		poverload[ii] = flag ? OverLoaded : 0;
		if(flag)
			atomicCAS(&balance[0], 1, 0);
		// if(!flag)
		// {
		// 	balance[0] = 1;
		// 	// printf("balance!\n");
		// }
		// else 
		// {
		// 	balance[0] = 0;
		// 	// printf("imbalance!\n");
		// }
	}
}

__global__ void compute_imb(const int nparts, int *pwgts, int *opt_pwgts, float *imb)
{
	int ii = threadIdx.x;

	extern __shared__ float imb_cache[];
	imb_cache[ii] = 0.0;
	__syncthreads();

	// printf("0 part=%10d imb_cache=%10.3f\n", ii, imb_cache[ii]);

	if(ii >= nparts)
		return ;
	
	int p = ii;
	int p_wgt = pwgts[p];
	int opt_p_wgt = opt_pwgts[p];
	float imb_val = (float)p_wgt / (float)opt_p_wgt;

	imb_cache[ii] = imb_val;
	for(int i = blockDim.x + ii;i < nparts;i += blockDim.x)
		imb_cache[ii] = max(imb_val, imb_cache[ii]);

	#pragma unroll
	for(int range = blockDim.x >> 1;range >= 32;range >>= 1)
	{
		if(ii < range)
			imb_cache[ii] = max(imb_cache[ii], imb_cache[ii + range]);
		
		__syncthreads();
	}

	if(ii < 32)
	{
		imb_val = imb_cache[ii];
		#pragma unroll
		for(int range = 16;range > 0;range >>= 1)
		{
			float tmp_imb = __shfl_down_sync(0xffffffff, imb_val, range, 32);
			if(ii < range)
				imb_val = max(imb_val, tmp_imb);
		}
	}

	if(ii == 0)
		imb[0] = imb_val;
}

__global__ void print_select(int nvtxs, char *select, int *to, int *gain, int *xadj)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && select[ii] == 1)
	{
		printf("ii=%10d to=%10d gain=%10d length=%10d\n", ii, to[ii], gain[ii], xadj[ii + 1] - xadj[ii]);
	}
}

__global__ void exam_pwgts(int nparts, int *pwgts, int *maxwgt, int *poverload)
{
	for(int i = 0;i < nparts;i++)
		printf("%10d ", pwgts[i]);
	printf("\n");
	for(int i = 0;i < nparts;i++)
		printf("%10d ", maxwgt[i]);
	printf("\n");
	for(int i = 0;i < nparts;i++)
		printf("%10d ", poverload[i]);
	printf("\n");
	for(int i = 0;i < nparts;i++)
		printf("%10.3lf ", (double)pwgts[i] / (double)maxwgt[i]);
	printf("\n");
	
}

void hunyuangraph_k_refinement_SC25(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int *level)
{
	// printf("hunyuangraph_k_refinement_SC25 begin\n");
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs  = graph->nvtxs;
	int nedges = graph->nedges;

	int edgecut;
	int balance;
	int select_sum;
	/*int best_cut, *best_where;
	float h_best_imb, *d_best_imb;

	best_cut = graph->mincut;
	graph->mincut = best_cut;

	h_best_imb = 0;
	if(GPU_Memory_Pool)
		d_best_imb = (float *)lmalloc_with_check(sizeof(float), "d_best_imb");
	else
		cudaMalloc((void **)&d_best_imb, sizeof(float));
	compute_imb<<<1, 1024, 1024 * sizeof(float)>>>(nparts, graph->cuda_pwgts, graph->cuda_opt_pwgts, d_best_imb);

	cudaMemcpy(&h_best_imb, d_best_imb, sizeof(float), cudaMemcpyDeviceToHost);

	if(GPU_Memory_Pool)
		lfree_with_check(d_best_imb, sizeof(float), "d_best_imb");
	else
		cudaFree(d_best_imb);

	if(GPU_Memory_Pool)
		best_where = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "best_where");
	else
		cudaMalloc((void **)&best_where, sizeof(int) * nvtxs);

	cudaMemcpy(best_where, graph->cuda_where, sizeof(int) * nvtxs, cudaMemcpyDeviceToDevice);*/

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
	Sum_maxmin_pwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_maxwgt, graph->cuda_tpwgts, graph->tvwgt[0],nparts);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_Sum_maxmin_pwgts += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
	init_moved<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_moved);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_select_init_select += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
		
	// cudaDeviceSynchronize();
	// exam_balance<<<1,1>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_minwgt);
	// cudaDeviceSynchronize();
	for(int iter = 0;iter < 10;iter++)
	{
		// init_connection<<<(nedges + 127) / 128, 128>>>(nedges, graph->cuda_connection, graph->cuda_connection_to);
		// cudaDeviceSynchronize();
		// exam_where<<<1, 1>>>(nvtxs, graph->cuda_where);
		// cudaDeviceSynchronize();

		int balance = 1;
		cudaMemcpy(graph->cuda_balance, &balance, sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		is_balance<<<(nparts + 31) / 32, 32>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_balance, graph->cuda_poverload);
		cudaDeviceSynchronize();

		// cudaDeviceSynchronize();
		// exam_pwgts<<<1, 1>>>(nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_poverload);
		// cudaDeviceSynchronize();

		cudaMemcpy(&balance, graph->cuda_balance, sizeof(int), cudaMemcpyDeviceToHost);
		// printf("balance=%d\n", balance[0]);

		//	if balance
		if(balance == 1)
		{
			// printf("to reduce edgecut\n");
			//	if balance, the lock for vertex moving can be unlocked 
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
			init_moved<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_moved);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_select_init_select += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
			init_select<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_select);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_select_init_select += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

			//	bnd and gain >= -0.15 * id
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
			for(int i = 1;i < 14;i++)
			{
				int num = graph->h_bin_offset[i + 1] - graph->h_bin_offset[i];
				// printf("select_bnd_vertices_subwarp %d begin num=%d %d\n", i, num, (int)pow(2, i));
				// cudaDeviceSynchronize();
				if(num != 0)
				switch(i)
				{
				case 1:
					select_bnd_vertices_subwarp<2><<<(num + 63) / 64, 128, sizeof(int) * (256 + 64)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 2:
					select_bnd_vertices_subwarp<4><<<(num + 31) / 32, 128, sizeof(int) * (256 + 32)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 3:
					select_bnd_vertices_subwarp<8><<<(num + 15) / 16, 128, sizeof(int) * (256 + 16)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 4:
					select_bnd_vertices_subwarp<16><<<(num + 7) / 8, 128, sizeof(int) * (256 + 8)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 5:
					select_bnd_vertices_subwarp<32><<<(num + 3) / 4, 128, sizeof(int) * (256 + 4)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 6:
					select_bnd_vertices_warp_bin<<<(num + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 7:
					select_bnd_vertices_warp_bin<<<(num + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 8:
					select_bnd_vertices_warp_bin<<<(num + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 9:
					select_bnd_vertices_warp_bin<<<(num + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 10:
					select_bnd_vertices_warp_bin<<<(num + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 11:
					select_bnd_vertices_warp_bin<<<(num + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 12:
					select_bnd_vertices_warp_bin<<<(num + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				case 13:
					select_bnd_vertices_warp_bin<<<(num + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(num, nparts, i, graph->bin_offset, graph->bin_idx, graph->cuda_xadj, graph->cuda_adjncy, \
						graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_moved);
					break;
				default:
					break;
				}
				// cudaDeviceSynchronize();
				// printf("select_bnd_vertices_subwarp %d end\n", i);
			}

			// select_bnd_vertices_warp<<<(nvtxs + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(nvtxs, nparts, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, \
			// 	graph->cuda_select, graph->cuda_gain, graph->cuda_to);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_select_bnd_vertices_warp += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
			// printf("select_bnd_vertices_warp end %10.3lf\n", uncoarsen_select_bnd_vertices_warp);
			// cudaDeviceSynchronize();
			// exam_where<<<1, 1>>>(nvtxs, graph->cuda_where);
			// cudaDeviceSynchronize();

			// cudaDeviceSynchronize();
			// print_select<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_select, graph->cuda_to, graph->cuda_gain, graph->cuda_xadj);
			// cudaDeviceSynchronize();

			// warp
			// cudaDeviceSynchronize();
			// init_select<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_select);
			// cudaDeviceSynchronize();

			// select_bnd_vertices_warp<<<(nvtxs + 3) / 4 , 128, (8 * nparts + 8) * sizeof(int)>>>(nvtxs, nparts, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, \
			// 	graph->cuda_select, graph->cuda_gain, graph->cuda_to);
			
			// cudaDeviceSynchronize();
			// select_sum = compute_graph_select_gpu(graph);
			// cudaDeviceSynchronize();
			// printf("warp second_select=%10d\n", select_sum);

			// cudaDeviceSynchronize();
			// print_select<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_select, graph->cuda_to, graph->cuda_gain, graph->cuda_xadj);
			// cudaDeviceSynchronize();

			//	update select
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
			moving_vertices_interaction_SC25<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, \
				graph->cuda_select, graph->cuda_gain, graph->cuda_to);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_moving_interaction += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
			
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
			update_select_SC25<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_select, graph->cuda_gain);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_update_select += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

			cudaDeviceSynchronize();
			int select_sum = compute_graph_select_gpu(graph);
			cudaDeviceSynchronize();
			// printf("subwarp second_select=%10d\n", select_sum);

			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
			execute_move<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_vwgt, graph->cuda_where, graph->cuda_pwgts, graph->cuda_select, graph->cuda_to, graph->cuda_moved);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_execute_move += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
		}
		else
		{
			// printf("to balance\n");
			//	if unbalance
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
			init_select<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_select);

			init_moved<<<(nparts + 128) / 128, 128>>>(nparts + 1, graph->cuda_kway_bin);
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_select_init_select += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

			cudaDeviceSynchronize();
			// select_balance_vertex<<<(nvtxs + 3) / 4, 128, 4 * nparts * sizeof(int)>>>(nvtxs, nparts, 0, graph->bin_offset, graph->bin_idx, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, \
			// 	graph->cuda_adjwgt, graph->cuda_where, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_poverload, &graph->cuda_kway_bin[1]);
			select_balance_vertex_to<<<(nvtxs + 3) / 4, 128, 4 * nparts * sizeof(int)>>>(nvtxs, nparts, 0, graph->bin_offset, graph->bin_idx, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, \
				graph->cuda_adjwgt, graph->cuda_where, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_poverload, &graph->cuda_kway_bin[1]);
			cudaDeviceSynchronize();

			cudaDeviceSynchronize();
			select_sum = compute_graph_select_gpu(graph);
			cudaDeviceSynchronize();
			// printf("second_select1=%10d\n", select_sum);

			if(select_sum == 0)
				break;

			cudaMemcpy(graph->h_kway_bin, graph->cuda_kway_bin, (nparts + 1) * sizeof(int), cudaMemcpyDeviceToHost);

			// printf("h_kway_bin:         0          1          2          3          4          5          6          7          8\n");
			// printf("h_kway_bin:");
			// for(int i = 0;i <= nparts;i++)
			// {
			// 	// int num = graph->h_kway_bin[i + 1] - graph->h_kway_bin[i];
			// 	printf("%10d ", graph->h_kway_bin[i]);
			// }
			// printf("\n");

			int *kway_bin_size;
			if(GPU_Memory_Pool)
			{
				prefixsum(graph->cuda_kway_bin + 1, graph->cuda_kway_bin + 1, nparts, prefixsum_blocksize, 1);	//0:lmalloc,1:rmalloc
				kway_bin_size = (int *)lmalloc_with_check(sizeof(int) * nparts, "kway_bin_size");
			}
			else
			{
				thrust::inclusive_scan(thrust::device, graph->cuda_kway_bin + 1, graph->cuda_kway_bin + 1 + nparts, graph->cuda_kway_bin + 1);
				cudaMalloc((void **)&kway_bin_size, sizeof(int) * nparts);
			}
			
			cudaMemcpy(graph->h_kway_bin, graph->cuda_kway_bin, (nparts + 1) * sizeof(int), cudaMemcpyDeviceToHost);

			// printf("h_kway_bin:");
			// for(int i = 0;i <= nparts;i++)
			// {
			// 	// int num = graph->h_kway_bin[i + 1] - graph->h_kway_bin[i];
			// 	printf("%10d ", graph->h_kway_bin[i]);
			// }
			// printf("\n");

			init_moved<<<(nparts + 127) / 128, 128>>>(nparts, kway_bin_size);

			// printf("set_kway_idx begin\n");
			// set_kway_idx<<<(nvtxs + 127) / 128, 128>>>(nvtxs, nparts, 0, graph->bin_offset, graph->bin_idx, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, \
			// 	graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_poverload, kway_bin_size, graph->cuda_kway_bin, graph->cuda_kway_idx, graph->cuda_kway_loss);
			set_kway_idx_to<<<(nvtxs + 127) / 128, 128>>>(nvtxs, nparts, 0, graph->bin_offset, graph->bin_idx, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, \
				graph->cuda_adjwgt, graph->cuda_where, graph->cuda_select, graph->cuda_gain, graph->cuda_to, graph->cuda_poverload, kway_bin_size, graph->cuda_kway_bin, graph->cuda_kway_idx, graph->cuda_kway_loss);

			// printf("sort begin\n");
			// for(int i = 0;i < nparts;i++)
			// {
			// 	int num = graph->h_kway_bin[i + 1] - graph->h_kway_bin[i];
			// 	if(num > 0)
			// 	{
			// 		thrust::sort_by_key(thrust::device, graph->cuda_kway_idx + graph->h_kway_bin[i], graph->cuda_kway_idx + graph->h_kway_bin[i + 1], graph->cuda_kway_loss + graph->h_kway_bin[i]);	//, thrust::greater<int>()
			// 		// sort_kway_idx<<<(num + 127) / 128, 128>>>(num, nparts, i, graph->cuda_kway_bin, graph->cuda_kway_idx, graph->cuda_kway_loss);
			// 		printf("nparts=%d:\n", i);
			// 		cudaDeviceSynchronize();
			// 		exam_loss_sort<<<1, 1>>>(num, nparts, i, graph->cuda_kway_bin, graph->cuda_kway_idx, graph->cuda_kway_loss);
			// 		cudaDeviceSynchronize();
			// 	}
			// }

			// count_move_num<<<nparts, 32>>>(nparts, graph->cuda_kway_bin, graph->cuda_kway_idx, graph->cuda_vwgt, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_poverload, kway_bin_size);

			cudaMemcpy(graph->h_kway_bin, kway_bin_size, nparts * sizeof(int), cudaMemcpyDeviceToHost);

			// printf("  kway_bin:         0          1          2          3          4          5          6          7          8\n");
			// printf("  kway_bin:");
			// for(int i = 0;i < nparts;i++)
			// {
			// 	// int num = graph->h_kway_bin[i + 1] - graph->h_kway_bin[i];
			// 	printf("%10d ", graph->h_kway_bin[i]);
			// }
			// printf("\n");

			gettimeofday(&begin_gpu_kway, NULL);
			init_select<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_select);
			cudaDeviceSynchronize();

			// execute_move_balance<<<nparts, 128>>>(nvtxs, nparts, graph->cuda_kway_bin, kway_bin_size, graph->cuda_kway_idx, graph->cuda_vwgt, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_where, graph->cuda_to, graph->cuda_poverload);
			execute_move_balance_to<<<nparts, 128>>>(nvtxs, nparts, graph->cuda_kway_bin, kway_bin_size, graph->cuda_kway_idx, graph->cuda_vwgt, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_where, graph->cuda_to, graph->cuda_select, graph->cuda_poverload);

			cudaDeviceSynchronize();
			select_sum = compute_graph_select_gpu(graph);
			cudaDeviceSynchronize();
			// printf("second_select2=%10d\n", select_sum);

			if(GPU_Memory_Pool)
				lfree_with_check(kway_bin_size, sizeof(int) * nparts, "kway_bin_size");
			else
				cudaFree(kway_bin_size);
		}
		
		// cudaDeviceSynchronize();
		// select_sum = compute_graph_select_gpu(graph);
		// cudaDeviceSynchronize();
		// printf("second_select=%10d\n", select_sum);
		
		// cudaDeviceSynchronize();
		// exam_where<<<1, 1>>>(nvtxs, graph->cuda_where);
		// cudaDeviceSynchronize();

		compute_edgecut_gpu(graph->nvtxs, &edgecut, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where);

		// printf("iter:%d %10d\n", iter, edgecut);

		// cudaDeviceSynchronize();
		// exam_pwgts<<<1, 1>>>(nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_poverload);
		// cudaDeviceSynchronize();

		// printf("iter:%d %10d\n", iter, edgecut);
		// cudaDeviceSynchronize();
		// is_balance<<<nparts, 32>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_balance);
		// cudaDeviceSynchronize();

		//copy current partition and relevant data to output partition if following conditions pass
		/*float h_curr_imb, *d_curr_imb;
		h_curr_imb = 0;
		if(GPU_Memory_Pool)
			d_curr_imb = (float *)lmalloc_with_check(sizeof(float), "d_curr_imb");
		else
			cudaMalloc((void **)&d_curr_imb, sizeof(float));

		compute_imb<<<1, 1024, 1024 * sizeof(float)>>>(nparts, graph->cuda_pwgts, graph->cuda_opt_pwgts, d_curr_imb);

		cudaMemcpy(&h_curr_imb, d_curr_imb, sizeof(float), cudaMemcpyDeviceToHost);

		if(GPU_Memory_Pool)
			lfree_with_check(d_curr_imb, sizeof(float), "d_curr_imb");
		else
			cudaFree(d_curr_imb);

		if(h_best_imb > IMB && h_curr_imb < h_best_imb)
		{
			h_best_imb = h_curr_imb;
			best_cut = graph->mincut;
			
			cudaMemcpy(best_where, graph->cuda_where, sizeof(int) * nvtxs, cudaMemcpyDeviceToDevice);
			// count = 0;
		}
		else if(graph->mincut < best_cut && (h_curr_imb <= IMB || h_curr_imb <= h_best_imb))
		{
			// if(graph->mincut < tol * best_cut)
			// 	count = 0;
			h_best_imb = h_curr_imb;
			best_cut = graph->mincut;
			
			cudaMemcpy(best_where, graph->cuda_where, sizeof(int) * nvtxs, cudaMemcpyDeviceToDevice);
		}*/

		// printf("count=%10d, h_curr_imb=%10f, h_best_imb=%10f, best_cut=%10d curr_cut=%10d \n", count, h_curr_imb, h_best_imb, best_cut, graph->mincut);
	}

	/*graph->mincut = best_cut;

	cudaMemcpy(graph->cuda_where, best_where, sizeof(int) * nvtxs, cudaMemcpyDeviceToDevice);

	if(GPU_Memory_Pool)
		lfree_with_check(best_where, sizeof(int) * nvtxs, "k_refine: best_where");
	else
		cudaFree(best_where);*/
}


__global__ void exam_gain(int nvtxs, int nparts, int *gain_offset, int *gain_val, int *gain_where, int *where)
{
	for(int i = 0;i < nvtxs;i++)
	{
		int begin = gain_offset[i];
		int end   = gain_offset[i + 1];

		printf("vertex: %10d %10d %10d %10d\n", i, where[i], gain_offset[i], gain_offset[i + 1]);

		for(int j = begin;j < end;j++)
			printf("%10d ", gain_where[j]);
		printf("\n");
		for(int j = begin;j < end;j++)
			printf("%10d ", gain_val[j]);
		printf("\n");
	}
}

__global__ void exam_partition(int nvtxs, int nparts, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int *where)
{
	for(int i = 0;i < nvtxs;i++)
	{
		int begin = xadj[i];
		int end   = xadj[i + 1];

		printf("vertex: %10d %10d %10d\n", i, xadj[i], xadj[i + 1]);

		for(int j = begin;j < end;j++)
			printf("%10d ", adjncy[j]);
		printf("\n");
		for(int j = begin;j < end;j++)
			printf("%10d ", where[adjncy[j]]);
		printf("\n");
		for(int j = begin;j < end;j++)
			printf("%10d ", adjwgt[j]);
		printf("\n");
	}
}

__global__ void exam_same(int num, int *right, int *check)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if(ii >= num)
		return ;
	
	if(right[ii] != check[ii])
		printf("ii=%10d right=%10d check=%10d\n", ii, right[ii], check[ii]);

}


template<class K>
__global__ void init_val(int length, K val, K *num)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if(ii >= length)
		return ;
	
	num[ii] = val;
}

__device__ void scan_prefixsum(int lane_id, int nparts, int *sub, int vertex)
{
    int ptr = lane_id;
    int iter = 0;
    int sum = 0; // 累计的总和，初始为0

    while (iter * 32 < nparts)
    {
        int sum_before_iter = sum;

        int vals = (ptr < nparts) ? sub[ptr] : 0;
        vals = prefixsum_warp(vals, lane_id);

        if (ptr < nparts)
            sub[ptr] = vals + sum_before_iter;

        int current_total = __shfl_sync(0xffffffff, vals, 31);

        sum = sum_before_iter + current_total;

        iter++;
        ptr += 32;
        __syncwarp();
    }
}

__global__ void select_dest_part(int nvtxs, double filter_ratio, int *dest_cache, int *dest_part, int *where, int *gain_offset, int *gain_val, int *gain_where, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int best = dest_cache[vertex];
	if(best != -1)
	{
		dest_part[vertex] = best;
		return ;
	}

	int p = where[vertex];
	best = p;
	int b_conn = 0;
    int p_conn = 0;
	int begin = gain_offset[vertex];
	int end   = gain_offset[vertex + 1];
	//finds potential destination as most connected part excluding p
	for(int j = begin;j < end;j++)
	{
		int j_conn = gain_val[j];
		if(j_conn == -1)
			break;
		int j_where = gain_where[j];
		if(j_conn > b_conn && j_where != p)
		{
			best = j_where;
			b_conn = j_conn;
		}
		else if(j_conn > 0 && j_where == p)
			p_conn = j_conn;
	}

	// printf("vertex=%10d best=%10d b_conn=%10d p_conn=%10d\n", vertex, best, b_conn, p_conn);

	gain[vertex] = 0;
	if(best != p)
	{
		// vertices must pass this filter in order to be considered further
        // b_conn >= p_conn may seem redundant but it is important
        // to address an edge case where floor(filter_ratio*p_conn) rounds to zero
		if(b_conn >= p_conn || ((p_conn - b_conn) < floor(filter_ratio * p_conn)))
			gain[vertex] = b_conn - p_conn;
		else	
			best = p;
	}
	dest_cache[vertex] = best;
	dest_part[vertex] = best;
}

__global__ void select_dest_part_filter_vertex(int nvtxs, double filter_ratio, int *dest_cache, int *dest_part, int *where, int *gain_offset, int *gain_val, int *gain_where, \
	char *lock, int *pregain, int *num_pos, int *filter_idx)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int best = dest_cache[vertex];
	bool skip_processing = (best != -1);
	bool write_flag = false;
    int tmp_gain = IDX_MIN;
	int local_counter, offset;
	if(skip_processing)
	{
		dest_part[vertex] = best;
		return ;
	}

	int p = where[vertex];
	best = p;
	int b_conn = 0;
    int p_conn = 0;
	int begin = gain_offset[vertex];
	int end   = gain_offset[vertex + 1];
	//finds potential destination as most connected part excluding p
	for(int j = begin;j < end;j++)
	{
		int j_where = gain_where[j];
		if(j_where == -1)
			break;
		int j_conn = gain_val[j];
		if(j_conn > b_conn && j_where != p)
		{
			best = j_where;
			b_conn = j_conn;
		}
		else if(j_conn > 0 && j_where == p)
			p_conn = j_conn;
	}

	// printf("vertex=%10d best=%10d b_conn=%10d p_conn=%10d\n", vertex, best, b_conn, p_conn);

	tmp_gain = IDX_MIN;
	if(best != p)
	{
		// vertices must pass this filter in order to be considered further
        // b_conn >= p_conn may seem redundant but it is important
        // to address an edge case where floor(filter_ratio*p_conn) rounds to zero
		if( (b_conn >= p_conn || ((p_conn - b_conn) < floor(filter_ratio * p_conn))) && lock[vertex] == 0)
		{
			int write_ptr = atomicAdd(num_pos, 1);
			filter_idx[write_ptr] = vertex;
			tmp_gain = b_conn - p_conn;
		}
		else	
			best = p;
	}

	dest_cache[vertex] = best;
	dest_part[vertex] = best;
	pregain[vertex] = tmp_gain;

}

__global__ void filter_potential_vertex(int nvtxs, int *where, int *dest_part, char *lock, int *pregain, int *gain, int *num_pos, int *filter_idx)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int p = where[vertex];
	int best = dest_part[vertex];

	if(p != best && lock[vertex] == 0)
	{
		int write_ptr = atomicAdd(num_pos, 1);
		// printf("num_pos=%10d write_ptr=%10d\n", num_pos[0], write_ptr);
		filter_idx[write_ptr] = vertex;
		pregain[vertex] = gain[vertex];
	}
	else
	{
		pregain[vertex] = IDX_MIN;
	}
}

__global__ void afterburner_heuristic(int num_pos, int *filter_idx, int *dest_part, int *where, int *pregain, \
	int *xadj, int *adjncy, int *adjwgt, char *select)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	int vertex = filter_idx[ii];
	int best = dest_part[vertex];
	int p = where[vertex];
	int igain = pregain[vertex];
	int begin = xadj[vertex];
	int end   = xadj[vertex + 1];

	int change = 0;
	for(int j = begin;j < end;j++)
	{
		int neighbor = adjncy[j];
		int nei_gain = pregain[neighbor];

		if(nei_gain > igain || (nei_gain == igain && neighbor < vertex))
		{
			int nei_where = dest_part[neighbor];
			int nei_wgt = adjwgt[j];

			if(nei_where == p)
				change -= nei_wgt;
			else if(nei_where == best)
				change += nei_wgt;
			
			nei_where = where[neighbor];
			if(nei_where == p)
				change += nei_wgt;
			else if(nei_where == best)
				change -= nei_wgt;
		}
		// printf("vertex=%10d where=%10d best=%10d neighbor=%10d igain=%10d nei_gain=%10d nei_dest=%10d nei_where=%10d adjwgt=%10d change=%10d\n", vertex, where[vertex], best, neighbor, igain, nei_gain, dest_part[neighbor], where[neighbor], adjwgt[j], change);
	}

	if(igain + change >= 0)
		select[vertex] = 1;
}

__global__ void filter_beneficial_moves(int num_pos, char *select, int *new_num_pos, int *filter_idx, int *pos_move)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	int vertex = filter_idx[ii];
	if(select[vertex] == 1)
	{
		int write_ptr = atomicAdd(new_num_pos, 1);
		pos_move[write_ptr] = vertex;
		select[vertex] = 0;
		// printf("vertex=%10d write_ptr=%10d pos_move=%10d\n", vertex, write_ptr, pos_move[write_ptr]);
	}
}

__global__ void filter_beneficial_moves_lock(int num_pos, char *select, int *new_num_pos, int *filter_idx, int *pos_move, char *lock)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	int vertex = filter_idx[ii];
	if(select[vertex] == 1)
	{
		int write_ptr = atomicAdd(new_num_pos, 1);
		pos_move[write_ptr] = vertex;
		select[vertex] = 0;
		lock[vertex] = 1;
		// printf("vertex=%10d write_ptr=%10d pos_move=%10d\n", vertex, write_ptr, pos_move[write_ptr]);
	}
}

__global__ void set_lock(int num_pos, int *pos_move, char *lock)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	int vertex = pos_move[ii];
	lock[vertex] = 1;
}

__global__ void exam_destpart(int nvtxs, int *where, int *dest_part, int *dest_cache)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	printf("vertex=%10d where=%10d dest_part=%10d dest_cache=%10d\n", ii, where[ii], dest_part[ii], dest_cache[ii]);
}

__global__ void exam_filteridx(int num_pos, int *filter_idx, int *pregain)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	printf("vertex=%10d filter_idx=%10d pregain=%10d\n", ii, filter_idx[ii], pregain[ii]);
}

__global__ void exam_afterburner(int num_pos, int *filter_idx, char *select, int *pregain)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	printf("vertex=%10d filter_idx=%10d select=%10d\n", ii, filter_idx[ii], select[filter_idx[ii]]);
}

int jetlp(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int level)
{
	// cudaError_t cuda_err;

	int nparts = hunyuangraph_admin->nparts;
	int nvtxs = graph->nvtxs;
	double filter_ratio = 0.75;
	if(level == 0)
		filter_ratio = 0.25;

	int *d_num_pos, *pregain, *filter_idx;
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		d_num_pos = (int *)lmalloc_with_check(sizeof(int), "jetlp: d_num_pos");
		pregain = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "jetlp: pregain");
		filter_idx = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "jetlp: filter_idx");
	}
	else
	{
		cudaMalloc((void **)&d_num_pos, sizeof(int));
		cudaMalloc((void **)&pregain, sizeof(int) * nvtxs);
		cudaMalloc((void **)&filter_idx, sizeof(int) * nvtxs);
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL); 
	double tmp_time = (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
	uncoarsen_lp -= tmp_time;
	uncoarsen_gpu_malloc += tmp_time;
#endif

	init_val<<<1, 1>>>(1, 0, d_num_pos);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	select_dest_part<<<(nvtxs + 127) / 128, 128>>>(nvtxs, filter_ratio, graph->dest_cache, graph->dest_part, graph->cuda_where, graph->gain_offset, \
		graph->gain_val, graph->gain_where, graph->cuda_gain);
	// select_dest_part_filter_vertex<<<(nvtxs + 127) / 128, 128>>>(nvtxs, filter_ratio, graph->dest_cache, graph->dest_part, graph->cuda_where, graph->gain_offset, \
	// 	graph->gain_val, graph->gain_where, graph->lock, pregain, d_num_pos, filter_idx);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	lp_select_dest_part += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
#endif
	// printf("select_dest_part end\n");
	
	// cudaDeviceSynchronize();
	// exam_destpart<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_where, graph->dest_part, graph->dest_cache);
	// cudaDeviceSynchronize();

	// cudaDeviceSynchronize();
	// gettimeofday(&begin_gpu_kway_pm, NULL);
	filter_potential_vertex<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_where, graph->dest_part, graph->lock, pregain, graph->cuda_gain, d_num_pos, filter_idx);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
	// cudaDeviceSynchronize();
	// gettimeofday(&end_gpu_kway_pm, NULL);
	// lp_filter_potential_vertex += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
	// printf("filter_potential_vertex end\n");
	// cudaDeviceSynchronize();

	int h_num_pos;
	cudaMemcpy(&h_num_pos, d_num_pos, sizeof(int), cudaMemcpyDeviceToHost);

	// printf("h_num_pos=%10d\n", h_num_pos);

	if(h_num_pos == 0)
	{
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
		if(GPU_Memory_Pool)
		{
			lfree_with_check(filter_idx, sizeof(int) * nvtxs, "jetlp: filter_idx");
			lfree_with_check(pregain, sizeof(int) * nvtxs, "jetlp: pregain");
			lfree_with_check(d_num_pos, sizeof(int), "jetlp: d_num_pos");
		}
		else
		{
			cudaFree(filter_idx);
			cudaFree(pregain);
			cudaFree(d_num_pos);
		}
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_kway_pm, NULL);
		tmp_time = (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
		uncoarsen_lp -= tmp_time;
		uncoarsen_gpu_free += tmp_time;
#endif

		return 0;
	}

	// cudaDeviceSynchronize();
	// exam_filteridx<<<(nvtxs + 127) / 128, 128>>>(nvtxs, filter_idx, pregain);
	// cudaDeviceSynchronize();

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	afterburner_heuristic<<<(h_num_pos + 127) / 128, 128>>>(h_num_pos, filter_idx, graph->dest_part, graph->cuda_where, pregain, graph->cuda_xadj, graph->cuda_adjncy, \
		graph->cuda_adjwgt, graph->cuda_select);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	lp_afterburner_heuristic += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
#endif
	// printf("afterburner_heuristic end\n");
	
	// cudaDeviceSynchronize();
	// exam_afterburner<<<(h_num_pos + 127) / 128, 128>>>(h_num_pos, filter_idx, graph->cuda_select, pregain);
	// cudaDeviceSynchronize();

	init_val<<<1, 1>>>(1, 0, d_num_pos);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	// init_val<char><<<(nvtxs + 127) / 128, 128>>>(nvtxs, 0, graph->lock);
	init_val<<<(nvtxs + 127) / 128, 128>>>(nvtxs, (char)0, graph->lock);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	lp_init += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	filter_beneficial_moves<<<(h_num_pos + 127) / 128, 128>>>(h_num_pos, graph->cuda_select, d_num_pos, filter_idx, graph->pos_move);
	// filter_beneficial_moves_lock<<<(h_num_pos + 127) / 128, 128>>>(h_num_pos, graph->cuda_select, d_num_pos, filter_idx, graph->pos_move, graph->lock);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
#endif
	lp_filter_beneficial_moves += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
	// printf("filter_beneficial_moves end\n");

	cudaMemcpy(&h_num_pos, d_num_pos, sizeof(int), cudaMemcpyDeviceToHost);

	// printf("h_num_pos=%10d\n", h_num_pos);

	if(h_num_pos == 0)
	{
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_kway_pm, NULL);	
#endif
		if(GPU_Memory_Pool)
		{
			lfree_with_check(filter_idx, sizeof(int) * nvtxs, "jetlp: filter_idx");
			lfree_with_check(pregain, sizeof(int) * nvtxs, "jetlp: pregain");
			lfree_with_check(d_num_pos, sizeof(int), "jetlp: d_num_pos");
		}
		else
		{
			cudaFree(filter_idx);
			cudaFree(pregain);
			cudaFree(d_num_pos);
		}
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_kway_pm, NULL);
		tmp_time = (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
		uncoarsen_lp -= tmp_time;
		uncoarsen_gpu_free += tmp_time;
#endif

		return 0;
	}

	// cudaDeviceSynchronize();
	// gettimeofday(&begin_gpu_kway_pm, NULL);
	set_lock<<<(h_num_pos + 127) / 128, 128>>>(h_num_pos, graph->pos_move, graph->lock);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
	// cudaDeviceSynchronize();
	// gettimeofday(&end_gpu_kway_pm, NULL);
	// lp_set_lock += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
	// printf("set_lock end\n");

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		lfree_with_check(filter_idx, sizeof(int) * nvtxs, "jetlp: filter_idx");
		lfree_with_check(pregain, sizeof(int) * nvtxs, "jetlp: pregain");
		lfree_with_check(d_num_pos, sizeof(int), "jetlp: d_num_pos");
	}
	else
	{
		cudaFree(filter_idx);
		cudaFree(pregain);
		cudaFree(d_num_pos);
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	tmp_time = (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
	uncoarsen_lp -= tmp_time;
	uncoarsen_gpu_free += tmp_time;
#endif

	return h_num_pos;
}

const int max_buckets = 50;
const int mid_bucket = 25;
const int max_sections = 128;

__device__ int gain_bucket(const int gain_vertex, const int vwgt)
{
	float gain = static_cast<float>(gain_vertex) / static_cast<float>(vwgt);
	int gain_type = 0;

	if(gain > 0.0)
		gain_type = 0;
	else if(gain == 0.0)
		gain_type = 1;
	else
	{
		gain_type = mid_bucket;
        gain = abs(gain);
		if(gain < 1.0)
		{
			while(gain < 1.0)
			{
				gain *= 1.5;
                gain_type--;
			}
			if(gain_type < 2)
                gain_type = 2;
		}
		else
		{
			while(gain > 1.0)
			{
				gain /= 1.5;
                gain_type++;
			}
			if(gain_type > max_buckets)
                gain_type = max_buckets - 1;
		}
	}

	return gain_type;
}

__global__ void assign_move_scores_part1(const int nvtxs, int *where, int *poverload, int *gain_offset, int *gain_val, int *gain_where, \
	int *vwgt, int *pwgts, int *opt_pwgts, int *bucket_sizes, const int sections)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int p = where[vertex];

	if(poverload[p] != OverLoaded)
		return ;
	
	int gain_begin = gain_offset[vertex];
	int gain_end   = gain_offset[vertex + 1];
	int p_gain = 0;
	int tk = 0;
	int tg = 0;
	for(int j = gain_begin;j < gain_end;j++)
	{
		int nei_where = gain_where[j];
		if(nei_where == p)
			p_gain = gain_val[j];
		else if(nei_where > -1)
		{
			if(poverload[p] != OverLoaded)
			{
				tg += gain_val[j];
				tk ++;
			}
		}
	}

	if(tk == 0)
		tk = 1;
	int tmp_gain = (tg / tk) - p_gain;
	int wgt_v = vwgt[vertex];
	int gain_type = gain_bucket(tmp_gain, wgt_v);

	// printf("vertex=%10d where=%10d vwgt=%10d gain_type=%10d pwgts=%10d opt_pwgts=%10d\n", vertex, p, wgt_v, gain_type, pwgts[p], opt_pwgts[p]);
	if(gain_type < max_buckets && wgt_v < 2 * (pwgts[p] - opt_pwgts[p]))
	{
		int g_id = (max_buckets * p + gain_type) * sections + (vertex % sections) + 1;
		atomicAdd(&bucket_sizes[g_id], 1);
		// printf("vertex=%10d where=%10d vwgt=%10d gain_type=%10d g_id=%10d\n", vertex, p, wgt_v, gain_type, g_id);
	}
}

__global__ void scan_scores_small(int num, int *bucket_sizes, int *bucket_offsets)
{

}

__global__ void scan_scores_big(int num, int *bucket_sizes, int *bucket_offsets)
{

}

__global__ void assign_move_scores_part2(const int nvtxs, int *where, int *poverload, int *gain_offset, int *gain_val, int *gain_where, \
	int *vwgt, int *pwgts, int *opt_pwgts, int *bucket_sizes, int *bucket_offsets, int *least_bad_moves, const int sections)
{
	int ii = blockDim.x* blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int p = where[vertex];
	if(poverload[p] != OverLoaded)
		return ;
	
	int gain_begin = gain_offset[vertex];
	int gain_end   = gain_offset[vertex + 1];
	int p_gain = 0;
	int tk = 0;
	int tg = 0;

	for(int j = gain_begin;j < gain_end;j++)
	{
		int nei_where = gain_where[j];
		if(nei_where == p)
			p_gain = gain_val[j];
		else if(nei_where > -1)
		{
			if(poverload[p] != OverLoaded)
			{
				tg += gain_val[j];
				tk ++;
			}
		}
	}

	if(tk == 0)
		tk = 1;
	int tmp_gain = (tg / tk) - p_gain;
	int wgt_v = vwgt[vertex];
	int gain_type = gain_bucket(tmp_gain, wgt_v);

	// printf("vertex=%10d where=%10d vwgt=%10d gain_type=%10d pwgts=%10d opt_pwgts=%10d\n", vertex, p, wgt_v, gain_type, pwgts[p], opt_pwgts[p]);
	if(gain_type < max_buckets && wgt_v < 2 * (pwgts[p] - opt_pwgts[p]))
	{
		int g_id = (max_buckets * p + gain_type) * sections + (vertex % sections) + 1;
		int offset = atomicAdd(&bucket_sizes[g_id], 1);
		int ptr = bucket_offsets[g_id] + offset;
		least_bad_moves[ptr] = vertex;
		// printf("vertex=%10d where=%10d vwgt=%10d gain_type=%10d g_id=%10d offset=%10d ptr=%10d least_bad_moves=%10d\n", vertex, p, wgt_v, gain_type, g_id, offset, ptr, least_bad_moves[ptr]);
	}
}

__global__ void assign_move_scores_part3(const int num_pos, int *least_bad_moves, int *balance_scan, int *vwgt)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii > num_pos)
		return ;
	
	if(ii == num_pos)
		balance_scan[0] = 0;
	else 
	{
		// printf("ii=%10d vertex=%10d\n", ii, least_bad_moves[ii]);
		int vertex = least_bad_moves[ii];
		balance_scan[ii + 1] = vwgt[vertex];
	}

	// printf("ii=%10d balance_scan=%10d\n", ii, balance_scan[ii]);
}

__global__ void find_score_cutoffs(const int nparts, int *bucket_offsets, int *poverload, int *pwgts, int *maxwgt, int *balance_scan, \
	int *evict_start, int *evict_end, const int sections)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nparts)
		return ;
	
	int idx = ii;
	evict_start[idx] = bucket_offsets[idx * max_buckets * sections];
	
	if(poverload[idx] == OverLoaded)
	{
		int evict_total = pwgts[idx] - maxwgt[idx];
		int bucket_start = bucket_offsets[idx * max_buckets * sections];
		int bucket_end   = bucket_offsets[(idx + 1) * max_buckets * sections];
		int find = balance_scan[bucket_start] + evict_total;
		int mid = (bucket_start + bucket_end) / 2;

		while(bucket_start + 1 < bucket_end)
		{
			if(balance_scan[mid] >= find)
				bucket_end = mid;
			else
				bucket_start = mid;
			mid = (bucket_start + bucket_end) / 2;
		}

		evict_end[idx] = bucket_end;
	}
	else
		evict_end[idx] = bucket_offsets[idx * max_buckets * sections];
	
	// printf("idx=%10d evict_start=%10d evict_end=%10d\n", idx, evict_start[idx], evict_end[idx]);
}

__global__ void filter_below_cutoffs(const int num_pos, int *least_bad_moves, int *where, int *evict_end, int *d_num_pos, int *pos_move)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	int vertex = least_bad_moves[ii];
	int p = where[vertex];
	if(ii < evict_end[p])
	{
		int write_ptr = atomicAdd(&d_num_pos[0], 1);
		pos_move[write_ptr] = vertex;

		// printf("vertex=%10d write_ptr=%10d pos_move=%10d\n", vertex, write_ptr, pos_move[write_ptr]);
	}
}

__global__ void balance_scan_evicted_vertices(const int num_pos, int *pos_move, int *vwgt, int *balance_scan)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii > num_pos)
		return ;
	
	if(ii == num_pos)
		balance_scan[0] = 0;
	else
	{
		int vertex = pos_move[ii];
		balance_scan[ii + 1] = vwgt[vertex];
	}

	// printf("ii=%10d balance_scan=%10d\n", ii, balance_scan[ii]);
}

__global__ void cookie_cutter(int *evict_start, int *evict_end, const int nparts, int *maxwgt, int *pwgts, const int num_pos, int *balance_scan)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii != 0)
		return ;
	
	evict_start[0] = 0;
	for(int p = 0;p < nparts;p++)
	{
		int select = 0;
		if(maxwgt[p] > pwgts[p])
			select = maxwgt[p] - pwgts[p];
		
		int start = evict_start[p];
		int end = num_pos;
		int find = balance_scan[start] + select;
		int mid = (start + end) / 2;
		while(start + 1 < end)
		{
			if(balance_scan[mid] >= find)
				end = mid;
			else
				start = mid;
			mid = (start + end) / 2;
		}

		if(abs(balance_scan[end] - find) < abs(balance_scan[start] - find))
			evict_end[p] = end;
        else 
			evict_end[p] = start;
        if(p + 1 < nparts)
			evict_start[p + 1] = evict_end[p];
		
		// printf("idx=%10d evict_start=%10d evict_end=%10d\n", p, evict_start[p], evict_end[p]);
	}
}

__global__ void select_dest_parts_rs(const int num_pos, const int nparts, int *evict_start, int *evict_end, int *dest_part, int *pos_move, int *where)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	int p = 0;
	while(p < nparts && evict_start[p] <= ii)
		p++;
	p--;

	int vertex = pos_move[ii];
	if(ii < evict_end[p])
		dest_part[vertex] = p;
	else
		dest_part[vertex] = where[vertex];
	// dest_part[vertex] = (ii < evict_end[p]) ? p : where[vertex];
	
	// printf("vertex=%10d where=%10d p=%10d evict_start=%10d evict_end=%10d dest_part=%10d\n", vertex, where[vertex], p, evict_start[p], evict_end[p], dest_part[vertex]);
}

__global__ void exam_balanceoffset(int num, int *bucket_offsets)
{
	for(int i = 0;i < num;i++)
		printf("i=%10d offset=%10d\n", i, bucket_offsets[i]);
}

int jetrs(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs = graph->nvtxs;
	int sections = max_sections;

	int section_size = (nvtxs + sections * nparts) / (sections * nparts);
    if(section_size < 4096)
	{
        section_size = 4096;
        sections = (nvtxs + section_size * nparts) / (section_size * nparts);
    }
	int t_minibuckets = max_buckets * nparts * sections;

	if(t_minibuckets == 0)
		return 0;

	// printf("sections=%10d t_minibuckets=%10d\n", sections, t_minibuckets);

	int *bucket_offsets, *bucket_sizes, *least_bad_moves;
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		bucket_offsets = (int *)lmalloc_with_check(sizeof(int) * (t_minibuckets + 1), "jetrs: bucket_offsets");
		bucket_sizes = (int *)lmalloc_with_check(sizeof(int) * t_minibuckets, "jetrs: bucket_sizes");
		least_bad_moves = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "jetrs: least_bad_moves");
	}
	else
	{
		cudaMalloc((void **)&bucket_offsets, sizeof(int) * (t_minibuckets + 1));
		cudaMalloc((void **)&bucket_sizes, sizeof(int) * t_minibuckets);
		cudaMalloc((void **)&least_bad_moves, sizeof(int) * nvtxs);
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	double tmp_time = (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
	uncoarsen_rs -= tmp_time;
	uncoarsen_gpu_malloc += tmp_time;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	init_val<<<(t_minibuckets + 1 + 127) / 128, 128>>>(t_minibuckets + 1, 0, bucket_offsets);
	init_val<<<(t_minibuckets + 127) / 128, 128>>>(t_minibuckets, 0, bucket_sizes);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_init += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	assign_move_scores_part1<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_where, graph->cuda_poverload, graph->gain_offset, graph->gain_val, graph->gain_where, \
		graph->cuda_vwgt, graph->cuda_pwgts, graph->cuda_opt_pwgts, &bucket_offsets[1], sections);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_assign_move_scores_part1 += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;

	// printf("assign_move_scores_part1 end\n");

	// cudaDeviceSynchronize();
	// exam_balanceoffset<<<1, 1>>>(t_minibuckets + 1, bucket_offsets);
	// cudaDeviceSynchronize();
	
	// if(t_minibuckets < 10000)
	// {
	// 	scan_scores_small<<<1, 1024>>>(t_minibuckets + 2, bucket_offsets, bucket_offsets);
	// }
	// else
	// {
	// 	scan_scores_big<<<(t_minibuckets + 2 + 127) / 128, 128>>>(t_minibuckets + 2, bucket_offsets, bucket_offsets);
	// }
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	if(GPU_Memory_Pool)
		prefixsum(&bucket_offsets[1], &bucket_offsets[1], t_minibuckets, prefixsum_blocksize, 0);		//0:lmalloc,1:rmalloc
	else 
		thrust::inclusive_scan(thrust::device, bucket_offsets + 1, bucket_offsets + t_minibuckets + 1, bucket_offsets + 1);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_prefixsum += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
#endif

	int h_num_pos = 0;
	cudaMemcpy(&h_num_pos, &bucket_offsets[t_minibuckets], sizeof(int), cudaMemcpyDeviceToHost);
	// printf("h_num_pos=%10d\n", h_num_pos);

	if(h_num_pos == 0)
	{
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
		if(GPU_Memory_Pool)
		{
			lfree_with_check(least_bad_moves, sizeof(int) * nvtxs, "jetrs: least_bad_moves");
			lfree_with_check(bucket_sizes, sizeof(int) * t_minibuckets, "jetrs: bucket_sizes");
			lfree_with_check(bucket_offsets, sizeof(int) * (t_minibuckets + 1), "jetrs: bucket_offsets");
		}
		else
		{
			cudaFree(least_bad_moves);
			cudaFree(bucket_sizes);
			cudaFree(bucket_offsets);
		}
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_kway_rs, NULL);
		tmp_time = (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
		uncoarsen_rs -= tmp_time;
		uncoarsen_gpu_free += tmp_time;
#endif
		return 0;
	}

	// cudaDeviceSynchronize();
	// exam_balanceoffset<<<1, 1>>>(t_minibuckets + 1, bucket_offsets);
	// cudaDeviceSynchronize();
	
	// cudaDeviceSynchronize();
	// gettimeofday(&begin_gpu_kway_rs, NULL);
	// init_val<<<(t_minibuckets + 127) / 128, 128>>>(t_minibuckets, 0, bucket_sizes);
	// cudaDeviceSynchronize();
	// gettimeofday(&end_gpu_kway_rs, NULL);
	// rs_init += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
	
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	assign_move_scores_part2<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_where, graph->cuda_poverload, graph->gain_offset, graph->gain_val, graph->gain_where, \
		graph->cuda_vwgt, graph->cuda_pwgts, graph->cuda_opt_pwgts, bucket_sizes, bucket_offsets, least_bad_moves, sections);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_assign_move_scores_part2 += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
#endif
	// printf("assign_move_scores_part2 end\n");

	int *balance_scan, *evict_start, *evict_end, *d_num_pos, balance_scan_size;
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		balance_scan_size = h_num_pos + 1;
		balance_scan = (int *)lmalloc_with_check(sizeof(int) * (h_num_pos + 1), "jetrs: balance_scan");
		evict_start = (int *)lmalloc_with_check(sizeof(int) * nparts, "jetrs: evict_start");
		evict_end = (int *)lmalloc_with_check(sizeof(int) * nparts, "jetrs: evict_end");
		d_num_pos = (int *)lmalloc_with_check(sizeof(int), "jetrs: d_num_pos");
	}
	else
	{
		cudaMalloc((void **)&balance_scan, sizeof(int) * (h_num_pos + 1));
		cudaMalloc((void **)&evict_start, sizeof(int) * nparts);
		cudaMalloc((void **)&evict_end, sizeof(int) * nparts);
		cudaMalloc((void **)&d_num_pos, sizeof(int));
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	tmp_time = (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
	uncoarsen_rs -= tmp_time;
	uncoarsen_gpu_malloc += tmp_time;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	// cudaMemset(balance_scan, 0, sizeof(int));
	assign_move_scores_part3<<<(h_num_pos + 1 + 127) / 128, 128>>>(h_num_pos, least_bad_moves, balance_scan, graph->cuda_vwgt);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_assign_move_scores_part3 += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
#endif
	// printf("assign_move_scores_part3 end\n");
	
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	if(GPU_Memory_Pool)
		prefixsum(&balance_scan[1], &balance_scan[1], h_num_pos, prefixsum_blocksize, 0);		//0:lmalloc,1:rmalloc
	else 
		thrust::inclusive_scan(thrust::device, balance_scan + 1, balance_scan + 1 + h_num_pos, balance_scan + 1);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_prefixsum += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	find_score_cutoffs<<<(nparts + 31) / 32, 32>>>(nparts, bucket_offsets, graph->cuda_poverload, graph->cuda_pwgts, graph->cuda_maxwgt, balance_scan, \
		evict_start, evict_end, sections);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_find_score_cutoffs += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
#endif
	// printf("find_score_cutoffs end\n");

	init_val<<<1, 1>>>(1, 0, d_num_pos);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	filter_below_cutoffs<<<(h_num_pos + 127) / 128, 128>>>(h_num_pos, least_bad_moves, graph->cuda_where, evict_end, d_num_pos, graph->pos_move);	
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_filter_below_cutoffs += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
#endif

	cudaMemcpy(&h_num_pos, d_num_pos, sizeof(int), cudaMemcpyDeviceToHost);
	// printf("h_num_pos=%10d\n", h_num_pos);

	if(h_num_pos == 0)
	{
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
		if(GPU_Memory_Pool)
		{
			lfree_with_check(d_num_pos, sizeof(int), "jetrs: d_num_pos");
			lfree_with_check(evict_end, sizeof(int) * nparts, "jetrs: evict_end");
			lfree_with_check(evict_start, sizeof(int) * nparts, "jetrs: evict_start");
			lfree_with_check(balance_scan, sizeof(int) * balance_scan_size, "jetrs: balance_scan");
			lfree_with_check(least_bad_moves, sizeof(int) * nvtxs, "jetrs: least_bad_moves");
			lfree_with_check(bucket_sizes, sizeof(int) * t_minibuckets, "jetrs: bucket_sizes");
			lfree_with_check(bucket_offsets, sizeof(int) * (t_minibuckets + 1), "jetrs: bucket_offsets");
		}
		else
		{
			cudaFree(d_num_pos);
			cudaFree(evict_end);
			cudaFree(evict_start);
			cudaFree(balance_scan);
			cudaFree(least_bad_moves);
			cudaFree(bucket_sizes);
			cudaFree(bucket_offsets);
		}
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_kway_rs, NULL);
		tmp_time = (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
		uncoarsen_rs -= tmp_time;
		uncoarsen_gpu_free += tmp_time;
#endif
		return 0;
	}

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	// cudaMemset(balance_scan, 0, sizeof(int));
	balance_scan_evicted_vertices<<<(h_num_pos + 1 + 127) / 128, 128>>>(h_num_pos, graph->pos_move, graph->cuda_vwgt, balance_scan);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_balance_scan_evicted_vertices += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
	// printf("balance_scan_evicted_vertices end\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	if(GPU_Memory_Pool)
		prefixsum(&balance_scan[1], &balance_scan[1], h_num_pos, prefixsum_blocksize, 0);		//0:lmalloc,1:rmalloc
	else 
		thrust::inclusive_scan(thrust::device, balance_scan + 1, balance_scan + 1 + h_num_pos, balance_scan + 1);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_prefixsum += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	cookie_cutter<<<1, 1>>>(evict_start, evict_end, nparts, graph->cuda_maxwgt, graph->cuda_pwgts, h_num_pos, balance_scan);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	rs_cookie_cutter += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;

	// printf("cookie_cutter end\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	select_dest_parts_rs<<<(h_num_pos + 127) / 128, 128>>>(h_num_pos, nparts, evict_start, evict_end, graph->dest_part, graph->pos_move, graph->cuda_where);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
#endif
	rs_select_dest_parts += (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
	// printf("select_dest_parts_rs end\n");

	h_num_pos = 0;
	cudaMemcpy(&h_num_pos, d_num_pos, sizeof(int), cudaMemcpyDeviceToHost);
	// printf("h_num_pos=%10d\n", h_num_pos);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rs, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		lfree_with_check(d_num_pos, sizeof(int), "jetrs: d_num_pos");
		lfree_with_check(evict_end, sizeof(int) * nparts, "jetrs: evict_end");
		lfree_with_check(evict_start, sizeof(int) * nparts, "jetrs: evict_start");
		lfree_with_check(balance_scan, sizeof(int) * balance_scan_size, "jetrs: balance_scan");
		lfree_with_check(least_bad_moves, sizeof(int) * nvtxs, "jetrs: least_bad_moves");
		lfree_with_check(bucket_sizes, sizeof(int) * t_minibuckets, "jetrs: bucket_sizes");
		lfree_with_check(bucket_offsets, sizeof(int) * (t_minibuckets + 1), "jetrs: bucket_offsets");
	}
	else
	{
		cudaFree(d_num_pos);
		cudaFree(evict_end);
		cudaFree(evict_start);
		cudaFree(balance_scan);
		cudaFree(least_bad_moves);
		cudaFree(bucket_sizes);
		cudaFree(bucket_offsets);
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rs, NULL);
	tmp_time = (end_gpu_kway_rs.tv_sec - begin_gpu_kway_rs.tv_sec) * 1000.0 + (end_gpu_kway_rs.tv_usec - begin_gpu_kway_rs.tv_usec) / 1000.0;
	uncoarsen_rs -= tmp_time;
	uncoarsen_gpu_free += tmp_time;
#endif
	return h_num_pos;
}

__global__ void set_maxdest(const int nparts, int *max_dest, int *maxwgt)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nparts)
		return ;
	
	int val = maxwgt[ii] * 0.99;
	if(val < maxwgt[ii] - 100)
		val = maxwgt[ii] - 100;
	max_dest[ii] = val;

	// printf("part=%10d max_dest=%10d\n", ii, max_dest[ii]);
}

__global__ void init_undersized_parts_list(const int nparts, int *pwgts, int *max_dest, int *undersized, int *total_undersized)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nparts)
		return ;
	
	if(pwgts[ii] < max_dest[ii])
	{
		int write_ptr = atomicAdd(total_undersized, 1);
		undersized[write_ptr] = ii;

		// printf("ptr=%10d undersized=%10d\n", write_ptr, undersized[write_ptr]);
	}
}

__global__ void set_maxdest_init_undersized_parts_list(const int nparts, int *maxwgt, int *pwgts, int *undersized, int *total_undersized)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nparts)
		return ;
	
	int val = maxwgt[ii] * 0.99;
	if(val < maxwgt[ii] - 100)
		val = maxwgt[ii] - 100;
	
	if(pwgts[ii] < val)
	{
		int write_ptr = atomicAdd(total_undersized, 1);
		undersized[write_ptr] = ii;

		// printf("ptr=%10d undersized=%10d\n", write_ptr, undersized[write_ptr]);
	}
}

__global__ void select_dest_parts_rw(const int nvtxs, int *where, int *vwgt, int *pwgts, int *maxwgt, int *opt_pwgts, int *gain_offset, \
	int *gain_where, int *gain_val, int *dest_part, int *save_gains, int *undersized, int *total_undersized)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int p = where[vertex];
	int p_gain = 0;
	int best = p;
	int gain = 0;
	int wgt_v = vwgt[vertex];

	if(pwgts[p] > maxwgt[p] && wgt_v < 1.5 * (pwgts[p] - opt_pwgts[p]))
	{
		int gain_start = gain_offset[vertex];
		int gain_end = gain_offset[vertex + 1];

		for(int j = gain_start; j < gain_end; j++)
		{
			int nei_where = gain_where[j];
			if(nei_where > -1 && pwgts[nei_where] < maxwgt[nei_where])
			{
				int nei_conn = gain_val[j];
				if(nei_conn > gain)
				{
					best = nei_where;
					gain = nei_conn;
				}
			}
			if(nei_where == p)
				p_gain = gain_val[j];
			
			// printf("vertex=%10d p=%10d p_gain=%10d nei_where=%10d pwgts=%10d maxwgt=%10d best=%10d gain=%10d\n", vertex, p, p_gain, nei_where, pwgts[nei_where], maxwgt[nei_where], best, gain);
		}

		if(gain > 0)
		{
			dest_part[vertex] = best;
			save_gains[vertex] = gain - p_gain;
		}
		else
		{
			best = undersized[vertex % total_undersized[0]];
            dest_part[vertex] = best;
            save_gains[vertex] = -p_gain;
		}

		// printf("vertex=%10d gain=%10d where=%10d dest_part=%10d save_gains=%10d\n", vertex, gain, p, dest_part[vertex], save_gains[vertex]);

		// if(p != best)
		// 	printf("vertex=%10d gain=%10d where=%10d dest_part=%10d save_gains=%10d\n", vertex, gain, p, dest_part[vertex], save_gains[vertex]);
	}
	else
		dest_part[vertex] = p;

}

__global__ void assign_move_scores(const int nvtxs, int *where, int *dest_part, int *bid, int *save_gains, int *vwgt, const int sections, \
	int *bucket_sizes, int *vscore)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int p = where[vertex];
	int best = dest_part[vertex];
	bid[vertex] = -1;

	if(p != best)
	{
		int gain = save_gains[vertex];
		int wgt_v = vwgt[vertex];
		int gain_type = gain_bucket(gain, wgt_v);
		int g_id = (max_buckets * p + gain_type) * sections + (vertex % sections);
		bid[vertex] = g_id;
		vscore[vertex] = atomicAdd(&bucket_sizes[g_id], wgt_v);

		// printf("vertex=%10d where=%10d gain=%10d vwgt=%10d gain_type=%10d g_id=%10d vscore=%10d\n", vertex, p, gain, wgt_v, gain_type, g_id, vscore[vertex]);
	}
}

__global__ void select_dest_parts_assign_move_scores(const int nvtxs, int *where, int *vwgt, int *pwgts, int *maxwgt, int *opt_pwgts, int *gain_offset, \
	int *gain_where, int *gain_val, int *dest_part, int *undersized, int *total_undersized, int *bid, const int sections, int *bucket_sizes, int *vscore)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int p = where[vertex];
	int p_gain = 0;
	int best = p;
	int gain = 0;
	int wgt_v = vwgt[vertex];
	bid[vertex] = -1;

	if(pwgts[p] > maxwgt[p] && wgt_v < 1.5 * (pwgts[p] - opt_pwgts[p]))
	{
		int gain_start = gain_offset[vertex];
		int gain_end = gain_offset[vertex + 1];

		for(int j = gain_start; j < gain_end; j++)
		{
			int nei_where = gain_where[j];
			if(nei_where > -1 && pwgts[nei_where] < maxwgt[nei_where])
			{
				int nei_conn = gain_val[j];
				if(nei_conn > gain)
				{
					best = nei_where;
					gain = nei_conn;
				}
			}
			if(nei_where == p)
				p_gain = gain_val[j];
			
			// printf("vertex=%10d p=%10d p_gain=%10d nei_where=%10d pwgts=%10d maxwgt=%10d best=%10d gain=%10d\n", vertex, p, p_gain, nei_where, pwgts[nei_where], maxwgt[nei_where], best, gain);
		}

		bool vaild = (gain > 0);
		// save_gains[vertex] = vaild ? gain - p_gain : -p_gain;
		int regi_gain = vaild ? gain - p_gain : -p_gain;
		if(!vaild)
			best = undersized[vertex % total_undersized[0]];
		dest_part[vertex] = best;

		// if(gain > 0)
		// {
		// 	dest_part[vertex] = best;
		// 	save_gains[vertex] = gain - p_gain;
		// }
		// else
		// {
		// 	best = undersized[vertex % total_undersized[0]];
        //     dest_part[vertex] = best;
        //     save_gains[vertex] = -p_gain;
		// }

		if(p != best)
		{
			int wgt_v = vwgt[vertex];
			int gain_type = gain_bucket(regi_gain, wgt_v);
			int g_id = (max_buckets * p + gain_type) * sections + (vertex % sections);
			bid[vertex] = g_id;
			vscore[vertex] = atomicAdd(&bucket_sizes[g_id], wgt_v);
		}

		// printf("vertex=%10d gain=%10d where=%10d dest_part=%10d save_gains=%10d\n", vertex, gain, p, dest_part[vertex], save_gains[vertex]);

		// if(p != best)
		// 	printf("vertex=%10d gain=%10d where=%10d dest_part=%10d save_gains=%10d\n", vertex, gain, p, dest_part[vertex], save_gains[vertex]);
	}
	else
		dest_part[vertex] = p;

}

__global__ void filter_scores_below_cutoff(const int nvtxs, int *bid, int *where, const int sections, int *vscore, int *bucket_offsets, \
	int *pwgts, int *maxwgt, int *num_pos, int *pos_move)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int b = bid[vertex];
	if(b == -1)
		return ;
	
	int p = where[vertex];
	int begin_bucket = max_buckets * p * sections;
	int score = vscore[vertex] + bucket_offsets[b] - bucket_offsets[begin_bucket];
	int limit = pwgts[p] - maxwgt[p];
	if(score < limit)
	{
		int write_ptr = atomicAdd(num_pos, 1);
		pos_move[write_ptr] = vertex;
		// printf("vertex=%10d b=%10d score=%10d limit=%10d write_ptr=%10d pos_move=%10d\n", vertex, b, score, limit, write_ptr, pos_move[write_ptr]);
	}
}

__global__ void exam_destpart_savegains(int nvtxs, int *dest_part, int *save_gains)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int best = dest_part[vertex];
	int gain = save_gains[vertex];

	// printf("vertex=%10d dest_part=%10d save_gains=%10d\n", vertex, best, gain);
}

int jetrw(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs = graph->nvtxs;
	int sections = max_sections;

	int section_size = (nvtxs + sections * nparts) / (sections * nparts);
    if(section_size < 4096)
	{
        section_size = 4096;
        sections = (nvtxs + section_size * nparts) / (section_size * nparts);
    }
	int t_minibuckets = max_buckets * nparts * sections;

	// printf("sections=%10d t_minibuckets=%10d\n", sections, t_minibuckets);

	int *max_dest, *bucket_offsets, *total_undersized, *undersized, *save_gains, *bid, *vscore, *d_num_pos;
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rw, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		max_dest = (int *)lmalloc_with_check(sizeof(int) * nparts, "jetrw: max_dest");
		bucket_offsets = (int *)lmalloc_with_check(sizeof(int) * (t_minibuckets + 1), "jetrw: bucket_offsets");
		total_undersized = (int *)lmalloc_with_check(sizeof(int), "jetrw: total_undersized");
		undersized = (int *)lmalloc_with_check(sizeof(int) * nparts, "jetrw: undersized");
		save_gains = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "jetrw: save_gains");
		bid = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "jetrw: bid");
		vscore = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "jetrw: vscore");
		d_num_pos = (int *)lmalloc_with_check(sizeof(int), "jetrw: d_num_pos");
	}
	else
	{
		cudaMalloc((void **)&max_dest, sizeof(int) * nparts);
		cudaMalloc((void **)&bucket_offsets, sizeof(int) * (t_minibuckets + 1));
		cudaMalloc((void **)&total_undersized, sizeof(int));
		cudaMalloc((void **)&undersized, sizeof(int) * nparts);
		cudaMalloc((void **)&save_gains, sizeof(int) * nvtxs);
		cudaMalloc((void **)&bid, sizeof(int) * nvtxs);
		cudaMalloc((void **)&vscore, sizeof(int) * nvtxs);
		cudaMalloc((void **)&d_num_pos, sizeof(int));
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rw, NULL);
	double tmp_time = (end_gpu_kway_rw.tv_sec - begin_gpu_kway_rw.tv_sec) * 1000.0 + (end_gpu_kway_rw.tv_usec - begin_gpu_kway_rw.tv_usec) / 1000.0;
	uncoarsen_rw -= tmp_time;
	uncoarsen_gpu_malloc += tmp_time;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rw, NULL);
#endif
	init_val<<<1, 1>>>(1, 0, total_undersized);
	init_val<<<(t_minibuckets + 1 + 127) / 128, 128>>>(t_minibuckets + 1, 0, bucket_offsets);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rw, NULL);
	rw_init += (end_gpu_kway_rw.tv_sec - begin_gpu_kway_rw.tv_sec) * 1000.0 + (end_gpu_kway_rw.tv_usec - begin_gpu_kway_rw.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rw, NULL);
#endif
	set_maxdest<<<(nparts + 31) / 32, 32>>>(nparts, max_dest, graph->cuda_maxwgt);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
	// printf("set_maxdest end\n");

	init_undersized_parts_list<<<(nparts + 31) / 32, 32>>>(nparts, graph->cuda_pwgts, max_dest, undersized, total_undersized);
	// set_maxdest_init_undersized_parts_list<<<(nparts + 31) / 32, 32>>>(nparts, graph->cuda_maxwgt, graph->cuda_pwgts, undersized, total_undersized);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rw, NULL);
	rw_parts += (end_gpu_kway_rw.tv_sec - begin_gpu_kway_rw.tv_sec) * 1000.0 + (end_gpu_kway_rw.tv_usec - begin_gpu_kway_rw.tv_usec) / 1000.0;
	// printf("init_undersized_parts_list end\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rw, NULL);
#endif
	select_dest_parts_rw<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_where, graph->cuda_vwgt, graph->cuda_pwgts, graph->cuda_maxwgt, \
		graph->cuda_opt_pwgts, graph->gain_offset, graph->gain_where, graph->gain_val, graph->dest_part, save_gains, undersized, \
		total_undersized);
	// select_dest_parts_assign_move_scores<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_where, graph->cuda_vwgt, graph->cuda_pwgts, graph->cuda_maxwgt, \
	// 	graph->cuda_opt_pwgts, graph->gain_offset, graph->gain_where, graph->gain_val, graph->dest_part, undersized, \
	// 	total_undersized, bid, sections, &bucket_offsets[1], vscore);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rw, NULL);
	rw_select_dest_parts += (end_gpu_kway_rw.tv_sec - begin_gpu_kway_rw.tv_sec) * 1000.0 + (end_gpu_kway_rw.tv_usec - begin_gpu_kway_rw.tv_usec) / 1000.0;
#endif

	// printf("select_dest_parts_rw end\n");
	
	// cudaDeviceSynchronize();
	// exam_destpart_savegains<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->dest_part, save_gains);
	// cudaDeviceSynchronize();

	// cudaDeviceSynchronize();
	// gettimeofday(&begin_gpu_kway_rw, NULL);
	assign_move_scores<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_where, graph->dest_part, bid, save_gains, graph->cuda_vwgt, sections, \
		&bucket_offsets[1], vscore);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
	// cudaDeviceSynchronize();
	// gettimeofday(&end_gpu_kway_rw, NULL);
	// rw_assign_move_scores += (end_gpu_kway_rw.tv_sec - begin_gpu_kway_rw.tv_sec) * 1000.0 + (end_gpu_kway_rw.tv_usec - begin_gpu_kway_rw.tv_usec) / 1000.0;
	// printf("assign_move_scores end\n");

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rw, NULL);
#endif
	if(GPU_Memory_Pool)
		prefixsum(&bucket_offsets[1], &bucket_offsets[1], t_minibuckets, prefixsum_blocksize, 0);		//0:lmalloc,1:rmalloc
	else 
		thrust::inclusive_scan(thrust::device, bucket_offsets + 1, bucket_offsets + t_minibuckets + 1, bucket_offsets + 1);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rw, NULL);
	rw_prefixsum += (end_gpu_kway_rw.tv_sec - begin_gpu_kway_rw.tv_sec) * 1000.0 + (end_gpu_kway_rw.tv_usec - begin_gpu_kway_rw.tv_usec) / 1000.0;
#endif

	init_val<<<1, 1>>>(1, 0, d_num_pos);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rw, NULL);
#endif
	filter_scores_below_cutoff<<<(nvtxs + 127) / 128, 128>>>(nvtxs, bid, graph->cuda_where, sections, vscore, bucket_offsets, \
		graph->cuda_pwgts, graph->cuda_maxwgt, d_num_pos, graph->pos_move);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rw, NULL);
	rw_filter_scores_below_cutoff += (end_gpu_kway_rw.tv_sec - begin_gpu_kway_rw.tv_sec) * 1000.0 + (end_gpu_kway_rw.tv_usec - begin_gpu_kway_rw.tv_usec) / 1000.0;
#endif
	// printf("filter_scores_below_cutoff end\n");
	
	int h_num_pos = 0;
	cudaMemcpy(&h_num_pos, d_num_pos, sizeof(int), cudaMemcpyDeviceToHost);
	// printf("h_num_pos=%10d\n", h_num_pos);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_rw, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		lfree_with_check(d_num_pos, sizeof(int), "jetrw: d_num_pos");
		lfree_with_check(vscore, sizeof(int) * nvtxs, "jetrw: vscore");
		lfree_with_check(bid, sizeof(int) * nvtxs, "jetrw: bid");
		lfree_with_check(save_gains, sizeof(int) * nvtxs, "jetrw: save_gains");
		lfree_with_check(undersized, sizeof(int) * nparts, "jetrw: undersized");
		lfree_with_check(total_undersized, sizeof(int), "jetrw: total_undersized");
		lfree_with_check(bucket_offsets, sizeof(int) * (t_minibuckets + 1), "jetrw: bucket_offsets");
		lfree_with_check(max_dest, sizeof(int) * nparts, "jetrw: max_dest");
	}
	else
	{
		cudaFree(d_num_pos);
		cudaFree(vscore);
		cudaFree(bid);
		cudaFree(save_gains);
		cudaFree(undersized);
		cudaFree(total_undersized);
		cudaFree(bucket_offsets);
		cudaFree(max_dest);
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_rw, NULL);
	tmp_time = (end_gpu_kway_rw.tv_sec - begin_gpu_kway_rw.tv_sec) * 1000.0 + (end_gpu_kway_rw.tv_usec - begin_gpu_kway_rw.tv_usec) / 1000.0;
	uncoarsen_rw -= tmp_time;
	uncoarsen_gpu_free += tmp_time;
#endif
	return h_num_pos;
}

__global__ void mark_adjacent(int num_pos, int *pos_move, int *xadj, int *adjncy, int *mark)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	int vertex = pos_move[ii];
	int begin = xadj[vertex];
	int end   = xadj[vertex + 1];

	for(int j = begin;j < end;j++)
	{
		int neighbor = adjncy[j];
		if(mark[neighbor] == 0)
			mark[neighbor] = 1;
	}
}

__global__ void reset_conn_DS(int nvtxs, int nparts, int *mark, int *gain_offset, int *dest_cache, int *gain_val, int *gain_where, \
	int *xadj, int *adjncy, int *adjwgt, int *where, int *vals_global)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	if(mark[vertex] == 0)
		return ;
	
	int write_ptr = gain_offset[vertex];
	int write_end   = gain_offset[vertex + 1];
	int write_length  = 0;
	dest_cache[vertex] = -1;

	for(int j = write_ptr;j < write_end;j++)
		gain_val[j] = 0;
	for(int j = write_ptr;j < write_end;j++)
		gain_where[j] = -1;
	__syncthreads();

	int begin = xadj[vertex];
	int end   = xadj[vertex + 1];
	int *vals = vals_global + nparts * vertex;
	for(int j = begin; j < end;j++)
	{
		int neighbor = adjncy[j];
		int nei_where = where[neighbor];
		vals[nei_where] += adjwgt[j];
	}

	for(int i = 0;i < nparts;i++)
	{
		if(vals[i] != 0)
		{
			gain_val[write_ptr + write_length] = vals[i];
			gain_where[write_ptr + write_length] = i;
			write_length++;
		}
	}

	while(write_ptr + write_length < write_end)
	{
		gain_val[write_ptr + write_length] = 0;
		gain_where[write_ptr + write_length] = -1;
		write_length++;
	}
}

__global__ void reset_conn_DS_warp(int nvtxs, int nparts, int *mark, int *gain_offset, int *dest_cache, int *gain_val, int *gain_where, \
	int *xadj, int *adjncy, int *adjwgt, int *where)
{
	const int blockwarp_id = threadIdx.x >> 5;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = blockIdx.x * 4 + blockwarp_id;

	extern __shared__ int cache_vals[];
	int *vals = cache_vals + blockwarp_id * 2 * nparts; // 每个warp占用2*nparts
    int *sub = vals + nparts;                           // sub紧随vals之后

	for(int i = lane_id;i < nparts;i += 32)
		vals[i] = 0;
	__syncwarp();

	if(warp_id >= nvtxs)
		return ;
	
	int vertex = warp_id;
	if(mark[vertex] == 0)
		return ;
	
#ifdef FIGURE14_EDGECUT
	dest_cache[vertex] = -1;
#endif

	int begin = xadj[vertex];
	int end   = xadj[vertex + 1];
	int gain_begin = gain_offset[vertex];
	int gain_end   = gain_offset[vertex + 1];

	for(int j = begin + lane_id;j < end;j += 32)
	{
		int neighbor = adjncy[j];
		int nei_where = where[neighbor];
		atomicAdd(&vals[nei_where], adjwgt[j]);
	}
	__syncwarp();

	// 	for(int i = lane_id;i < nparts;i += 32)
	// 	{
	// 		int tmp_val = vals[i];
	// 		printf("vertex=%10d i=%10d lane_id=%10d gain_where=%10d vals=%10d\n", vertex, i, lane_id, i, tmp_val);
	// 	}
	// __syncwarp();

	for(int i = lane_id;i < nparts;i += 32)
		sub[i] = (vals[i] == 0) ? 1 : 0;
	__syncwarp();

	// 	for(int i = lane_id;i < nparts;i += 32)
	// 	{
	// 		printf("scan_prefixsum begin vertex=%10d i=%10d sub=%10d\n", vertex, i, sub[i]);
	// 	}
	// __syncwarp();

	scan_prefixsum(lane_id, nparts, sub, vertex);
	__syncwarp();

	// 	for(int i = lane_id;i < nparts;i += 32)
	// 	{
	// 		printf("scan_prefixsum end   vertex=%10d i=%10d sub=%10d\n", vertex, i, sub[i]);
	// 	}
	// __syncwarp();

	for(int i = lane_id;i < nparts;i += 32)
	{
		if(vals[i] != 0)
		{
			int offset = sub[i];
			gain_where[gain_begin + i - offset] = i;
			gain_val[gain_begin + i - offset] = vals[i];
		}
	}

	// __syncwarp();
	// 	for(int i = gain_begin + lane_id;i < gain_end;i += 32)
	// 	{
	// 		printf("mid gain_end - gain_begin=%10d vertex=%10d i=%10d gain_where=%10d gain_vals=%10d ptr=%10d offset=%10d\n", gain_end - gain_begin, vertex, i, gain_where[i], gain_val[i], gain_begin + nparts - sub[nparts - 1], sub[nparts - 1]);
	// 	}
	// __syncwarp();

	for(int i = gain_begin + nparts - sub[nparts - 1] + lane_id;i < gain_end;i += 32)
	{
		gain_where[i] = -1;
		gain_val[i] = 0;
	}

	// __syncwarp();
	// for(int i = gain_begin + lane_id;i < gain_end;i += 32)
	// {
	// 	printf("gain_end - gain_begin=%10d vertex=%10d i=%10d gain_where=%10d gain_vals=%10d\n", gain_end - gain_begin, vertex, i, gain_where[i], gain_val[i]);
	// }

}

void update_large(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int h_num_pos)
{
	int nvtxs = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;

	int *mark, *vals;
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm_update, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		mark = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "update_large: mark");
		// vals = (int *)lmalloc_with_check(sizeof(int) * nvtxs * nparts, "update_large: vals");
	}
	else
	{
		cudaMalloc((void **)&mark, sizeof(int) * nvtxs);
		// cudaMalloc((void **)&vals, sizeof(int) * nvtxs * nparts);
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm_update, NULL);
	double tmp_time = (end_gpu_kway_pm_update.tv_sec - begin_gpu_kway_pm_update.tv_sec) * 1000.0 + (end_gpu_kway_pm_update.tv_usec - begin_gpu_kway_pm_update.tv_usec) / 1000.0;
	pm_update_large -= tmp_time;
	uncoarsen_gpu_malloc += tmp_time;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm_update, NULL);
#endif
	init_val<<<(nvtxs + 127) / 128, 128>>>(nvtxs, 0, mark);
	// init_val<<<(nvtxs * nparts + 127) / 128, 128>>>(nvtxs * nparts, 0, vals);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm_update, NULL);
	pm_update_init += (end_gpu_kway_pm_update.tv_sec - begin_gpu_kway_pm_update.tv_sec) * 1000.0 + (end_gpu_kway_pm_update.tv_usec - begin_gpu_kway_pm_update.tv_usec) / 1000.0;
	// printf("update_large init_val end\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm_update, NULL);
#endif
	mark_adjacent<<<(h_num_pos + 127) / 128, 128>>>(h_num_pos, graph->pos_move, graph->cuda_xadj, graph->cuda_adjncy, mark);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm_update, NULL);
	pm_update_mark_adjacent += (end_gpu_kway_pm_update.tv_sec - begin_gpu_kway_pm_update.tv_sec) * 1000.0 + (end_gpu_kway_pm_update.tv_usec - begin_gpu_kway_pm_update.tv_usec) / 1000.0;
	// printf("mark_adjacent end\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm_update, NULL);
#endif

	/*printf("-------------------------------------------------------------------------------------\n");
	int gain_size;
	cudaMemcpy(&gain_size, &graph->gain_offset[nvtxs], sizeof(int), cudaMemcpyDeviceToHost);
	int *g_val, *g_where;
	if(GPU_Memory_Pool)
	{
		g_val = (int *)lmalloc_with_check(sizeof(int) * gain_size, "k_refine: g_val");
		g_where = (int *)lmalloc_with_check(sizeof(int) * gain_size, "k_refine: g_where");
	}
	else
	{
		cudaMalloc((void **)&g_val, sizeof(int) * gain_size);
		cudaMalloc((void **)&g_where, sizeof(int) * gain_size);
	}
	cudaMemcpy(g_val, graph->gain_val, sizeof(int) * gain_size, cudaMemcpyDeviceToDevice);
	cudaMemcpy(g_where, graph->gain_where, sizeof(int) * gain_size, cudaMemcpyDeviceToDevice);*/

	// reset_conn_DS<<<(nvtxs + 127) / 128, 128>>>(nvtxs, nparts, mark, graph->gain_offset, graph->dest_cache, graph->gain_val, \
	// 	graph->gain_where, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, vals);
	reset_conn_DS_warp<<<(nvtxs + 3) / 4, 128, 8 * nparts * sizeof(int)>>>(nvtxs, nparts, mark, graph->gain_offset, graph->dest_cache, graph->gain_val, \
		graph->gain_where, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());

	// cudaDeviceSynchronize();
	// exam_gain<<<1, 1>>>(nvtxs, nparts, graph->gain_offset, graph->gain_val, graph->gain_where, graph->cuda_where);
	// cudaDeviceSynchronize();
	// printf("-------------------------------------------------------------------------------------\n");
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm_update, NULL);
	pm_update_reset_conn_DS += (end_gpu_kway_pm_update.tv_sec - begin_gpu_kway_pm_update.tv_sec) * 1000.0 + (end_gpu_kway_pm_update.tv_usec - begin_gpu_kway_pm_update.tv_usec) / 1000.0;
	// printf("reset_conn_DS end\n");
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm_update, NULL);
#endif
	
	/*cudaDeviceSynchronize();
	reset_conn_DS_warp<<<(nvtxs + 3) / 4, 128, 8 * nparts * sizeof(int)>>>(nvtxs, nparts, mark, graph->gain_offset, graph->dest_cache, g_val, \
		g_where, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where);
	CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
	cudaDeviceSynchronize();
	
	// cudaDeviceSynchronize();
	// exam_gain<<<1, 1>>>(nvtxs, nparts, graph->gain_offset, g_val, g_where, graph->cuda_where);
	// cudaDeviceSynchronize();
	
	printf("update_large reset_conn_DS_warp\n");
	cudaDeviceSynchronize();
	exam_same<<<(gain_size + 127) / 128, 128>>>(gain_size, graph->gain_val, g_val);
	cudaDeviceSynchronize();
	printf("-------------------------------------------------------------------------------------\n");
	cudaDeviceSynchronize();
	exam_same<<<(gain_size + 127) / 128, 128>>>(gain_size, graph->gain_where, g_where);
	cudaDeviceSynchronize();

	if(GPU_Memory_Pool)
	{
		lfree_with_check(g_where, sizeof(int) * gain_size, "k_refine: g_where");
		lfree_with_check(g_val, sizeof(int) * gain_size, "k_refine: g_val");
	}
	else
	{
		cudaFree(g_where);
		cudaFree(g_val);
	}*/

	if(GPU_Memory_Pool)
	{
		// lfree_with_check(vals, sizeof(int) * nvtxs * nparts, "update_large: vals");
		lfree_with_check(mark, sizeof(int) * nvtxs, "update_large: mark");
	}
	else
	{
		// cudaFree(vals);
		cudaFree(mark);
	}

	// exit(0)
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm_update, NULL);
	tmp_time = (end_gpu_kway_pm_update.tv_sec - begin_gpu_kway_pm_update.tv_sec) * 1000.0 + (end_gpu_kway_pm_update.tv_usec - begin_gpu_kway_pm_update.tv_usec) / 1000.0;
	pm_update_large -= tmp_time;
	uncoarsen_gpu_free += tmp_time;
#endif

}

void update_small(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int h_num_pos)
{
	
}

__device__ int lookup(int *val, int *where, int key, int length)
{
	for(int i = 0;i < length;i++)
	{
		if(where[i] == key)
			return val[i];
		else if(where[i] == -1)
			return 0;
	}
	return 0;
}

__device__ int scan_add_warp(int val, int lane_id)
{
	int range = 16;
	
	#pragma unroll
	while(range >= 32)
	{
		int tmp_val = __shfl_down_sync(0xffffffff, val, range, 32);
		if(lane_id < range)
			val += tmp_val;

		// printf("wwarp lane_id=%10d range=%10d val=%10d\n", lane_id, range, val);
		range >>= 1;
	}

	return val;
}

__device__ int scan_add_block(int nparts, int val, int *val_cache)
{
	int range = blockDim.x >> 1;

	val_cache[threadIdx.x] = val;
	__syncthreads();

	//	block
	#pragma unroll
	while(range >= 1)
	{
		if(threadIdx.x < range)
			val_cache[threadIdx.x] += val_cache[threadIdx.x + range];
		
		// if(threadIdx.x < range)
		// 	printf("block threadIdx.x=%10d range=%10d val_cache=%10d\n", threadIdx.x, range, val_cache[threadIdx.x]);
		
		range >>= 1;

		__syncthreads();
	}

	return val_cache[threadIdx.x];
}

__global__ void count_edgecut_change1(int num_pos, int nparts, int *pos_move, int *dest_part, int *where, int *gain_offset, int *gain_val, int *gain_where, int *d_change)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;

	extern __shared__ int val_cache[];
	int change = 0;

	if(ii < num_pos)
	{
		int vertex = pos_move[ii];
		int best = dest_part[vertex];
		int p = where[vertex];
		int begin = gain_offset[vertex];
		int end = gain_offset[vertex + 1];
		int length = end - begin;
		int p_con = lookup(gain_val + begin, gain_where + begin, p   , end - begin);
		int b_con = lookup(gain_val + begin, gain_where + begin, best, end - begin);
		change = b_con - p_con;

		// printf("veretx=%10d p=%10d best=%10d p_con=%10d b_con=%10d change=%10d\n", vertex, p, best, p_con, b_con, change);
	}
	__syncthreads();
	
	// reduce change(really need?)
	change = scan_add_block(nparts, change, val_cache);

	if(threadIdx.x == 0)
		atomicAdd(&d_change[0], change);
}

__global__ void perform_moves_cuda(int num_pos, int *pos_move, int *dest_part, int *where, int *vwgt, int *dest_cahce, int *pwgts)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= num_pos)
		return ;
	
	int vertex = pos_move[ii];
	int best = dest_part[vertex];
	int p = where[vertex];
	int wgt = vwgt[vertex];
	dest_cahce[vertex] = -1;
	atomicSub(&pwgts[p], wgt);
	atomicAdd(&pwgts[best], wgt);
	where[vertex] = best;
	dest_part[vertex] = p;

	// printf("vertex=%10d p=%10d best=%10d wgt=%10d\n", vertex, p, best, wgt);
}

__global__ void count_edgecut_change2(int num_pos, int nparts, int *pos_move, int *dest_part, int *where, int *gain_offset, int *gain_val, int *gain_where, int *d_change)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;

	extern __shared__ int val_cache[];
	int change = 0;

	if(ii < num_pos)
	{	
		int vertex = pos_move[ii];
		int p = dest_part[vertex];
		int best = where[vertex];
		int gain_begin = gain_offset[vertex];
		int gain_end   = gain_offset[vertex + 1];
		int p_con = lookup(gain_val + gain_begin, gain_where + gain_begin, p   , gain_end - gain_begin);
		int b_con = lookup(gain_val + gain_begin, gain_where + gain_begin, best, gain_end - gain_begin);
		change = b_con - p_con;

		// printf("veretx=%10d p=%10d best=%10d p_con=%10d b_con=%10d change=%10d\n", vertex, p, best, p_con, b_con, change);
	}
	__syncthreads();
	
	// reduce change(really need?)
	change = scan_add_block(nparts, change, val_cache);

	if(threadIdx.x == 0)
		atomicAdd(&d_change[1], change);
}

__global__ void exam_update(int nvtxs, int nparts, int *pwgts, int *where, int *gain_offset, int *gain_val, int *gain_where)
{
	printf("pwgts: ");
	for(int i = 0;i < nparts;i++)
		printf("%10d ", pwgts[i]);
	printf("\n");

	for(int i = 0;i < nvtxs;i++)
	{
		int begin = gain_offset[i];
		int end   = gain_offset[i + 1];

		printf("vertex: %10d %10d %10d %10d\n", i, where[i], gain_offset[i], gain_offset[i + 1]);

		for(int j = begin;j < end;j++)
			printf("%10d ", gain_where[j]);
		printf("\n");
		for(int j = begin;j < end;j++)
			printf("%10d ", gain_val[j]);
		printf("\n");
	}
}

void perform_moves(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int h_num_pos)
{
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs = graph->nvtxs;

	if(h_num_pos == 0)
		return ;

	int *d_change;
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	if(GPU_Memory_Pool)
		d_change = (int *)lmalloc_with_check(sizeof(int) * 2, "perform_moves: d_change");
	else
		cudaMalloc((void **)&d_change, sizeof(int) * 2);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	double tmp_time = (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
	uncoarsen_pm -= tmp_time;
	uncoarsen_gpu_malloc += tmp_time;
#endif

	init_val<<<1, 2>>>(2, 0, d_change);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	count_edgecut_change1<<<(h_num_pos + 127) / 128, 128, sizeof(int) * 128>>>(h_num_pos, nparts, graph->pos_move, graph->dest_part, graph->cuda_where, graph->gain_offset, \
		graph->gain_val, graph->gain_where, d_change);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	pm_count_change1 += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
#endif
	// printf("count_edgecut_change1 end\n");

	int h_change[2];
	cudaMemcpy(&h_change, d_change, sizeof(int) * 2, cudaMemcpyDeviceToHost);
	// printf("h_change=%10d %10d\n", h_change[0], h_change[1]);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	perform_moves_cuda<<<(h_num_pos + 127) / 128, 128>>>(h_num_pos, graph->pos_move, graph->dest_part, graph->cuda_where, graph->cuda_vwgt, graph->dest_cache, graph->cuda_pwgts);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	pm_pm_cuda += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
	// printf("perform_moves end\n");

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	update_large(hunyuangraph_admin, graph, h_num_pos);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	pm_update_large += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
#endif
	// printf("update_large end\n");

	// cudaDeviceSynchronize();
	// exam_partition<<<1, 1>>>(nvtxs, nparts, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where);
	// cudaDeviceSynchronize();

	// cudaDeviceSynchronize();
	// exam_update<<<1, 1>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_where, graph->gain_offset, graph->gain_val, graph->gain_where);
	// cudaDeviceSynchronize();
	// update_small();

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	count_edgecut_change2<<<(h_num_pos + 127) / 128, 128, sizeof(int) * 128>>>(h_num_pos, nparts, graph->pos_move, graph->dest_part, graph->cuda_where, graph->gain_offset, \
		graph->gain_val, graph->gain_where, d_change);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	pm_count_change2 += (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
#endif
	// printf("count_edgecut_change2 end\n");

	cudaMemcpy(&h_change, d_change, sizeof(int) * 2, cudaMemcpyDeviceToHost);
	// printf("h_change=%10d %10d\n", h_change[0], h_change[1]);

	int edgecut = graph->mincut - (h_change[0] + h_change[1]) / 2;
	graph->mincut = edgecut;

	// cudaDeviceSynchronize();
	// compute_edgecut_gpu(graph->nvtxs, &edgecut, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where);
	// cudaDeviceSynchronize();
	// printf("real edgecut=%10d\n", edgecut);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway_pm, NULL);
#endif
	if(GPU_Memory_Pool)
		lfree_with_check(d_change, sizeof(int) * 2, "perform_moves: d_change");
	else
		cudaFree(d_change);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway_pm, NULL);
	tmp_time = (end_gpu_kway_pm.tv_sec - begin_gpu_kway_pm.tv_sec) * 1000.0 + (end_gpu_kway_pm.tv_usec - begin_gpu_kway_pm.tv_usec) / 1000.0;
	uncoarsen_pm -= tmp_time;
	uncoarsen_gpu_free += tmp_time;
#endif

}

int next_pow2(int n) 
{
    if (n <= 0) return 1;
    n--;
    n |= n >> 1; n |= n >> 2;
    n |= n >> 4; n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

__global__ void compute_imb_opt(const int nparts, const int* __restrict__ pwgts,
                            const int* __restrict__ opt_pwgts, float* imb) 
{
    const int tid = threadIdx.x;
    extern __shared__ float imb_cache[];
    
    // 阶段1: 并行加载并计算初始最大值
    float my_max = -INFINITY;
    for (int i = tid; i < nparts; i += blockDim.x) 
	{
        float imb_val = (float)pwgts[i] / (float)opt_pwgts[i];
        my_max = fmaxf(my_max, imb_val);
    }
    imb_cache[tid] = my_max;
    __syncthreads();

    // 阶段2: 并行归约 (支持任意block size)
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride && (tid + stride) < blockDim.x) {
            imb_cache[tid] = fmaxf(imb_cache[tid], imb_cache[tid + stride]);
        }
        __syncthreads();
    }

    // 阶段3: Warp级归约 (处理最后32个元素)
    if (tid < 32) {
        float val = imb_cache[tid];
        for (int offset = 16; offset >= 1; offset >>= 1) 
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        
        if (tid == 0) 
            imb[0] = val;
    }
}

__global__ void set_gain_offset(int nvtxs, int nparts, int *length_vertex, int *gain_offset)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int l = length_vertex[vertex];

	if(l > nparts)
		gain_offset[vertex] = nparts;
	else
		gain_offset[vertex] = l;

	// if(gain_offset[vertex] < 0)
	// 	printf("vertx=%10d gain_offset=%10d\n", vertex, gain_offset[vertex]);
}

__global__ void set_gain_idx(int nvtxs, int nparts, const int *gain_offset, const int *gain_bin, int *gain_idx, int *binsize)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int conn = __ldg(&gain_offset[ii]);
		
	if(conn < 1)
	{
		int ptr = __ldg(&gain_bin[ 0]) + atomicAdd(&binsize[ 0], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 2)
	{
		int ptr = __ldg(&gain_bin[ 1]) + atomicAdd(&binsize[ 1], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 4)
	{
		int ptr = __ldg(&gain_bin[ 2]) + atomicAdd(&binsize[ 2], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 8)
	{
		int ptr = __ldg(&gain_bin[ 3]) + atomicAdd(&binsize[ 3], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 16)
	{
		int ptr = __ldg(&gain_bin[ 4]) + atomicAdd(&binsize[ 4], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 32)
	{
		int ptr = __ldg(&gain_bin[ 5]) + atomicAdd(&binsize[ 5], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 64)
	{
		int ptr = __ldg(&gain_bin[ 6]) + atomicAdd(&binsize[ 6], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 128)
	{
		int ptr = __ldg(&gain_bin[ 7]) + atomicAdd(&binsize[ 7], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 256)
	{
		int ptr = __ldg(&gain_bin[ 8]) + atomicAdd(&binsize[ 8], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 512)
	{
		int ptr = __ldg(&gain_bin[ 9]) + atomicAdd(&binsize[ 9], 1);
		gain_idx[ptr] = vertex;
	}
	else if(conn < 1024)
	{
		int ptr = __ldg(&gain_bin[10]) + atomicAdd(&binsize[10], 1);
		gain_idx[ptr] = vertex;
	}
	else
	{
		int ptr = __ldg(&gain_bin[11]) + atomicAdd(&binsize[11], 1);
		gain_idx[ptr] = vertex;
	}
}	

__global__ void set_gain_val(int nvtxs, int nparts, int *xadj, int *adjncy, int *adjwgt, int *where, int *length_vertex, \
	int *gain_offset, int *gain_val, int *gain_where, int *vals_global)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if(ii >= nvtxs)
		return ;
	
	int vertex = ii;
	int begin = xadj[vertex];
	int end   = xadj[vertex + 1];
	int write_ptr = gain_offset[vertex];
	int write_end = gain_offset[vertex + 1];
	int write_length = 0;
	int *vals = vals_global + vertex * nparts;

	for(int j = begin; j < end;j++)
	{
		int neighbor = adjncy[j];
		int nei_where = where[neighbor];
		vals[nei_where] += adjwgt[j];
	}

	for(int i = 0;i < nparts;i++)
	{
		if(vals[i] != 0)
		{
			gain_val[write_ptr + write_length] = vals[i];
			gain_where[write_ptr + write_length] = i;
			write_length++;
		}
	}

	while(write_ptr + write_length < write_end)
	{
		gain_val[write_ptr + write_length] = 0;
		gain_where[write_ptr + write_length] = -1;
		write_length++;
	}

}

__global__ void set_gain_val_warp(const int nvtxs, const int nparts, int *xadj, int *adjncy, int *adjwgt, int *where, \
	int *gain_offset, int *gain_val, int *gain_where)
{
	// const int ii = blockIdx.x * blockDim.x + threadIdx.x;
	const int blockwarp_id = threadIdx.x >> 5;
	const int lane_id = threadIdx.x & 31;
	const int warp_id = blockIdx.x * 4 + blockwarp_id;

	extern __shared__ int cache_vals[];
	int *vals = cache_vals + blockwarp_id * 2 * nparts; // 每个warp占用2*nparts
    int *sub = vals + nparts;                           // sub紧随vals之后

	for(int i = lane_id;i < nparts;i += 32)
		vals[i] = 0;
	__syncwarp();

	if(warp_id >= nvtxs)
		return ;
	
	int vertex = warp_id;
	int begin = xadj[vertex];
	int end   = xadj[vertex + 1];
	int gain_begin = gain_offset[vertex];
	int gain_end   = gain_offset[vertex + 1];

	for(int j = begin + lane_id;j < end;j += 32)
	{
		int neighbor = adjncy[j];
		int nei_where = where[neighbor];
		atomicAdd(&vals[nei_where], adjwgt[j]);
	}
	__syncwarp();

	// if(vertex == 0)
	// 	for(int i = lane_id;i < nparts;i += 32)
	// 	{
	// 		int tmp_val = vals[i];
	// 		printf("vertex=%10d i=%10d lane_id=%10d gain_where=%10d vals=%10d\n", vertex, i, lane_id, i, tmp_val);
	// 	}
	// __syncwarp();


	for(int i = lane_id;i < nparts;i += 32)
		sub[i] = (vals[i] == 0) ? 1 : 0;
	__syncwarp();

	// if(vertex == 0)
	// 	for(int i = lane_id;i < nparts;i += 32)
	// 	{
	// 		printf("scan_prefixsum begin vertex=%10d i=%10d sub=%10d\n", vertex, i, sub[i]);
	// 	}
	// __syncwarp();

	//	scan_prefixsum
	scan_prefixsum(lane_id, nparts, sub, vertex);
	__syncwarp();
		
	// if(vertex == 0)
	// 	for(int i = lane_id;i < nparts;i += 32)
	// 	{
	// 		printf("scan_prefixsum end   vertex=%10d i=%10d sub=%10d\n", vertex, i, sub[i]);
	// 	}
	// __syncwarp();

	for(int i = lane_id;i < nparts;i += 32)
	{
		if(vals[i] != 0)
		{
			int offset = sub[i];
			gain_where[gain_begin + i - offset] = i;
			gain_val[gain_begin + i - offset] = vals[i];
		}
	}
	// __syncwarp();

	// if(gain_end - gain_begin <= nparts)
	// {
	// 	for(int i = gain_begin + lane_id;i < gain_end;i += 32)
	// 	{
	// 		printf("mid gain_end - gain_begin=%10d vertex=%10d i=%10d gain_where=%10d gain_vals=%10d ptr=%10d offset=%10d\n", gain_end - gain_begin, vertex, i, gain_where[i], gain_val[i], gain_begin + nparts - sub[nparts - 1], sub[nparts - 1]);
	// 	}
	// }
	// __syncwarp();

	for(int i = gain_begin + nparts - sub[nparts - 1] + lane_id;i < gain_end;i += 32)
	{
		gain_where[i] = -1;
		gain_val[i] = 0;
	}
	// __syncwarp();

	// if(blockIdx.x == 0)
	// if(gain_end - gain_begin < nparts)
	// {
	// 	for(int i = gain_begin + lane_id;i < gain_end;i += 32)
	// 	{
	// 		printf("gain_end - gain_begin=%10d vertex=%10d i=%10d gain_where=%10d gain_vals=%10d\n", gain_end - gain_begin, vertex, i, gain_where[i], gain_val[i]);
	// 	}
	// }

}

void k_refine(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int *level)
{
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs = graph->nvtxs;
	int best_cut, *best_where;
	float h_best_imb, *d_best_imb;

	best_cut = graph->mincut;

	// printf("level=%10d nvtxs=%10d nedges=%10d\n", level[0], nvtxs, graph->nedges);
	// printf("real edgecut=%10d\n", best_cut);

	graph->mincut = best_cut;

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	h_best_imb = 0;
	if(GPU_Memory_Pool)
		d_best_imb = (float *)lmalloc_with_check(sizeof(float), "d_best_imb");
	else
		cudaMalloc((void **)&d_best_imb, sizeof(float));
	compute_imb<<<1, 1024, 1024 * sizeof(float)>>>(nparts, graph->cuda_pwgts, graph->cuda_opt_pwgts, d_best_imb);
	// int block_size = next_pow2(nparts);
	// if(block_size > 1024)
	// 	compute_imb<<<1, 1024, 1024 * sizeof(float)>>>(nparts, graph->cuda_pwgts, graph->cuda_opt_pwgts, d_best_imb);
	// else 
	// {
	// 	block_size = min(1024, next_pow2(nparts)); 
	// 	compute_imb<<<1, block_size, block_size * sizeof(float)>>>(nparts, graph->cuda_pwgts, graph->cuda_opt_pwgts, d_best_imb);
	// }
	cudaMemcpy(&h_best_imb, d_best_imb, sizeof(float), cudaMemcpyDeviceToHost);

	if(GPU_Memory_Pool)
		lfree_with_check(d_best_imb, sizeof(float), "d_best_imb");
	else
		cudaFree(d_best_imb);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_compute_imb += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

	// printf("h_best_imb=%10.3f\n", h_best_imb);
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	if(GPU_Memory_Pool)
		best_where = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "best_where");
	else
		cudaMalloc((void **)&best_where, sizeof(int) * nvtxs);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_gpu_malloc += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	cudaMemcpy(best_where, graph->cuda_where, sizeof(int) * nvtxs, cudaMemcpyDeviceToDevice);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_gpu_memcpy += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif

	// cudaDeviceSynchronize();
	// exam_partition<<<1, 1>>>(nvtxs, nparts, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where);
	// cudaDeviceSynchronize();

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	init_val<<<1, 1>>>(1, 0, graph->gain_offset);
	set_gain_offset<<<(nvtxs + 127) / 128, 128>>>(nvtxs, nparts, graph->length_vertex, &graph->gain_offset[1]);
	// CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_set_gain_offset += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
	
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	if(GPU_Memory_Pool)
    {
        prefixsum(graph->gain_offset + 1, graph->gain_offset + 1, nvtxs, prefixsum_blocksize, 1);	//0:lmalloc,1:rmalloc
    }
    else
    {
        thrust::inclusive_scan(thrust::device, graph->gain_offset + 1, graph->gain_offset + 1 + nvtxs, graph->gain_offset + 1);
    }
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_prefixsum += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif

	int gain_size;
	cudaMemcpy(&gain_size, &graph->gain_offset[nvtxs], sizeof(int), cudaMemcpyDeviceToHost);

	// printf("gain_size=%d\n", gain_size);

	// int *gain_where, *gain_val, *dest_cache;
	int *vals;
	if(GPU_Memory_Pool)
	{
		graph->gain_val = (int *)lmalloc_with_check(sizeof(int) * gain_size, "k_refine: gain_val");
		graph->gain_where = (int *)lmalloc_with_check(sizeof(int) * gain_size, "k_refine: gain_where");
		graph->dest_cache = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "k_refine: dest_cache");
		// vals = (int *)lmalloc_with_check(sizeof(int) * nvtxs * nparts, "vals");
	}
	else
	{
		cudaMalloc((void **)&graph->gain_val, sizeof(int) * gain_size);
		cudaMalloc((void **)&graph->gain_where, sizeof(int) * gain_size);
		cudaMalloc((void **)&graph->dest_cache, sizeof(int) * nvtxs);
		// cudaMalloc((void **)&vals, sizeof(int) * nvtxs * nparts);
	}

	// cudaDeviceSynchronize();
	// gettimeofday(&begin_gpu_kway, NULL);
	// init_val<<<(nvtxs * nparts + 127) / 128, 128>>>(nvtxs * nparts, 0, vals);
	// cudaDeviceSynchronize();
	// gettimeofday(&end_gpu_kway, NULL);
	// uncoarsen_init_vals += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	// set_gain_val<<<(nvtxs + 127) / 128, 128>>>(nvtxs, nparts, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, \
	// 	graph->length_vertex, graph->gain_offset, graph->gain_val, graph->gain_where, vals);
	set_gain_val_warp<<<(nvtxs + 3) / 4, 128, 8 * nparts * sizeof(int)>>>(nvtxs, nparts, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, \
		graph->gain_offset, graph->gain_val, graph->gain_where);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_set_gain_val += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif
	// printf("set_gain_val_warp end\n");
	
	// exit(0);

	/*if(GPU_Memory_Pool)
		lfree_with_check(vals, sizeof(int) * nvtxs * nparts, "vals");
	else
		cudaFree(vals);
	
	// cudaDeviceSynchronize();
	// exam_gain<<<1, 1>>>(nvtxs, nparts, graph->gain_offset, graph->gain_val, graph->gain_where, graph->cuda_where);
	// cudaDeviceSynchronize();

	printf("-------------------------------------------------------------------------------------\n");

	int *g_val, *g_where;
	if(GPU_Memory_Pool)
	{
		g_val = (int *)lmalloc_with_check(sizeof(int) * gain_size, "k_refine: g_val");
		g_where = (int *)lmalloc_with_check(sizeof(int) * gain_size, "k_refine: g_where");
	}
	else
	{
		cudaMalloc((void **)&g_val, sizeof(int) * gain_size);
		cudaMalloc((void **)&g_where, sizeof(int) * gain_size);
	}

	cudaDeviceSynchronize();
	set_gain_val_warp<<<(nvtxs + 3) / 4, 128, 8 * nparts * sizeof(int)>>>(nvtxs, nparts, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->cuda_where, \
		graph->gain_offset, g_val, g_where);
	CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
	cudaDeviceSynchronize();
	
	// cudaDeviceSynchronize();
	// exam_gain<<<1, 1>>>(nvtxs, nparts, graph->gain_offset, g_val, g_where, graph->cuda_where);
	// cudaDeviceSynchronize();
	
	printf("level=%10d\n", level[0]);
	cudaDeviceSynchronize();
	exam_same<<<(gain_size + 127) / 128, 128>>>(gain_size, graph->gain_val, g_val);
	cudaDeviceSynchronize();
	printf("-------------------------------------------------------------------------------------\n");
	cudaDeviceSynchronize();
	exam_same<<<(gain_size + 127) / 128, 128>>>(gain_size, graph->gain_where, g_where);
	cudaDeviceSynchronize();

	if(GPU_Memory_Pool)
	{
		lfree_with_check(g_where, sizeof(int) * gain_size, "k_refine: g_where");
		lfree_with_check(g_val, sizeof(int) * gain_size, "k_refine: g_val");
	}
	else
	{
		cudaFree(g_where);
		cudaFree(g_val);
	}

	// exit(0);*/

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	init_val<<<(nvtxs + 127) / 128, 128>>>(nvtxs, -1, graph->dest_cache);
	init_val<<<(nvtxs + 127) / 128, 128>>>(nvtxs, (char)0, graph->lock);
	init_val<<<(nvtxs + 127) / 128, 128>>>(nvtxs, (char)0, graph->cuda_select);
	// init_val<char><<<(nvtxs + 127) / 128, 128>>>(nvtxs, 0, graph->lock);
	// init_val<char><<<(nvtxs + 127) / 128, 128>>>(nvtxs, 0, graph->cuda_select);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_init_vals += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif

	int count = 0;
    int iter_count = 0;
    int balance_counter = 0;
    int lab_counter = 0;
    double tol = 0.999;

#ifdef FIGURE14_EDGECUT
	while(count++ <= 11)
#else 
	while(count++ <= 5)
#endif

	{
		int balance = 1;
		cudaMemcpy(graph->cuda_balance, &balance, sizeof(int), cudaMemcpyHostToDevice);

#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_kway, NULL);
#endif
		is_balance<<<(nparts + 31) / 32, 32>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_balance, graph->cuda_poverload);
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_kway, NULL);
		uncoarsen_is_balance += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif

		// cudaDeviceSynchronize();
		// exam_pwgts<<<1, 1>>>(nparts, graph->cuda_pwgts, graph->cuda_maxwgt, graph->cuda_poverload);
		// cudaDeviceSynchronize();

		cudaMemcpy(&balance, graph->cuda_balance, sizeof(int), cudaMemcpyDeviceToHost);

		// printf("balance=%10d\n", balance);
		// balance = 1;
		int h_num_pos = 0;
		//	  balance -> LP
		if(balance == 1)
		{
			// printf("jetlp ");
			// printf("jetlp begin\n");
#ifdef TIMER
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
#endif
			h_num_pos = jetlp(hunyuangraph_admin, graph, level[0]);
#ifdef TIMER
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_lp += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif
			// printf("jetlp end h_num_pos=%10d\n", h_num_pos);
			balance_counter = 0;
			lab_counter = 0;
		}

		 //	unbalance -> jetr
		else
		{
			// printf("jetrw\n");
			// h_num_pos = jetrw(hunyuangraph_admin, graph);
			if(balance_counter < 2)
			{
				// printf("jetrw ");
#ifdef TIMER
				cudaDeviceSynchronize();
				gettimeofday(&begin_gpu_kway, NULL);
#endif
				h_num_pos = jetrw(hunyuangraph_admin, graph);
#ifdef TIMER
				cudaDeviceSynchronize();
				gettimeofday(&end_gpu_kway, NULL);
				uncoarsen_rw += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif

            } else 
			{
				// printf("jetrs ");
#ifdef TIMER
				cudaDeviceSynchronize();
				gettimeofday(&begin_gpu_kway, NULL);
#endif
				h_num_pos = jetrs(hunyuangraph_admin, graph);
#ifdef TIMER
				cudaDeviceSynchronize();
				gettimeofday(&end_gpu_kway, NULL);
				uncoarsen_rs += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif
            }
            balance_counter++;
		}

		// printf("perform_moves h_num_pos=%10d ", h_num_pos);
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_kway, NULL);
#endif
		perform_moves(hunyuangraph_admin, graph, h_num_pos);
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_kway, NULL);
		uncoarsen_pm += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

		//copy current partition and relevant data to output partition if following conditions pass
		cudaDeviceSynchronize();
		gettimeofday(&begin_gpu_kway, NULL);
#endif
		float h_curr_imb, *d_curr_imb;
		h_curr_imb = 0;
		if(GPU_Memory_Pool)
			d_curr_imb = (float *)lmalloc_with_check(sizeof(float), "d_curr_imb");
		else
			cudaMalloc((void **)&d_curr_imb, sizeof(float));
		compute_imb<<<1, 1024, 1024 * sizeof(float)>>>(nparts, graph->cuda_pwgts, graph->cuda_opt_pwgts, d_curr_imb);
		// // int block_size = next_pow2(nparts);
		// if(block_size > 1024)
		// 	compute_imb<<<1, 1024, 1024 * sizeof(float)>>>(nparts, graph->cuda_pwgts, graph->cuda_opt_pwgts, d_curr_imb);
		// else 
		// {
		// 	// block_size = min(1024, next_pow2(nparts)); 
		// 	compute_imb<<<1, block_size, block_size * sizeof(float)>>>(nparts, graph->cuda_pwgts, graph->cuda_opt_pwgts, d_curr_imb);
		// }
		cudaMemcpy(&h_curr_imb, d_curr_imb, sizeof(float), cudaMemcpyDeviceToHost);

		if(GPU_Memory_Pool)
			lfree_with_check(d_curr_imb, sizeof(float), "d_curr_imb");
		else
			cudaFree(d_curr_imb);
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&end_gpu_kway, NULL);
		uncoarsen_compute_imb += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif

		// cudaDeviceSynchronize();
		// gettimeofday(&begin_gpu_kway, NULL);
		// float h_curr_imb, *d_curr_imb;
		// h_curr_imb = 0;
		// if(GPU_Memory_Pool)
		// 	d_curr_imb = (float *)lmalloc_with_check(sizeof(float), "d_curr_imb");
		// else
		// 	cudaMalloc((void **)&d_curr_imb, sizeof(float));
		// compute_imb<<<1, 1024, 1024 * sizeof(float)>>>(nparts, graph->cuda_pwgts, graph->cuda_opt_pwgts, d_curr_imb);
		// cudaMemcpy(&h_curr_imb, d_curr_imb, sizeof(float), cudaMemcpyDeviceToHost);
		// cudaDeviceSynchronize();
		// gettimeofday(&end_gpu_kway, NULL);
		// uncoarsen_compute_imb += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

		// if(GPU_Memory_Pool)
		// 	lfree_with_check(d_curr_imb, sizeof(float), "d_curr_imb");
		// else
		// 	cudaFree(d_curr_imb);

		if(h_best_imb > IMB && h_curr_imb < h_best_imb)
		{
			h_best_imb = h_curr_imb;
			best_cut = graph->mincut;
			
#ifdef TIMER
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
#endif
			cudaMemcpy(best_where, graph->cuda_where, sizeof(int) * nvtxs, cudaMemcpyDeviceToDevice);
#ifdef TIMER
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_gpu_memcpy += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif
			count = 0;
		}
		else if(graph->mincut < best_cut && (h_curr_imb <= IMB || h_curr_imb <= h_best_imb))
		{
			if(graph->mincut < tol * best_cut)
				count = 0;
			h_best_imb = h_curr_imb;
			best_cut = graph->mincut;
			
#ifdef TIMER
			cudaDeviceSynchronize();
			gettimeofday(&begin_gpu_kway, NULL);
#endif
			cudaMemcpy(best_where, graph->cuda_where, sizeof(int) * nvtxs, cudaMemcpyDeviceToDevice);
#ifdef TIMER
			cudaDeviceSynchronize();
			gettimeofday(&end_gpu_kway, NULL);
			uncoarsen_gpu_memcpy += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif
		}

		// printf("count=%10d, h_curr_imb=%10f, h_best_imb=%10f, best_cut=%10d curr_cut=%10d \n", count, h_curr_imb, h_best_imb, best_cut, graph->mincut);
	}

	graph->mincut = best_cut;

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	cudaMemcpy(graph->cuda_where, best_where, sizeof(int) * nvtxs, cudaMemcpyDeviceToDevice);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_gpu_memcpy += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif
	// printf("level=%10d best_cut=%10d\n", level[0], best_cut);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		lfree_with_check(graph->dest_cache, sizeof(int) * nvtxs, "k_refine: dest_cache");
		lfree_with_check(graph->gain_where, sizeof(int) * gain_size, "k_refine: gain_where");
		lfree_with_check(graph->gain_val, sizeof(int) * gain_size, "k_refine: gain_val");
		lfree_with_check(best_where, sizeof(int) * nvtxs, "k_refine: best_where");
	}
	else
	{
		cudaFree(graph->dest_cache);
		cudaFree(graph->gain_where);
		cudaFree(graph->gain_val);
		cudaFree(best_where);
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_gpu_free += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif

}

#endif