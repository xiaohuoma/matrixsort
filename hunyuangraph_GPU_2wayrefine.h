#ifndef _H_GPU_2WAYREFINE
#define _H_GPU_2WAYREFINE

#include "hunyuangraph_struct.h"

__global__ void select_bnd(int nvtxs, int *xadj, int *adjncy, int *where, int *bnd)
{
}

__global__ void select_bnd(int nvtxs, int *xadj, int *adjncy, int *where, int *bnd)
{
	int iii = threadIdx.x / 32;
	int ii  = blockIdx.x * 4 + iii;
    int tid = threadIdx.x % 32;

	// 启动�?1个Block�?128线程
	// 1个Warp处理一个顶点，将处理顶点的where存放于共享内�?
	// 可以减少32倍的where全局访问，但是会不会引起Bank冲突�?
	__shared__ int cache_twhere[4];		// 目的是减少global memory访问
	__shared__ int cache_flag[4];		// 目的是warp内通信

	if(tid == 0) cache_twhere[iii] = where[ii];
	if(tid == 1) cache_flag[iii]   = 0;
	__syncthreads();

	if(ii < nvtxs)
	{
		int i, j, wj, me, begin, end;

		me    = cache_twhere[iii];						// 会不会引起Bank冲突
		begin = xadj[ii];
		end   = xadj[ii + 1];

		for(i = begin + tid;i < end;i += 32)
		{
			j  = adjncy[i];
			wj = where[j];
			if(wj != me) cache_flag[iii]++;
			if(cache_flag[iii] != 0) break;
		}

		// 是边界顶点，bnd值为1；不是边界顶点，bnd值为1
		// 可以通过reduce计算边界顶点的数�?
		if(tid == 0)
		{
			if(cache_flag[iii] != 0) bnd[ii] = 1;
			else bnd[ii] = 0;
		}
	}
}

__global__ void compute_gain_shared(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *bnd, int *gain)
{
	int iii = threadIdx.x / 32;
	int ii  = blockIdx.x * 4 + iii;
	int tid = threadIdx.x % 32;

	// 启动�?1个Block�?128线程
	// 1个Warp处理一个顶点，将处理顶点的where存放于共享内�?
	// 可以减少32倍的where全局访问，但是会不会引起Bank冲突�?
	__shared__ int cache_twhere[4];		// 目的是减少global memory访问
	__shared__ int cache_ed[128];		// 目的是warp内reduce计算负责顶点的ed
	__shared__ int cache_id[128];		// 目的是warp内reduce计算负责顶点的id

	if(tid == 0) cache_twhere[iii] = where[ii];
	cache_ed[threadIdx.x] = 0;
	cache_id[threadIdx.x] = 0;
	__syncthreads();

	if(ii < nvtxs && bnd[ii] == 1)		//bnd[ii]理论上也可以像cache_twhere减少32倍的全局访问
	{
		int i, j, wj, me, begin ,end;
		me    = cache_twhere[iii];
		begin = xadj[ii];
		end   = xadj[ii + 1];

		for(i = begin;i < end;i += 32)
		{
			j  = adjncy[i];
			wj = where[j];
			if(me == wj) cache_id[threadIdx.x] += adjwgt[i];
			else cache_ed[threadIdx.x] += adjwgt[i];
		}

		// 对cache_id和cache_ed进行Warp内reduce
		if(tid < 16) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 16];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 16];
		}
		if(tid < 8) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 8];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 8];
		}
		if(tid < 4) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 4];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 4];
		}
		if(tid < 2) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 2];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 2];
		}
		if(tid < 1) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 1];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 1];
		}

		// 计算gain并存�?
		if(tid == 0) gain[ii] = cache_ed[threadIdx.x] - cache_id[threadIdx.x];
	}
}

__global__ void compute_infor_shared(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *gain)
{
	int iii = threadIdx.x / 32;
	int ii  = blockIdx.x * 4 + iii;
  	int tid = threadIdx.x % 32;

	// 启动�?1个Block�?128线程
	// 1个Warp处理一个顶点，将处理顶点的where存放于共享内�?
	// 可以减少32倍的where全局访问，但是会不会引起Bank冲突�?
	// __shared__ int cache_twhere[4];		// 目的是减少global memory访问
	__shared__ int cache_ed[128];		// 目的是warp内reduce计算负责顶点的ed
	__shared__ int cache_id[128];		// 目的是warp内reduce计算负责顶点的id

	// if(tid == 0) cache_twhere[iii] = where[ii];
	// cache_ed[threadIdx.x] = 0;
	// cache_id[threadIdx.x] = 0;
	// __syncthreads();

	if(ii < nvtxs)
	{
		// if(tid == 0) cache_twhere[iii] = where[ii];
		cache_ed[threadIdx.x] = 0;
		cache_id[threadIdx.x] = 0;
		__syncthreads();

		int i, j, wj, me, begin ,end, t;
		// me    = cache_twhere[iii];
		me    = where[ii];
		begin = xadj[ii];
		end   = xadj[ii + 1];

		for(i = begin;i < end;i += 32)
		{
			j  = adjncy[i];
			wj = where[j];
			if(me == wj) cache_id[threadIdx.x] += adjwgt[i];
			else cache_ed[threadIdx.x] += adjwgt[i];
		}

		// 对cache_id和cache_ed进行Warp内reduce
		if(tid < 16) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 16];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 16];
		}
		if(tid < 8) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 8];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 8];
		}
		if(tid < 4) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 4];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 4];
		}
		if(tid < 2) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 2];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 2];
		}
		if(tid < 1) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 1];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 1];
		}

		// 计算gain并存放（移动也可以放在这里）
		if(tid == 0) 
		{
			t = cache_ed[threadIdx.x] - cache_id[threadIdx.x];
			// printf("ii=%d gain=%d\n",ii,t);
			// t = 0;
			// gain[ii] = t;
			if(t > 0) where[ii] = (me + 1) % 2;
		}
	}
}

__global__ void compute_infor(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, k, me, wk, begin, end, ed, id, t;
		me    = where[ii];
		begin = xadj[ii];
		end   = xadj[ii + 1];
		ed    = 0;
		id    = 0;

		for(i = begin;i < end;i++)
		{
			j  = adjncy[i];
			wk = where[j];

			if(me == wk) id += adjwgt[i];
			else ed += adjwgt[i];
		}
		
		t = ed - id;
		gain[ii] = t;
		if(t > 0) where[ii] = (me + 1) % 2;
	}
}

__global__ void compute_infor_atomic(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, k, me, wk, begin, end, ed, id, t, p;
		p = 10;
		while(p >= 0)
		{
			me    = where[ii];
			begin = xadj[ii];
			end   = xadj[ii + 1];
			ed    = 0;
			id    = 0;

			for(i = begin;i < end;i++)
			{
				j  = adjncy[i];
				// wk = atomicAdd(&where[j], 0);
				wk = where[j];

				if(me == wk) id += adjwgt[i];
				else ed += adjwgt[i];
			}
			
			t = ed - id;
			// gain[ii] = t;
			if(t > 0) atomicExch(&where[ii], (me + 1) % 2);
			p --;
		}
	}
}

__global__ void greedy_refine(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *step)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, k, me, wk, begin, end, ed, id, t, p, s;
		
		s = nvtxs % 100;
		p = 0;

		while(p < s)
		{
			me    = where[ii];
			begin = xadj[ii];
			end   = xadj[ii + 1];
			ed    = 0;
			id    = 0;

			for(i = begin;i < end;i++)
			{
				j  = adjncy[i];
				wk = where[j];

				if(me == wk) id += adjwgt[i];
				else ed += adjwgt[i];
			}
			
			t = ed - id;

			if(t > 0)
			{
				if(atomicAdd(step, 1) <= p * 3)
				{
					atomicExch(&where[ii], (me + 1) % 2);
					break;
				}
			}
			
			p++;
		}
	}
}

__global__ void compute_balance(int nvtxs, int *vwgt, int *where, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(where[ii] == 0) gain[ii] = vwgt[ii];
		else gain[ii] = 0;
	}
}

__global__ void move_vertex(int nvtxs, int *where, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		if(gain[ii] > 0) 
		{
			int me = where[ii];

			// 取余快还是if快？
			if(me == 0) where[ii] = 1;
			else where[ii] = 0;
		}
	}
}

__global__ void compute_infor_rebalance_shared(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, kp_t *gain_kp)
{
	int iii = threadIdx.x / 32;
	int ii  = blockIdx.x * 4 + iii;
  	int tid = threadIdx.x % 32;

	// 启动�?1个Block�?128线程
	// 1个Warp处理一个顶点，将处理顶点的where存放于共享内�?
	// 可以减少32倍的where全局访问，但是会不会引起Bank冲突�?
	// __shared__ int cache_twhere[4];		// 目的是减少global memory访问
	__shared__ int cache_ed[128];		// 目的是warp内reduce计算负责顶点的ed
	__shared__ int cache_id[128];		// 目的是warp内reduce计算负责顶点的id

	// if(tid == 0) cache_twhere[iii] = where[ii];
	// cache_ed[threadIdx.x] = 0;
	// cache_id[threadIdx.x] = 0;
	// __syncthreads();

	if(ii < nvtxs)
	{
		// if(tid == 0) cache_twhere[iii] = where[ii];
		cache_ed[threadIdx.x] = 0;
		cache_id[threadIdx.x] = 0;
		__syncthreads();

		int i, j, wj, me, begin ,end, t;
		// me    = cache_twhere[iii];
		me    = where[ii];
		begin = xadj[ii];
		end   = xadj[ii + 1];

		for(i = begin;i < end;i += 32)
		{
			j  = adjncy[i];
			wj = where[j];
			if(me == wj) cache_id[threadIdx.x] += adjwgt[i];
			else cache_ed[threadIdx.x] += adjwgt[i];
		}

		// 对cache_id和cache_ed进行Warp内reduce
		if(tid < 16) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 16];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 16];
		}
		if(tid < 8) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 8];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 8];
		}
		if(tid < 4) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 4];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 4];
		}
		if(tid < 2) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 2];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 2];
		}
		if(tid < 1) 
		{
			cache_ed[threadIdx.x] += cache_ed[threadIdx.x + 1];
			cache_id[threadIdx.x] += cache_id[threadIdx.x + 1];
		}

		// 计算gain并存放（移动也可以放在这里）
		if(tid == 0) 
		{
			gain_kp[ii].key = cache_ed[threadIdx.x] - cache_id[threadIdx.x];
			gain_kp[ii].ptr = ii;
		}
	}
}

__global__ void compute_infor_rebalance(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, kp_t *gain_kp)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int i, j, k, me, wk, begin, end, ed, id, t;
		me    = where[ii];
		begin = xadj[ii];
		end   = xadj[ii + 1];
		ed    = 0;
		id    = 0;

		for(i = begin;i < end;i++)
		{
			j  = adjncy[i];
			wk = where[j];

			if(me == wk) id += adjwgt[i];
			else ed += adjwgt[i];
		}

		gain_kp[ii].key = ed - id;
		gain_kp[ii].ptr = ii;
	}
}

__global__ void compute_vwgt(int nvtxs, int *vwgt, kp_t *gain_kp, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int temp = gain_kp[ii].ptr;
		gain[ii] = vwgt[temp];
	}
}

__global__ void rebalance(int nvtxs, int to, int val, int *where, kp_t *gain_kp, int *gain)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int temp, ptr;
		temp = gain[ii];

		if(temp < val)
		{
			ptr = gain_kp[ii].key;
			where[ptr] = to;
		}
	}
}

//利用原子操作实现等待 向CPU串行方向靠拢
void FM_GPU(hunyuangraph_graph_t *graph)
{
	int nvtxs, nbnd, from, to, val;
	int *bnd, *gain, *moved, *step;
	kp_t *gain_kp;

	nvtxs = graph->nvtxs;

	cudaMalloc((void **)&gain,sizeof(int) * nvtxs);
	cudaMalloc((void **)&step,sizeof(int));

	// select_bnd和compute_gain_shared可以合并成一个计算全部顶点的ed，id，gain，bnd
	// 合并select_bnd和compute_gain_shared可做进一步优化，可以减少全局访存

	// 目前没有标记边界顶点（bnd�?
	// for(int i = 0;i < 5;i++)	// 将这个循环放入kernel
	// 	// compute_infor_shared<<<(nvtxs + 3) / 4, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,\
	// 		graph->cuda_where, gain);
		// compute_infor<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,\
		// 	graph->cuda_where, gain);
	// compute_infor_atomic<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,\
		graph->cuda_where, gain);

	greedy_refine<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,\
		graph->cuda_where, step);

	// 停止移动后观察是否平�?
	compute_balance<<<(nvtxs + 127) / 128,128>>>(nvtxs,graph->cuda_vwgt,graph->cuda_where,gain);

	graph->pwgts[0] = thrust::reduce(thrust::device, gain, gain + nvtxs);
	// for(int l = nvtxs;l != 1;l = (l + 512 - 1) / 512)
    //     reduction6<<<(l + 512 - 1) / 512,256>>>(gain,l);
	// cudaMemcpy(&graph->pwgts[0], &gain[0],sizeof(int),cudaMemcpyDeviceToHost);

	graph->pwgts[1] = graph->tvwgt[0] - graph->pwgts[0];

	// 若平衡，不操�?
	if((graph->pwgts[0] >= graph->tvwgt[0] * 0.5 / 1.03 && graph->pwgts[0] <= graph->tvwgt[0] * 0.5 * 1.03) && \
		(graph->pwgts[1] >= graph->tvwgt[0] * 0.5 / 1.03 && graph->pwgts[1] <= graph->tvwgt[0] * 0.5 * 1.03)) ;

	// 若不平衡则移动顶点至平衡
	else
	{
		printf("rebalance\n");

		// 从权重大的分区向权重小的分区移动顶点
		if(graph->pwgts[0] > graph->pwgts[1]) from = 0;
		else from = 1;
		to = (from + 1) % 2;

		// 计算当前分区状态的gain
		// compute_infor_rebalance_shared<<<(nvtxs + 3) / 4, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,\
			graph->cuda_where, gain_kp);
		cudaMalloc((void **)&gain_kp,sizeof(kp_t) * nvtxs);
		compute_infor_rebalance<<<(nvtxs + 127) / 128, 128>>>(nvtxs,graph->cuda_xadj,graph->cuda_adjncy,graph->cuda_adjwgt,\
			graph->cuda_where, gain_kp);

		// 按照gain的数值从大到小进行排�?
		thrust::sort(thrust::device, gain_kp, gain_kp + nvtxs, compRule());

		// 通过前缀和计算排序后顶点权重的和，据此判断哪些顶点移动就可以满足平衡
		compute_vwgt<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_vwgt, gain_kp, gain);

		thrust::inclusive_scan(thrust::device, gain, gain + nvtxs, gain);

		// 移动顶点至平�?
		val = (graph->pwgts[from] - graph->pwgts[to]) / 2;
		rebalance<<<(nvtxs + 127) / 128, 128>>>(nvtxs, to, val, graph->cuda_where, gain_kp, gain);
	}

	cudaFree(gain);
	cudaFree(step);
	cudaFree(gain_kp);
}

__global__ void init_moved(int nvtxs, int *moved)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
		moved[ii] = -1;
}

__global__ void compute_gain(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where, int *gain, int *moved)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs && moved[ii] == -1)
	{
		int i, j, k, me, wk, begin, end, ed, id, t;
		me    = where[ii];
		begin = xadj[ii];
		end   = xadj[ii + 1];
		ed    = 0;
		id    = 0;

		for(i = begin;i < end;i++)
		{
			j  = adjncy[i];
			wk = where[j];

			if(me == wk) id += adjwgt[i];
			else ed += adjwgt[i];
		}
		
		t = ed - id;
		gain[ii] = t;
	}
}

__global__ void judge_move(int id, int *mincut, int *newcut, int *max_element_ptr, int *tpwgts, int *pwgts, int *origdiff, \
	int *avgvwgt, int *mindiff, int *mincutorder, int *nswaps, int *limit, int *flag, int *where, int *moved, int *swaps)
{
	newcut[0] = mincut[0] - max_element_ptr[0];
	if((newcut[0] < mincut[0]&&abs(tpwgts[0]-pwgts[0])<=origdiff[0]+avgvwgt[0])|| 
        (newcut[0] == mincut[0]&&abs(tpwgts[0]-pwgts[0])<mindiff[0])){
        mincut[0]=newcut[0];
        mindiff[0]=abs(tpwgts[0]-pwgts[0]);
        mincutorder[0]=nswaps[0];

		// 
		int me = where[id];
		if(me == 0) where[id] = 1;
		else where[id] = 0;
		moved[id]=nswaps[0];
		swaps[nswaps[0]]=id;
    }
    else if(nswaps[0] - mincutorder[0] > limit[0]){ 
        flag[0] = 1;
    }
}

__global__ void move_max_index(int id, int *where, int *moved)
{
	int me = where[id];
	moved[id] = 1;
	if(me == 0) where[id] = 1;
	else where[id] = 0;
}

void Greedy_GPU(hunyuangraph_graph_t *graph, float *ntpwgts)
{
	int *gain, *moved, *swaps;
	int *max_element_ptr, max_index;

	int nvtxs  = graph->nvtxs;
	int *pwgts = graph->pwgts;

	int mincut, initcut, newcut;
	mincut = initcut = newcut = graph->mincut;

	int tpwgts[2];
	tpwgts[0]=graph->tvwgt[0]*ntpwgts[0];
  	tpwgts[1]=graph->tvwgt[0]-tpwgts[0];

	int limit=hunyuangraph_min(hunyuangraph_max(0.01*nvtxs,15),100);
  	int avgvwgt=hunyuangraph_min((pwgts[0]+pwgts[1])/20,2*(pwgts[0]+pwgts[1])/nvtxs);
	int origdiff=abs(tpwgts[0]-pwgts[0]);

	cudaMalloc((void **)&gain,sizeof(int) * nvtxs);
	cudaMalloc((void **)&moved,sizeof(int) * nvtxs);
	cudaMalloc((void **)&swaps,sizeof(int) * nvtxs);

	init_moved<<<(nvtxs + 127) / 128, 128>>>(nvtxs, moved);

	// 计算gain
	compute_gain<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt,\
		graph->cuda_where, gain, moved);
	
	// 挑选最优顶�? 使用thrust::max_element获取最大值的迭代�?
	max_element_ptr = thrust::max_element(thrust::device, gain, gain + nvtxs);
	max_index =  max_element_ptr - gain;

	// 判断最优顶点是否移�?
	// judge_move<<<1,1>>>(max_index,);
	/* CPU判断规则
	if((newcut<mincut&&abs(tpwgts[0]-pwgts[0])<=origdiff+avgvwgt)|| 
        (newcut==mincut&&abs(tpwgts[0]-pwgts[0])<mindiff)){
        mincut=newcut;
        mindiff=abs(tpwgts[0]-pwgts[0]);
        mincutorder=nswaps;
    }
    else if(nswaps-mincutorder>limit){ 
        newcut+=(ed[higain]-id[higain]);
        hunyuangraph_add_sub(pwgts[from],pwgts[to],vwgt[higain]);
        break;
      }
	*/

	// 移动最优顶�?
	move_max_index<<<1,1>>>(max_index, graph->cuda_where, moved);
}

void FM_2WayCutRefine_GPU(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, float *ntpwgts)
{
	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_2way, NULL);
	// cudaMalloc((void**)&graph->cuda_xadj,sizeof(int) * (graph->nvtxs + 1));
	// cudaMalloc((void**)&graph->cuda_adjncy,sizeof(int) * graph->nedges);
	// cudaMalloc((void**)&graph->cuda_adjwgt,sizeof(int) * graph->nedges);
	// cudaMalloc((void**)&graph->cuda_vwgt,sizeof(int) * graph->nvtxs);
	cudaMalloc((void**)&graph->cuda_where,sizeof(int) * graph->nvtxs);

	// cudaMemcpy(graph->cuda_xadj,graph->xadj,sizeof(int) * (graph->nvtxs + 1), cudaMemcpyHostToDevice);
	// cudaMemcpy(graph->cuda_adjncy,graph->adjncy,sizeof(int) * graph->nedges, cudaMemcpyHostToDevice);
	// cudaMemcpy(graph->cuda_adjwgt,graph->adjwgt,sizeof(int) * graph->nedges, cudaMemcpyHostToDevice);
	// cudaMemcpy(graph->cuda_vwgt,graph->vwgt,sizeof(int) * graph->nvtxs, cudaMemcpyHostToDevice);
	cudaMemcpy(graph->cuda_where,graph->where,sizeof(int) * graph->nvtxs, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_2way, NULL);
	malloc_2way += (end_malloc_2way.tv_sec - begin_malloc_2way.tv_sec) * 1000 + (end_malloc_2way.tv_usec - begin_malloc_2way.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_2way, NULL);
	// jetLP(graph);
	FM_GPU(graph);
	// Greedy_GPU(graph, ntpwgts);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_2way, NULL);
	gpu_2way += (end_gpu_2way.tv_sec - begin_gpu_2way.tv_sec) * 1000 + (end_gpu_2way.tv_usec - begin_gpu_2way.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_malloc_2way, NULL);
	cudaMemcpy(graph->where,graph->cuda_where,sizeof(int) * graph->nvtxs, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	gettimeofday(&end_malloc_2way, NULL);
	malloc_2way += (end_malloc_2way.tv_sec - begin_malloc_2way.tv_sec) * 1000 + (end_malloc_2way.tv_usec - begin_malloc_2way.tv_usec) / 1000.0;
}

#endif