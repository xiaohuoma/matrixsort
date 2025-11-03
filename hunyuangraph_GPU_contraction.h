#ifndef _H_GPU_CONSTRUCTION
#define _H_GPU_CONSTRUCTION

#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
#include "hunyuangraph_GPU_prefixsum.h"
#include "bb_segsort.h"

/*CUDA-set each vertex pair adjacency list and weight params*/
__global__ void set_tadjncy_tadjwgt(int *txadj, int *xadj, int *match, int *adjncy,\
    int *cmap, int *tadjncy, int *tadjwgt, int *adjwgt, int nvtxs)
{
    // long long int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int ii  = blockIdx.x * 4 + threadIdx.x / 32;
    int tid = threadIdx.x % 32;

    if(ii < nvtxs)
    {
        int u, t, pp, k, ptr, begin, end, iii;

        u = match[ii];
        t = cmap[ii];

        pp = txadj[t];
        if(ii > u)
        {
            begin = xadj[u];
            end   = xadj[u + 1];
            pp   += end - begin;
        }

        begin = xadj[ii];
        end   = xadj[ii + 1];

        for(iii = begin + tid, ptr = pp + tid;iii < end;iii += 32, ptr += 32)
        {
            k   = adjncy[iii];

            tadjncy[ptr] = cmap[k];
            tadjwgt[ptr] = adjwgt[iii];
        }
    }
}

//  nvtxs!!!!!
template<int SUBWARP_SIZE>
__global__ void set_tadjncy_tadjwgt_subwarp(    const int bin, const int *bin_offset, const int *bin_idx, \
                                                const int *txadj, const int *xadj, const int *match, const int *adjncy, const int *adjwgt, \
                                                const int *cmap, int *tadjncy, int *tadjwgt, const int nvtxs)
{
    int lane_id = threadIdx.x & (SUBWARP_SIZE - 1);
	int blockwarp_id = threadIdx.x / SUBWARP_SIZE;
	int subwarp_num = blockDim.x / SUBWARP_SIZE;
	int subwarp_id = blockIdx.x * subwarp_num + blockwarp_id;

    //  先不使用shared memory
    if(subwarp_id >= nvtxs) 
        return ;
    
    int bin_row_offset = __ldg(&bin_offset[bin]) + subwarp_id; 
    int vertex = __ldg(&bin_idx[bin_row_offset]);
    int vertex_match = __ldg(&match[vertex]);
    int map_cvertex = __ldg(&cmap[vertex]);
    
    int tbegin = __ldg(&txadj[map_cvertex]);
    bool is_vertex_larger = (vertex > vertex_match);
    tbegin += (is_vertex_larger) ? (__ldg(&xadj[vertex_match + 1]) - __ldg(&xadj[vertex_match])) : 0;
    int begin = __ldg(&xadj[vertex]);
    int end = __ldg(&xadj[vertex + 1]);

    int j = begin + lane_id;
    if(j >= end) return ;

    int k = __ldg(&adjncy[j]);
    int wgt_k = __ldg(&adjwgt[j]);
    int k_cmap = __ldg(&cmap[k]);

    tadjwgt[tbegin + lane_id] = wgt_k;
    tadjncy[tbegin + lane_id] = k_cmap;
}

template<int WARP_SIZE>
__global__ void set_tadjncy_tadjwgt_warp_bin(   const int bin, const int *bin_offset, const int *bin_idx, \
                                                const int *txadj, const int *xadj, const int *match, const int *adjncy, const int *adjwgt, \
                                                const int *cmap, int *tadjncy, int *tadjwgt, int nvtxs)
{
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
	int blockwarp_id = threadIdx.x / WARP_SIZE;
	int subwarp_num = blockDim.x / WARP_SIZE;
	int subwarp_id = blockIdx.x * subwarp_num + blockwarp_id;

    //  先不使用shared memory
    if(subwarp_id >= nvtxs) 
        return ;
    
    int bin_row_offset = __ldg(&bin_offset[bin]) + subwarp_id; 
    int vertex = __ldg(&bin_idx[bin_row_offset]);
    int vertex_match = __ldg(&match[vertex]);
    int map_cvertex = __ldg(&cmap[vertex]);
    
    int tbegin = __ldg(&txadj[map_cvertex]);
    bool is_vertex_larger = (vertex > vertex_match);
    tbegin += (is_vertex_larger) ? (__ldg(&xadj[vertex_match + 1]) - __ldg(&xadj[vertex_match])) : 0;
    int begin = __ldg(&xadj[vertex]);
    int end = __ldg(&xadj[vertex + 1]);

    for(int j = begin + lane_id, ptr = tbegin + lane_id;j < end;j += WARP_SIZE, ptr += WARP_SIZE)
    {
        int k = __ldg(&adjncy[j]);
        int wgt_k = __ldg(&adjwgt[j]);
        int k_cmap = __ldg(&cmap[k]);

        tadjwgt[ptr] = wgt_k;
        tadjncy[ptr] = k_cmap;
    }
}

template<int BLOCK_SIZE>
__global__ void set_tadjncy_tadjwgt_block_bin(  const int bin, const int *bin_offset, const int *bin_idx, \
                                                const int *txadj, const int *xadj, const int *match, const int *adjncy, const int *adjwgt, \
                                                const int *cmap, int *tadjncy, int *tadjwgt, int nvtxs)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;

    int bin_row_offset = __ldg(&bin_offset[bin]) + block_id; 
    int vertex = __ldg(&bin_idx[bin_row_offset]);
    int vertex_match = __ldg(&match[vertex]);
    int map_cvertex = __ldg(&cmap[vertex]);
    
    int tbegin = __ldg(&txadj[map_cvertex]);
    bool is_vertex_larger = (vertex > vertex_match);
    tbegin += (is_vertex_larger) ? (__ldg(&xadj[vertex_match + 1]) - __ldg(&xadj[vertex_match])) : 0;
    int begin = __ldg(&xadj[vertex]);
    int end = __ldg(&xadj[vertex + 1]);

    for(int j = begin + thread_id, ptr = tbegin + thread_id;j < end;j += BLOCK_SIZE, ptr += BLOCK_SIZE)
    {
        int k = __ldg(&adjncy[j]);
        int wgt_k = __ldg(&adjwgt[j]);
        int k_cmap = __ldg(&cmap[k]);

        tadjwgt[ptr] = wgt_k;
        tadjncy[ptr] = k_cmap;
    }
}

__global__ void segment_sort(int *tadjncy, int *tadjwgt, int nedges, int *txadj, int cnvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < cnvtxs)
	{
		int begin, end, ptr, val;
		int i, j, k;
		begin = txadj[ii];
		end   = txadj[ii + 1];

		for(i = begin;i < end;i++)
		{
			ptr = i;
			val = tadjwgt[ptr];
			for(j = i + 1;j < end;j++)
				if(tadjwgt[j] < val) ptr = j, val = tadjwgt[ptr];
			val = tadjncy[ptr], tadjncy[ptr] = tadjncy[i], tadjncy[i] = val;
			val = tadjwgt[ptr], tadjwgt[ptr] = tadjwgt[i], tadjwgt[i] = val;
		}
	}
}

//Sort_cnedges2_part1<<<(cnvtxs + 3) / 4,128>>>(graph->tadjncy,graph->txadj,temp_scan,cnvtxs);
/*CUDA-Segmentation sorting part1-set scan array value 0 or 1*/
__global__ void mark_edges(int *tadjncy, int *txadj, int *temp_scan, int cnvtxs)
{
    // long long int i   = blockIdx.x * blockDim.x + threadIdx.x;
	int ii  = blockIdx.x * 4 + threadIdx.x / 32;
	int tid = threadIdx.x % 32;

	if(ii < cnvtxs)
	{
		int j, begin, end, iii;

		begin  = txadj[ii];
		end    = txadj[ii + 1];

		for(iii = begin + tid;iii < end;iii += 32)
		{
			j   = tadjncy[iii];
				
			if(iii == begin)
			{
				if(j == ii) temp_scan[iii] = 0;
				else temp_scan[iii] = 1;
			}
			else 
			{
				if(j == ii) temp_scan[iii] = 0;
				else
				{
					if(j == tadjncy[iii - 1]) temp_scan[iii] = 0;
					else temp_scan[iii] = 1;
				}
			}
		}
	}
}

__global__ void mark_edges_shfl(int *tadjncy, int *txadj, int *temp_scan, int cnvtxs)
{
    int lane_id = threadIdx.x & 31;
    int blockwarp_id = threadIdx.x >> 5;
    int warp_id = blockIdx.x * 4 + blockwarp_id;

    if(warp_id >= cnvtxs)
        return ;

    int vertex, begin, end;
    vertex = warp_id;
    begin = txadj[vertex];
    end   = txadj[vertex + 1];
    
    for(int j = begin + lane_id;j < end;j += 32)
    {
        int k = tadjncy[j];

        int k_front1 = __shfl_up_sync(0xffffffff, k, 1);
        if(j != begin)
            k_front1 = tadjncy[j - 1];
        
        //  0:k == vertex || (j != begin && k == k_front1)
        //  1:(j == begin && k != vertex) || (j != begin && k != vertex && k != k_front1)
        bool is_valid = (k != vertex) && ( (j == begin) || (k != k_front1) );
        temp_scan[j] = is_valid ? 1 : 0;
    }
}

template<int WARP_SIZE>
__global__ void mark_edges_warp_bin(const int tbin, const int *tbin_offset, const int *tbin_idx, const int *tadjncy, const int *txadj, int *temp_scan, int num)
{
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
	int blockwarp_id = threadIdx.x / WARP_SIZE;
	int subwarp_num = blockDim.x / WARP_SIZE;
	int subwarp_id = blockIdx.x * subwarp_num + blockwarp_id;

    if(subwarp_id >= num) 
        return ;
    
    int tbin_row_offset = __ldg(&tbin_offset[tbin]) + subwarp_id; 
    int vertex = __ldg(&tbin_idx[tbin_row_offset]);
    int begin = __ldg(&txadj[vertex]);
    int end   = __ldg(&txadj[vertex + 1]);

    for(int j = begin + lane_id;j < end;j += WARP_SIZE)
    {
        int k = __ldg(&tadjncy[j]);

        int k_front1 = __shfl_up_sync(0xffffffff, k, 1);
        if(j != begin)
            k_front1 = __ldg(&tadjncy[j - 1]);
        
        //  0:k == vertex || (j != begin && k == k_front1)
        //  1:(j == begin && k != vertex) || (j != begin && k != vertex && k != k_front1)
        bool is_valid = (k != vertex) && ( (j == begin) || (k != k_front1) );
        temp_scan[j] = is_valid ? 1 : 0;
    }
}

template<int SUBWARP_SIZE>
__global__ void mark_edges_subwarp(const int tbin, const int *tbin_offset, const int *tbin_idx, const int *tadjncy, const int *txadj, int *temp_scan, int num)
{
    int lane_id = threadIdx.x & (SUBWARP_SIZE - 1);
	int blockwarp_id = threadIdx.x / SUBWARP_SIZE;
	int subwarp_num = blockDim.x / SUBWARP_SIZE;
	int subwarp_id = blockIdx.x * subwarp_num + blockwarp_id;

    if(subwarp_id >= num) 
        return ;
    
    int tbin_row_offset =tbin_offset[tbin] + subwarp_id; 
    int vertex =tbin_idx[tbin_row_offset];
    int begin =txadj[vertex];
    int end   =txadj[vertex + 1];

    int j = begin + lane_id;

    // 新逻辑：
    // const int physical_lane = threadIdx.x % 32;  // 物理warp内的lane编号
    // const int subwarp_index = physical_lane / SUBWARP_SIZE; // 子warp索引
    // unsigned mask = (SUBWARP_SIZE == 32) ? 0xffffffff : ((1u << SUBWARP_SIZE) - 1) << (subwarp_index * SUBWARP_SIZE); // 子warp掩码

    // unsigned mask = (SUBWARP_SIZE == 32) ? 0xffffffff : ((1u << SUBWARP_SIZE) - 1);
    if(j >= end) 
    {
        // __syncwarp(mask);
        return ;
    }
    
    int k =tadjncy[j];
    // unsigned mask = (SUBWARP_SIZE == 32) ? 0xffffffff : ((1u << SUBWARP_SIZE) - 1);
    const int physical_lane = threadIdx.x % 32;  // 物理warp内的lane编号
    const int subwarp_index = physical_lane / SUBWARP_SIZE; // 子warp索引
    unsigned mask = (SUBWARP_SIZE == 32) ? 0xffffffff : ((1u << SUBWARP_SIZE) - 1) << (subwarp_index * SUBWARP_SIZE); // 子warp掩码
    int k_front1 = __shfl_up_sync(mask, k, 1, SUBWARP_SIZE);
    // int k_front1 = __shfl_up_sync(mask, k, 1);
    bool is_first = (j == begin);
    bool is_valid = (k != vertex) && ( is_first || (k != k_front1) );
    temp_scan[j] = is_valid ? 1 : 0;
}

template<int KWARP_SIZE>
__global__ void mark_edges_kwarp(const int tbin, const int *tbin_offset, const int *tbin_idx, const int *tadjncy, const int *txadj, int *temp_scan, int num)
{
    int lane_id = threadIdx.x & (KWARP_SIZE - 1);
    int physical_lane_id = threadIdx.x & 31;
	int blockwarp_id = threadIdx.x / KWARP_SIZE;
	int subwarp_num = blockDim.x / KWARP_SIZE;
	int subwarp_id = blockIdx.x * subwarp_num + blockwarp_id;

    if(subwarp_id >= num) 
        return ;
    
    int tbin_row_offset = __ldg(&tbin_offset[tbin]) + subwarp_id; 
    int vertex = __ldg(&tbin_idx[tbin_row_offset]);
    int begin = __ldg(&txadj[vertex]);
    int end   = __ldg(&txadj[vertex + 1]);

    int j = begin + lane_id;
    if(j >= end) return ;
    
    int k = __ldg(&tadjncy[j]);
    int k_front1 = __shfl_up_sync(0xffffffff, k, 1);
    if(physical_lane_id == 0 && lane_id != 0)
        k_front1 = __ldg(&tadjncy[j - 1]);
    bool is_first = (j == begin);
    bool is_valid = (k != vertex) && ( is_first || (k != k_front1) );
    temp_scan[j] = is_valid ? 1 : 0;
}

template<int BLOCK_SIZE>
__global__ void mark_edges_block(const int tbin, const int *tbin_offset, const int *tbin_idx, const int *tadjncy, const int *txadj, int *temp_scan, int num)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    int physical_lane_id = threadIdx.x & 31;
    
    int tbin_row_offset = __ldg(&tbin_offset[tbin]) + block_id; 
    int vertex = __ldg(&tbin_idx[tbin_row_offset]);
    int begin = __ldg(&txadj[vertex]);
    int end   = __ldg(&txadj[vertex + 1]);

    for(int j = begin + thread_id;j < end;j += BLOCK_SIZE)
    {
        int k = __ldg(&tadjncy[j]);
        int k_front1 = __shfl_up_sync(0xffffffff, k, 1);
        if(physical_lane_id == 0 && j != begin)
            k_front1 = __ldg(&tadjncy[j - 1]);
        bool is_first = (j == begin);
        bool is_valid = (k != vertex) && ( is_first || (k != k_front1) );
        temp_scan[j] = is_valid ? 1 : 0;
    }
}

__global__ void Sort_cnedges2_part1_shared(int *tadjncy, int *txadj, int *temp_scan, int cnvtxs)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int ii  = i / 32;
    int tid = i - ii * 32;

    __shared__ int cache_txadj[8];
    // __shared__ int cache_tadjncy[4][32];
    int pid = ii % 4;

    if(ii < cnvtxs)
    {
        int j, begin, end;

        if(tid == 0) cache_txadj[pid] = txadj[ii];
        if(tid == 1) cache_txadj[pid + 1] = txadj[ii + 1];

        __syncthreads();

        // begin  = txadj[ii];
        // end    = txadj[ii + 1];

        //bank conflict
        begin = cache_txadj[pid];
        end   = cache_txadj[pid + 1];

        for(i = begin + tid;i < end;i += 32)
        {
            j   = tadjncy[i];
            // cache_tadjncy[pid][tid] = j;
            // __syncthreads();
            
            if(i == begin)
            {
                if(j == ii) temp_scan[i] = 0;
                else temp_scan[i] = 1;
            }
            else 
            {
                if(j == ii) temp_scan[i] = 0;
                else
                {
                    if(j == tadjncy[i - 1]) temp_scan[i] = 0;
                    // if(tid != 0 && j == cache_tadjncy[pid][tid - 1]) temp_scan[i] = 0;
                    // else if(tid == 0 && j == tadjncy[i - 1]) temp_scan[i] = 0;
                    else temp_scan[i] = 1;
                }
            }
        }
    }
}

/*CUDA-Segmentation sorting part2-set cxadj*/
__global__ void set_cxadj(const int *txadj, const int *temp_scan, int *cxadj, int cnvtxs)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < cnvtxs)
    { 
        int ppp = __ldg(&txadj[ii + 1]);
        cxadj[ii + 1] = (ppp > 0) ? __ldg(&temp_scan[ppp - 1]) : 0;

        // cxadj[ii + 1] = temp_scan[ppp - 1];
        // if(ppp > 0)
        //     cxadj[ii + 1] = temp_scan[ppp - 1];
        // else 
        //     cxadj[ii + 1] = 0;
    }
    else if(ii == cnvtxs) cxadj[0] = 0;
} 

/*CUDA-Segmentation sorting part2.5-init cadjwgt and cadjncy*/
__global__ void init_cadjwgt(int *cadjwgt, int cnedges)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;  

    if(ii < cnedges)
        cadjwgt[ii] = 0;
}

/*CUDA-Segmentation sorting part3-deduplication and accumulation*/
__global__ void set_cadjncy_cadjwgt(int *tadjncy,int *txadj, int *tadjwgt,int *temp_scan, int *cxadj,int *cadjncy, int *cadjwgt, int cnvtxs)
{
    // long long int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int ii  = blockIdx.x * 4 + threadIdx.x / 32;
    int tid = threadIdx.x % 32;

    // if(ii < cnvtxs)
    // {
    //     int ptr, j, begin, end;

    //     begin = txadj[ii];
    //     end   = txadj[ii + 1];

    //     for(i = begin + tid;i < end;i += 32)
    //     {
    //         j   = tadjncy[i];
    //         ptr = temp_scan[i] - 1;

    //         if(j != ii)
    //         {
    //             atomicAdd(&cadjwgt[ptr],tadjwgt[i]);

    //             if(i == begin) cadjncy[ptr] = j;
    //             else
    //             {
    //                 if(j != tadjncy[i - 1]) cadjncy[ptr] = j;
    //             }
    //         }
    //     }
    // }

	if(ii < cnvtxs)
	{
		int begin, end, j, k, iii;

		begin = txadj[ii];
        end   = txadj[ii + 1];

		for(iii = begin + tid;iii < end;iii += 32)
		{
			j = tadjncy[iii];
			k = temp_scan[iii] - 1;

			if(iii == begin)
			{
				if(j != ii)
				{
					cadjncy[k] = j;
					atomicAdd(&cadjwgt[k],tadjwgt[iii]);
				}
			}
			else 
			{
				if(j != ii)
				{
					if(j != tadjncy[iii - 1])
					{
						cadjncy[k] = j;
						atomicAdd(&cadjwgt[k],tadjwgt[iii]);
					}
					else atomicAdd(&cadjwgt[k],tadjwgt[iii]);
				}
			}
		}
	}
}

__global__ void set_cadjncy_cadjwgt_shfl(int *tadjncy,int *txadj, int *tadjwgt,int *temp_scan, int *cxadj,int *cadjncy, int *cadjwgt, int cnvtxs)
{
    // long long int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;
    int blockwarp_id = threadIdx.x >> 5;
    int warp_id = blockIdx.x * 4 + blockwarp_id;

    if(warp_id >= cnvtxs)
        return ;

    int vertex, begin, end;
    vertex = warp_id;
    begin = txadj[vertex];
    end   = txadj[vertex + 1];
    
    for(int j = begin + lane_id;j < end;j += 32)
    {
        int k = tadjncy[j];
        int mark_k = temp_scan[j] - 1;

        int k_front1 = __shfl_up_sync(0xffffffff, k, 1);
        if(j != begin)
            k_front1 = tadjncy[j - 1];
        
        bool is_valid = (k != vertex);
        bool need_add = (j == begin) || (k != k_front1);

        if(is_valid)
        {
            if(need_add)
                cadjncy[mark_k] = k;
            
            atomicAdd(&cadjwgt[mark_k], tadjwgt[j]);
        }
    }
}

template<int WARP_SIZE>
__global__ void set_cadjncy_cadjwgt_warp_bin(   const int tbin, const int *tbin_offset, const int *tbin_idx, \
                                                const int *tadjncy, const int *txadj, const int *tadjwgt, const int *temp_scan, \
                                                const int *cxadj, int *cadjncy, int *cadjwgt, int num)
{
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
	int blockwarp_id = threadIdx.x / WARP_SIZE;
	int subwarp_num = blockDim.x / WARP_SIZE;
	int subwarp_id = blockIdx.x * subwarp_num + blockwarp_id;

    if(subwarp_id >= num) 
        return ;
    
    int tbin_row_offset = __ldg(&tbin_offset[tbin]) + subwarp_id; 
    int vertex = __ldg(&tbin_idx[tbin_row_offset]);
    int begin = __ldg(&txadj[vertex]);
    int end   = __ldg(&txadj[vertex + 1]);

    for(int j = begin + lane_id;j < end;j += WARP_SIZE)
    {
        int k = __ldg(&tadjncy[j]);
        int mark_k = __ldg(&temp_scan[j]) - 1;

        int k_front1 = __shfl_up_sync(0xffffffff, k, 1);
        if(j != begin)
            k_front1 = __ldg(&tadjncy[j - 1]);
        
        bool is_valid = (k != vertex);
        bool need_add = (j == begin) || (k != k_front1);

        if(is_valid)
        {
            if(need_add)
                cadjncy[mark_k] = k;
            
            atomicAdd(&cadjwgt[mark_k], __ldg(&tadjwgt[j]));
        }
    }
}
    // extern __shared__ int cache_cadjwgt[];
    // cache_cadjwgt[lane_id] = 0;
template<int SUBWARP_SIZE>
__global__ void set_cadjncy_cadjwgt_subwarp(const int tbin, const int *tbin_offset, const int *tbin_idx, \
                                            const int *tadjncy, const int *txadj, const int *tadjwgt, const int *temp_scan, \
                                            const int *cxadj, int *cadjncy, int *cadjwgt, int num)
{
    int lane_id = threadIdx.x & (SUBWARP_SIZE - 1);
	int blockwarp_id = threadIdx.x / SUBWARP_SIZE;
	int subwarp_num = blockDim.x / SUBWARP_SIZE;
	int subwarp_id = blockIdx.x * subwarp_num + blockwarp_id;

    if(subwarp_id >= num) 
        return ;

    int tbin_row_offset = tbin_offset[tbin] + subwarp_id; 
    int vertex = tbin_idx[tbin_row_offset];
    int begin = txadj[vertex];
    int end   = txadj[vertex + 1];

    int j = begin + lane_id;
    if(j >= end) return ;

    int k = tadjncy[j];
    int mark_k = temp_scan[j] - 1;
    const int physical_lane = threadIdx.x % 32;  // 物理warp内的lane编号
    const int subwarp_index = physical_lane / SUBWARP_SIZE; // 子warp索引
    unsigned mask = (SUBWARP_SIZE == 32) ? 0xffffffff : ((1u << SUBWARP_SIZE) - 1) << (subwarp_index * SUBWARP_SIZE); // 子warp掩码
    // int k_front1 = __shfl_up_sync(mask, k, 1, SUBWARP_SIZE);
    // unsigned mask = (SUBWARP_SIZE == 32) ? 0xffffffff : ((1u << SUBWARP_SIZE) - 1);
    int k_front1 = __shfl_up_sync(mask, k, 1, SUBWARP_SIZE);

    bool is_valid = (k != vertex);
    bool need_add = (j == begin) || (k != k_front1);

    if(is_valid)
    {
        if(need_add)
            cadjncy[mark_k] = k;
            
        atomicAdd(&cadjwgt[mark_k], tadjwgt[j]);
    }
}

template<int KWARP_SIZE>
__global__ void set_cadjncy_cadjwgt_kwarp(  const int tbin, const int *tbin_offset, const int *tbin_idx, \
                                            const int *tadjncy, const int *txadj, const int *tadjwgt, const int *temp_scan, \
                                            const int *cxadj, int *cadjncy, int *cadjwgt, int num)
{
    int lane_id = threadIdx.x & (KWARP_SIZE - 1);
    int physical_lane_id = threadIdx.x & 31;
	int blockwarp_id = threadIdx.x / KWARP_SIZE;
	int subwarp_num = blockDim.x / KWARP_SIZE;
	int subwarp_id = blockIdx.x * subwarp_num + blockwarp_id;

    if(subwarp_id >= num) 
        return ;

    int tbin_row_offset = __ldg(&tbin_offset[tbin]) + subwarp_id; 
    int vertex = __ldg(&tbin_idx[tbin_row_offset]);
    int begin = __ldg(&txadj[vertex]);
    int end   = __ldg(&txadj[vertex + 1]);

    int j = begin + lane_id;
    if(j >= end) return ;

    int k = __ldg(&tadjncy[j]);
    int mark_k = __ldg(&temp_scan[j]) - 1;
    int k_front1 = __shfl_up_sync(0xffffffff, k, 1);
    if(physical_lane_id == 0 && lane_id != 0)
        k_front1 = tadjncy[j - 1];
    bool is_valid = (k != vertex);
    bool need_add = (j == begin) || (k != k_front1);

    if(is_valid)
    {
        if(need_add)
            cadjncy[mark_k] = k;
            
        atomicAdd(&cadjwgt[mark_k], __ldg(&tadjwgt[j]));
    }
}

template<int BLOCK_SIZE>
__global__ void set_cadjncy_cadjwgt_block(  const int tbin, const int *tbin_offset, const int *tbin_idx, \
                                            const int *tadjncy, const int *txadj, const int *tadjwgt, const int *temp_scan, \
                                            const int *cxadj, int *cadjncy, int *cadjwgt, int num)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    int physical_lane_id = threadIdx.x & 31;

    int tbin_row_offset = __ldg(&tbin_offset[tbin]) + block_id; 
    int vertex = __ldg(&tbin_idx[tbin_row_offset]);
    int begin = __ldg(&txadj[vertex]);
    int end   = __ldg(&txadj[vertex + 1]);

    for(int j = begin + thread_id;j < end;j += BLOCK_SIZE)
    {
        int k = __ldg(&tadjncy[j]);
        int mark_k = __ldg(&temp_scan[j]) - 1;
        int k_front1 = __shfl_up_sync(0xffffffff, k, 1);
        if(physical_lane_id == 0 && j != begin)
            k_front1 = __ldg(&tadjncy[j - 1]);
        
        bool is_valid = (k != vertex);
        bool need_add = (j == begin) || (k != k_front1);

        if(is_valid)
        {
            if(need_add)
                cadjncy[mark_k] = k;
                
            atomicAdd(&cadjwgt[mark_k], __ldg(&tadjwgt[j]));
        }
    }
}

__global__ void print_xadj(int nvtxs, int *xadj)
{
	int i;
	printf("xadj:");
	for(i = 0;i < nvtxs;i++)
		printf("%10d ", xadj[i]);
	printf("\n");
}

/*Create gpu coarsen graph by contract*/
void hunyuangraph_gpu_create_cgraph(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, hunyuangraph_graph_t *cgraph)
{
    int nvtxs  = graph->nvtxs;
    int nedges = graph->nedges;
    int cnvtxs = cgraph->nvtxs;

    // printf("txadj\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(11, graph->txadj);
	// cudaDeviceSynchronize();

#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
#endif
    if(GPU_Memory_Pool)
    {
        prefixsum(graph->txadj + 1, graph->txadj + 1, cnvtxs, prefixsum_blocksize, 1);	//0:lmalloc,1:rmalloc
    }
    else
    {
        thrust::inclusive_scan(thrust::device, graph->txadj, graph->txadj + cnvtxs + 1, graph->txadj);
        // thrust::exclusive_scan(thrust::device, graph->txadj, graph->txadj + cnvtxs + 1, graph->txadj);
    }
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    exclusive_scan_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
#endif  
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(11, graph->txadj);
	// cudaDeviceSynchronize();

    // printf("prefixsum end\n");
    // cudaMalloc((void**)&graph->tadjncy,nedges * sizeof(int));
    // cudaMalloc((void**)&graph->tadjwgt,nedges * sizeof(int));
	int *bb_keysB_d, *bb_valsB_d;
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc,NULL);
#endif
    if(GPU_Memory_Pool)
    {
        bb_keysB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"hunyuangraph_gpu_create_cgraph: bb_keysB_d");
        bb_valsB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"hunyuangraph_gpu_create_cgraph: bb_valsB_d");
        graph->tadjncy = (int *)rmalloc_with_check(sizeof(int) * nedges,"hunyuangraph_gpu_create_cgraph: tadjncy");
        graph->tadjwgt = (int *)rmalloc_with_check(sizeof(int) * nedges,"hunyuangraph_gpu_create_cgraph: tadjwgt");
    }
    else
    {
        cudaMalloc((void**)&graph->tadjncy, sizeof(int) * nedges);
        cudaMalloc((void**)&graph->tadjwgt, sizeof(int) * nedges);
    }
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc,NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;

    // for(int i = 1;i < 14;i++)
	// {
	// 	int num = graph->h_bin_offset[i + 1] - graph->h_bin_offset[i];
    //     if(num != 0)
	// 	    printf("set_tadjncy_tadjwgt_subwarp %d begin num=%d %d\n", i, num, (int)pow(2, i));
    // }

    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
#endif
    // set_tadjncy_tadjwgt<<<(nvtxs + 3) / 4,128>>>(graph->txadj,graph->cuda_xadj,graph->cuda_match,graph->cuda_adjncy,graph->cuda_cmap,\
    //     graph->tadjncy,graph->tadjwgt,graph->cuda_adjwgt,nvtxs);
    for(int i = 1;i < 14;i++)
	{
		int num = graph->h_bin_offset[i + 1] - graph->h_bin_offset[i];
        // if(num != 0)
		//     printf("set_tadjncy_tadjwgt_subwarp %d begin num=%d %d\n", i, num, (int)pow(2, i));
		// cudaDeviceSynchronize();
		if(num != 0)
		switch(i)
		{
		case 1:
            set_tadjncy_tadjwgt_subwarp<2><<<(num + 63) / 64, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy, \
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 2:
			set_tadjncy_tadjwgt_subwarp<4><<<(num + 31) / 32, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 3:
			set_tadjncy_tadjwgt_subwarp<8><<<(num + 15) / 16, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 4:
			set_tadjncy_tadjwgt_subwarp<16><<<(num + 7) / 8, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 5:
			set_tadjncy_tadjwgt_subwarp<32><<<(num + 3) / 4, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 6:
            set_tadjncy_tadjwgt_subwarp<64><<<(num + 1) / 2 , 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			// set_tadjncy_tadjwgt_warp_bin<32><<<(num + 3) / 4 , 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
            //     graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 7:
            set_tadjncy_tadjwgt_subwarp<128><<<num, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 8:
            set_tadjncy_tadjwgt_block_bin<128><<<num, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 9:
            set_tadjncy_tadjwgt_block_bin<128><<<num, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 10:
            set_tadjncy_tadjwgt_block_bin<128><<<num, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 11:
            set_tadjncy_tadjwgt_block_bin<128><<<num, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 12:
            set_tadjncy_tadjwgt_block_bin<128><<<num, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		case 13:
            set_tadjncy_tadjwgt_block_bin<128><<<num, 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
                graph->cuda_adjwgt, graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, num);
			break;
		default:
            break;
		}
	}
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    set_tadjncy_tadjwgt_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
#endif
    // printf("set_tadjncy_tadjwgt end\n");
    // printf("tadjncy/tadjwgt\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(160, graph->tadjncy);
	// cudaDeviceSynchronize();
    // print_xadj<<<1, 1>>>(160, graph->tadjwgt);
	// cudaDeviceSynchronize();

    if(GPU_Memory_Pool)
    {
        int *bb_counter, *bb_id;
#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&begin_malloc,NULL);
#endif
        // bb_keysB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"bb_keysB_d");
        // bb_valsB_d = (int *)rmalloc_with_check(sizeof(int) * nedges,"bb_valsB_d");
        bb_id      = (int *)rmalloc_with_check(sizeof(int) * cnvtxs,"bb_id");
        bb_counter = (int *)rmalloc_with_check(sizeof(int) * 13,"bb_counter");
#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&end_malloc,NULL);
        coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;

        // printf("hunyuangraph_segmengtsort malloc end\n");

        cudaDeviceSynchronize();
        gettimeofday(&begin_gpu_contraction,NULL);
#endif
        hunyuangraph_segmengtsort(graph->tadjncy, graph->tadjwgt, nedges, graph->txadj, cnvtxs, bb_counter, bb_id, bb_keysB_d, bb_valsB_d);
        // segment_sort<<<(cnvtxs + 127) / 128, 128>>>(graph->tadjncy, graph->tadjwgt, nedges, graph->txadj, cnvtxs);
#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&end_gpu_contraction,NULL);
        ncy_segmentsort_gpu_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;

        // printf("hunyuangraph_segmengtsort end\n");

        cudaDeviceSynchronize();
        gettimeofday(&begin_free,NULL);
#endif
        rfree_with_check((void *)bb_counter, sizeof(int) * 13,"bb_counter");		//bb_counter
        rfree_with_check((void *)bb_id, sizeof(int) * cnvtxs,"bb_id");				//bb_id
        graph->tadjncy = bb_keysB_d;
        graph->tadjwgt = bb_valsB_d;
        rfree_with_check((void *)graph->tadjwgt, sizeof(int) * nedges,"bb_valsB_d");		//tadjwgt
        rfree_with_check((void *)graph->tadjncy, sizeof(int) * nedges,"bb_keysB_d");		//tadjncy
#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&end_free,NULL);
        coarsen_free += (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
#endif
    }
    else
    {
#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&begin_gpu_contraction,NULL);
#endif
        bb_segsort(graph->tadjncy, graph->tadjwgt, nedges, graph->txadj, cnvtxs);
#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&end_gpu_contraction,NULL);
        ncy_segmentsort_gpu_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
#endif
    }
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(160, graph->tadjncy);
	// cudaDeviceSynchronize();
    // print_xadj<<<1, 1>>>(160, graph->tadjwgt);
	// cudaDeviceSynchronize();

    int *temp_scan;
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc,NULL);
#endif
    // cudaMalloc((void**)&temp_scan, nedges * sizeof(int));
    if(GPU_Memory_Pool)
    	temp_scan = (int *)rmalloc_with_check(sizeof(int) * nedges,"hunyuangraph_gpu_create_cgraph: temp_scan");
    else    
        cudaMalloc((void**)&temp_scan, sizeof(int) * nedges);
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc,NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;

    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
#endif
    // mark_edges<<<(cnvtxs + 3) / 4,128>>>(graph->tadjncy,graph->txadj,temp_scan,cnvtxs);
    // mark_edges_shfl<<<(cnvtxs + 3) / 4,128>>>(graph->tadjncy, graph->txadj, temp_scan, cnvtxs);
    for(int i = 1;i < 14;i++)
	{
		int num = graph->h_tbin_offset[i + 1] - graph->h_tbin_offset[i];
        // if(num != 0)
		//     printf("set_tadjncy_tadjwgt_subwarp %d begin num=%d %d\n", i, num, (int)pow(2, i));
		// cudaDeviceSynchronize();
		if(num != 0)
		switch(i)
		{
		case 1:
            mark_edges_subwarp<2><<<(num + 63) / 64, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 2:
			mark_edges_subwarp<4><<<(num + 31) / 32, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 3:
			mark_edges_subwarp<8><<<(num + 15) / 16, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 4:
			mark_edges_subwarp<16><<<(num + 7) / 8, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 5:
			mark_edges_subwarp<32><<<(num + 3) / 4, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 6:
            mark_edges_kwarp<64><<<(num + 1) / 2, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			// set_tadjncy_tadjwgt_warp_bin<32><<<(num + 3) / 4 , 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
            //     graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, graph->cuda_adjwgt, num);
			break;
		case 7:
            mark_edges_kwarp<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 8:
            mark_edges_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 9:
            mark_edges_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 10:
            mark_edges_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 11:
            mark_edges_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 12:
            mark_edges_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		case 13:
            mark_edges_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, temp_scan, num);
			break;
		default:
            break;
		}
	}
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    mark_edges_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    // printf("mark_edges end\n");
    // printf("temp_scan\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(160, temp_scan);
	// cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
#endif

    if(GPU_Memory_Pool)
        prefixsum(temp_scan,temp_scan,nedges,prefixsum_blocksize,1);	//0:lmalloc,1:rmalloc
    else
        thrust::inclusive_scan(thrust::device,temp_scan, temp_scan + nedges, temp_scan);
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    inclusive_scan_time2 += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    // printf("prefixsum end\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(160, temp_scan);
	// cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc,NULL);
#endif
    // cudaMalloc((void**)&cgraph->cuda_xadj, (cnvtxs+1)*sizeof(int));
    if(GPU_Memory_Pool)
    	cgraph->cuda_xadj = (int *)lmalloc_with_check(sizeof(int) * (cnvtxs + 1),"hunyuangraph_gpu_create_cgraph: xadj");
    else 
        cudaMalloc((void**)&cgraph->cuda_xadj, sizeof(int) * (cnvtxs + 1));
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc,NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;

    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
#endif
    set_cxadj<<<(cnvtxs + 128) / 128,128>>>(graph->txadj,temp_scan,cgraph->cuda_xadj,cnvtxs);
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    set_cxadj_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
#endif
    // printf("set_cxadj end\n");
    // printf("cxadj\n");
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(11, cgraph->cuda_xadj);
	// cudaDeviceSynchronize();

    cudaMemcpy(&cgraph->nedges, &cgraph->cuda_xadj[cnvtxs], sizeof(int), cudaMemcpyDeviceToHost);
    int cnedges = cgraph->nedges;

#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc,NULL);
#endif
    // cudaMalloc((void**)&cgraph->cuda_adjncy, cgraph->nedges * sizeof(int));
    // cudaMalloc((void**)&cgraph->cuda_adjwgt, cgraph->nedges * sizeof(int));
    if(GPU_Memory_Pool)
    {
        cgraph->cuda_adjncy = (int *)lmalloc_with_check(sizeof(int) * cnedges,"hunyuangraph_gpu_create_cgraph: adjncy");
        cgraph->cuda_adjwgt = (int *)lmalloc_with_check(sizeof(int) * cnedges,"hunyuangraph_gpu_create_cgraph: adjwgt");

        //  Mask the memcpy time after sorting
            //  The time of memcpy after sorting needs to be masked only if coarsening can continue
        if( cnvtxs > hunyuangraph_admin->Coarsen_threshold &&
            cnvtxs < 0.85 * nvtxs && cnedges > cnvtxs / 2)
        {
            size_t Spacing_distance = 0;
            size_t tmp;

            //  length_vertex
            tmp = cnvtxs * sizeof(int);
            Spacing_distance += (tmp + hunyuangraph_GPU_cacheline - 1) / hunyuangraph_GPU_cacheline * hunyuangraph_GPU_cacheline;
            //  bin_offset
            tmp = 15 * sizeof(int);
            Spacing_distance += (tmp + hunyuangraph_GPU_cacheline - 1) / hunyuangraph_GPU_cacheline * hunyuangraph_GPU_cacheline;
            //  bin_idx
            tmp = cnvtxs * sizeof(int);
            Spacing_distance += (tmp + hunyuangraph_GPU_cacheline - 1) / hunyuangraph_GPU_cacheline * hunyuangraph_GPU_cacheline;
            //  cmap
            tmp = cnvtxs * sizeof(int);
            Spacing_distance += (tmp + hunyuangraph_GPU_cacheline - 1) / hunyuangraph_GPU_cacheline * hunyuangraph_GPU_cacheline;
            //  where
            tmp = cnvtxs * sizeof(int);
            Spacing_distance += (tmp + hunyuangraph_GPU_cacheline - 1) / hunyuangraph_GPU_cacheline * hunyuangraph_GPU_cacheline;

            cgraph->bb_cvalsB_d = (int *)lmalloc_with_mandatory_space(sizeof(int) * cnedges, Spacing_distance,"hunyuangraph_gpu_create_cgraph: bb_cvalsB_d");
            
            //  cgraph->bb_cvalsB_d
            tmp = cnedges * sizeof(int);
            Spacing_distance += (tmp + hunyuangraph_GPU_cacheline - 1) / hunyuangraph_GPU_cacheline * hunyuangraph_GPU_cacheline;
            
            cgraph->bb_ckeysB_d = (int *)lmalloc_with_mandatory_space(sizeof(int) * cnedges, Spacing_distance,"hunyuangraph_gpu_create_cgraph: bb_ckeysB_d");

            //  swap 
            int *p;
            p = cgraph->bb_cvalsB_d, cgraph->bb_cvalsB_d = cgraph->cuda_adjncy, cgraph->cuda_adjncy = p;
            p = cgraph->bb_ckeysB_d, cgraph->bb_ckeysB_d = cgraph->cuda_adjwgt, cgraph->cuda_adjwgt = p;
        }
    }
    else
    {
        cudaMalloc((void**)&cgraph->cuda_adjncy, sizeof(int) * cnedges);
        cudaMalloc((void**)&cgraph->cuda_adjwgt, sizeof(int) * cnedges);
    }
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc,NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
    
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
#endif
    init_cadjwgt<<<(cnedges + 127) / 128,128>>>(cgraph->cuda_adjwgt,cnedges);
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    init_cadjwgt_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
    // printf("init_cadjwgt end\n");
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_contraction,NULL);
#endif
    // set_cadjncy_cadjwgt<<<(cnvtxs + 3) / 4,128>>>(graph->tadjncy,graph->txadj,\
    //     graph->tadjwgt,temp_scan,cgraph->cuda_xadj,cgraph->cuda_adjncy,cgraph->cuda_adjwgt,cnvtxs);
    // set_cadjncy_cadjwgt_shfl<<<(cnvtxs + 3) / 4,128>>>(graph->tadjncy,graph->txadj,\
    //     graph->tadjwgt,temp_scan,cgraph->cuda_xadj,cgraph->cuda_adjncy,cgraph->cuda_adjwgt,cnvtxs);

    for(int i = 1;i < 14;i++)
	{
		int num = graph->h_tbin_offset[i + 1] - graph->h_tbin_offset[i];
        // if(num != 0)
		//     printf("set_tadjncy_tadjwgt_subwarp %d begin num=%d %d\n", i, num, (int)pow(2, i));
		// cudaDeviceSynchronize();
		if(num != 0)
		switch(i)
		{
		case 1:
            set_cadjncy_cadjwgt_subwarp<2><<<(num + 63) / 64, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 2:
			set_cadjncy_cadjwgt_subwarp<4><<<(num + 31) / 32, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 3:
			set_cadjncy_cadjwgt_subwarp<8><<<(num + 15) / 16, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 4:
			set_cadjncy_cadjwgt_subwarp<16><<<(num + 7) / 8, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 5:
			set_cadjncy_cadjwgt_subwarp<32><<<(num + 3) / 4, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 6:
            set_cadjncy_cadjwgt_kwarp<64><<<(num + 1) / 2, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			// set_tadjncy_tadjwgt_warp_bin<32><<<(num + 3) / 4 , 128>>>(i, graph->bin_offset, graph->bin_idx, graph->txadj, graph->cuda_xadj, graph->cuda_match, graph->cuda_adjncy,\
            //     graph->cuda_cmap, graph->tadjncy, graph->tadjwgt, graph->cuda_adjwgt, num);
			break;
		case 7:
            set_cadjncy_cadjwgt_kwarp<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 8:
            set_cadjncy_cadjwgt_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 9:
            set_cadjncy_cadjwgt_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 10:
            set_cadjncy_cadjwgt_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 11:
            set_cadjncy_cadjwgt_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 12:
            set_cadjncy_cadjwgt_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		case 13:
            set_cadjncy_cadjwgt_block<128><<<num, 128>>>(i, graph->tbin_offset, graph->tbin_idx, graph->tadjncy, graph->txadj, graph->tadjwgt, temp_scan, cgraph->cuda_xadj, \
                cgraph->cuda_adjncy,cgraph->cuda_adjwgt, num);
			break;
		default:
            break;
		}
	}
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_contraction,NULL);
    set_cadjncy_cadjwgt_time += (end_gpu_contraction.tv_sec - begin_gpu_contraction.tv_sec) * 1000 + (end_gpu_contraction.tv_usec - begin_gpu_contraction.tv_usec) / 1000.0;
#endif
    // cudaDeviceSynchronize();
	// print_xadj<<<1, 1>>>(11, cgraph->cuda_xadj);
	// cudaDeviceSynchronize();
	cgraph->tvwgt[0] = graph->tvwgt[0];  

	// printf("cnvtxs=%d cnedges=%d ",cnvtxs,cgraph->nedges);

#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_free,NULL);
#endif
    if(GPU_Memory_Pool)
    {
        rfree_with_check((void *)temp_scan, sizeof(int) * nedges,"hunyuangraph_gpu_create_cgraph: temp_scan");			//temp_scan
        rfree_with_check((void *)graph->tadjwgt, sizeof(int) * nedges,"hunyuangraph_gpu_create_cgraph: tadjwgt");		//tadjwgt
        rfree_with_check((void *)graph->tadjncy, sizeof(int) * nedges,"hunyuangraph_gpu_create_cgraph: tadjncy");		//tadjncy
        rfree_with_check((void *)graph->tbin_idx, sizeof(int) * cnvtxs,"hunyuangraph_gpu_create_cgraph: tbin_idx");		//tbin_idx
        rfree_with_check((void *)graph->tbin_offset, sizeof(int) * 15,"hunyuangraph_gpu_create_cgraph: tbin_offset");	//tbin_offset
        rfree_with_check((void *)graph->txadj, sizeof(int) * (cnvtxs + 1),"hunyuangraph_gpu_create_cgraph: txadj");		//txadj
        rfree_with_check((void *)graph->cuda_match, sizeof(int) * nvtxs,"hunyuangraph_gpu_create_cgraph: match");		//match
    }
    else
    {
        cudaFree(temp_scan);
        cudaFree(graph->tadjwgt);
        cudaFree(graph->tadjncy);
        cudaFree(graph->tbin_idx);
        cudaFree(graph->tbin_offset);
        cudaFree(graph->txadj);
        cudaFree(graph->cuda_match);
    }
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_free,NULL);
    coarsen_free += (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
#endif

    free(graph->h_tbin_offset);
}

#endif