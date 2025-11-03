#ifndef _H_GPU_INITIALPARTITION
#define _H_GPU_INITIALPARTITION

#include "hunyuangraph_struct.h"
#include "hunyuangraph_common.h"
#include "hunyuangraph_admin.h"
#include "hunyuangraph_GPU_priorityqueue.h"
#include "hunyuangraph_GPU_splitgraph.h"

#include <cuda_runtime.h>
#include <cstdint> 
// #include "device_launch_parameters.h"
#include "curand_kernel.h"

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err));
        // You may also choose to exit or handle the error differently here.
    }
}

__device__ void compute_v_ed_id_bnd(int v, int *xadj, int *adjncy, int *adjwgt, hunyuangraph_int8_t *where, int *ed, int *id, hunyuangraph_int8_t *bnd)
{
    int begin, end, ted, tid;
    hunyuangraph_int8_t me, other;
    begin = xadj[v];
    end = xadj[v + 1];
    me = where[v];
    // if(me != 0 && me != 1)
    //     printf("Error: v = %d, where[v] = %d\n", v, me);
    ted = 0;
    tid = 0;

    if(begin == end)
    {
        bnd[v] = 1;
        ed[v] = 0;
        id[v] = 0;
        return ;
    }

    for(int i = begin;i < end; i++)
    {
        int j = adjncy[i];
        int wgt = adjwgt[i];
        other = where[j];

        if(me != other) 
            ted += wgt;
        else 
            tid += wgt;
    }

    ed[v] = ted;
    id[v] = tid;

    if(ted > 0)
        bnd[v] = 1;
    else
        bnd[v] = 0;
}

__device__ void warpReduction(volatile int *reduce_num, int tid ,int blocksize)
{
    if(blocksize >= 64) reduce_num[tid] += reduce_num[tid + 32];
    if(blocksize >= 32) reduce_num[tid] += reduce_num[tid + 16];
    if(blocksize >= 16) reduce_num[tid] += reduce_num[tid + 8];
    if(blocksize >= 8) reduce_num[tid] += reduce_num[tid + 4];
    if(blocksize >= 4) reduce_num[tid] += reduce_num[tid + 2];
    if(blocksize >= 2) reduce_num[tid] += reduce_num[tid + 1];
}

__device__ void WarpGetMax(volatile int *warp_reduce, int lane_id, int val)
{
    int compare = val;
    compare = __shfl_down_sync(0xffffffff, val, 16, 32);
    if(lane_id < 16)
    {
        if(compare > val)
        {
            val = compare;
            compare = val;
            warp_reduce[lane_id] = warp_reduce[lane_id + 16];
        }
    }
    compare = __shfl_down_sync(0xffffffff, val, 8, 32);
    if(lane_id < 8)
    {
        if(compare > val)
        {
            val = compare;
            compare = val;
            warp_reduce[lane_id] = warp_reduce[lane_id + 8];
        }
    }
    compare = __shfl_down_sync(0xffffffff, val, 4, 32);
    if(lane_id < 4)
    {
        if(compare > val)
        {
            val = compare;
            compare = val;
            warp_reduce[lane_id] = warp_reduce[lane_id + 4];
        }
    }
    compare = __shfl_down_sync(0xffffffff, val, 2, 32);
    if(lane_id < 2)
    {
        if(compare > val)
        {
            val = compare;
            compare = val;
            warp_reduce[lane_id] = warp_reduce[lane_id + 2];
        }
    }
    compare = __shfl_down_sync(0xffffffff, val, 1, 32);
    if(lane_id < 1)
    {
        if(compare > val)
        {
            val = compare;
            compare = val;
            warp_reduce[lane_id] = warp_reduce[lane_id + 1];
        }
    }
}

__device__ int hunyuangraph_gpu_int_min(int first, int second)
{
    if(first <= second)
        return first;
    else
        return second;
}

__device__ int hunyuangraph_gpu_int_max(int first, int second)
{
    if(first >= second)
        return first;
    else
        return second;
}

__device__ int hunyuangraph_gpu_int_abs(int a)
{
    if(a < 0)
        return -a;
    else
        return a;
}

__device__ void hunyuangraph_gpu_int_swap(int *a, int *b)
{
    int t = a[0];
    a[0] = b[0];
    b[0] = t;
}

__device__ void FM_2way_cut_refinement(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, hunyuangraph_int8_t *where, int *pwgts, int *ed, int *id, \
    hunyuangraph_int8_t *bnd, hunyuangraph_int8_t *moved, int *swaps, int tvwgt, int edgecut, double ntpwgts0, priority_queue_t *queues, int niter)
{
    int tpwgts[2], limit, avgvwgt, origdiff;
    int mincutorder, newcut, mincut, initcut, mindiff;
    int pass, nswaps, from, to, vertex, begin, end, i, j, k, connect_partition, kwgt;

    tpwgts[0] = tvwgt * ntpwgts0;
    tpwgts[1] = tvwgt - tpwgts[0];
    limit = hunyuangraph_gpu_int_min(hunyuangraph_gpu_int_max(0.01 * nvtxs, 15), 100);
    avgvwgt = hunyuangraph_gpu_int_min((pwgts[0] + pwgts[1]) / 20, 2 * (pwgts[0] + pwgts[1]) / nvtxs);
    origdiff = hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]);

    for(pass = 0; pass < niter; pass++)
    {
        priority_queue_Reset(&queues[0], nvtxs);
        priority_queue_Reset(&queues[1], nvtxs);

        mincutorder = -1;
        newcut = mincut = initcut = edgecut;
        mindiff = hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]);

        for(i = 0;i < nvtxs;i++)
        {
            if(bnd[i] != 0)
                priority_queue_Insert(&queues[where[i]], i, ed[i] - id[i]);
        }

        for(nswaps = 0; nswaps < nvtxs; nswaps++)
        {
            from = (tpwgts[0] - pwgts[0] < tpwgts[1] - pwgts[1] ? 0 : 1);
            to = from ^ 1;

            vertex = priority_queue_GetTop(&queues[from]);
            if(vertex == -1)
                break;
            
            newcut -= (ed[vertex] - id[vertex]);

            if((newcut < mincut && hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]) <= origdiff + avgvwgt)
                || (newcut == mincut && hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]) < mindiff))
            {
                mincut  = newcut;
                mindiff = hunyuangraph_gpu_int_abs(tpwgts[0] - pwgts[0]);
                mincutorder = nswaps;
            }
            else if(nswaps - mincutorder > limit)
            { 
                newcut+=(ed[vertex] - id[vertex]);
                // pwgts[from] += vwgt[vertex];
                // pwgts[to]   -= vwgt[vertex];
                break;
            }

            begin = xadj[vertex];
            end   = xadj[vertex + 1];
            where[vertex] = to;
            pwgts[to]   += vwgt[vertex];
            pwgts[from] -= vwgt[vertex];
            moved[vertex] = nswaps;
            swaps[nswaps] = vertex;
            hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
            if(ed[vertex] == 0 && end != begin)
                bnd[vertex] = 0;
            
            for(j = begin; j < end; j++)
            {
                k = adjncy[j];
                connect_partition = where[k];
                if(to == connect_partition)
                    kwgt = adjwgt[j];
                else 
                    kwgt = -adjwgt[j];

                id[k] += kwgt;
                ed[k] -= kwgt;

                if(bnd[k] == 1)
                {
                    if(ed[k] == 0)
                    {
                        bnd[k] = 0;
                        if(moved[k] == -1)
                            priority_queue_Delete(&queues[connect_partition], k);
                    }
                    else
                    {
                        if(moved[k] == -1)
                            priority_queue_Update(&queues[connect_partition], k, ed[k] - id[k]);
                    }
                }
                else
                {
                    if(ed[k] > 0)
                    {
                        bnd[k] = 1;
                        if(moved[k] == -1)
                            priority_queue_Insert(&queues[connect_partition], k, ed[k] - id[k]);
                    }
                }
            }

        }

        for(i = 0;i < nswaps;i++)
            moved[swaps[i]] = -1;

        //  roll back        
        for(nswaps--;nswaps > mincutorder;nswaps--)
        {
            vertex = swaps[nswaps];
            from = where[vertex];
            to = from ^ 1;
            hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
            begin = xadj[vertex];
            end   = xadj[vertex + 1];
            if(ed[vertex] == 0 && bnd[vertex] == 1 && begin != end)
                bnd[vertex] = 0;
            else if(ed[vertex] > 0 && bnd[vertex] == 0)
                bnd[vertex] = 1;
            pwgts[to] += vwgt[vertex];
            pwgts[from] -= vwgt[vertex];

            for(j = begin; j < end;j++)
            {
                k = adjncy[j];
                connect_partition = where[k];
                if(to == connect_partition)
                    kwgt = adjwgt[j];
                else 
                    kwgt = -adjwgt[j];
                id[k] += kwgt;
                ed[k] -= kwgt;

                if(bnd[k] == 1 && ed[k] == 0)
                    bnd[k] = 0;
                if(bnd[k] == 0 && ed[k] > 0)
                    bnd[k] = 1;
            }
        }

        edgecut = mincut;

        if(mincutorder <= 0 || mincut == initcut)
            break;
    }

    //  free
}

__global__ void hunyuangraph_gpu_Bisection(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    int *global_edgecut, hunyuangraph_int8_t *global_where, priority_queue_t *queues, int *key, int *val, int *locator, int oneminpwgt, int onemaxpwgt, curandState *state)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
	int ii = blockIdx.x;
    int tid = threadIdx.x;

    // printf("ii=%d tid=%d\n",ii, tid);

    // int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    extern __shared__ hunyuangraph_int8_t num[];
    // __shared__ hunyuangraph_int8_t twhere[nvtxs];
    // __shared__ hunyuangraph_int8_t moved[nvtxs];
    // __shared__ hunyuangraph_int8_t bnd[nvtxs];
    // __shared__ int queue[nvtxs];
    // __shared__ int ed[nvtxs];
    // __shared__ int swaps[nvtxs]; 
    // __shared__ int tpwgts[2];
    
    int *queue = (int *)num;
    int *ed = queue + nvtxs;
    int *swaps = ed + nvtxs;
    int *tpwgts = (int *)(swaps + nvtxs);
    hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
    hunyuangraph_int8_t *moved = twhere + nvtxs;
    hunyuangraph_int8_t *bnd = moved + nvtxs;
    /*if(tid == 0)
    {
        printf("nvtxs=%d\n", nvtxs);
        printf("num=%p\n", num);
        printf("queue=%p\n", queue);
        printf("ed=%p\n", ed); 
        printf("swaps=%p\n", swaps);
        printf("tpwgts=%p\n", tpwgts);
        printf("twhere=%p\n", twhere);
        printf("moved=%p\n", moved);
        printf("bnd=%p\n", bnd);
    }*/
    // int *queue = (int *)(bnd + nvtxs);
    // int *ed = queue + nvtxs;
    // int *swaps = ed + nvtxs;
    // int *tpwgts = (int *)(swaps + nvtxs);

    for(int i = tid;i < nvtxs; i += blockDim.x)
    {
        twhere[i] = 1;
        moved[i] = 0;
    }
    if(tid == 0)
        tpwgts[0] = 0;
    else if(tid == 1)
        tpwgts[1] = tvwgt;
    __syncthreads();

    if(tid == 0)
    {
        int vertex, first, last, nleft, drain;
        int v, k, begin, end;

        vertex = ii;
        queue[0] = vertex;
        moved[queue[0]] = 1;
        first = 0;
        last = 1;
        nleft = nvtxs - 1;
        drain = 0;

        for(;;)
        {
            if (first == last) 
            {
                if (nleft == 0 || drain)
                    break;

                k = get_random_number_range(nleft, &state[ii]);
                // printf("inbfs=%"PRIDX" k=%"PRIDX"\n",inbfs,k);

                for (v = 0; v < nvtxs; v++) 
                {
                    if (moved[v] == 0) 
                    {
                        if (k == 0)
                            break;
                        else
                            k--;
                    }
                }

                queue[0] = v;
                moved[v] = 1;
                first    = 0; 
                last     = 1;
                nleft--;
            }
            
            v = queue[first];
            first++;
            // printf("v=%d\n", v);
            if(tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
            {
                drain = 1;
                continue;
            }

            twhere[v] = 0;
            tpwgts[0] += vwgt[v];
            tpwgts[1] -= vwgt[v];
            if(tpwgts[1] <= onemaxpwgt)
                break;
            
            drain = 0;
            end = xadj[v + 1];
            for(begin = xadj[v]; begin < end; begin++)
            {
                k = adjncy[begin];
                if(moved[k] == 0)
                {
                    queue[last] = k;
                    moved[k] = 1;
                    last++;
                    nleft--;
                }
            }
        }
    }
    // printf("tid=%d\n", tid);
    __syncthreads();

    // for(int i = tid;i < nvtxs;i += blockDim.x)
    //     global_where[i] = twhere[i];
    hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
    // if(ii == 1 && tid == 0)
    //     printf("ptr=%p\n", ptr);
    for(int i = tid;i < nvtxs; i += blockDim.x)
        ptr[i] = twhere[i];
    __syncthreads();
    // if(tid == 0)
    //     printf("gpu p=%d v=%d\n", p, ii);

    // printf("tid=%d\n", tid);
    
    //  compute ed, id, bnd
    int *id;
    id = queue;
    for(int i = tid;i < nvtxs; i += blockDim.x)
    {
        compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
    }
    __syncthreads();

    //  reduce ed to acquire the edgecut
    int edgecut;
        //  add to the first blockDim.x threads
    int *reduce_num = swaps;
    if(tid < nvtxs)
        reduce_num[tid] = ed[tid];
    for(int i = tid + blockDim.x;i < nvtxs; i += blockDim.x)
        reduce_num[tid] += ed[i];
    __syncthreads();

        //  if nvtxs < blockDim.x
    if(tid < nvtxs)
    {
        if(blockDim.x >= 512) 
        {
            if(tid < 256) reduce_num[tid] += reduce_num[tid + 256];
            __syncthreads();
        }
        if(blockDim.x >= 256) 
        {
            if(tid < 128) reduce_num[tid] += reduce_num[tid + 128];
            __syncthreads();
        }
        if(blockDim.x >= 128) 
        {
            if(tid < 64) reduce_num[tid] += reduce_num[tid + 64];
            __syncthreads();
        }

        if(tid < 32) warpReduction(reduce_num, tid, blockDim.x);

        if(tid == 0) edgecut = reduce_num[0];
    }
    // if(tid == 0)
    //     printf("gpu p=%d v=%d edgecut=%d\n", p, ii, edgecut);
    /*
    //  Balance the partition
        // if it is not necessary
    
    //  FM_2WayCutRefine
        //  serial
            //  init array moved
    for(int i = tid;i < nvtxs; i += blockDim.x)
        moved[i] = -1;
    if(tid == 0)
    {
        // allocate queues
        queues[ii * 2].key = &key[ii * 2];
        queues[ii * 2].val = &val[ii * 2];
        queues[ii * 2].locator = &locator[ii * 2];
        queues[ii * 2 + 1].key = &key[ii * 2 + 1];
        queues[ii * 2 + 1].val = &val[ii * 2 + 1];
        queues[ii * 2 + 1].locator = &locator[ii * 2 + 1];

        FM_2way_cut_refinement(nvtxs, vwgt, xadj, adjncy, adjwgt, twhere, tpwgts, ed, id, bnd, moved, swaps, tvwgt, edgecut, tpwgts0, &queues[ii * 2], 10);

        global_edgecut[ii] = edgecut;
    }

    hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
    for(int i = tid;i < nvtxs; i += blockDim.x)
        ptr[i] = twhere[i];
    
    if(tid == 0)
    {
        for(int i = 0;i < nvtxs;i++)
            printf("i=%d\n", i);
    }*/
}

__global__ void hunyuangraph_gpu_Bisection_global(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, hunyuangraph_int8_t *global_where, int oneminpwgt, int onemaxpwgt, curandState *state)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
	int ii = blockIdx.x;
    int tid = threadIdx.x;

    // printf("ii=%d tid=%d\n",ii, tid);

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;
    // extern __shared__ hunyuangraph_int8_t num[];
    // __shared__ hunyuangraph_int8_t twhere[nvtxs];
    // __shared__ hunyuangraph_int8_t moved[nvtxs];
    // __shared__ hunyuangraph_int8_t bnd[nvtxs];
    // __shared__ int queue[nvtxs];
    // __shared__ int ed[nvtxs];
    // __shared__ int swaps[nvtxs]; 
    // __shared__ int tpwgts[2];
    num += ii * shared_size;
    int *queue = (int *)num;
    int *ed = queue + nvtxs;
    int *swaps = ed + nvtxs;
    int *tpwgts = (int *)(swaps + nvtxs);
    hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
    hunyuangraph_int8_t *moved = twhere + nvtxs;
    hunyuangraph_int8_t *bnd = moved + nvtxs;
    // if(tid == 0)
    //     printf("gpu i=%d where=%p\n", ii, twhere);
    /*if(tid == 0)
    {
        printf("nvtxs=%d\n", nvtxs);
        printf("num=%p\n", num);
        printf("queue=%p\n", queue);
        printf("ed=%p\n", ed); 
        printf("swaps=%p\n", swaps);
        printf("tpwgts=%p\n", tpwgts);
        printf("twhere=%p\n", twhere);
        printf("moved=%p\n", moved);
        printf("bnd=%p\n", bnd);
    }*/
    // int *queue = (int *)(bnd + nvtxs);
    // int *ed = queue + nvtxs;
    // int *swaps = ed + nvtxs;
    // int *tpwgts = (int *)(swaps + nvtxs);

    for(int i = tid;i < nvtxs; i += blockDim.x)
    {
        twhere[i] = 1;
        moved[i] = 0;
    }
    if(tid == 0)
        tpwgts[0] = 0;
    else if(tid == 1)
        tpwgts[1] = tvwgt;
    __syncthreads();

    if(tid == 0)
    {
        int vertex, first, last, nleft, drain;
        int v, k, begin, end;

        vertex = ii;
        queue[0] = vertex;
        moved[queue[0]] = 1;
        first = 0;
        last = 1;
        nleft = nvtxs - 1;
        drain = 0;

        for(;;)
        {
            if (first == last) 
            {
                if (nleft == 0 || drain)
                    break;

                k = get_random_number_range(nleft, &state[ii]);
                // printf("inbfs=%"PRIDX" k=%"PRIDX"\n",inbfs,k);

                for (v = 0; v < nvtxs; v++) 
                {
                    if (moved[v] == 0) 
                    {
                        if (k == 0)
                            break;
                        else
                            k--;
                    }
                }

                queue[0] = v;
                moved[v] = 1;
                first    = 0; 
                last     = 1;
                nleft--;
            }
            
            v = queue[first];
            first++;
            // printf("v=%d\n", v);
            if(tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
            {
                drain = 1;
                continue;
            }

            twhere[v] = 0;
            tpwgts[0] += vwgt[v];
            tpwgts[1] -= vwgt[v];
            if(tpwgts[1] <= onemaxpwgt)
                break;
            
            drain = 0;
            end = xadj[v + 1];
            for(begin = xadj[v]; begin < end; begin++)
            {
                k = adjncy[begin];
                if(moved[k] == 0)
                {
                    queue[last] = k;
                    moved[k] = 1;
                    last++;
                    nleft--;
                }
            }
        }
    }
    // printf("tid=%d\n", tid);
    __syncthreads();

    // for(int i = tid;i < nvtxs;i += blockDim.x)
    //     global_where[i] = twhere[i];
    /*hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
    // if(ii == 1 && tid == 0)
    //     printf("ptr=%p\n", ptr);
    for(int i = tid;i < nvtxs; i += blockDim.x)
        ptr[i] = twhere[i];
    __syncthreads();
    // if(tid == 0)
    //     printf("gpu p=%d v=%d\n", p, ii);

    // printf("tid=%d\n", tid);
    
    //  compute ed, id, bnd
    int *id;
    id = queue;
    for(int i = tid;i < nvtxs; i += blockDim.x)
    {
        compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
    }
    __syncthreads();

    //  reduce ed to acquire the edgecut
    int edgecut;
        //  add to the first blockDim.x threads
    // int *reduce_num = swaps;
    extern __shared__ int reduce_num[];
    if(tid < nvtxs)
        reduce_num[tid] = ed[tid];
    for(int i = tid + blockDim.x;i < nvtxs; i += blockDim.x)
        reduce_num[tid] += ed[i];
    __syncthreads();

        //  if nvtxs < blockDim.x
    if(tid < nvtxs)
    {
        if(blockDim.x >= 512) 
        {
            if(tid < 256) reduce_num[tid] += reduce_num[tid + 256];
            __syncthreads();
        }
        if(blockDim.x >= 256) 
        {
            if(tid < 128) reduce_num[tid] += reduce_num[tid + 128];
            __syncthreads();
        }
        if(blockDim.x >= 128) 
        {
            if(tid < 64) reduce_num[tid] += reduce_num[tid + 64];
            __syncthreads();
        }

        if(tid < 32) warpReduction(reduce_num, tid, blockDim.x);

        if(tid == 0) edgecut = reduce_num[0];
    }*/

    // if(tid == 0)
    //     printf("gpu p=%d v=%d edgecut=%d\n", p, ii, edgecut);
}

__global__ void hunyuangraph_gpu_Bisection_warp(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, hunyuangraph_int8_t *global_where, int oneminpwgt, int onemaxpwgt, curandState *state)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + (tid >> 5);
    // int exam = blockIdx.x;
    // printf("p=%d exam=%d tid=%d ii=%d\n", p, exam, tid, ii);

    // printf("ii=%d tid=%d\n",ii, tid);

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;
    // extern __shared__ hunyuangraph_int8_t num[];
    // __shared__ hunyuangraph_int8_t twhere[nvtxs];
    // __shared__ hunyuangraph_int8_t moved[nvtxs];
    // __shared__ hunyuangraph_int8_t bnd[nvtxs];
    // __shared__ int queue[nvtxs];
    // __shared__ int ed[nvtxs];
    // __shared__ int swaps[nvtxs]; 
    // __shared__ int tpwgts[2];
    if(ii < nvtxs)
    {
        num += ii * shared_size;
        int *queue = (int *)num;
        int *ed = queue + nvtxs;
        int *swaps = ed + nvtxs;
        int *tpwgts = (int *)(swaps + nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
        hunyuangraph_int8_t *moved = twhere + nvtxs;
        hunyuangraph_int8_t *bnd = moved + nvtxs;
        // if(lane_id == 0)
        //     printf("gpu i=%d where=%p\n", ii, twhere);
        /*if(blockIdx.x == 0 && lane_id == 0)
        {
            printf("gpu i=%d nvtxs=%d\n", ii, nvtxs);
            printf("gpu i=%d num=%p\n", ii, num);
            printf("gpu i=%d queue=%p\n", ii, queue);
            printf("gpu i=%d ed=%p\n", ii, ed); 
            printf("gpu i=%d swaps=%p\n", ii, swaps);
            printf("gpu i=%d tpwgts=%p\n", ii, tpwgts);
            printf("gpu i=%d twhere=%p\n", ii, twhere);
            printf("gpu i=%d moved=%p\n", ii, moved);
            printf("gpu i=%d bnd=%p\n", ii, bnd);
        }*/
        // int *queue = (int *)(bnd + nvtxs);
        // int *ed = queue + nvtxs;
        // int *swaps = ed + nvtxs;
        // int *tpwgts = (int *)(swaps + nvtxs);

        for(int i = lane_id;i < nvtxs; i += 32)
        {
            twhere[i] = 1;
            moved[i] = 0;
        }
        if(lane_id == 0)
            tpwgts[0] = 0;
        else if(lane_id == 1)
            tpwgts[1] = tvwgt;
        __syncthreads();

        if(lane_id == 0)
        {
            int vertex, first, last, nleft, drain;
            int v, k, begin, end;

            vertex = ii;
            queue[0] = vertex;
            moved[queue[0]] = 1;
            first = 0;
            last = 1;
            nleft = nvtxs - 1;
            drain = 0;

            for(;;)
            {
                if (first == last) 
                {
                    if (nleft == 0 || drain)
                        break;

                    k = get_random_number_range(nleft, &state[ii]);
                    // printf("inbfs=%"PRIDX" k=%"PRIDX"\n",inbfs,k);

                    for (v = 0; v < nvtxs; v++) 
                    {
                        if (moved[v] == 0) 
                        {
                            if (k == 0)
                                break;
                            else
                                k--;
                        }
                    }

                    queue[0] = v;
                    moved[v] = 1;
                    first    = 0; 
                    last     = 1;
                    nleft--;
                }
                
                v = queue[first];
                first++;
                // printf("v=%d\n", v);
                if(tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
                {
                    drain = 1;
                    continue;
                }

                twhere[v] = 0;
                tpwgts[0] += vwgt[v];
                tpwgts[1] -= vwgt[v];
                if(tpwgts[1] <= onemaxpwgt)
                    break;
                
                drain = 0;
                end = xadj[v + 1];
                for(begin = xadj[v]; begin < end; begin++)
                {
                    k = adjncy[begin];
                    if(moved[k] == 0)
                    {
                        queue[last] = k;
                        moved[k] = 1;
                        last++;
                        nleft--;
                    }
                }
            }
        }
        // printf("tid=%d\n", tid);
        /*__syncthreads();

        // for(int i = tid;i < nvtxs;i += blockDim.x)
        //     global_where[i] = twhere[i];
        // hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
        // if(ii == 1 && tid == 0)
        //     printf("ptr=%p\n", ptr);
        // for(int i = lane_id;i < nvtxs; i += 32)
        //     ptr[i] = twhere[i];
        // __syncthreads();
        // if(tid == 0)
        //     printf("gpu p=%d v=%d\n", p, ii);

        // printf("tid=%d\n", tid);
        // if(lane_id == 0)
        // {
        //     for(int i = 0;i < nvtxs;i++)
        //         printf("i=%d where=%d\n",i, twhere[i]);
        //         printf("pu i=%d where=%p\n", ii, twhere);
        // }
        
        //  compute ed, id, bnd
        int *id;
        id = queue;
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
        }
        __syncthreads();

        //  reduce ed to acquire the edgecut
        int edgecut;
            //  add to the first blockDim.x threads
        // int *reduce_num = swaps;
        extern __shared__ int reduce_num[];
        if(lane_id < nvtxs)
            reduce_num[tid] = ed[lane_id];
        else 
            reduce_num[tid] = 0;
        for(int i = lane_id + 32;i < nvtxs; i += 32)
            reduce_num[tid] += ed[i];
        __syncthreads();

            //  if nvtxs < 32
        // if(lane_id < 32 && lane_id < nvtxs) 
        warpReduction(reduce_num, tid, 32);
        if(lane_id == 0) edgecut = reduce_num[(tid >> 5) * 32];

        // if(lane_id == 0)
        //     printf("gpu p=%d v=%d edgecut=%d\n", lane_id, ii, edgecut);*/
    }
}

__device__ bool can_moved(int *tpwgts, int vertex, int onemaxpwgt, int oneminpwgt, int *vwgt, hunyuangraph_int8_t *moved, int *drain)
{
    if(moved[vertex] != 1)
    {
        // printf("moved[%d]=%d\n",vertex, moved[vertex]);
        // moved[vertex] = 1;
        return false;
    }
    if(tpwgts[0] + vwgt[vertex] > onemaxpwgt || tpwgts[1] - vwgt[vertex] < oneminpwgt)
    {
        // printf("tpwgts[0] > 0 && tpwgts[1] - vwgt[vertex] < oneminpwgt\n");
        drain[0] = 1;
        return false;
        // continue;
        // return ;
    }
    return true;
}

__device__ bool can_moved_end(int *tpwgts, int vertex, int onemaxpwgt, int oneminpwgt, int *vwgt, hunyuangraph_int8_t *moved)
{
    // printf("%p\n", &moved[vertex]);
    if(moved[vertex] != 1)
    {
        return false;
    }
    if(tpwgts[0] + vwgt[vertex] > onemaxpwgt || tpwgts[1] - vwgt[vertex] < oneminpwgt)
    {
        return false;
    }
    return true;
}

__device__ bool is_balanced(int *tpwgts, int onemaxpwgt, int oneminpwgt)
{
    if(tpwgts[1] <= onemaxpwgt && tpwgts[1] >= oneminpwgt
        && tpwgts[0] <= onemaxpwgt && tpwgts[0] >= oneminpwgt)
        return true;
    else 
        return false;
}

__global__ void hunyuangraph_gpu_BFS_warp(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, hunyuangraph_int8_t *global_where, int oneminpwgt, int onemaxpwgt, curandState *state)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + (tid >> 5);
    // int exam = blockIdx.x;
    // printf("p=%d exam=%d tid=%d ii=%d\n", p, exam, tid, ii);

    // printf("ii=%d tid=%d\n",ii, tid);

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;
    // extern __shared__ hunyuangraph_int8_t num[];
    // __shared__ hunyuangraph_int8_t twhere[nvtxs];
    // __shared__ hunyuangraph_int8_t moved[nvtxs];
    // __shared__ hunyuangraph_int8_t bnd[nvtxs];
    // __shared__ int queue[nvtxs];
    // __shared__ int ed[nvtxs];
    // __shared__ int swaps[nvtxs]; 
    // __shared__ int tpwgts[2];
    if(ii < nvtxs)
    {
        num += ii * shared_size;
        int *queue = (int *)num;
        int *ed = queue + nvtxs;
        int *swaps = ed + nvtxs;
        int *tpwgts = (int *)(swaps + nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
        hunyuangraph_int8_t *moved = twhere + nvtxs;
        hunyuangraph_int8_t *bnd = moved + nvtxs;
        // if(lane_id == 0)
        //     printf("gpu i=%d where=%p\n", ii, twhere);
        /*if(blockIdx.x == 0 && lane_id == 0)
        {
            printf("gpu i=%d nvtxs=%d\n", ii, nvtxs);
            printf("gpu i=%d num=%p\n", ii, num);
            printf("gpu i=%d queue=%p\n", ii, queue);
            printf("gpu i=%d ed=%p\n", ii, ed); 
            printf("gpu i=%d swaps=%p\n", ii, swaps);
            printf("gpu i=%d tpwgts=%p\n", ii, tpwgts);
            printf("gpu i=%d twhere=%p\n", ii, twhere);
            printf("gpu i=%d moved=%p\n", ii, moved);
            printf("gpu i=%d bnd=%p\n", ii, bnd);
        }*/
        // int *queue = (int *)(bnd + nvtxs);
        // int *ed = queue + nvtxs;
        // int *swaps = ed + nvtxs;
        // int *tpwgts = (int *)(swaps + nvtxs);

        for(int i = lane_id;i < nvtxs; i += 32)
        {
            twhere[i] = 1;
            moved[i] = 0;
        }
        if(lane_id == 0)
            tpwgts[0] = 0;
        else if(lane_id == 1)
            tpwgts[1] = tvwgt;
        __syncthreads();

        int vertex, first, last, nleft, drain;
        int v, k, begin, end, length, flag;
        
        if(lane_id == 0)
        {
            vertex = ii;
            queue[0] = vertex;
            first = 0;
            last = 1;
            drain = 0;
        }
        while(1)
        {
            __syncwarp();
            first  = __shfl_sync(0xffffffff, first, 0, 32);
            last   = __shfl_sync(0xffffffff, last, 0, 32);
            if(first == last)
                break;
            if(lane_id == 0)
            {
                vertex = queue[first];
                first++;
                flag = 0;

                if(!can_moved(tpwgts, vertex, onemaxpwgt, oneminpwgt, vwgt, moved, &drain))
                    flag = 1;
            }
            // if(ii == 0 && lane_id == 0)
            //     printf("flag=%d\n", flag);
            __syncwarp();
            flag   = __shfl_sync(0xffffffff, flag, 0, 32);
            // if(ii == 0 && lane_id == 0)
            //     printf("flag=%d\n", flag);
            if(flag)
                continue;
            if(lane_id == 0)
            {
                twhere[vertex] = 0;
                tpwgts[0] += vwgt[vertex];
                tpwgts[1] -= vwgt[vertex];
            }
            // __syncwarp();
            // printf("p=%d 1207\n", p);
            __syncwarp();
            if(is_balanced(tpwgts, onemaxpwgt, oneminpwgt))
                break;
            
            vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
            begin = xadj[vertex];
            end   = xadj[vertex + 1];
            length = end - begin;
            first  = __shfl_sync(0xffffffff, first, 0, 32);
            last   = __shfl_sync(0xffffffff, last, 0, 32);
            last += length;
            // printf("p=%d 1218\n", p);
            //  push_queue
            for(int i = lane_id;i < length; i += 32)
            {
                k = adjncy[begin + i];
                queue[first + i] = k;
                moved[k]++;
            }

            // printf("p=%d 1227\n", p);
        }

        // printf("tid=%d\n", tid);
        __syncwarp();

        // for(int i = tid;i < nvtxs;i += blockDim.x)
        //     global_where[i] = twhere[i];
        // hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
        // if(ii == 1 && tid == 0)
        //     printf("ptr=%p\n", ptr);
        // for(int i = lane_id;i < nvtxs; i += 32)
        //     ptr[i] = twhere[i];
        // __syncthreads();
        // if(tid == 0)
        //     printf("gpu p=%d v=%d\n", p, ii);

        // printf("tid=%d\n", tid);
        // if(lane_id == 0)
        // {
        //     printf("ii=%d tpwgts[0]=%d tpwgts[1]=%d\n", ii, tpwgts[0], tpwgts[1]);
        //     for(int i = 0;i < nvtxs;i++)
        //         printf("i=%d where=%d\n",i, twhere[i]);
        // //         printf("pu i=%d where=%p\n", ii, twhere);
        // }
        
        //  compute ed, id, bnd
        int *id;
        id = queue;
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
        }
        __syncwarp();

        //  reduce ed to acquire the edgecut
        int edgecut;
            //  add to the first blockDim.x threads
        // int *reduce_num = swaps;
        extern __shared__ int reduce_num[];
        if(lane_id < nvtxs)
            reduce_num[tid] = ed[lane_id];
        else 
            reduce_num[tid] = 0;
        for(int i = lane_id + 32;i < nvtxs; i += 32)
            reduce_num[tid] += ed[i];
        __syncwarp();

            //  if nvtxs < 32
        // if(lane_id < 32 && lane_id < nvtxs) 
        warpReduction(reduce_num, tid, 32);
        if(lane_id == 0) edgecut = reduce_num[(tid >> 5) * 32];

        // if(lane_id == 0)
        //     printf("gpu p=%d v=%d edgecut=%d\n", lane_id, ii, edgecut);*/
    }
}

__device__ bool fm_can_balanced(int *tpwgts, int vertex, int to, int from, int onemaxpwgt, int oneminpwgt, int *vwgt)
{
    if(tpwgts[to] + vwgt[vertex] <= onemaxpwgt && tpwgts[to] + vwgt[vertex] >= oneminpwgt
        && tpwgts[from] - vwgt[vertex] <= onemaxpwgt && tpwgts[from] - vwgt[vertex] >= oneminpwgt)
    {
        // printf("tpwgts[0] > 0 && tpwgts[1] - vwgt[vertex] < oneminpwgt\n");
        return true;
        // continue;
        // return ;
    }
    return false;
}

__device__ bool can_continue_partition_BFS(int to, int from, int nparts, int nvtxs0, int nvtxs1)
{
    int less0 = nparts >> 1;
    int less1 = nparts - less0;

    if(less1 > (nvtxs1 - 1))
        return false;
    else
        return true;
}

__device__ bool can_continue_partition(int to, int from, int nparts, int nvtxs0, int nvtxs1)
{
    int less0 = nparts >> 1;
    int less1 = nparts - less0;

    if(to == 0)
    {
        if(less0 > (nvtxs0 + 1) || less1 > (nvtxs1 - 1))
            return false;
        else
            return true;
    }
    else
    {
        if(less1 > (nvtxs1 + 1) || less0 > (nvtxs0 - 1))
            return false;
        else
            return true;
    }
}

__global__ void hunyuangraph_gpu_BFS_warp_2wayrefine(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, int oneminpwgt, int onemaxpwgt, curandState *state)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + (tid >> 5);
    int exam = blockIdx.x;

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;
    // if(lane_id == 0)
    //     printf("p=%d exam=%d tid=%d ii=%d %d shared_size=%d\n", p, exam, tid, ii, exam * 4 + tid / 32, shared_size);
    // __syncwarp();
    // extern __shared__ hunyuangraph_int8_t num[];
    // __shared__ hunyuangraph_int8_t twhere[nvtxs];
    // __shared__ hunyuangraph_int8_t moved[nvtxs];
    // __shared__ hunyuangraph_int8_t bnd[nvtxs];
    // __shared__ int queue[nvtxs];
    // __shared__ int ed[nvtxs];
    // __shared__ int swaps[nvtxs]; 
    // __shared__ int tpwgts[2];
    if(ii < nvtxs)
    {
        num += ii * shared_size;
        int *queue = (int *)num;
        int *ed = queue + nvtxs;
        int *swaps = ed + nvtxs;
        int *tpwgts = (int *)(swaps + nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
        hunyuangraph_int8_t *moved = twhere + nvtxs;
        hunyuangraph_int8_t *bnd = moved + nvtxs;

        // if(lane_id == 0)
        //     printf("gpu i=%d num=%p\n", ii, num);
        //     printf("gpu i=%d where=%p\n", ii, twhere);
        // if(ii == 18 && lane_id == 0)
        // {
        //     // printf("gpu i=%d nvtxs=%d\n", ii, nvtxs);
        //     printf("p=%d exam=%d tid=%d ii=%d %d shared_size=%d\n", p, exam, tid, ii, exam * 4 + tid / 32, shared_size);
        //     printf("gpu i=%d num=%p\n", ii, num);
        //     printf("gpu i=%d queue=%p\n", ii, queue);
        //     printf("gpu i=%d ed=%p\n", ii, ed); 
        //     printf("gpu i=%d swaps=%p\n", ii, swaps);
        //     printf("gpu i=%d tpwgts=%p\n", ii, tpwgts);
        //     printf("gpu i=%d twhere=%p\n", ii, twhere);
        //     printf("gpu i=%d moved=%p\n", ii, moved);
        //     printf("gpu i=%d bnd=%p\n", ii, bnd);
        // }
        // int *queue = (int *)(bnd + nvtxs);
        // int *ed = queue + nvtxs;
        // int *swaps = ed + nvtxs;
        // int *tpwgts = (int *)(swaps + nvtxs);
        __syncwarp();
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            twhere[i] = 1;
            moved[i] = 0;
        }
        if(lane_id == 0)
            tpwgts[0] = 0;
        else if(lane_id == 1)
            tpwgts[1] = tvwgt;
        __syncthreads();
        // if(lane_id == 0)
        // {
        //     for(int i = 0;i < nvtxs;i++)
        //         if(twhere[i] != 1)
        //             printf("init v=%d i=%d where is wrong where=%d\n", ii, i, twhere[i]);
        // }

        int vertex, first, nleft;
        // int drain;
        int v, k, begin, end, length, flag;
        // extern __shared__ int reduce_num[];
        __shared__ int last_id[4];
        int *last = last_id + (tid >> 5);
        // int *last = reduce_num + (tid >> 5);
        // if(lane_id == 0)
        //     printf("ii=%d last=%p\n", ii, last);
        
        if(lane_id == 0)
        {
            vertex = ii;
            if(vertex >= nvtxs)
                printf("error ii=%d vertex=%d nvtxs=%d begin \n", ii, vertex, nvtxs);
            queue[0] = vertex;
            moved[vertex] = 1;
            first = 0;
            last[0] = 1;
            nleft = nvtxs - 1;
            // drain = 0;
        }
        // if(ii == 0 && lane_id == 0)
        // {
        //     for(int i = 0;i < nvtxs;i++)
        //         if(twhere[i] != 1)
        //             printf("begin v=%d i=%d where is wrong where=%d\n", ii, i, twhere[i]);
        // }
        int mark = 0;
        __syncwarp();
        int step = 0;
        while(1)
        {
            //  moved: 0: not moved, 1: pushed, 2: moved
            // if(lane_id == 0)
            //     printf("begin ii=%d step=%d first=%d last=%d tpwgts0=%d tpwgts1=%d\n", ii, step, first, last[0], tpwgts[0], tpwgts[1]);
            step++;
            if(step >= 2 * nvtxs)
                break;
            __syncwarp();
            first  = __shfl_sync(0xffffffff, first, 0, 32);
            // last   = __shfl_sync(0xffffffff, last, 0, 32);

            __syncwarp();
            flag = 0;
            if(lane_id == 0)
            {
                if(first == last[0])
                {
                    k = get_random_number_range(nleft, &state[ii]);

                    for(v = 0;v < nvtxs;v++)
                    {
                        if(moved[v] == 0)
                        {
                            if(k == 0)
                                break;
                            else 
                                k--;
                        }
                    }
                    vertex = v;
                    if(vertex >= nvtxs)
                        printf("error ii=%d vertex=%d nvtxs=%d == \n", ii, vertex, nvtxs);

                    if(flag == 0)
                    {
                        queue[0] = vertex;
                        moved[vertex] = 1;
                        first = 0;
                        last[0] = 1;
                        nleft--;

                        // flag = 1;
                        mark = 1;
                        // break;
                    }
                //     // else
                //     // {
                //     //     int sum = 0, moved1 = 0, moved2 = 0;
                //     //     for(int i = 0;i < nvtxs;i++)
                //     //     {
                //     //         if(moved[i] == 1)
                //     //             moved1++;
                //     //         else if(moved[i] == 2)
                //     //             moved2++;
                //     //         sum += moved[i];
                //     //     }
                //     //     // printf("ii=%d nvtxs=%d first=%d last=%d moved1=%d moved2=%d sum=%d\n", ii, nvtxs, first, last[0], moved1, moved2, sum);
                //     // }
                }
            }
            __syncwarp();
            flag = __shfl_sync(0xffffffff, flag, 0, 32);
            if(flag)
                break;

            __syncwarp();
            first  = __shfl_sync(0xffffffff, first, 0, 32);

            flag = 0;
            if(lane_id == 0)
            {
                // if(ii == 0)
                //     printf("first=%d last=%d begin\n", first, last[0]);
                // if(first < nvtxs)
                vertex = queue[first];
                if(vertex >= nvtxs)
                    printf("error ii=%d vertex=%d nvtxs=%d judge\n", ii, vertex, nvtxs);
                // printf("ii=%d vertex=%d first=%d last=%d begin\n", ii, vertex, first, last[0]);
                first++;

                // if(ii == 0 && moved[vertex] >= 1)
                //     printf("moved[%d]=%d\n", vertex, moved[vertex]);
                if(vertex == -1 || !can_moved_end(tpwgts, vertex, onemaxpwgt, oneminpwgt, vwgt, moved))
                    flag = 1;
                // printf("ii=%d vertex=%d tpwgts0=%d tpwgts1=%d vwgt=%d flag=%d\n", ii, vertex, tpwgts[0], tpwgts[1], vwgt[vertex], flag);
            }
            // if(lane_id == 0)
            //     printf("flag=%d\n", flag);
            __syncwarp();
            flag   = __shfl_sync(0xffffffff, flag, 0, 32);
            // if(ii == 0 && lane_id == 0)
            //     printf("flag=%d\n", flag);
            if(flag)
                continue;
            flag = 0;
            if(lane_id == 0)
            {
                if(vertex >= nvtxs)
                    printf("error ii=%d vertex=%d nvtxs=%d moved\n", ii, vertex, nvtxs);
                twhere[vertex] = 0;
                moved[vertex] = 2;
                tpwgts[0] += vwgt[vertex];
                tpwgts[1] -= vwgt[vertex];
                if(is_balanced(tpwgts, onemaxpwgt, oneminpwgt))
                    flag = 1;
            }
            __syncwarp();
            flag   = __shfl_sync(0xffffffff, flag, 0, 32);
            if(flag)
            {
                mark = 2;
                break;
            }
            // printf("p=%d 1207\n", p);
            
            __syncwarp();
            vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
            __syncwarp();
            if(vertex >= nvtxs)
                printf("error ii=%d vertex=%d nvtxs=%d broadcast\n", ii, vertex, nvtxs);
            begin = xadj[vertex];
            end   = xadj[vertex + 1];
            length = end - begin;
            first  = __shfl_sync(0xffffffff, first, 0, 32);
            if(lane_id == 0)
                printf("ii=%d vertex=%d \n", ii, vertex);
            // last   = __shfl_sync(0xffffffff, last, 0, 32);
            // if(lane_id == 0)
            //     printf("BFS first=%d v=%d vertex=%d moved=%d where=%x\n", first, ii, vertex, moved[vertex], twhere[vertex]);
            // if(ii == 0 && lane_id == 0)
            // {
            //     printf("BFS first=%d vertex=%d moved=%d\n", first, vertex, moved[vertex]);
            // }
            // printf("p=%d 1218\n", p);
            //  push_queue
            __syncwarp();
            for(int i = lane_id;i < length; i += 32)
            {
                k = adjncy[begin + i];
                printf("ii=%d k=%d nvtxs=%d vertex=%d i=%d length=%d begin + i=%d last=%d movedk=%d\n", ii, k, nvtxs, vertex, i, length, begin + i, last[0], moved[k]);
                if(k >= nvtxs)
                    printf("push_error ii=%d vertex=%d k=%d nvtxs=%d i=%d length=%d begin + i=%d first=%d last=%d\n", ii, vertex, k, nvtxs, i, length, begin + i, first, last[0]);
                if(moved[k] == 0)
                {
                    int ptr = atomicAdd(&last[0], 1);
                    // printf("ii=%d ptr=%d\n", ii, ptr);
                    queue[ptr] = k;
                    moved[k] = 1;
                }
            }
            // last[0] += length;
            // if(lane_id == 0)
            // {
            //     printf("first=%d last=%d begin\n", first, last[0]);
            //     // for(int t = first;t < last;t++)
            //     //     printf("queue[%d]=%d\n", t, queue[t]);
            // }
            __syncwarp();
            if(lane_id == 0)
                printf("ii=%d vertex=%d first=%d last=%d end\n", ii, vertex, first, last[0]);
            // first  = __shfl_sync(0xffffffff, first, 0, 32);
            // if(lane_id == 0)
            // {
            //     int sum = 0, moved1 = 0, moved2 = 0;
            //     for(int i = 0;i < nvtxs;i++)
            //     {
            //         if(moved[i] == 1)
            //             moved1++;
            //         else if(moved[i] == 2)
            //             moved2++;
            //     }
            //     // printf("ii=%d nvtxs=%d first=%d last=%d moved1=%d moved2=%d end\n", ii, nvtxs, first, last[0], moved1, moved2);
            // }

            // printf("p=%d 1227\n", p);
        }

        // printf("tid=%d\n", tid);
        // __syncwarp();

        // if(lane_id == 0)
        // {
        //     for(int i = 0;i < nvtxs;i++)
        //         if(twhere[i] != 0 && twhere[i] != 1)
        //             printf("v=%d i=%d where=%p where is wrong where=%x\n", ii, i, twhere, twhere[i]);
        // }

        // if(lane_id == 0)
        // {
        //     printf("ii=%d mark=%d\n", ii, mark);
        // }

        // for(int i = tid;i < nvtxs;i += blockDim.x)
        //     global_where[i] = twhere[i];
        // hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
        // if(ii == 1 && tid == 0)
        //     printf("ptr=%p\n", ptr);
        // for(int i = lane_id;i < nvtxs; i += 32)
        //     ptr[i] = twhere[i];
        // __syncthreads();
        // if(tid == 0)
        //     printf("gpu p=%d v=%d\n", p, ii);

        // printf("tid=%d\n", tid);
        // __syncwarp();
        // if(lane_id == 0)
        // {
        //     printf("ii=%d tpwgts[0]=%d tpwgts[1]=%d\n", ii, tpwgts[0], tpwgts[1]);
        //     // for(int i = 0;i < nvtxs;i++)
        //     //     printf("i=%d where=%d\n",i, twhere[i]);
        // //         printf("pu i=%d where=%p\n", ii, twhere);
        // }
        
        //  compute ed, id, bnd
        __syncwarp();
        int *id;
        id = queue;
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
            // int begin, end, ted, tid;
            // hunyuangraph_int8_t me, other;
            // begin = xadj[i];
            // end = xadj[i + 1];
            // me = twhere[i];
            // if(me != 0 && me != 1)
            //     printf("Error: v = %d, where[v] = %d\n", i, me);
            // ted = 0;
            // tid = 0;
            // if(ii == 0 && i == 3971)
            //     printf("begin=%d end=%d\n", begin, end);
            // if(begin == end)
            // {
            //     bnd[i] = 1;
            //     ed[i] = 0;
            //     id[i] = 0;
            //     return ;
            // }

            // for(int ptr = begin;ptr < end; ptr++)
            // {
            //     int j = adjncy[ptr];
            //     int wgt = adjwgt[ptr];
            //     other = twhere[j];

            //     if(me != other) 
            //         ted += wgt;
            //     else 
            //         tid += wgt;
            // }

            // ed[i] = ted;
            // id[i] = tid;

            // if(ted > 0)
            //     bnd[i] = 1;
            // else
            //     bnd[i] = 0;
        }
        __syncwarp();

        //  reduce ed to acquire the edgecut
        int edgecut = 0;
            //  add to the first blockDim.x threads
        // int *reduce_num = swaps;
        extern __shared__ int reduce_num[];
        int *warp_reduce = reduce_num + (tid >> 5) * 32;
        if(lane_id < nvtxs)
            warp_reduce[lane_id] = ed[lane_id];
        else 
            warp_reduce[lane_id] = 0;
        for(int i = lane_id + 32;i < nvtxs; i += 32)
            warp_reduce[lane_id] += ed[i];
        __syncwarp();

        // if(lane_id == 0 && ii == 0) 
        // {
        //     for(int i = 0;i < 32;i++)
        //         printf("%d warp_reduce=%d\n", i, warp_reduce[i]);
        //     for(int i = 1;i < 32;i++)
        //         warp_reduce[0] += warp_reduce[i];
        //     for(int i = 0;i < 32;i++)
        //         printf("%d warp_reduce=%d\n", i, warp_reduce[i]);
        // }

            //  if nvtxs < 32
        // if(lane_id < 32 && lane_id < nvtxs) 
        warpReduction(warp_reduce, lane_id, 32);
        if(lane_id == 0) 
        {
            // for(int i = 0;i < nvtxs;i++)
            // {
            //     edgecut += ed[i];
            //     for(int j = xadj[i];j < xadj[i + 1];j++)
            //     {
            //         int jj = adjncy[j];
            //         if(twhere[jj] != twhere[i])
            //             edgecut += adjwgt[j];
            //     }
            // }
            // edgecut /= 2;
            edgecut = warp_reduce[0] / 2;
            // global_edgecut[ii] = edgecut;
        }

        edgecut = __shfl_sync(0xffffffff, edgecut, 0, 32);
        // if(lane_id == 0)
        //     printf("gpu p=%d v=%d edgecut=%d mark=%d first\n", lane_id, ii, edgecut, mark);

        //  2wayrefine
        // if(lane_id == 0)
        //     printf("v=%d warp_reduce=%p reduce_num=%p\n", ii, warp_reduce, reduce_num);
        for(int i = lane_id;i < nvtxs; i += 32)
            moved[i] = 0;
        int ideal_pwgts[2];
        int nswaps, from, to, val, big_num;
        int limit, avgvwgt, origdiff;
        int newcut, mincut, initcut, mincutorder, mindiff;
        ideal_pwgts[0] = tvwgt * tpwgts0;
        ideal_pwgts[1] = tvwgt - ideal_pwgts[0];

        limit = hunyuangraph_gpu_int_min(hunyuangraph_gpu_int_max(0.01 * nvtxs, 15), 100);
        avgvwgt = hunyuangraph_gpu_int_min((tpwgts[0] + tpwgts[1]) / 20, 2 * (tpwgts[0]+tpwgts[1]) / nvtxs);
        origdiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
        big_num = -xadj[nvtxs];
        
        __syncwarp();
        for(int pass = 0;pass < 1;pass++)
        {
            mincutorder=-1;
            newcut = mincut = initcut = edgecut;
            mindiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
            flag = 0;
            
            // if(lane_id == 0)
            //     printf("pass=%d v=%d 1658\n", pass, ii);
            
            for(nswaps = 0;nswaps < nvtxs;nswaps++)
            {
                // if(nswaps > 16)
                //     break;
                //  partition
                from = (ideal_pwgts[0] - tpwgts[0] < ideal_pwgts[1] - tpwgts[1] ? 0 : 1);
                to = from ^ 1;

                //  select vertex
                warp_reduce[lane_id] = -1;
                val = big_num;
                __syncwarp();
                // if(lane_id == 0)
                //     printf("pass=%d nswaps=%d v=%d 1673\n", pass, nswaps, ii);
                // printf("init p=%d v=%d lane_id=%d warp_reduce=%d val=%d\n", p, ii, lane_id, warp_reduce[lane_id], val);
                for(int i = lane_id;i < nvtxs; i += 32)
                {
                    if(twhere[i] == from && bnd[i] == 1 && moved[i] == 0)
                    {
                        int t = warp_reduce[lane_id];
                        if(t == -1 || val < ed[i] - id[i])
                        {
                            warp_reduce[lane_id] = i;
                            val = ed[i] - id[i];
                        }
                    }
                    // printf("vertex=%d twhere=%d bnd=%d moved=%d p=%d v=%d lane_id=%d warp_reduce=%d val=%d\n", i, twhere[i], bnd[i], moved[i], p, ii, lane_id, warp_reduce[lane_id], val);
                }
                // printf("p=%d v=%d lane_id=%d warp_reduce=%d\n", p, ii, lane_id, warp_reduce[lane_id]);
                __syncwarp();

                // if(lane_id == 0)
                //     printf("pass=%d nswaps=%d v=%d 1690\n", pass, nswaps, ii);

                WarpGetMax(warp_reduce, lane_id, val);
                if(lane_id == 0)
                    vertex = warp_reduce[0];
                // if(lane_id == 0)
                // {
                //     printf("v=%d vertex=%d where=%d length=%d\n", ii, vertex, twhere[vertex], xadj[vertex + 1] - xadj[vertex]);
                //     if(ii == 3)
                //     {
                //         for(int i = xadj[vertex];i < xadj[vertex + 1];i++)
                //             printf("v=%d i=%d adjncy=%d adjwgt=%d where=%d\n", vertex, i - xadj[vertex], adjncy[i], adjwgt[i], twhere[adjncy[i]]);
                //     }
                // }

                //  boardcast vertex
                __syncwarp();
                vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
                flag = 0;
                if(lane_id == 0)
                {
                    if(vertex == -1)
                    {
                        // if(lane_id == 0)
                        // {
                        //     printf("v=%d vertex=-1 from=%d to=%d\n", ii, from, to);
                        //     for(int x = 0;x < nvtxs;x++)
                        //     {
                        //         printf("x=%d where_x=%d ed=%d id=%d bnd=%d moved=%d\n", x, twhere[x], ed[x], id[x], bnd[x], moved[x]);
                        //     }
                        // }
                        // printf("p=%d v=%d lane_id=%d warp_reduce=%d\n", p, ii, lane_id, warp_reduce[lane_id]);
                        flag = 1;
                    }
                }
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                if(flag)
                    break;

                // if(lane_id == 0)
                //     printf("pass=%d nswaps=%d v=%d 1726\n", pass, nswaps, ii);
                
                //  judge the vertex
                flag = 0;
                if(lane_id == 0)
                {
                    // int answer = 0;
                    // printf("v=%d ed0-id0=%d\n", ii, ed[answer] - id[answer]);
                    // for(int t = 1;t < nvtxs;t++)
                    // {
                    //     if(twhere[t] == from && bnd[t] == 1 && moved[t] == 0)
                    //     {
                    //         int tt = ed[t] - id[t];
                    //         printf("v=%d t=%d tt=%d\n", v, t, tt);
                    //         if(tt > ed[answer] - id[answer])
                    //         {
                    //             answer = t;
                    //         }
                    //     }
                    // }
                    // printf("v=%d answer=%d end\n", ii, answer);

                    newcut -= (ed[vertex] - id[vertex]);
                    if(fm_can_balanced(tpwgts, vertex, to, from, onemaxpwgt, oneminpwgt, vwgt) 
                        && ((newcut < mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) <= origdiff + avgvwgt)
                        || (newcut == mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) < mindiff)))
                    {
                        mincut  = newcut;
                        // if(ii == 500)
                        //     printf("nswaps=%d v=%d mincut=%d vertex=%d ed=%d id=%d\n", nswaps, ii, mincut, vertex, ed[vertex], id[vertex]);
                        mindiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
                        mincutorder = nswaps;
                    }
                    else if(nswaps - mincutorder > limit)
                    {
                        // printf("v=%d nswaps=%d mincutorder=%d limit=%d\n", ii, nswaps, mincutorder, limit);
                        flag = 1;
                        newcut += (ed[vertex] - id[vertex]);
                    }
                    // if(ii == 0)
                    //     printf("nswaps=%d mincutorder=%d newcut=%d\n", nswaps, mincutorder, newcut);
                    // if(ii == 500)
                    //     printf("nswaps=%d v=%d newcut=%d vertex=%d ed=%d id=%d\n", nswaps, ii, newcut, vertex, ed[vertex], id[vertex]);
                        
                }
                // if(lane_id == 0)
                //     printf("v=%d mincut=%d\n", v, mincut);
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                // if(lane_id == 0)
                //     printf("gpu p=%d v=%d flag=%d\n", p, ii, flag);
                if(flag)
                    break;
                
                // if(lane_id == 0)
                //     printf("pass=%d nswaps=%d v=%d 1778\n", pass, nswaps, ii);
                
                //  move the vertex
                if(lane_id == 0)
                {
                    // printf("gpu p=%d v=%d vertex=%d vwgt=%d tpwgts0=%d tpwgts1=%d twhere=%d moved=%d ed=%d id=%d bnd=%d\n", \
                    //     p, ii, vertex, vwgt[vertex], tpwgts[0], tpwgts[1], twhere[vertex], moved[vertex], ed[vertex], id[vertex], bnd[vertex]);
                    tpwgts[to] += vwgt[vertex];
                    tpwgts[from] -= vwgt[vertex];
                    twhere[vertex] = to;
                    moved[vertex] = 1;
                    swaps[nswaps] = vertex;
                    // if(ii == 0)
                    //     printf("tpwgts0=%d tpwgts1=%d\n", tpwgts[0], tpwgts[1]);
                    hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
                    
                    if(ed[vertex] == 0 && xadj[vertex] < xadj[vertex + 1])
                        bnd[vertex] = 0;
                    // if(ii == 500)
                    //     printf("gpu p=%d v=%d vertex=%d vwgt=%d tpwgts0=%d tpwgts1=%d twhere=%d moved=%d ed=%d id=%d bnd=%d\n", \
                    //         p, ii, vertex, vwgt[vertex], tpwgts[0], tpwgts[1], twhere[vertex], moved[vertex], ed[vertex], id[vertex], bnd[vertex]);
                }
                __syncwarp();

                //  the adj vertex
                begin = xadj[vertex];
                end = xadj[vertex + 1];
                length = end - begin;
                for(int i = lane_id;i < length;i += 32)
                {
                    int adj_vertex = adjncy[begin + i];
                    int kwgt;
                    if(twhere[adj_vertex] == from)
                        kwgt = -adjwgt[begin + i];
                    else 
                        kwgt = adjwgt[begin + i];
                    // if(ii == 500 && adj_vertex == 328)
                    //     printf("nswaps=%d v=%d k=%d kwgt=%d ed=%d id=%d bnd=%d before\n", nswaps, ii, adj_vertex, kwgt, ed[adj_vertex], id[adj_vertex], bnd[adj_vertex]);
                    id[adj_vertex] += kwgt;
                    ed[adj_vertex] -= kwgt;

                    if(bnd[adj_vertex] == 1)
                    {
                        if(ed[adj_vertex] == 0)
                        {
                            bnd[adj_vertex] = 0;
                        }
                    }
                    else
                    {
                        if(ed[adj_vertex] > 0)
                        {
                            bnd[adj_vertex] = 1;
                        }
                    }
                    // if(ii == 500 && adj_vertex == 328)
                    // if(ii == 500)
                    //     printf("nswaps=%d v=%d k=%d kwgt=%d ed=%d id=%d bnd=%d after\n", nswaps, ii, adj_vertex, kwgt, ed[adj_vertex], id[adj_vertex], bnd[adj_vertex]);
                }
                __syncwarp();
                // if(lane_id == 0 && ii == 0)
                //     printf("gpu p=%d v=%d edgecut=%d nswaps=%d\n", lane_id, ii, mincut, nswaps);
            }
            __syncwarp();

            // if(lane_id == 0)
            //     printf("pass=%d v=%d 1839\n", pass, ii);

            // for(int i = lane_id;i < nswaps;i += 32)
            //     moved[swaps[i]] = 0;
            
            // for(nswaps--; nswaps > mincutorder; nswaps--)
            // {
            //     if(lane_id == 0)
            //     {
            //         vertex = swaps[nswaps];
            //         from = twhere[vertex];
            //         to = from ^ 1;
            //         twhere[vertex] = to;
            //         // int t = ed[vertex];
            //         // ed[vertex] = id[vertex];
            //         // id[vertex] = t;
            //         // hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);

            //         // if(ed[vertex] == 0 && bnd[vertex] == 1 && xadj[vertex] < xadj[vertex + 1])
            //         //     bnd[vertex] = 0;
            //         // else if(ed[vertex] > 0 && bnd[vertex] == 0)
            //         //     bnd[vertex] = 1;
                    
            //         tpwgts[to] += vwgt[vertex];
            //         tpwgts[from] -= vwgt[vertex];
            //         // if(ii == 0)
            //         //     printf("rollback tpwgts0=%d tpwgts1=%d\n", tpwgts[0], tpwgts[1]);
            //     }
            // }

            // for(nswaps--; nswaps > mincutorder; nswaps--)
            // {
            //     if(lane_id == 0)
            //     {
            //         vertex = swaps[nswaps];
            //         from = twhere[vertex];
            //         to = from ^ 1;
            //         twhere[vertex] = to;
            //         // int t = ed[vertex];
            //         // ed[vertex] = id[vertex];
            //         // id[vertex] = t;
            //         hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);

            //         if(ed[vertex] == 0 && bnd[vertex] == 1 && xadj[vertex] < xadj[vertex + 1])
            //             bnd[vertex] = 0;
            //         else if(ed[vertex] > 0 && bnd[vertex] == 0)
            //             bnd[vertex] = 1;
                    
            //         tpwgts[to] += vwgt[vertex];
            //         tpwgts[from] -= vwgt[vertex];
            //     }
                
            //     __syncwarp();
            //     vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
            //     begin = xadj[vertex];
            //     end = xadj[vertex + 1];
            //     length = end - begin;
            //     for(int i = lane_id;i < length; i += 32)
            //     {
            //         int adj_vertex = adjncy[begin + i];
            //         int kwgt;
            //         if(twhere[adj_vertex] == from)
            //             kwgt = -adjwgt[begin + i];
            //         else 
            //             kwgt = adjwgt[begin + i];
                    
            //         if(bnd[adj_vertex] == 1 && ed[adj_vertex] == 0)
            //             bnd[adj_vertex] = 0;
            //         else if(bnd[adj_vertex] == 0 && ed[adj_vertex] > 0)
            //             bnd[adj_vertex] = 1;
            //     }
            // }

            // if(lane_id == 0)
            //     printf("pass=%d v=%d 1887\n", pass, ii);

            flag = 0;
            if(lane_id == 0)
            {
                // edgecut = mincut;
                edgecut = newcut;   // rollback is no exist
                // printf("gpu p=%d v=%d edgecut=%d nswaps=%d\n", lane_id, ii, edgecut, nswaps);

                if(mincutorder <= 0 || mincut == initcut)
                    flag = 1;
            }
            __syncwarp();
            flag    = __shfl_sync(0xffffffff, flag, 0, 32);
            edgecut = __shfl_sync(0xffffffff, edgecut, 0, 32);
            if(flag)
                break;
            
            // if(lane_id == 0)
            //     printf("pass=%d v=%d 1904\n", pass, ii);
        }
        
        // if(lane_id == 0 && ii == 0)
        // {
        //     for(int i = 0;i < nvtxs;i++)
        //     {
        //         printf("i=%d where=%d\n", i, twhere[i]);
        //     }
        // }
        __syncwarp();
        if(lane_id == 0)
        {
            // printf("gpu p=%d v=%d edgecut=%d end\n", lane_id, ii, edgecut);
            global_edgecut[ii] = edgecut;
            // global_edgecut[ii] = ii;
        }
        // hunyuangraph_int8_t *ptr = global_where + ii * nvtxs;
        // for(int i = lane_id;i < nvtxs;i += 32)
        // {
        //     ptr[i] = twhere[i];
        // }
        // if(lane_id == 0)
        //     printf("ii=%d tpwgts[0]=%d tpwgts[1]=%d\n", ii, tpwgts[0], tpwgts[1]);
        // if(lane_id == 0 && ii == 0)
        // {
        //     // printf("gpu i=%d id=%p ed=%p\n", ii, id, ed);
        //     // printf("cpu i=%d id=%p ed=%p\n", a, queue, ted);
        //     // for(int i = 0;i < nvtxs;i++)
        //     // {
        //     //     printf("i=%7d ed=%7d id=%7d\n", i, ed[i], id[i]);
        //     // }
        //     printf("tpwgts0=%d tpwgts1=%d\n", tpwgts[0], tpwgts[1]);
        // }
        
    }
}

__global__ void hunyuangraph_gpu_BFS_warp_2wayrefine_memorytest(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, int oneminpwgt, int onemaxpwgt, curandState *devStates, int nparts, \
    int *temp1, int *temp2, int *temp3, int *temp4, hunyuangraph_int8_t *temp5, hunyuangraph_int8_t *temp6, hunyuangraph_int8_t *temp7)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + (tid >> 5);
    int exam = blockIdx.x;

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;

    if(ii < nvtxs)
    {
        if(lane_id == 0)
            curand_init(-1, 0, nvtxs, &devStates[ii]);

        num += ii * shared_size;
        int *queue = (int *)num;
        int *ed = queue + nvtxs;
        int *swaps = ed + nvtxs;
        int *tpwgts = (int *)(swaps + nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
        hunyuangraph_int8_t *moved = twhere + nvtxs;
        hunyuangraph_int8_t *bnd = moved + nvtxs;
        
        // int *queue = temp1 + nvtxs * ii;
        // int *ed = temp2 + nvtxs * ii;
        // int *swaps = temp3 + nvtxs * ii;
        // int *tpwgts = temp4 + 2 * ii;
        // hunyuangraph_int8_t *twhere = temp5 + nvtxs * ii;
        // hunyuangraph_int8_t *moved = temp6 + nvtxs * ii;
        // hunyuangraph_int8_t *bnd = temp7 + nvtxs * ii;

        __syncwarp();
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            twhere[i] = 1;
            moved[i] = 0;
        }
        if(lane_id == 0)
            tpwgts[0] = 0;
        else if(lane_id == 1)
            tpwgts[1] = tvwgt;
        __syncthreads();

        int vertex, first, last, nleft, drain, nvtxs0, nvtxs1;
        int v, k, begin, end, length, flag, mark;
        nvtxs0 = 0;
        nvtxs1 = nvtxs;
        
        if(lane_id == 0)
        {
            mark = 0;
            
            vertex = ii;
            queue[0] = vertex;
            moved[vertex] = 1;
            first = 0;
            last = 1;
            nleft = nvtxs - 1;
            drain = 0;

            for(;;)
            {
                if(first == last)
                {
                    // if (nleft == 0 || drain)
                    if (nleft == 0)
                    {
                        // printf("ii=%d nleft=%d drain=%d\n", ii, nleft, drain);
                        mark = 1;
                        break;
                    }

                    k = get_random_number_range(nleft, &devStates[ii]);

                    for (v = 0; v < nvtxs; v++) 
                    {
                        if (moved[v] == 0) 
                        {
                            if (k == 0)
                                break;
                            else
                                k--;
                        }
                    }

                    queue[0] = v;
                    moved[v] = 1;
                    first    = 0; 
                    last     = 1;
                    nleft--;
                }

                v = queue[first];
                // if(nvtxs < 10)
                //     printf("ii=%d v=%d tpwgts0=%d tpwgts1=%d vwgt=%d nleft=%d\n", ii, v, tpwgts[0], tpwgts[1], vwgt[v], nleft);
                first++;
                if(nvtxs0 >= (nparts >> 1) && tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
                {
                    // drain = 1;
                    continue;
                }

                if(!can_continue_partition_BFS(0, 1, nparts, nvtxs0, nvtxs1))
                {
                    // printf("BFS ii=%d nvtxs0=%d nvtxs1=%d nparts=%d\n", ii, nvtxs0, nvtxs1, nparts);
                    break;
                }
                if(nvtxs1 == nparts - (nparts >> 1))
                    break;
                
                twhere[v] = 0;
                nvtxs0++;
                nvtxs1--;
                tpwgts[0] += vwgt[v];
                tpwgts[1] -= vwgt[v];
                if(nvtxs0 >= (nparts >> 1) && tpwgts[1] <= onemaxpwgt)
                {
                    mark = 2;
                    break;
                }
                
                drain = 0;
                end = xadj[v + 1];
                for(begin = xadj[v]; begin < end; begin++)
                {
                    k = adjncy[begin];
                    if(moved[k] == 0)
                    {
                        queue[last] = k;
                        moved[k] = 1;
                        last++;
                        nleft--;
                    }
                }
            }
            // printf("ii=%d mark=%d tpwgts0=%d tpwgts1=%d\n", ii, mark, tpwgts[0], tpwgts[1]);
        }

        __syncwarp();
        int *id;
        id = queue;
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
        }
        __syncwarp();

        //  reduce ed to acquire the edgecut
        int edgecut = 0;
            //  add to the first blockDim.x threadss
        extern __shared__ int reduce_num[];
        int *warp_reduce = reduce_num + (tid >> 5) * 32;
        if(lane_id < nvtxs)
            warp_reduce[lane_id] = ed[lane_id];
        else 
            warp_reduce[lane_id] = 0;
        for(int i = lane_id + 32;i < nvtxs; i += 32)
            warp_reduce[lane_id] += ed[i];
        __syncwarp();

            //  if nvtxs < 32
        // if(lane_id < 32 && lane_id < nvtxs) 
        warpReduction(warp_reduce, lane_id, 32);
        if(lane_id == 0) 
        {
            edgecut = warp_reduce[0] / 2;
            // global_edgecut[ii] = edgecut;
            // if(ii == 330)
                // printf("ii=%d mark=%d tpwgts0=%d tpwgts1=%d edgecut=%d nvtxs=%d nvtxs0=%d nvtxs1=%d\n", ii, mark, tpwgts[0], tpwgts[1], edgecut, nvtxs, nvtxs0, nvtxs1);
        }
        __syncwarp();
        // if(ii == 3 && lane_id == 0)
        //     {
        //         for(int i = 0;i < nvtxs;i++)
        //             printf("before nvtxs=%d i=%d twhere=%d vwgt=%d\n", nvtxs, i, twhere[i], vwgt[i]);
        //     }

        __syncwarp();
        edgecut = __shfl_sync(0xffffffff, edgecut, 0, 32);

         //  2wayrefine
        // if(lane_id == 0)
        //     printf("v=%d warp_reduce=%p reduce_num=%p\n", ii, warp_reduce, reduce_num);
        for(int i = lane_id;i < nvtxs; i += 32)
            moved[i] = 0;
        int ideal_pwgts[2];
        int nswaps, from, to, val, big_num;
        int limit, avgvwgt, origdiff;
        int newcut, mincut, initcut, mincutorder, mindiff;
        ideal_pwgts[0] = tvwgt * tpwgts0;
        ideal_pwgts[1] = tvwgt - ideal_pwgts[0];

        limit = hunyuangraph_gpu_int_min(hunyuangraph_gpu_int_max(0.01 * nvtxs, 15), 100);
        avgvwgt = hunyuangraph_gpu_int_min((tpwgts[0] + tpwgts[1]) / 20, 2 * (tpwgts[0]+tpwgts[1]) / nvtxs);
        origdiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
        big_num = -2147483648;
        // big_num = -2000000000;
        
        __syncwarp();
        for(int pass = 0;pass < 10;pass++)
        {
            mincutorder = -1;
            newcut = mincut = initcut = edgecut;
            mindiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
            flag = 0;
            
            // if(ii == 330 && lane_id == 0)
            //     printf("pass=%d \n", pass);

            mark = 0;
            for(nswaps = 0;nswaps < nvtxs;nswaps++)
            {
                //  partition
                from = (ideal_pwgts[0] - tpwgts[0] < ideal_pwgts[1] - tpwgts[1] ? 0 : 1);
                // if(ii == 368 && lane_id == 0)
                // if(lane_id == 0)
                //     printf("ii=%d from=%d nparts=%d nparts>>1=%d nvtxs=%d nvtxs0=%d nvtxs1=%d\n", ii, from, nparts, nparts >> 1, nvtxs, nvtxs0, nvtxs1);
                // if(from == 0)
                // {
                //     if(nvtxs0 == (nparts >> 1))
                //         from = 1;
                // }
                // else 
                // {
                //     if(nvtxs1 == nparts - (nparts >> 1))
                //         from = 0;
                // }
                to = from ^ 1;
                // if(ii == 3 && lane_id == 0)
                // {
                //     if(from == 0)
                //         printf("ii=%d nvtxs=%d nvtxs0=%d nvtxs1=%d from=%d\n", ii, nvtxs, nvtxs0, nvtxs1, from);
                // }

                //  select vertex
                warp_reduce[lane_id] = -1;
                val = big_num;
                // if(ii == 330 && lane_id == 0)
                // {
                //     int num_bndfrom = 0;
                //     for(int i = 0;i < nvtxs;i++)
                //         if(twhere[i] == from && bnd[i] == 1)
                //             num_bndfrom++;
                //     printf("pass=%d nswaps=%d num_bndfrom=%d\n", pass, nswaps, num_bndfrom);
                //     int num_movefrom = 0;
                //     for(int i = 0;i < nvtxs;i++)
                //         if(twhere[i] == from && bnd[i] == 1 && moved[i] == 0)
                //             num_movefrom++;
                //     printf("pass=%d nswaps=%d num_movefrom=%d\n", pass, nswaps, num_movefrom);
                // }
                __syncwarp();
                // __syncthreads();

                for(int i = lane_id;i < nvtxs; i += 32)
                {
                    if(twhere[i] == from && bnd[i] == 1 && moved[i] == 0)
                    {
                        int t = warp_reduce[lane_id];
                        if(t == -1 || val < ed[i] - id[i])
                        {
                            warp_reduce[lane_id] = i;
                            val = ed[i] - id[i];
                        }
                    }
                }
                __syncwarp();
                // if(ii == 330 && pass == 1)
                // {
                //     if(lane_id == 0)
                //     {
                //         for(int i = 0;i < 32;i++)
                //             printf("%d %d %d\n", warp_reduce[i], ed[warp_reduce[i]] - id[warp_reduce[i]], val);
                //         printf("\n");
                //     }
                // }
                // __syncwarp();

                WarpGetMax(warp_reduce, lane_id, val);
                if(lane_id == 0)
                    vertex = warp_reduce[0];

                //  boardcast vertex
                __syncwarp();
                vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
                flag = 0;
                if(lane_id == 0)
                {
                    if(vertex == -1)
                    {
                        flag = 1;
                        mark = 1;
                    }
                }
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                if(flag)
                    break;
                
                //  judge the vertex
                flag = 0;
                if(lane_id == 0)
                {
                    newcut -= (ed[vertex] - id[vertex]);
                    if(fm_can_balanced(tpwgts, vertex, to, from, onemaxpwgt, oneminpwgt, vwgt) 
                        && can_continue_partition(to, from, nparts, nvtxs0, nvtxs1)
                        && ((newcut < mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) <= origdiff + avgvwgt)
                        || (newcut == mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) < mindiff)))
                    {
                        mincut  = newcut;
                        mindiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
                        mincutorder = nswaps;
                    }
                    else if(nswaps - mincutorder > limit)
                    {
                        flag = 1;
                        mark = 2;
                        newcut += (ed[vertex] - id[vertex]);
                    }
                        
                }
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                if(flag)
                    break;
                
                //  move the vertex
                if(lane_id == 0)
                {
                    // if(ii == 368)
                        // printf("ii=%d newcut=%d nvtxs0=%d nvtxs1=%d tpwgts0=%d tpwgts1=%d vertex=%d vwgt=%d\n", ii, newcut, nvtxs0, nvtxs1, tpwgts[0], tpwgts[1], vertex, vwgt[vertex]);
                    tpwgts[to] += vwgt[vertex];
                    tpwgts[from] -= vwgt[vertex];
                    twhere[vertex] = to;
                    if(to == 0)
                        nvtxs0++, nvtxs1--;
                    else
                        nvtxs0--, nvtxs1++;
                    moved[vertex] = 1;
                    swaps[nswaps] = vertex;
                    hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
                    
                    if(ed[vertex] == 0 && xadj[vertex] < xadj[vertex + 1])
                        bnd[vertex] = 0;
                }
                __syncwarp();

                //  the adj vertex
                begin = xadj[vertex];
                end = xadj[vertex + 1];
                length = end - begin;
                for(int i = lane_id;i < length;i += 32)
                {
                    int adj_vertex = adjncy[begin + i];
                    int kwgt;
                    if(twhere[adj_vertex] == from)
                        kwgt = -adjwgt[begin + i];
                    else 
                        kwgt = adjwgt[begin + i];
                    id[adj_vertex] += kwgt;
                    ed[adj_vertex] -= kwgt;

                    if(bnd[adj_vertex] == 1)
                    {
                        if(ed[adj_vertex] == 0)
                        {
                            bnd[adj_vertex] = 0;
                        }
                    }
                    else
                    {
                        if(ed[adj_vertex] > 0)
                        {
                            bnd[adj_vertex] = 1;
                        }
                    }
                }
                __syncwarp();
            }
            __syncwarp();
            
            for(int i = lane_id;i < nswaps;i += 32)
                moved[swaps[i]] = 0;
            
            // if(ii == 330 && lane_id == 0)
            // {
            //     int sum_moved = 0;
            //     for(int i = 0;i < nvtxs;i++)
            //         sum_moved += moved[i];
            //     printf("sum_moved=%d\n", sum_moved);
            // }
            
            __syncwarp();
            if(lane_id == 0)
            // for(nswaps--; nswaps > mincutorder; nswaps--)
            {
                // printf("ii=%d mincutorder=%d\n", ii, mincutorder);
                for(nswaps--; nswaps > mincutorder; nswaps--)
                {
                    vertex = swaps[nswaps];
                    from = twhere[vertex];
                    to = from ^ 1;
                    twhere[vertex] = to;
                    if(to == 0)
                        nvtxs0++, nvtxs1--;
                    else
                        nvtxs0--, nvtxs1++;
                    
                    hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);

                    if(ed[vertex] == 0 && bnd[vertex] == 1 && xadj[vertex] < xadj[vertex + 1])
                        bnd[vertex] = 0;
                    else if(ed[vertex] > 0 && bnd[vertex] == 0)
                        bnd[vertex] = 1;
                    
                    tpwgts[to] += vwgt[vertex];
                    tpwgts[from] -= vwgt[vertex];

                    begin = xadj[vertex];
                    end = xadj[vertex + 1];
                    for(int j = begin;j < end;j++)
                    {
                        int adj_vertex = adjncy[j];
                        int kwgt;
                        if(twhere[adj_vertex] == from)
                            kwgt = -adjwgt[j];
                        else 
                            kwgt = adjwgt[j];
                        id[adj_vertex] += kwgt;
                        ed[adj_vertex] -= kwgt;
                        
                        if(bnd[adj_vertex] == 1 && ed[adj_vertex] == 0)
                            bnd[adj_vertex] = 0;
                        else if(bnd[adj_vertex] == 0 && ed[adj_vertex] > 0)
                            bnd[adj_vertex] = 1;
                    }
                }
            }
            // if(ii == 368 && lane_id == 0)
            // if(lane_id == 0)
            //     printf("ii=%d mark=%d nvtxs0=%d nvtxs1=%d\n", ii, mark, nvtxs0, nvtxs1);
            // if(ii == 3 && lane_id == 0)
            // {
            //     for(int i = 0;i < nvtxs;i++)
            //         printf("nvtxs=%d i=%d twhere=%d vwgt=%d\n", nvtxs, i, twhere[i], vwgt[i]);
            // }

            __syncwarp();
            flag = 0;
            if(lane_id == 0)
            {
                edgecut = mincut;
                // edgecut = newcut;   // rollback is no exist

                if(mincutorder <= 0 || mincut == initcut)
                {
                    flag = 1;
                    // if(ii == 330)
                    //     printf("ii=%d pass=%d edgecut=%d %d %d\n", ii, pass, edgecut, mincutorder <= 0, mincut == initcut);
                }
            }
            // if(ii == 330 && lane_id == 0)
            //     printf("ii=%d pass=%d edgecut=%d\n", ii, pass, edgecut);
            __syncwarp();
            flag    = __shfl_sync(0xffffffff, flag, 0, 32);
            edgecut = __shfl_sync(0xffffffff, edgecut, 0, 32);
            if(flag)
                break;
        }
        
        __syncwarp();
        if(lane_id == 0)
        {
            global_edgecut[ii] = edgecut;
            // printf("ii=%d nvtxs0=%d nvtxs1=%d\n", ii, nvtxs0, nvtxs1);
        }
    }
}

__global__ void hunyuangraph_gpu_BFS_warp_2wayrefine_noedit(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, int oneminpwgt, int onemaxpwgt, curandState *devStates, int nparts)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + (tid >> 5);
    int exam = blockIdx.x;

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;

    if(ii < nvtxs)
    {
        if(lane_id == 0)
            curand_init(-1, 0, nvtxs, &devStates[ii]);

        num += ii * shared_size;
        int *queue = (int *)num;
        int *ed = queue + nvtxs;
        int *swaps = ed + nvtxs;
        int *tpwgts = (int *)(swaps + nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
        hunyuangraph_int8_t *moved = twhere + nvtxs;
        hunyuangraph_int8_t *bnd = moved + nvtxs;

        __syncwarp();
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            twhere[i] = 1;
            moved[i] = 0;
        }
        if(lane_id == 0)
            tpwgts[0] = 0;
        else if(lane_id == 1)
            tpwgts[1] = tvwgt;
        __syncthreads();

        int vertex, first, last, nleft, drain, nvtxs0, nvtxs1;
        int v, k, begin, end, length, flag, mark;
        nvtxs0 = 0;
        nvtxs1 = nvtxs;
        
        if(lane_id == 0)
        {
            mark = 0;
            
            vertex = ii;
            queue[0] = vertex;
            moved[vertex] = 1;
            first = 0;
            last = 1;
            nleft = nvtxs - 1;
            drain = 0;

            for(;;)
            {
                if(first == last)
                {
                    if (nleft == 0 || drain)
                    // if (nleft == 0)
                    {
                        // printf("ii=%d nleft=%d drain=%d\n", ii, nleft, drain);
                        mark = 1;
                        break;
                    }

                    k = get_random_number_range(nleft, &devStates[ii]);

                    for (v = 0; v < nvtxs; v++) 
                    {
                        if (moved[v] == 0) 
                        {
                            if (k == 0)
                                break;
                            else
                                k--;
                        }
                    }

                    queue[0] = v;
                    moved[v] = 1;
                    first    = 0; 
                    last     = 1;
                    nleft--;
                }

                v = queue[first];
                first++;
                // if(nvtxs0 >= (nparts >> 1) && tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
                if(tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
                {
                    drain = 1;
                    continue;
                }

                // if(!can_continue_partition_BFS(0, 1, nparts, nvtxs0, nvtxs1))
                // {
                //     break;
                // }
                // if(nvtxs1 == nparts - (nparts >> 1))
                //     break;
                
                twhere[v] = 0;
                // nvtxs0++;
                // nvtxs1--;
                tpwgts[0] += vwgt[v];
                tpwgts[1] -= vwgt[v];
                // if(nvtxs0 >= (nparts >> 1) && tpwgts[1] <= onemaxpwgt)
                if(tpwgts[1] <= onemaxpwgt)
                {
                    mark = 2;
                    break;
                }
                
                drain = 0;
                end = xadj[v + 1];
                for(begin = xadj[v]; begin < end; begin++)
                {
                    k = adjncy[begin];
                    if(moved[k] == 0)
                    {
                        queue[last] = k;
                        moved[k] = 1;
                        last++;
                        nleft--;
                    }
                }
            }
        }

        __syncwarp();
        int *id;
        id = queue;
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
        }
        __syncwarp();

        //  reduce ed to acquire the edgecut
        int edgecut = 0;
            //  add to the first blockDim.x threadss
        extern __shared__ int reduce_num[];
        int *warp_reduce = reduce_num + (tid >> 5) * 32;
        if(lane_id < nvtxs)
            warp_reduce[lane_id] = ed[lane_id];
        else 
            warp_reduce[lane_id] = 0;
        for(int i = lane_id + 32;i < nvtxs; i += 32)
            warp_reduce[lane_id] += ed[i];
        __syncwarp();

            //  if nvtxs < 32
        // if(lane_id < 32 && lane_id < nvtxs) 
        // warpReduction(warp_reduce, lane_id, 32);
        edgecut = warp_reduce[lane_id];
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 16);
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 8);
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 4);
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 2);
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 1);
        if(lane_id == 0) 
        {
            // edgecut = warp_reduce[0] / 2;
            edgecut /= 2;
            // global_edgecut[ii] = edgecut;
            // if(ii == 330)
                // printf("ii=%d mark=%d tpwgts0=%d tpwgts1=%d edgecut=%d nvtxs=%d nvtxs0=%d nvtxs1=%d\n", ii, mark, tpwgts[0], tpwgts[1], edgecut, nvtxs, nvtxs0, nvtxs1);
        }
        __syncwarp();
        // if(ii == 3 && lane_id == 0)
        //     {
        //         for(int i = 0;i < nvtxs;i++)
        //             printf("before nvtxs=%d i=%d twhere=%d vwgt=%d\n", nvtxs, i, twhere[i], vwgt[i]);
        //     }

        __syncwarp();
        edgecut = __shfl_sync(0xffffffff, edgecut, 0, 32);
        // if(lane_id == 0)
        //     printf("ii=%10d edgecut=%10d\n", ii, edgecut);
         //  2wayrefine
        // if(lane_id == 0)
        //     printf("v=%d warp_reduce=%p reduce_num=%p\n", ii, warp_reduce, reduce_num);
        for(int i = lane_id;i < nvtxs; i += 32)
            moved[i] = 0;
        int ideal_pwgts[2];
        int nswaps, from, to, val, big_num;
        int limit, avgvwgt, origdiff;
        int newcut, mincut, initcut, mincutorder, mindiff;
        ideal_pwgts[0] = tvwgt * tpwgts0;
        ideal_pwgts[1] = tvwgt - ideal_pwgts[0];

        limit = hunyuangraph_gpu_int_min(hunyuangraph_gpu_int_max(0.01 * nvtxs, 15), 100);
        avgvwgt = hunyuangraph_gpu_int_min((tpwgts[0] + tpwgts[1]) / 20, 2 * (tpwgts[0] + tpwgts[1]) / nvtxs);
        origdiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
        big_num = -2147483648;
        
        __syncwarp();
        for(int pass = 0;pass < 10;pass++)
        {
            mincutorder = -1;
            newcut = mincut = initcut = edgecut;
            mindiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
            flag = 0;
            
            // if(ii == 330 && lane_id == 0)
            //     printf("pass=%d \n", pass);

            mark = 0;
            for(nswaps = 0;nswaps < nvtxs;nswaps++)
            {
                //  partition
                from = (ideal_pwgts[0] - tpwgts[0] < ideal_pwgts[1] - tpwgts[1] ? 0 : 1);

                to = from ^ 1;

                //  select vertex
                warp_reduce[lane_id] = -1;
                val = big_num;
                // if(ii == 330 && lane_id == 0)
                // {
                //     int num_bndfrom = 0;
                //     for(int i = 0;i < nvtxs;i++)
                //         if(twhere[i] == from && bnd[i] == 1)
                //             num_bndfrom++;
                //     printf("pass=%d nswaps=%d num_bndfrom=%d\n", pass, nswaps, num_bndfrom);
                //     int num_movefrom = 0;
                //     for(int i = 0;i < nvtxs;i++)
                //         if(twhere[i] == from && bnd[i] == 1 && moved[i] == 0)
                //             num_movefrom++;
                //     printf("pass=%d nswaps=%d num_movefrom=%d\n", pass, nswaps, num_movefrom);
                // }
                __syncwarp();

                for(int i = lane_id;i < nvtxs; i += 32)
                {
                    if(twhere[i] == from && bnd[i] == 1 && moved[i] == 0)
                    {
                        int t = warp_reduce[lane_id];
                        if(t == -1 || val < ed[i] - id[i])
                        {
                            warp_reduce[lane_id] = i;
                            val = ed[i] - id[i];
                        }
                    }
                }
                __syncwarp();
                // if(ii == 330 && pass == 1)
                // {
                //     if(lane_id == 0)
                //     {
                //         for(int i = 0;i < 32;i++)
                //             printf("%d %d %d\n", warp_reduce[i], ed[warp_reduce[i]] - id[warp_reduce[i]], val);
                //         printf("\n");
                //     }
                // }
                // __syncwarp();

                WarpGetMax(warp_reduce, lane_id, val);
                if(lane_id == 0)
                    vertex = warp_reduce[0];

                //  boardcast vertex
                __syncwarp();
                vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
                flag = 0;
                if(lane_id == 0)
                {
                    if(vertex == -1)
                    {
                        flag = 1;
                        mark = 1;
                    }
                }
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                if(flag)
                    break;
                
                //  judge the vertex
                flag = 0;
                if(lane_id == 0)
                {
                    // if(ii == 64 && nswaps == 0)
                    //     printf("ii=%d nswaps=%d mincutorder=%d newcut=%d nvtxs0=%d nvtxs1=%d tpwgts0=%d tpwgts1=%d vertex=%d vwgt=%d\n", ii, nswaps, mincutorder, newcut, nvtxs0, nvtxs1, tpwgts[0], tpwgts[1], vertex, vwgt[vertex]);
                
                    newcut -= (ed[vertex] - id[vertex]);
                    if((newcut < mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) <= origdiff + avgvwgt)
                        || (newcut == mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) < mindiff))
                    // if(can_continue_partition(to, from, nparts, nvtxs0, nvtxs1)
                    //     && ((newcut < mincut && tpwgts[to] + vwgt[vertex] <= onemaxpwgt && tpwgts[from] - vwgt[vertex] <= onemaxpwgt)
                    //     || (newcut == mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) < mindiff)))
                    {
                        mincut  = newcut;
                        mindiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
                        mincutorder = nswaps;
                    }
                    else if(nswaps - mincutorder > limit)
                    {
                        flag = 1;
                        mark = 2;
                        newcut += (ed[vertex] - id[vertex]);
                    }
                        
                }
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                if(flag)
                    break;
                
                //  move the vertex
                if(lane_id == 0)
                {
                    tpwgts[to] += vwgt[vertex];
                    tpwgts[from] -= vwgt[vertex];
                    twhere[vertex] = to;
                    // if(to == 0)
                    //     nvtxs0++, nvtxs1--;
                    // else
                    //     nvtxs0--, nvtxs1++;
                    moved[vertex] = 1;
                    swaps[nswaps] = vertex;
                    hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
                    
                    if(ed[vertex] == 0 && xadj[vertex] < xadj[vertex + 1])
                        bnd[vertex] = 0;
                    
                    // if(ii == 64)
                    //     printf("ii=%d nswaps=%d mincutorder=%d newcut=%d nvtxs0=%d nvtxs1=%d tpwgts0=%d tpwgts1=%d vertex=%d vwgt=%d\n", ii, nswaps, mincutorder, newcut, nvtxs0, nvtxs1, tpwgts[0], tpwgts[1], vertex, vwgt[vertex]);
                }
                __syncwarp();

                //  the adj vertex
                begin = xadj[vertex];
                end = xadj[vertex + 1];
                length = end - begin;
                for(int i = lane_id;i < length;i += 32)
                {
                    int adj_vertex = adjncy[begin + i];
                    int kwgt;
                    if(twhere[adj_vertex] == from)
                        kwgt = -adjwgt[begin + i];
                    else 
                        kwgt = adjwgt[begin + i];
                    id[adj_vertex] += kwgt;
                    ed[adj_vertex] -= kwgt;

                    if(bnd[adj_vertex] == 1)
                    {
                        if(ed[adj_vertex] == 0)
                        {
                            bnd[adj_vertex] = 0;
                        }
                    }
                    else
                    {
                        if(ed[adj_vertex] > 0)
                        {
                            bnd[adj_vertex] = 1;
                        }
                    }
                }
                __syncwarp();
            }
            __syncwarp();
            
            for(int i = lane_id;i < nswaps;i += 32)
                moved[swaps[i]] = 0;
            
            // if(ii == 330 && lane_id == 0)
            // {
            //     int sum_moved = 0;
            //     for(int i = 0;i < nvtxs;i++)
            //         sum_moved += moved[i];
            //     printf("sum_moved=%d\n", sum_moved);
            // }
            
            __syncwarp();
            if(lane_id == 0)
            // for(nswaps--; nswaps > mincutorder; nswaps--)
            {
                // printf("ii=%d mincutorder=%d\n", ii, mincutorder);
                for(nswaps--; nswaps > mincutorder; nswaps--)
                {
                    vertex = swaps[nswaps];
                    from = twhere[vertex];
                    to = from ^ 1;
                    twhere[vertex] = to;
                    // if(to == 0)
                    //     nvtxs0++, nvtxs1--;
                    // else
                    //     nvtxs0--, nvtxs1++;
                    
                    hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);

                    if(ed[vertex] == 0 && bnd[vertex] == 1 && xadj[vertex] < xadj[vertex + 1])
                        bnd[vertex] = 0;
                    else if(ed[vertex] > 0 && bnd[vertex] == 0)
                        bnd[vertex] = 1;
                    
                    tpwgts[to] += vwgt[vertex];
                    tpwgts[from] -= vwgt[vertex];

                    begin = xadj[vertex];
                    end = xadj[vertex + 1];
                    for(int j = begin;j < end;j++)
                    {
                        int adj_vertex = adjncy[j];
                        int kwgt;
                        if(twhere[adj_vertex] == from)
                            kwgt = -adjwgt[j];
                        else 
                            kwgt = adjwgt[j];
                        id[adj_vertex] += kwgt;
                        ed[adj_vertex] -= kwgt;
                        
                        if(bnd[adj_vertex] == 1 && ed[adj_vertex] == 0)
                            bnd[adj_vertex] = 0;
                        else if(bnd[adj_vertex] == 0 && ed[adj_vertex] > 0)
                            bnd[adj_vertex] = 1;
                    }
                }
            }
            // if(ii == 368 && lane_id == 0)
            // if(lane_id == 0)
            //     printf("ii=%d mark=%d nvtxs0=%d nvtxs1=%d\n", ii, mark, nvtxs0, nvtxs1);
            // if(ii == 3 && lane_id == 0)
            // {
            //     for(int i = 0;i < nvtxs;i++)
            //         printf("nvtxs=%d i=%d twhere=%d vwgt=%d\n", nvtxs, i, twhere[i], vwgt[i]);
            // }

            __syncwarp();
            flag = 0;
            if(lane_id == 0)
            {
                edgecut = mincut;
                // edgecut = newcut;   // rollback is no exist

                if(mincutorder <= 0 || mincut == initcut)
                {
                    flag = 1;
                    // if(ii == 330)
                    //     printf("ii=%d pass=%d edgecut=%d %d %d\n", ii, pass, edgecut, mincutorder <= 0, mincut == initcut);
                }
            }
            // if(ii == 330 && lane_id == 0)
            //     printf("ii=%d pass=%d edgecut=%d\n", ii, pass, edgecut);
            __syncwarp();
            flag    = __shfl_sync(0xffffffff, flag, 0, 32);
            edgecut = __shfl_sync(0xffffffff, edgecut, 0, 32);
            if(flag)
                break;
        }

        //  forced balance for vertex numbers
        /*
        flag = 0;
        int steps;
        if(lane_id == 0)
        {
            if(nvtxs0 < (nparts >> 1))
            {
                to = 0;
                from = 1;
                steps = (nparts >> 1) - nvtxs0;
            }
            else if(nvtxs1 < nparts - (nparts >> 1))
            {
                to = 1;
                from = 0;
                steps = nparts - (nparts >> 1) - nvtxs1;
            }
            else 
            {
                flag = 1;
            }
        }

        __syncwarp();
        flag = __shfl_sync(0xffffffff, flag, 0, 32);
        if(flag == 0)
        {
            for(int step = 0;step < steps;step++)
            {
                warp_reduce[lane_id] = -1;
                val = big_num;
                __syncwarp();

                for(int i = lane_id;i < nvtxs; i += 32)
                {
                    if(twhere[i] == from && bnd[i] == 1 && moved[i] == 0)
                    {
                        int t = warp_reduce[lane_id];
                        if(t == -1 || val < ed[i] - id[i])
                        {
                            warp_reduce[lane_id] = i;
                            val = ed[i] - id[i];
                        }
                    }
                }
                __syncwarp();

                WarpGetMax(warp_reduce, lane_id, val);
                if(lane_id == 0)
                    vertex = warp_reduce[0];

                //  boardcast vertex
                __syncwarp();
                vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
                flag = 0;
                if(lane_id == 0)
                {
                    // printf("ii=%d step=%d steps=%d nvtxs0=%d nvtxs1=%d vertex=%d\n", ii, step, steps, nvtxs0, nvtxs1, vertex);
                    if(vertex == -1)
                    {
                        k = get_random_number_range(nvtxs, &devStates[ii]);
                        int t = 0;
                        while(t < nvtxs)
                        {
                            if(twhere[k] == from)
                                break;
                            else
                            {
                                k++;
                                t++;
                                if(k == nvtxs)
                                    k = 0;
                            }
                        }
                        
                        vertex = k;
                        // flag = 1;
                        mark = 1;
                        // printf("vertex=-1 ii=%d step=%d steps=%d nvtxs0=%d nvtxs1=%d vertex=%d\n", ii, step, steps, nvtxs0, nvtxs1, vertex);
                    }
                }
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                if(flag == 1)
                {
                    newcut = big_num;
                    break;
                }
                if(lane_id == 0)
                {
                    newcut -= (ed[vertex] - id[vertex]);
                    tpwgts[to] += vwgt[vertex];
                    tpwgts[from] -= vwgt[vertex];
                    twhere[vertex] = to;
                    if(to == 0)
                        nvtxs0++, nvtxs1--;
                    else
                        nvtxs0--, nvtxs1++;
                    moved[vertex] = 1;
                    swaps[nswaps] = vertex;
                    hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
                    
                    if(ed[vertex] == 0 && xadj[vertex] < xadj[vertex + 1])
                        bnd[vertex] = 0;
                }
                __syncwarp();
                vertex = __shfl_sync(0xffffffff, vertex, 0, 32);

                    //  the adj vertex
                begin = xadj[vertex];
                end = xadj[vertex + 1];
                length = end - begin;
                for(int i = lane_id;i < length;i += 32)
                {
                    int adj_vertex = adjncy[begin + i];
                    int kwgt;
                    if(twhere[adj_vertex] == from)
                        kwgt = -adjwgt[begin + i];
                    else 
                        kwgt = adjwgt[begin + i];
                    id[adj_vertex] += kwgt;
                    ed[adj_vertex] -= kwgt;

                    if(bnd[adj_vertex] == 1)
                    {
                        if(ed[adj_vertex] == 0)
                        {
                            bnd[adj_vertex] = 0;
                        }
                    }
                    else
                    {
                        if(ed[adj_vertex] > 0)
                        {
                            bnd[adj_vertex] = 1;
                        }
                    }
                }
                __syncwarp();
            }
        }
        */
        
        __syncwarp();
        if(lane_id == 0)
        {
            global_edgecut[ii] = edgecut;
            // printf("ii=%d nvtxs0=%d nvtxs1=%d\n", ii, nvtxs0, nvtxs1);
        }
    }
}

__global__ void hunyuangraph_gpu_BFS_warp_2wayrefine_noedit_sampling(int start_num, int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    hunyuangraph_int8_t *num, int *global_edgecut, int *global_id, int *global_num, int oneminpwgt, int onemaxpwgt, curandState *devStates, int nparts)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x & 31;
	int ii = blockIdx.x * 4 + (tid >> 5);
    int exam = blockIdx.x;

    int shared_size = sizeof(hunyuangraph_int8_t) * nvtxs * 3 + sizeof(int) * (nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;

    if(ii < start_num)
    {
        if(lane_id == 0)
            curand_init(-1, 0, start_num, &devStates[ii]);

        num += ii * shared_size;
        int *queue = (int *)num;
        int *ed = queue + nvtxs;
        int *swaps = ed + nvtxs;
        int *tpwgts = (int *)(swaps + nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
        hunyuangraph_int8_t *moved = twhere + nvtxs;
        hunyuangraph_int8_t *bnd = moved + nvtxs;

        __syncwarp();
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            twhere[i] = 1;
            moved[i] = 0;
        }
        if(lane_id == 0)
            tpwgts[0] = 0;
        else if(lane_id == 1)
            tpwgts[1] = tvwgt;
        __syncthreads();

        int vertex, first, last, nleft, drain, nvtxs0, nvtxs1;
        int v, k, begin, end, length, flag, mark;
        int range_left, range_right, remainder;
        length = nvtxs / start_num;
        remainder = nvtxs % start_num;
        if(ii < remainder)
        {
            length++;
            range_left = ii * length;
            range_right = range_left + length;
        }
        else
        {
            range_left = remainder * (length + 1) + (ii - remainder) * length;
            range_right = range_left + length;
        }
        nvtxs0 = 0;
        nvtxs1 = nvtxs;
        
        if(lane_id == 0)
        {
            mark = 0;
            
            // k = get_random_number_range(nleft, &devStates[ii]);
            vertex = range_left + get_random_number_range(length, &devStates[ii]);
            // global_id[ii] = vertex;
            // printf("ii=%5d start_num=%5d nvtxs=%5d range_left=%5d range_right=%5d vertex=%5d\n", ii, start_num, nvtxs, range_left, range_right, vertex);
            // vertex = ii;
            queue[0] = vertex;
            moved[vertex] = 1;
            first = 0;
            last = 1;
            nleft = nvtxs - 1;
            drain = 0;

            for(;;)
            {
                if(first == last)
                {
                    if (nleft == 0 || drain)
                    // if (nleft == 0)
                    {
                        // printf("ii=%d nleft=%d drain=%d\n", ii, nleft, drain);
                        mark = 1;
                        break;
                    }

                    k = get_random_number_range(nleft, &devStates[ii]);

                    for (v = 0; v < nvtxs; v++) 
                    {
                        if (moved[v] == 0) 
                        {
                            if (k == 0)
                                break;
                            else
                                k--;
                        }
                    }

                    queue[0] = v;
                    moved[v] = 1;
                    first    = 0; 
                    last     = 1;
                    nleft--;
                }

                v = queue[first];
                first++;
                // if(nvtxs0 >= (nparts >> 1) && tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
                if(tpwgts[0] > 0 && tpwgts[1] - vwgt[v] < oneminpwgt)
                {
                    drain = 1;
                    continue;
                }

                // if(!can_continue_partition_BFS(0, 1, nparts, nvtxs0, nvtxs1))
                // {
                //     break;
                // }
                // if(nvtxs1 == nparts - (nparts >> 1))
                //     break;
                
                twhere[v] = 0;
                // nvtxs0++;
                // nvtxs1--;
                tpwgts[0] += vwgt[v];
                tpwgts[1] -= vwgt[v];
                // if(nvtxs0 >= (nparts >> 1) && tpwgts[1] <= onemaxpwgt)
                if(tpwgts[1] <= onemaxpwgt)
                {
                    mark = 2;
                    break;
                }
                
                drain = 0;
                end = xadj[v + 1];
                for(begin = xadj[v]; begin < end; begin++)
                {
                    k = adjncy[begin];
                    if(moved[k] == 0)
                    {
                        queue[last] = k;
                        moved[k] = 1;
                        last++;
                        nleft--;
                    }
                }
            }
        }

        __syncwarp();
        int *id;
        id = queue;
        for(int i = lane_id;i < nvtxs; i += 32)
        {
            compute_v_ed_id_bnd(i, xadj, adjncy, adjwgt, twhere, ed, id, bnd);
        }
        __syncwarp();
        
        //  reduce ed to acquire the edgecut
        int edgecut = 0;
            //  add to the first blockDim.x threadss
        extern __shared__ int reduce_num[];
        int *warp_reduce = reduce_num + (tid >> 5) * 32;
        if(lane_id < nvtxs)
            warp_reduce[lane_id] = ed[lane_id];
        else 
            warp_reduce[lane_id] = 0;
        for(int i = lane_id + 32;i < nvtxs; i += 32)
            warp_reduce[lane_id] += ed[i];
        __syncwarp();
        
            //  if nvtxs < 32
        // warpReduction_shlf(warp_reduce, lane_id, 32);
        edgecut = warp_reduce[lane_id];
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 16);
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 8);
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 4);
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 2);
        edgecut += __shfl_down_sync(0xffffffff, edgecut, 1);
        // __syncwarp();
        if(lane_id == 0) 
        {
            // edgecut = warp_reduce[0] / 2;
            // global_edgecut[ii] = edgecut;
            edgecut /= 2;
            // if(ii == 330)
            //     printf("ii=%d mark=%d tpwgts0=%d tpwgts1=%d edgecut=%d nvtxs=%d nvtxs0=%d nvtxs1=%d\n", ii, mark, tpwgts[0], tpwgts[1], edgecut, nvtxs, nvtxs0, nvtxs1);
        }
        // __syncwarp();
        // if(ii == 3 && lane_id == 0)
        //     {
        //         for(int i = 0;i < nvtxs;i++)
        //             printf("before nvtxs=%d i=%d twhere=%d vwgt=%d\n", nvtxs, i, twhere[i], vwgt[i]);
        //     }
        edgecut = __shfl_sync(0xffffffff, edgecut, 0, 32);
        __syncwarp();

         //  2wayrefine
        // if(lane_id == 0)
        //     printf("v=%d warp_reduce=%p reduce_num=%p\n", ii, warp_reduce, reduce_num);
        for(int i = lane_id;i < nvtxs; i += 32)
            moved[i] = 0;
        int ideal_pwgts[2];
        int nswaps, from, to, val, big_num;
        int limit, avgvwgt, origdiff;
        int newcut, mincut, initcut, mincutorder, mindiff;
        ideal_pwgts[0] = tvwgt * tpwgts0;
        ideal_pwgts[1] = tvwgt - ideal_pwgts[0];

        limit = hunyuangraph_gpu_int_min(hunyuangraph_gpu_int_max(0.01 * nvtxs, 15), 100);
        avgvwgt = hunyuangraph_gpu_int_min((tpwgts[0] + tpwgts[1]) / 20, 2 * (tpwgts[0] + tpwgts[1]) / nvtxs);
        origdiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
        big_num = -2147483648;
        
        __syncwarp();
        for(int pass = 0;pass < 10;pass++)
        {
            mincutorder = -1;
            newcut = mincut = initcut = edgecut;
            mindiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
            flag = 0;
            
            // if(ii == 330 && lane_id == 0)
            //     printf("pass=%d \n", pass);

            mark = 0;
            for(nswaps = 0;nswaps < nvtxs;nswaps++)
            {
                //  partition
                from = (ideal_pwgts[0] - tpwgts[0] < ideal_pwgts[1] - tpwgts[1] ? 0 : 1);

                to = from ^ 1;

                //  select vertex
                warp_reduce[lane_id] = -1;
                val = big_num;
                // if(ii == 330 && lane_id == 0)
                // {
                //     int num_bndfrom = 0;
                //     for(int i = 0;i < nvtxs;i++)
                //         if(twhere[i] == from && bnd[i] == 1)
                //             num_bndfrom++;
                //     printf("pass=%d nswaps=%d num_bndfrom=%d\n", pass, nswaps, num_bndfrom);
                //     int num_movefrom = 0;
                //     for(int i = 0;i < nvtxs;i++)
                //         if(twhere[i] == from && bnd[i] == 1 && moved[i] == 0)
                //             num_movefrom++;
                //     printf("pass=%d nswaps=%d num_movefrom=%d\n", pass, nswaps, num_movefrom);
                // }
                __syncwarp();

                for(int i = lane_id;i < nvtxs; i += 32)
                {
                    if(twhere[i] == from && bnd[i] == 1 && moved[i] == 0)
                    {
                        int t = warp_reduce[lane_id];
                        if(t == -1 || val < ed[i] - id[i])
                        {
                            warp_reduce[lane_id] = i;
                            val = ed[i] - id[i];
                        }
                    }
                }
                __syncwarp();
                // if(ii == 330 && pass == 1)
                // {
                //     if(lane_id == 0)
                //     {
                //         for(int i = 0;i < 32;i++)
                //             printf("%d %d %d\n", warp_reduce[i], ed[warp_reduce[i]] - id[warp_reduce[i]], val);
                //         printf("\n");
                //     }
                // }
                // __syncwarp();

                WarpGetMax(warp_reduce, lane_id, val);
                if(lane_id == 0)
                    vertex = warp_reduce[0];

                //  boardcast vertex
                __syncwarp();
                vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
                flag = 0;
                if(lane_id == 0)
                {
                    if(vertex == -1)
                    {
                        flag = 1;
                        mark = 1;
                    }
                }
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                if(flag)
                    break;
                
                //  judge the vertex
                flag = 0;
                if(lane_id == 0)
                {
                    // if(ii == 64 && nswaps == 0)
                    //     printf("ii=%d nswaps=%d mincutorder=%d newcut=%d nvtxs0=%d nvtxs1=%d tpwgts0=%d tpwgts1=%d vertex=%d vwgt=%d\n", ii, nswaps, mincutorder, newcut, nvtxs0, nvtxs1, tpwgts[0], tpwgts[1], vertex, vwgt[vertex]);
                
                    newcut -= (ed[vertex] - id[vertex]);
                    // if(can_continue_partition(to, from, nparts, nvtxs0, nvtxs1)
                    //     && ((newcut < mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) <= origdiff + avgvwgt)
                    //     || (newcut == mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) < mindiff)))
                    if((newcut < mincut && tpwgts[to] + vwgt[vertex] <= onemaxpwgt && tpwgts[from] - vwgt[vertex] <= onemaxpwgt)
                        || (newcut == mincut && hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]) < mindiff))
                    {
                        mincut  = newcut;
                        mindiff = hunyuangraph_gpu_int_abs(ideal_pwgts[0] - tpwgts[0]);
                        mincutorder = nswaps;
                    }
                    else if(nswaps - mincutorder > limit)
                    {
                        flag = 1;
                        mark = 2;
                        newcut += (ed[vertex] - id[vertex]);
                    }
                        
                }
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                if(flag)
                    break;
                
                //  move the vertex
                if(lane_id == 0)
                {
                    tpwgts[to] += vwgt[vertex];
                    tpwgts[from] -= vwgt[vertex];
                    twhere[vertex] = to;
                    // if(to == 0)
                    //     nvtxs0++, nvtxs1--;
                    // else
                    //     nvtxs0--, nvtxs1++;
                    moved[vertex] = 1;
                    swaps[nswaps] = vertex;
                    hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
                    
                    if(ed[vertex] == 0 && xadj[vertex] < xadj[vertex + 1])
                        bnd[vertex] = 0;
                    
                    // if(ii == 64)
                    //     printf("ii=%d nswaps=%d mincutorder=%d newcut=%d nvtxs0=%d nvtxs1=%d tpwgts0=%d tpwgts1=%d vertex=%d vwgt=%d\n", ii, nswaps, mincutorder, newcut, nvtxs0, nvtxs1, tpwgts[0], tpwgts[1], vertex, vwgt[vertex]);
                }
                __syncwarp();

                //  the adj vertex
                begin = xadj[vertex];
                end = xadj[vertex + 1];
                length = end - begin;
                for(int i = lane_id;i < length;i += 32)
                {
                    int adj_vertex = adjncy[begin + i];
                    int kwgt;
                    if(twhere[adj_vertex] == from)
                        kwgt = -adjwgt[begin + i];
                    else 
                        kwgt = adjwgt[begin + i];
                    id[adj_vertex] += kwgt;
                    ed[adj_vertex] -= kwgt;

                    if(bnd[adj_vertex] == 1)
                    {
                        if(ed[adj_vertex] == 0)
                        {
                            bnd[adj_vertex] = 0;
                        }
                    }
                    else
                    {
                        if(ed[adj_vertex] > 0)
                        {
                            bnd[adj_vertex] = 1;
                        }
                    }
                }
                __syncwarp();
            }
            __syncwarp();
            
            for(int i = lane_id;i < nswaps;i += 32)
                moved[swaps[i]] = 0;
            
            // if(ii == 330 && lane_id == 0)
            // {
            //     int sum_moved = 0;
            //     for(int i = 0;i < nvtxs;i++)
            //         sum_moved += moved[i];
            //     printf("sum_moved=%d\n", sum_moved);
            // }
            
            __syncwarp();
            if(lane_id == 0)
            // for(nswaps--; nswaps > mincutorder; nswaps--)
            {
                // printf("ii=%d mincutorder=%d\n", ii, mincutorder);
                for(nswaps--; nswaps > mincutorder; nswaps--)
                {
                    vertex = swaps[nswaps];
                    from = twhere[vertex];
                    to = from ^ 1;
                    twhere[vertex] = to;
                    // if(to == 0)
                    //     nvtxs0++, nvtxs1--;
                    // else
                    //     nvtxs0--, nvtxs1++;
                    
                    hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);

                    if(ed[vertex] == 0 && bnd[vertex] == 1 && xadj[vertex] < xadj[vertex + 1])
                        bnd[vertex] = 0;
                    else if(ed[vertex] > 0 && bnd[vertex] == 0)
                        bnd[vertex] = 1;
                    
                    tpwgts[to] += vwgt[vertex];
                    tpwgts[from] -= vwgt[vertex];

                    begin = xadj[vertex];
                    end = xadj[vertex + 1];
                    for(int j = begin;j < end;j++)
                    {
                        int adj_vertex = adjncy[j];
                        int kwgt;
                        if(twhere[adj_vertex] == from)
                            kwgt = -adjwgt[j];
                        else 
                            kwgt = adjwgt[j];
                        id[adj_vertex] += kwgt;
                        ed[adj_vertex] -= kwgt;
                        
                        if(bnd[adj_vertex] == 1 && ed[adj_vertex] == 0)
                            bnd[adj_vertex] = 0;
                        else if(bnd[adj_vertex] == 0 && ed[adj_vertex] > 0)
                            bnd[adj_vertex] = 1;
                    }
                }
            }
            // if(ii == 368 && lane_id == 0)
            // if(lane_id == 0)
            //     printf("ii=%d mark=%d nvtxs0=%d nvtxs1=%d\n", ii, mark, nvtxs0, nvtxs1);
            // if(ii == 3 && lane_id == 0)
            // {
            //     for(int i = 0;i < nvtxs;i++)
            //         printf("nvtxs=%d i=%d twhere=%d vwgt=%d\n", nvtxs, i, twhere[i], vwgt[i]);
            // }

            __syncwarp();
            flag = 0;
            if(lane_id == 0)
            {
                edgecut = mincut;
                // edgecut = newcut;   // rollback is no exist

                if(mincutorder <= 0 || mincut == initcut)
                {
                    flag = 1;
                    // if(ii == 330)
                    //     printf("ii=%d pass=%d edgecut=%d %d %d\n", ii, pass, edgecut, mincutorder <= 0, mincut == initcut);
                }
            }
            // if(ii == 330 && lane_id == 0)
            //     printf("ii=%d pass=%d edgecut=%d\n", ii, pass, edgecut);
            __syncwarp();
            flag    = __shfl_sync(0xffffffff, flag, 0, 32);
            edgecut = __shfl_sync(0xffffffff, edgecut, 0, 32);
            if(flag)
                break;
        }

        //  forced balance for vertex numbers
        /*
        flag = 0;
        int steps;
        if(lane_id == 0)
        {
            if(nvtxs0 < (nparts >> 1))
            {
                to = 0;
                from = 1;
                steps = (nparts >> 1) - nvtxs0;
            }
            else if(nvtxs1 < nparts - (nparts >> 1))
            {
                to = 1;
                from = 0;
                steps = nparts - (nparts >> 1) - nvtxs1;
            }
            else 
            {
                flag = 1;
            }
        }

        __syncwarp();
        flag = __shfl_sync(0xffffffff, flag, 0, 32);
        if(flag == 0)
        {
            for(int step = 0;step < steps;step++)
            {
                warp_reduce[lane_id] = -1;
                val = big_num;
                __syncwarp();

                for(int i = lane_id;i < nvtxs; i += 32)
                {
                    if(twhere[i] == from && bnd[i] == 1 && moved[i] == 0)
                    {
                        int t = warp_reduce[lane_id];
                        if(t == -1 || val < ed[i] - id[i])
                        {
                            warp_reduce[lane_id] = i;
                            val = ed[i] - id[i];
                        }
                    }
                }
                __syncwarp();

                WarpGetMax(warp_reduce, lane_id, val);
                if(lane_id == 0)
                    vertex = warp_reduce[0];

                //  boardcast vertex
                __syncwarp();
                vertex = __shfl_sync(0xffffffff, vertex, 0, 32);
                flag = 0;
                if(lane_id == 0)
                {
                    // printf("ii=%d step=%d steps=%d nvtxs0=%d nvtxs1=%d vertex=%d\n", ii, step, steps, nvtxs0, nvtxs1, vertex);
                    if(vertex == -1)
                    {
                        k = get_random_number_range(nvtxs, &devStates[ii]);
                        int t = 0;
                        while(t < nvtxs)
                        {
                            if(twhere[k] == from)
                                break;
                            else
                            {
                                k++;
                                t++;
                                if(k == nvtxs)
                                    k = 0;
                            }
                        }
                        
                        vertex = k;
                        // flag = 1;
                        mark = 1;
                        // printf("vertex=-1 ii=%d step=%d steps=%d nvtxs0=%d nvtxs1=%d vertex=%d\n", ii, step, steps, nvtxs0, nvtxs1, vertex);
                    }
                }
                __syncwarp();
                flag   = __shfl_sync(0xffffffff, flag, 0, 32);
                if(flag == 1)
                {
                    newcut = big_num;
                    break;
                }
                if(lane_id == 0)
                {
                    newcut -= (ed[vertex] - id[vertex]);
                    tpwgts[to] += vwgt[vertex];
                    tpwgts[from] -= vwgt[vertex];
                    twhere[vertex] = to;
                    if(to == 0)
                        nvtxs0++, nvtxs1--;
                    else
                        nvtxs0--, nvtxs1++;
                    moved[vertex] = 1;
                    swaps[nswaps] = vertex;
                    hunyuangraph_gpu_int_swap(&ed[vertex], &id[vertex]);
                    
                    if(ed[vertex] == 0 && xadj[vertex] < xadj[vertex + 1])
                        bnd[vertex] = 0;
                }
                __syncwarp();
                vertex = __shfl_sync(0xffffffff, vertex, 0, 32);

                    //  the adj vertex
                begin = xadj[vertex];
                end = xadj[vertex + 1];
                length = end - begin;
                for(int i = lane_id;i < length;i += 32)
                {
                    int adj_vertex = adjncy[begin + i];
                    int kwgt;
                    if(twhere[adj_vertex] == from)
                        kwgt = -adjwgt[begin + i];
                    else 
                        kwgt = adjwgt[begin + i];
                    id[adj_vertex] += kwgt;
                    ed[adj_vertex] -= kwgt;

                    if(bnd[adj_vertex] == 1)
                    {
                        if(ed[adj_vertex] == 0)
                        {
                            bnd[adj_vertex] = 0;
                        }
                    }
                    else
                    {
                        if(ed[adj_vertex] > 0)
                        {
                            bnd[adj_vertex] = 1;
                        }
                    }
                }
                __syncwarp();
            }
        }
        */
        
        __syncwarp();
        if(lane_id == 0)
        {
            global_edgecut[ii] = edgecut;
            global_id[atomicAdd(global_num, 1)] = edgecut;
            // printf("ii=%5d nvtxs0=%5d nvtxs1=%5d global_edgecut=%10d\n", ii, nvtxs0, nvtxs1, global_edgecut[ii]);
        }
    }
}

__device__ void warpGetMin(int *shared_edgecut, int *shared_id, int tid, int blocksize)
{
    if(blocksize >= 64)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 32])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 32];
            shared_id[tid] = shared_id[tid + 32];
        }
    }
    if(blocksize >= 32)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 16])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 16];
            shared_id[tid] = shared_id[tid + 16];
        }
    }
    if(blocksize >= 16)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 8])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 8];
            shared_id[tid] = shared_id[tid + 8];
        }
    }
    if(blocksize >= 8)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 4])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 4];
            shared_id[tid] = shared_id[tid + 4];
        }
    }
    if(blocksize >= 4)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 2])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 2];
            shared_id[tid] = shared_id[tid + 2];
        }
    }
    if(blocksize >= 2)
    {
        if(shared_edgecut[tid] > shared_edgecut[tid + 1])
        {
            shared_edgecut[tid] = shared_edgecut[tid + 1];
            shared_id[tid] = shared_id[tid + 1];
        }
    }
}

__global__ void hunyuangraph_gpu_select_where(int start_num, int *global_edgecut, int *best_id)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int num[2048];
    int *shared_edgecut, *shared_id;
    shared_edgecut = num;
    shared_id = shared_edgecut + blockDim.x;

    if(ii < start_num)
    {
        shared_edgecut[tid] = global_edgecut[tid];
        // shared_id[tid] = global_id[ii];
        shared_id[tid] = tid;
        // printf("tid=%d shared_id=%d shared_edgecut=%d\n", tid, shared_id[tid], shared_edgecut[tid]);
    }
    else
    {
        shared_edgecut[tid] = 2147483647;
        shared_id[tid] = -1;
    }
    __syncthreads();

    // printf("tid=%d edgecut=%d id=%d\n", tid, shared_edgecut[tid], shared_id[tid]);
    // __syncthreads();

    for(int i = tid + blockDim.x;i < start_num;i += blockDim.x)
    {
        if(shared_edgecut[tid] > global_edgecut[i])
        {
            shared_edgecut[tid] = global_edgecut[i];
            shared_id[tid] = i;
        }
    }
    __syncthreads();

    if(tid < start_num)
    {
        if(blockDim.x >= 1024) 
        {
            if(tid < 512) 
            {
                if(shared_edgecut[tid] > shared_edgecut[tid + 512])
                {
                    shared_edgecut[tid] = shared_edgecut[tid + 512];
                    shared_id[tid] = shared_id[tid + 512];
                }
            }
            __syncthreads();
        }
        if(blockDim.x >= 512) 
        {
            if(tid < 256) 
            {
                if(shared_edgecut[tid] > shared_edgecut[tid + 256])
                {
                    shared_edgecut[tid] = shared_edgecut[tid + 256];
                    shared_id[tid] = shared_id[tid + 256];
                }
            }
            __syncthreads();
        }
        if(blockDim.x >= 256) 
        {
            if(tid < 128) 
            {
                if(shared_edgecut[tid] > shared_edgecut[tid + 128])
                {
                    shared_edgecut[tid] = shared_edgecut[tid + 128];
                    shared_id[tid] = shared_id[tid + 128];
                }
            }
            __syncthreads();
        }
        if(blockDim.x >= 128) 
        {
            if(tid < 64) 
            {
                if(shared_edgecut[tid] > shared_edgecut[tid + 64])
                {
                    shared_edgecut[tid] = shared_edgecut[tid + 64];
                    shared_id[tid] = shared_id[tid + 64];
                }
            }
            __syncthreads();
        }

        if(tid < 32) 
            warpGetMin(shared_edgecut, shared_id, tid, blockDim.x);
        
        if(tid == 0)
            best_id[0] = shared_id[0];
        
        // if(tid == 0)
        // {
        //     printf("best_id=%d best_edgecut=%d\n", best_id[0], global_edgecut[best_id[0]]);
        // }
    }
}

__global__ void exam_where(int nvtxs, int *where, int *xadj, int *adjncy)
{
    // for(int i = 0;i < nvtxs;i++)
    // {
    //     if(where[i] == 0)
    //     {
    //         printf("%d %d\n", xadj[i], xadj[i + 1]);
    //         for(int j = xadj[i];j < xadj[i + 1];j++)
    //             printf("%d ", adjncy[j]);
    //         printf("\n");
    //     }
    // }
    for(int i = 0;i < nvtxs;i++)
        printf("i=%d where=%d\n", i, where[i]);
}

__global__ void exam_pwgts(int *pwgts, int oneminpwgts, int onemaxpwgts)
{
    printf("oneminpwgts=%d onemaxpwgts=%d\n", oneminpwgts, onemaxpwgts);
    printf("%d %d\n", pwgts[0], pwgts[1]);

    if(pwgts[0] <= onemaxpwgts && pwgts[1] <= onemaxpwgts)
        printf("balance!!!\n");
    else    
        printf("unbalance!!!\n");
}

__global__ void exam_answer(int start_num, int *answer)
{
    for(int i = 0;i < start_num;i++)
        // printf("%7d %10d\n", i, answer[i]);
        printf("i=%5d edgecut=%10d\n", i, answer[i]);
}

__global__ void hunyuangraph_gpu_update_where(int nvtxs, int shared_size, int *where, int *pwgts, hunyuangraph_int8_t *num, int best_id)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < nvtxs)
    {
        num += best_id * shared_size;
        int *queue = (int *)num;
        int *ed = queue + nvtxs;
        int *swaps = ed + nvtxs;
        int *tpwgts = (int *)(swaps + nvtxs);
        hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);

        // if(ii == 0)
        // {
        //     // printf("gpu i=%d nvtxs=%d\n", ii, nvtxs);
        //     printf("gpu i=%d num=%p\n", ii, num);
        //     printf("gpu i=%d queue=%p\n", ii, queue);
        //     printf("gpu i=%d ed=%p\n", ii, ed); 
        //     printf("gpu i=%d swaps=%p\n", ii, swaps);
        //     printf("gpu i=%d tpwgts=%p\n", ii, tpwgts);
        //     printf("gpu i=%d twhere=%p\n", ii, twhere);
        // }

        if(ii < 2)
            pwgts[ii] = tpwgts[ii];
        
        where[ii] = (int)twhere[ii];
        // printf("gpu i=%d where=%d\n", ii, where[ii]);
    }
}

__global__ void hunyuangraph_gpu_update_where_memorytest(int nvtxs, int *where, int *pwgts, int * temp4, hunyuangraph_int8_t *temp5, int best_id)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < nvtxs)
    {
        int *tpwgts = temp4 + 2 * best_id;
        hunyuangraph_int8_t *twhere = temp5 + nvtxs * best_id;

        if(ii < 2)
            pwgts[ii] = tpwgts[ii];
        
        where[ii] = (int)twhere[ii];
    }
}

__global__ void hunyuangraph_update_answer(int nvtxs, int fpart, int *where, int *answer, int *label)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < nvtxs)
        answer[label[ii]] = where[ii] + fpart;
}

__global__ void init_0(int length, int *num)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < length)
        num[ii] = 0;
}

// NVIDIA GeForce RTX 3060 Laptop GPU SM -> 30
void hunyuangraph_gpu_RecursiveBisection(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int nparts, double *ubvec, double *tpwgts, int *answer, int fpart, int level)
{
    // printf("nparts=%d fpart=%d level=%d nvtxs=%d\n", nparts, fpart, level, graph->nvtxs);

    if(graph->nvtxs < nparts)
    {
        printf("****You are trying to partition too many parts!****\n");
        printf("    nparts=%d nvtxs=%d\n", nparts, graph->nvtxs);
		// exit(0);
    }

    int *global_edgecut;
    hunyuangraph_int8_t *global_where;
    priority_queue_t *queues;
    int oneminpwgt, onemaxpwgt;
	double *tpwgts2, tpwgts0, tpwgts1;

	tpwgts2 = (double *)malloc(sizeof(double) * 2);
	tpwgts2[0] = hunyuangraph_double_sum(nparts >> 1, tpwgts);
	tpwgts2[1] = 1.0 - tpwgts2[0];
    // printf("%lf %lf %d\n", tpwgts2[0], tpwgts2[1], graph->tvwgt[0]);
    tpwgts0    = tpwgts2[0];
    // exit(0);

    int start_num = SM_NUM * 4;
    int sampling = 1;
    if(start_num > graph->nvtxs)
    {
        start_num = graph->nvtxs;
        sampling = 0;
    }
    // printf("start_num=%d nvtxs=%d\n", start_num, graph->nvtxs);
    //  test
#ifdef FIGURE10_EXHAUSTIVE
    start_num = graph->nvtxs;
    sampling = 0;
#endif

    
	//	GPU Bisection
    // graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_where");
    // graph->cuda_pwgts = (int *)lmalloc_with_check(sizeof(int) * 2, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_pwgts");
    // graph->cuda_ed    = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_ed");
    // graph->cuda_id    = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_id");
    // graph->cuda_bnd   = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_bnd");

    int shared_size = sizeof(hunyuangraph_int8_t) * graph->nvtxs * 3 + sizeof(int) * (graph->nvtxs * 3 + 2);
    shared_size = shared_size + hunyuangraph_GPU_cacheline - shared_size % hunyuangraph_GPU_cacheline;
    hunyuangraph_int8_t *tnum;
    int *global_id, *global_num;
    if(GPU_Memory_Pool)
    {
        // tnum = (hunyuangraph_int8_t *)lmalloc_with_check(shared_size * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: tnum");
        tnum = (hunyuangraph_int8_t *)lmalloc_with_check(shared_size * start_num, "hunyuangraph_gpu_RecursiveBisection: tnum");
        // printf("tnum=%p\n", tnum);
        
        // global_edgecut = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: global_edgecut");
        global_edgecut = (int *)lmalloc_with_check(sizeof(int) * start_num, "hunyuangraph_gpu_RecursiveBisection: global_edgecut");
        global_id = (int *)lmalloc_with_check(sizeof(int) * start_num, "hunyuangraph_gpu_RecursiveBisection: global_id");
        global_num = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_gpu_RecursiveBisection: global_num");
    }
    else
    {
        cudaMalloc((void **)&tnum, shared_size * start_num);
        cudaMalloc((void **)&global_edgecut, sizeof(int) * start_num);
        cudaMalloc((void **)&global_id, sizeof(int) * start_num);
        cudaMalloc((void **)&global_num, sizeof(int));
    }

    init_0<<<1, 1>>>(1, global_num);
    // global_where   = (hunyuangraph_int8_t *)lmalloc_with_check(sizeof(hunyuangraph_int8_t) * graph->nvtxs * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: global_where");
    // printf("global_where=%p\n", global_where);
    // printf("shared_size=%d\n", shared_size);
    onemaxpwgt = ubvec[0] * graph->tvwgt[0] * tpwgts2[1];
    oneminpwgt = (1.0 / ubvec[0])*graph->tvwgt[0] * tpwgts2[1];
    // printf("oneminpwgt=%d onemaxpwgt=%d\n", oneminpwgt, onemaxpwgt);

        //  CUDA Random number
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_bisection, NULL);
#endif
    curandState *devStates;
    // cudaMalloc(&devStates, graph->nvtxs * sizeof(curandState));
    // devStates = (curandState *)lmalloc_with_check(sizeof(curandState) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: devStates");
    
    if(GPU_Memory_Pool)
        devStates = (curandState *)lmalloc_with_check(sizeof(curandState) * start_num, "hunyuangraph_gpu_RecursiveBisection: devStates");
    else 
        cudaMalloc((void **)&devStates, sizeof(curandState) * start_num);

    // initializeCurand<<<(graph->nvtxs + 127) / 128, 128>>>(-1, 0, graph->nvtxs, devStates);
    // curand_init(-1, 0, graph->nvtxs, devStates);
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_bisection, NULL);
    initcurand_gpu_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;
#endif

    // __global__ void hunyuangraph_gpu_Bisection(int nvtxs, int *vwgt, int *xadj, int *adjncy, int *adjwgt, int tvwgt, double tpwgts0, \
    //     int *global_edgecut, int *global_where, priority_queue_t **queues, int oneminpwgt, int onemaxpwgt, curandState *state)
	// exit(0);
    // printf("hunyuangraph_gpu_Bisection begin\n");
    // cudaError_t err = cudaGetLastError();
    // cudaDeviceSynchronize();
    // exam_csr<<<1,1>>>(graph->nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt);
    // cudaDeviceSynchronize();

#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_bisection, NULL);
#endif
    // hunyuangraph_gpu_Bisection<<<graph->nvtxs, 128, sizeof(hunyuangraph_int8_t) * graph->nvtxs * 3 + sizeof(int) * (graph->nvtxs * 3 + 2)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     global_edgecut, global_where, queues, key, val, locator, oneminpwgt, onemaxpwgt, devStates);
    // hunyuangraph_gpu_Bisection_global<<<graph->nvtxs, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     tnum, global_edgecut, global_where, oneminpwgt, onemaxpwgt, devStates);
    // hunyuangraph_gpu_Bisection_warp<<<(graph->nvtxs + 3) / 4, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     tnum, global_edgecut, global_where, oneminpwgt, onemaxpwgt, devStates);
    // hunyuangraph_gpu_BFS_warp<<<(graph->nvtxs + 3) / 4, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     tnum, global_edgecut, global_where, oneminpwgt, onemaxpwgt, devStates);
    // hunyuangraph_gpu_BFS_warp_2wayrefine<<<(graph->nvtxs + 3) / 4, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     tnum, global_edgecut, oneminpwgt, onemaxpwgt, devStates);
    // int *temp1, *temp2, *temp3, *temp4;
    // hunyuangraph_int8_t *temp5, *temp6, *temp7;
    // cudaMalloc((void**)&temp1, graph->nvtxs * graph->nvtxs * sizeof(int));  //  queue
    // cudaMalloc((void**)&temp2, graph->nvtxs * graph->nvtxs * sizeof(int));  //  ed
    // cudaMalloc((void**)&temp3, graph->nvtxs * graph->nvtxs * sizeof(int));  //  swaps
    // cudaMalloc((void**)&temp4, graph->nvtxs * 2 * sizeof(int));             //  tpwgts
    // cudaMalloc((void**)&temp5, graph->nvtxs * graph->nvtxs * sizeof(hunyuangraph_int8_t));  //  twhere
    // cudaMalloc((void**)&temp6, graph->nvtxs * graph->nvtxs * sizeof(hunyuangraph_int8_t));  //  moved
    // cudaMalloc((void**)&temp7, graph->nvtxs * graph->nvtxs * sizeof(hunyuangraph_int8_t));  //  bnd
    // hunyuangraph_gpu_BFS_warp_2wayrefine_memorytest<<<(graph->nvtxs + 3) / 4, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     tnum, global_edgecut, oneminpwgt, onemaxpwgt, devStates, nparts, temp1, temp2, temp3, temp4, temp5, temp6, temp7);
    // hunyuangraph_gpu_BFS_warp_2wayrefine<<<2, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
    //     tnum, global_edgecut, oneminpwgt, onemaxpwgt, devStates);
    if(!sampling)
        hunyuangraph_gpu_BFS_warp_2wayrefine_noedit<<<(graph->nvtxs + 3) / 4, 128, 128 * sizeof(int)>>>(graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
            tnum, global_edgecut, oneminpwgt, onemaxpwgt, devStates, nparts);
    else
        hunyuangraph_gpu_BFS_warp_2wayrefine_noedit_sampling<<<(start_num + 3) / 4, 128, 128 * sizeof(int)>>>(start_num, graph->nvtxs, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt, graph->tvwgt[0], tpwgts0,\
            tnum, global_edgecut, global_id, global_num, oneminpwgt, onemaxpwgt, devStates, nparts);
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_bisection, NULL);
    bisection_gpu_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;
#endif
    // cudaDeviceSynchronize();
    // exam_answer<<<1, 1>>>(start_num, global_id);
    // cudaDeviceSynchronize();

    // err = cudaDeviceSynchronize();
    // checkCudaError(err, "hunyuangraph_gpu_Bisection");
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Kernel execution failed (error code %s)!\n", cudaGetErrorString(err));
    //     // Handle error as appropriate (e.g., exit, throw exception, etc.)
    // }
    // printf("hunyuangraph_gpu_Bisection end\n");
    // printf("        gpu_Bisection_time         %10.3lf %7.3lf%\n", bisection_gpu_time, bisection_gpu_time / bisection_gpu_time * 100);
    
    // exit(0);
    // cudaDeviceSynchronize();
    // int *gpu_edgecut;
    // gpu_edgecut = (int *)malloc(sizeof(int) * graph->nvtxs);
    // cudaMemcpy(gpu_edgecut, global_edgecut, sizeof(int) * graph->nvtxs, cudaMemcpyDeviceToHost);
    // int min_edgecut, min_edgecut_id;
    // for(int a = 0;a < graph->nvtxs;a++)
    // // for(int a = 0;a < 8;a++)
    // {
    //     hunyuangraph_int8_t *num;
    //     num = (hunyuangraph_int8_t *)malloc(sizeof(hunyuangraph_int8_t) * graph->nvtxs);
        
    //     int *queue = (int *)tnum;
    //     int *ed = queue + graph->nvtxs;
    //     int *swaps = ed + graph->nvtxs;
    //     int *tpwgts = (int *)(swaps + graph->nvtxs);
    //     hunyuangraph_int8_t *twhere = (hunyuangraph_int8_t *)(tpwgts + 2);
    //     twhere += a * shared_size;
    //     // queue  += a * shared_size;
    //     // queue = (int *)((int *)twhere - graph->nvtxs * 3  - 2);
    //     // cudaMemcpy(tid, queue, sizeof(int) * graph->nvtxs * 2, cudaMemcpyDeviceToHost);
    //     // if(a == 500)
    //     //     printf("cpu i=%d id=%p ed=%p\n", a, queue, queue + graph->nvtxs);
    //     cudaMemcpy(num, twhere, sizeof(hunyuangraph_int8_t) * graph->nvtxs, cudaMemcpyDeviceToHost);
    //     // cudaMemcpy(num, &global_where[a * graph->nvtxs], sizeof(hunyuangraph_int8_t) * graph->nvtxs, cudaMemcpyDeviceToHost);
    //     if(a == 330)
    //     {
    //         int lnvtxs = 0;
    //         for(int i = 0;i < graph->nvtxs;i++)
    //             if(num[i] == 0)
    //                 lnvtxs++;
    //         printf("lnvtxs=%d\n", lnvtxs);
    //     //         printf("i=%d where=%d\n", i, num[i]);
    //     //     int *ted, *tid;
    //     //     ted = (int *)malloc(sizeof(int) * graph->nvtxs);
    //     //     tid = (int *)malloc(sizeof(int) * graph->nvtxs);
    //     //     for(int i = 0;i < graph->nvtxs;i++)
    //     //     {
    //     //         hunyuangraph_int8_t me = num[i];
    //     //         ted[i] = 0, tid[i] = 0;
    //     //         for(int j = graph->xadj[i];j < graph->xadj[i + 1];j++)
    //     //         {
    //     //             hunyuangraph_int8_t other = num[graph->adjncy[j]];
    //     //             if(me == other)
    //     //                 tid[i] += graph->adjwgt[j];
    //     //             else 
    //     //                 ted[i] += graph->adjwgt[j];
    //     //             if(i == 3971)
    //     //             {
    //     //                 printf("i=%d j=%d me=%d other=%d adjwgt=%d\n", i, j, me, other, graph->adjwgt[j]);
    //     //             }
    //     //         }
    //     //         printf("i=%7d ed=%7d id=%7d\n", i, ted[i], tid[i]);
    //     //     }
    //     //     free(ted);
    //     //     free(tid);
    //     }
    //     int e = 0;
    //     int pwgts[2];
    //     pwgts[0] = 0;
    //     pwgts[1] = 0;
    //     for(int i = 0;i < graph->nvtxs;i++)
    //     {
    //         hunyuangraph_int8_t me = num[i];
    //         if(me == 0) pwgts[0] += graph->vwgt[i];
    //         else pwgts[1] += graph->vwgt[i];
    //         for(int j = graph->xadj[i];j < graph->xadj[i + 1];j++)
    //         {
    //             hunyuangraph_int8_t other = num[graph->adjncy[j]];
    //             if(other != me)
    //                 e += graph->adjwgt[j];
    //         }
    //     }
    //     int flag = 0;
    //     if(pwgts[0] >= oneminpwgt && pwgts[1] >= oneminpwgt
    //         && pwgts[0] <= onemaxpwgt && pwgts[1] <= onemaxpwgt)
    //         flag = 1;
    //     // printf("gpu p=%d v=%d edgecut=%d\n", a, a, gpu_edgecut[a]);
    //     // if(a == 0)
    //         printf("cpu v=%d edgecut=%d pwgts0=%d pwgts1=%d flag=%d\n", a, e / 2, pwgts[0], pwgts[1], flag);
    //     if(a == 0) 
    //     {
    //         min_edgecut = e / 2;
    //         min_edgecut_id = a;
    //     }
    //     else 
    //     {
    //         if(min_edgecut > e / 2)
    //         {    
    //             min_edgecut = e / 2;
    //             min_edgecut_id = a;
    //         }
    //     }
    //     free(num);
    // }
    // free(gpu_edgecut);
    // printf("min_edgecut=%d min_edgecut_id=%d\n", min_edgecut, min_edgecut_id);

    // exit(0);

#ifdef FIGURE10_EXHAUSTIVE
    cudaDeviceSynchronize();
    exam_answer<<<1, 1>>>(start_num, global_edgecut);
    cudaDeviceSynchronize();
#endif
#ifdef FIGURE10_SAMPLING
    cudaDeviceSynchronize();
    exam_answer<<<1, 1>>>(start_num, global_edgecut);
    cudaDeviceSynchronize();
#endif
    //  select the best where
    int *best_id_gpu, best_id;
    if(GPU_Memory_Pool)
        best_id_gpu = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_gpu_RecursiveBisection: best_id_gpu");
    else
        cudaMalloc((void **)&best_id_gpu, sizeof(int));

#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_bisection, NULL);
#endif
    // hunyuangraph_gpu_select_where<<<1, 1024, 2048 * sizeof(int)>>>(graph->nvtxs, global_edgecut, best_id_gpu);
    hunyuangraph_gpu_select_where<<<1, 1024, 2048 * sizeof(int)>>>(start_num, global_edgecut, best_id_gpu);
    // hunyuangraph_gpu_select_where<<<1, 1024, 2048 * sizeof(int)>>>(8, global_edgecut, best_id_gpu);
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_bisection, NULL);
    select_where_gpu_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;
#endif

    cudaMemcpy(&best_id, best_id_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("best_id=%d\n", best_id);

    int *temp_where;
    if(GPU_Memory_Pool)
    {
        temp_where = (int *)rmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: temp_where");
        graph->cuda_pwgts = (int *)rmalloc_with_check(sizeof(int) * 2, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_pwgts");
    }
    else
    {
        cudaMalloc((void **)&temp_where, sizeof(int) * graph->nvtxs);
        cudaMalloc((void **)&graph->cuda_pwgts, sizeof(int) * 2);
    }

    //	update where
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_bisection, NULL);
#endif
    hunyuangraph_gpu_update_where<<<(graph->nvtxs + 127) / 128, 128>>>(graph->nvtxs, shared_size, temp_where, graph->cuda_pwgts, tnum, best_id);
    // hunyuangraph_gpu_update_where_memorytest<<<(graph->nvtxs + 127) / 128, 128>>>(graph->nvtxs, graph->cuda_where, graph->cuda_pwgts, temp4, temp5, best_id);
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_bisection, NULL);
    update_where_gpu_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;
#endif

    // cudaDeviceSynchronize();
    // // exam_where<<<1, 1>>>(graph->nvtxs, graph->cuda_where, graph->cuda_xadj, graph->cuda_adjncy);
    // exam_pwgts<<<1, 1>>>(graph->cuda_pwgts, oneminpwgt, onemaxpwgt);
    // cudaDeviceSynchronize();
    // hunyuangraph_int8_t *answer = (hunyuangraph_int8_t *)malloc(sizeof(hunyuangraph_int8_t) * graph->nvtxs);
    if(GPU_Memory_Pool)
    {
        lfree_with_check((void *)best_id_gpu, sizeof(int), "hunyuangraph_gpu_RecursiveBisection: best_id_gpu");                         // best_id_gpu 
        // lfree_with_check(sizeof(curandState) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: devStates"); // devStates
        // lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: global_edgecut");    // global_edgecut
        // lfree_with_check(shared_size * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: tnum");              // tnum
        lfree_with_check((void *)devStates, sizeof(curandState) * start_num, "hunyuangraph_gpu_RecursiveBisection: devStates");         // devStates
        lfree_with_check((void *)global_num, sizeof(int), "hunyuangraph_gpu_RecursiveBisection: global_num");                           // global_num
        lfree_with_check((void *)global_id, sizeof(int) * start_num, "hunyuangraph_gpu_RecursiveBisection: global_id");                 // global_id
        lfree_with_check((void *)global_edgecut, sizeof(int) * start_num, "hunyuangraph_gpu_RecursiveBisection: global_edgecut");       // global_edgecut
        lfree_with_check((void *)tnum, shared_size * start_num, "hunyuangraph_gpu_RecursiveBisection: tnum");                           // tnum
    }
    else
    {
        cudaFree(best_id_gpu);
        cudaFree(devStates);
        cudaFree(global_num);
        cudaFree(global_id);
        cudaFree(global_edgecut);
        cudaFree(tnum);
    }
    // cudaFree(temp1);
    // cudaFree(temp2);
    // cudaFree(temp3);
    // cudaFree(temp4);
    // cudaFree(temp5);
    // cudaFree(temp6);
    // cudaFree(temp7);

    //  update answer
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_bisection, NULL);
#endif
    hunyuangraph_update_answer<<<(graph->nvtxs + 127) / 128, 128>>>(graph->nvtxs, fpart, temp_where, answer, graph->cuda_label);
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_bisection, NULL);
    update_answer_gpu_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;
#endif
    // exit(0);

	//	SplitGraph
    hunyuangraph_graph_t *lgraph, *rgraph;
    if(nparts > 2)
    {
        // hunyuangraph_splitgraph(hunyuangraph_admin, graph, &hunyuangraph_admin->lgraph, &hunyuangraph_admin->rgraph);
        //  first is right subgraph, second is left subgraph
#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&begin_gpu_bisection, NULL);
#endif
        hunyuangraph_gpu_SplitGraph_intersect(hunyuangraph_admin, graph, temp_where, &lgraph, &rgraph);
        // hunyuangraph_gpu_SplitGraph_separate(hunyuangraph_admin, graph, &lgraph, &rgraph);
#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&end_gpu_bisection, NULL);
        splitgraph_gpu_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;
#endif
        // printf("lgraph->nvtxs=%d rgraph->nvtxs=%d\n", lgraph->nvtxs, rgraph->nvtxs);
        if(lgraph->nvtxs < (nparts >> 1))
        {
            printf("lgraph is too small, lgraph->nvtxs=%d should greater than or equal to %d\n", lgraph->nvtxs, (nparts >> 1));
            // exit(0);
        }
        if(rgraph->nvtxs < nparts - (nparts >> 1))
        {
            printf("rgraph is too small, rgraph->nvtxs=%d should greater than or equal to %d\n", rgraph->nvtxs, nparts - (nparts >> 1));
            // exit(0);
        }

        // cudaMemcpy(graph->where, graph->cuda_where, sizeof(int) * graph->nvtxs, cudaMemcpyDeviceToHost);
        // exam_cpu_subgraph(graph);
    }

    if(GPU_Memory_Pool)
    {
        rfree_with_check((void *)graph->cuda_pwgts, sizeof(int) * 2, "hunyuangraph_gpu_RecursiveBisection: graph->cuda_pwgts");     // graph->cuda_pwgts
        rfree_with_check((void *)temp_where, sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_RecursiveBisection: temp_where");        // temp_where
    }
    else
    {
        cudaFree(graph->cuda_pwgts);
        cudaFree(temp_where);
    }

    //  free grapg, only retain lgraph and rgraph
    if(!GPU_Memory_Pool && level != 0)
    {
        cudaFree(graph->cuda_vwgt);
        cudaFree(graph->cuda_xadj);
        cudaFree(graph->cuda_adjncy);
        cudaFree(graph->cuda_adjwgt);
        cudaFree(graph->label);
    }

    // if(nparts == 8)
    //     exit(0);

	//	free graph
    // intersect: no free
    // separate: no free idea

    //  update tpwgts
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_bisection, NULL);
#endif
    double multi0 = 1.0 / tpwgts2[0];
    double multi1 = 1.0 / tpwgts2[1];
    for(int i = 0;i < nparts - (nparts >> 1);i++)
    {
        double *ptr = tpwgts + (nparts >> 1);
        ptr[i] = ptr[i] * multi0;
    }
    for(int i = 0;i < nparts >> 1;i++)
    {
        double *ptr = tpwgts;
        ptr[i] = ptr[i] * multi1;
    }
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_bisection, NULL);
    update_tpwgts_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;
#endif
    // printf("tpwgts: ");
    // for(int i = 0;i < nparts;i++)
    //     printf("%lf ", tpwgts[i]);
    // printf("\n");

	//	Recursive
	if(nparts > 3)
	{
        // for(int i = 0;i < nparts - (nparts >> 1);i++)
        // {
        //     double *ptr = tpwgts + (nparts >> 1);
        //     printf("%lf ", ptr[i]);
        // }
        // printf("\n");
        // exit(0);
        hunyuangraph_gpu_RecursiveBisection(hunyuangraph_admin, rgraph, nparts - (nparts >> 1), ubvec, tpwgts + (nparts >> 1), answer, fpart + (nparts >> 1), level + 1);
        hunyuangraph_gpu_RecursiveBisection(hunyuangraph_admin, lgraph, (nparts >> 1), ubvec, tpwgts, answer, fpart, level + 1);
	}
	else if(nparts == 3)
	{
		hunyuangraph_gpu_RecursiveBisection(hunyuangraph_admin, rgraph, nparts - (nparts >> 1), ubvec, tpwgts + (nparts >> 1), answer, fpart + (nparts >> 1), level + 1);
	}

	free(tpwgts2);
}

__global__ void set_label(int nvtxs, int *label)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < nvtxs)
        label[ii] = ii;
}

__global__ void set_where(int nvtxs, int *where)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < nvtxs)
    {
        where[ii] = 0;
    }
}

void hunyuangraph_gpu_initialpartition(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	// allocate memory
	// GPU已分配 vwgt, xadj, adjncy, adjwgt
	// CPU已含有 nvtxs, nedges, tvwgt
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_gpu_bisection, NULL);
#endif 
	// hunyuangraph_graph_t *t_graph = hunyuangraph_create_cpu_graph();

	// t_graph->nvtxs = graph->nvtxs;
	// t_graph->nedges = graph->nedges;
	// t_graph->tvwgt = (int *)malloc(sizeof(int));
	// t_graph->tvwgt[0] = graph->tvwgt[0];

    if(GPU_Memory_Pool)
    	graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_initialpartition: graph->cuda_where");
    else 
        cudaMalloc((void **)&graph->cuda_where, sizeof(int) * graph->nvtxs);

	// t_graph->cuda_vwgt   = (int *)lmalloc_with_check(sizeof(int) * t_graph->nvtxs, "hunyuangraph_gpu_initialpartition: t_graph->cuda_vwgt");
	// t_graph->cuda_xadj   = (int *)lmalloc_with_check(sizeof(int) * (t_graph->nvtxs + 1), "hunyuangraph_gpu_initialpartition: t_graph->cuda_xadj");
	// t_graph->cuda_adjncy = (int *)lmalloc_with_check(sizeof(int) * t_graph->nedges, "hunyuangraph_gpu_initialpartition: t_graph->cuda_adjncy");
	// t_graph->cuda_adjwgt = (int *)lmalloc_with_check(sizeof(int) * t_graph->nedges, "hunyuangraph_gpu_initialpartition: t_graph->cuda_adjwgt");
	// t_graph->cuda_label = (int *)lmalloc_with_check(sizeof(int) * t_graph->nvtxs, "hunyuangraph_gpu_initialpartition: t_graph->cuda_label");
    // t_graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * t_graph->nvtxs, "hunyuangraph_gpu_initialpartition: t_graph->cuda_where");

	// cudaMemcpy(t_graph->cuda_vwgt, graph->cuda_vwgt, sizeof(int) * t_graph->nvtxs, cudaMemcpyDeviceToDevice);
	// cudaMemcpy(t_graph->cuda_xadj, graph->cuda_xadj, sizeof(int) * (t_graph->nvtxs + 1), cudaMemcpyDeviceToDevice);
	// cudaMemcpy(t_graph->cuda_adjncy, graph->cuda_adjncy, sizeof(int) * t_graph->nedges, cudaMemcpyDeviceToDevice);
	// cudaMemcpy(t_graph->cuda_adjwgt, graph->cuda_adjwgt, sizeof(int) * t_graph->nedges, cudaMemcpyDeviceToDevice);

    if(hunyuangraph_admin->nparts < 1)
        printf("nparts(nparts < 1) is error, please input right nparts(nparts >= 1)!!!\n");
    if(hunyuangraph_admin->nparts == 1)
    {
        set_where<<<(graph->nvtxs + 127) / 128, 128>>>(graph->nvtxs, graph->cuda_where);
        return ;
    }

    // t_graph->vwgt = (int *)malloc(sizeof(int) * t_graph->nvtxs);
    // t_graph->xadj = (int *)malloc(sizeof(int) * (t_graph->nvtxs + 1));
    // t_graph->adjncy = (int *)malloc(sizeof(int) * t_graph->nedges);
    // t_graph->adjwgt = (int *)malloc(sizeof(int) * t_graph->nedges);
    // t_graph->where = (int *)malloc(sizeof(int) * t_graph->nvtxs);
    // t_graph->label = (int *)malloc(sizeof(int) * t_graph->nvtxs);
    // for(int i = 0; i < t_graph->nvtxs; i++)
    //     t_graph->label[i] = i;

    // memcpy(t_graph->vwgt, graph->vwgt, sizeof(int) * t_graph->nvtxs);
    // memcpy(t_graph->xadj, graph->xadj, sizeof(int) * (t_graph->nvtxs + 1));
    // memcpy(t_graph->adjncy, graph->adjncy, sizeof(int) * t_graph->nedges);
    // memcpy(t_graph->adjwgt, graph->adjwgt, sizeof(int) * t_graph->nedges);
    // printf("used_by_me_now=%d l=%d r=%d\n", used_by_me_now, lused, rused);
    void *record_lporinter;
    if(GPU_Memory_Pool)
        record_lporinter = record_lmove_pointer();

    if(GPU_Memory_Pool)
        graph->cuda_label = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_gpu_initialpartition: graph->cuda_label");
    else
        cudaMalloc((void **)&graph->cuda_label, sizeof(int) * graph->nvtxs);

    // set_label<<<(t_graph->nvtxs + 127) / 128, 128>>>(t_graph->nvtxs, t_graph->cuda_label);
    set_label<<<(graph->nvtxs + 127) / 128, 128>>>(graph->nvtxs, graph->cuda_label);

	double *ubvec, *tpwgts;
	ubvec  = (double *)malloc(sizeof(double));
	tpwgts = (double *)malloc(sizeof(double) * hunyuangraph_admin->nparts);
	
	ubvec[0] = (double)pow(hunyuangraph_admin->ubfactors[0], 1.0 / log(hunyuangraph_admin->nparts));
	for(int i = 0; i < hunyuangraph_admin->nparts; i++)
		tpwgts[i] = (double)hunyuangraph_admin->tpwgts[i];

#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_gpu_bisection, NULL);
    set_initgraph_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;
#endif

    // printf("hunyuangraph_gpu_RecursiveBisection begin\n");
	hunyuangraph_gpu_RecursiveBisection(hunyuangraph_admin, graph, hunyuangraph_admin->nparts, ubvec, tpwgts, graph->cuda_where, 0, 0);
    // printf("hunyuangraph_gpu_RecursiveBisection end\n");
	free(ubvec);
	free(tpwgts);

    if(GPU_Memory_Pool)
        return_lmove_pointer(record_lporinter);

    // printf("used_by_me_now=%d l=%d r=%d\n", used_by_me_now, lused, rused);
}


#endif