#ifndef _H_GPU_COARSEN
#define _H_GPU_COARSEN

#include "hunyuangraph_struct.h"
#include "hunyuangraph_GPU_match.h"
#include "hunyuangraph_GPU_contraction.h"

/*Malloc gpu coarsen graph params*/
void hunyuangraph_malloc_coarseninfo(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int level)
{
    int nvtxs = graph->nvtxs;
    int nedges = graph->nedges;

#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc, NULL);
#endif
    // cudaMalloc((void**)&graph->cuda_match,nvtxs * sizeof(int));
    if(GPU_Memory_Pool)
    {
        graph->cuda_match = (int *)rmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_coarseninfo: match");
        if(level != 0)
        {
            graph->length_vertex = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_coarseninfo: graph->length_vertex");
            graph->bin_offset = (int *)lmalloc_with_check(sizeof(int) * 15, "hunyuangraph_malloc_coarseninfo: graph->bin_offset");
            graph->bin_idx    = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_coarseninfo: graph->bin_idx");
        }
        graph->cuda_cmap = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_coarseninfo: cmap");
        graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_coarseninfo: where"); // 不可在k-way refinement再申请空间，会破坏栈的原则
    }
    else
    {
        cudaMalloc((void**)&graph->cuda_match, sizeof(int) * nvtxs);
        if(level != 0)
        {
            cudaMalloc((void**)&graph->length_vertex, sizeof(int) * nvtxs);
            cudaMalloc((void**)&graph->bin_offset, sizeof(int) * 15);
            cudaMalloc((void**)&graph->bin_idx, sizeof(int) * nvtxs);
        }
        cudaMalloc((void**)&graph->cuda_cmap, sizeof(int) * nvtxs);
        cudaMalloc((void**)&graph->cuda_where, sizeof(int) * nvtxs);
    }
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc, NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
#endif

    if(level != 0)
        graph->h_bin_offset = (int *)malloc(sizeof(int) * 15);
}

void hunyuangraph_memcpy_coarsentoinit(hunyuangraph_graph_t *graph)
{
    int nvtxs = graph->nvtxs;
    int nedges = graph->nedges;

#ifdef FIGURE10_CGRAPH
    graph->xadj = (int *)malloc(sizeof(int) * (nvtxs + 1));
    graph->vwgt = (int *)malloc(sizeof(int) * nvtxs);
    graph->adjncy = (int *)malloc(sizeof(int) * nedges);
    graph->adjwgt = (int *)malloc(sizeof(int) * nedges);
    // graph->where = (int *)malloc(sizeof(int) * nvtxs);

    // cudaDeviceSynchronize();
    // gettimeofday(&begin_coarsen_memcpy, NULL);
    cudaMemcpy(graph->xadj, graph->cuda_xadj, (nvtxs + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(graph->vwgt, graph->cuda_vwgt, nvtxs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(graph->adjncy, graph->cuda_adjncy, nedges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(graph->adjwgt, graph->cuda_adjwgt, nedges * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // gettimeofday(&end_coarsen_memcpy, NULL);
    // coarsen_memcpy += (end_coarsen_memcpy.tv_sec - begin_coarsen_memcpy.tv_sec) * 1000 + (end_coarsen_memcpy.tv_usec - begin_coarsen_memcpy.tv_usec) / 1000.0;
#endif

    int *length_bin, *bin_size;
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_malloc, NULL);
#endif
    if(GPU_Memory_Pool)
	{
		graph->length_vertex = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_memcpy_coarsentoinit: graph->length_vertex");
        graph->bin_offset = (int *)lmalloc_with_check(sizeof(int) * 15, "hunyuangraph_memcpy_coarsentoinit: graph->bin_offset");
        graph->bin_idx    = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_memcpy_coarsentoinit: graph->bin_idx");
        length_bin = (int *)rmalloc_with_check(sizeof(int) * 14, "hunyuangraph_memcpy_coarsentoinit: length_bin");
		bin_size   = (int *)rmalloc_with_check(sizeof(int) * 14, "hunyuangraph_memcpy_coarsentoinit: bin_size");
	}
	else
	{
		cudaMalloc((void**)&graph->length_vertex, sizeof(int) * nvtxs);
        cudaMalloc((void**)&graph->bin_offset, sizeof(int) * 15);
        cudaMalloc((void**)&graph->bin_idx, sizeof(int) * nvtxs);
        cudaMalloc((void**)&length_bin, sizeof(int) * 14);
		cudaMalloc((void**)&bin_size, sizeof(int) * 14);
	}
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_malloc, NULL);
    coarsen_malloc += (end_malloc.tv_sec - begin_malloc.tv_sec) * 1000 + (end_malloc.tv_usec - begin_malloc.tv_usec) / 1000.0;
#endif

	init_bin<<<1, 14>>>(14, length_bin);
	init_bin<<<1, 14>>>(15, graph->bin_offset);
	init_bin<<<1, 14>>>(14, bin_size);

	check_length<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->length_vertex, length_bin);

	cudaMemcpy(&graph->bin_offset[1], length_bin, sizeof(int) * 14, cudaMemcpyDeviceToDevice);

	if(GPU_Memory_Pool)
	{
		prefixsum(graph->bin_offset, graph->bin_offset, 15, prefixsum_blocksize, 1);	//0:lmalloc,1:rmalloc
	}
	else
	{
		thrust::inclusive_scan(thrust::device,graph-> bin_offset, graph->bin_offset + 15, graph->bin_offset);
	}

	set_bin<<<(nvtxs + 127) / 128, 128>>>(nvtxs, graph->length_vertex, bin_size, graph->bin_offset, graph->bin_idx);
    
    graph->h_bin_offset = (int *)malloc(sizeof(int) * 15);
    cudaMemcpy(graph->h_bin_offset, graph->bin_offset, sizeof(int) * 15, cudaMemcpyDeviceToHost);

#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&begin_free,NULL);
#endif
    if(GPU_Memory_Pool)
	{
		rfree_with_check(bin_size, sizeof(int) * 14, "hunyuangraph_memcpy_coarsentoinit: bin_size");
		rfree_with_check(length_bin, sizeof(int) * 14, "hunyuangraph_memcpy_coarsentoinit: length_bin");
	}
	else 
	{
		cudaFree(length_bin);
		cudaFree(bin_size);
	}
#ifdef TIMER
    cudaDeviceSynchronize();
    gettimeofday(&end_free,NULL);
    coarsen_free += (end_free.tv_sec - begin_free.tv_sec) * 1000 + (end_free.tv_usec - begin_free.tv_usec) / 1000.0;
#endif
}

__global__ void exam_cvwgt(int nvtxs, int nparts, int tvwgt, int *vwgt, int *flag)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    if(ii >= nvtxs)
        return ;

    int max_allow = tvwgt / nparts * 1.03;
    // if(ii == 0)
    //     printf("tvwgt=%d nparts=%d max_allow=%d\n", tvwgt, nparts, max_allow);
    if(vwgt[ii] > max_allow)
    {
        printf("vertex=%d vwgt=%d max_allow=%d\n", ii, vwgt[ii], max_allow);
        flag[0] = 1;
    }
}

/*Gpu multilevel coarsen*/
hunyuangraph_graph_t *hunyuangarph_coarsen(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int *level)
{
    hunyuangraph_admin->maxvwgt = 1.5 * graph->tvwgt[0] / hunyuangraph_admin->Coarsen_threshold;

    // printf("level %2d: nvtxs %10d nedges %10d nedges/nvtxs=%7.2lf adjwgtsum %12d\n",level, graph->nvtxs, graph->nedges, (double)graph->nedges / (double)graph->nvtxs, compute_graph_adjwgtsum_gpu(graph));
    // printf("         0|         1|         2|         3|         4|         5|         6|         7|         8|         9|        10|        11|        12|        13    \n");
    // printf("        =0|       <=2|       <=4|       <=8|      <=16|      <=32|      <=64|     <=128|     <=256|     <=512|    <=1024|    <=2048|    <=4096|     >4096    \n");
    do
    {
        hunyuangraph_malloc_coarseninfo(hunyuangraph_admin, graph, level[0]);
        // printf("hunyuangraph_malloc_coarseninfo end\n");

#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&begin_part_match, NULL);
#endif
        hunyuangraph_graph_t *cgraph = hunyuangraph_gpu_match(hunyuangraph_admin, graph, level[0]);
#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&end_part_match, NULL);
        part_match += (end_part_match.tv_sec - begin_part_match.tv_sec) * 1000 + (end_part_match.tv_usec - begin_part_match.tv_usec) / 1000.0;
#endif
        // printf("hunyuangraph_gpu_match end\n");

#ifdef TIMER
        cudaDeviceSynchronize();
        gettimeofday(&begin_part_contruction, NULL);
#endif
        hunyuangraph_gpu_create_cgraph(hunyuangraph_admin, graph, cgraph);
#ifdef TIMER
       cudaDeviceSynchronize();
        gettimeofday(&end_part_contruction, NULL);
        part_contruction += (end_part_contruction.tv_sec - begin_part_contruction.tv_sec) * 1000 + (end_part_contruction.tv_usec - begin_part_contruction.tv_usec) / 1000.0;
#endif

#ifdef FIGURE9_TIME
        cudaDeviceSynchronize();
        gettimeofday(&end_part_coarsen, NULL);
        hunyuangraph_admin->time_coarsen[level[0]] = (end_part_coarsen.tv_sec - begin_part_coarsen.tv_sec) * 1000 + (end_part_coarsen.tv_usec - begin_part_coarsen.tv_usec) / 1000.0;
#endif
        graph = graph->coarser;
        level[0]++;

        // int h_flag, *d_flag;
        // h_flag = 0;
        // if(GPU_Memory_Pool)
        //     d_flag = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_GPU_coarsen: d_flag");
        // else 
        //     cudaMalloc((void**)&d_flag, sizeof(int));
        // cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);
        // exam_cvwgt<<<(graph->nvtxs + 127) / 128, 128>>>(graph->nvtxs, hunyuangraph_admin->nparts, graph->tvwgt[0], graph->cuda_vwgt, d_flag);
        // cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        // if(GPU_Memory_Pool)
        //     lfree_with_check(d_flag, sizeof(int), "hunyuangraph_GPU_coarsen: d_flag");
        // else 
        //     cudaFree(d_flag);
        // if(h_flag == 1)
        // {
        //     printf("hunyuangraph_GPU_coarsen: warning: coarsen graph has a vertex with weight > nvtxs / nparts * 1.03\n");
        //     // break;
        // }

        // if(level[0] == 22)
        //     break;

        // cudaDeviceSynchronize();
        // print_graph<<<1, 1>>>(graph->nvtxs, graph->nedges, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt);
        // cudaDeviceSynchronize();

        // cudaDeviceSynchronize();
        // check_graph<<<(graph->nvtxs + 127) / 128, 128>>>(graph->nvtxs, graph->nedges, graph->cuda_vwgt, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt);
	    // cudaDeviceSynchronize();

        // if(level[0] < 2)
        // printf("level %2d: time: %10.3lf\n", level[0], part_coarsen);

        // break;
#ifdef FIGURE9_SUM
        printf("level %2d: nvtxs %10d nedges %10d nedges/nvtxs=%7.2lf adjwgtsum %12d\n", level[0], graph->nvtxs, graph->nedges, (double)graph->nedges / (double)graph->nvtxs, compute_graph_adjwgtsum_gpu(graph));
#endif

    } while (
        graph->nvtxs > hunyuangraph_admin->Coarsen_threshold &&
        graph->nvtxs < 0.85 * graph->finer->nvtxs &&
        graph->nedges > graph->nvtxs / 2);
    // printf("do while end\n");

    hunyuangraph_memcpy_coarsentoinit(graph);

    return graph;
}

#endif