#ifndef _H_GPU_UNCOARSEN
#define _H_GPU_UNCOARSEN

#include "hunyuangraph_struct.h"
#include "hunyuangraph_GPU_memory.h"
#include "hunyuangraph_GPU_coarsen.h"
#include "hunyuangraph_GPU_krefine.h"

/*CUDA-init pwgts array*/
__global__ void initpwgts(int *cuda_pwgts, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nparts)
		cuda_pwgts[ii] = 0;
}

/*Compute sum of pwgts*/
__global__ void calculateSum(int nvtxs, int nparts, int *pwgts, int *where, int *vwgt)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int cache_d[];
	for (int i = threadIdx.x; i < nparts; i += 128)
		cache_d[i] = 0;
	__syncthreads();

	int t;
	if (ii < nvtxs)
	{
		t = where[ii];
		atomicAdd(&cache_d[t], vwgt[ii]);
	}
	__syncthreads();

	int val;
	for (int i = threadIdx.x; i < nparts; i += 128)
	{
		val = cache_d[i];
		if (val > 0)
		{
			atomicAdd(&pwgts[i], val);
		}
	}
}

/*CUDA-init pwgts array*/
__global__ void inittpwgts(float *tpwgts, float temp, int nparts)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nparts)
		tpwgts[ii] = temp;
}

/*Malloc initial partition phase to refine phase params*/
void Mallocinit_refineinfo(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;
	// int num = 0;

	// printf("hunyuangraph_malloc_refineinfo nvtxs=%d\n", nvtxs);

	// cudaMalloc((void**)&graph->cuda_where,nvtxs * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_bnd,nvtxs * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_bndnum,sizeof(int));
	// cudaMalloc((void**)&graph->cuda_pwgts,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_tpwgts,nparts * sizeof(float));
	// cudaMalloc((void**)&graph->cuda_maxwgt,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_minwgt,nparts * sizeof(int));


	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
	if(GPU_Memory_Pool)
	{
		// graph->cuda_where = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "Mallocinit_refineinfo: where");
		// graph->cuda_bnd = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "Mallocinit_refineinfo: bnd");
		graph->cuda_pwgts = (int *)lmalloc_with_check(sizeof(int) * nparts, "Mallocinit_refineinfo: pwgts");
		graph->cuda_tpwgts = (float *)lmalloc_with_check(sizeof(float) * nparts, "Mallocinit_refineinfo: tpwgts");
		graph->cuda_maxwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "Mallocinit_refineinfo: maxwgt");
		graph->cuda_poverload = (int *)lmalloc_with_check(sizeof(int) * nparts, "Mallocinit_refineinfo: cuda_poverload");
		// graph->cuda_bndnum = (int *)lmalloc_with_check(sizeof(int), "Mallocinit_refineinfo: bndnum");

		graph->cuda_balance = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_malloc_refineinfo: cuda_balance");
		// graph->cuda_bn = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_bn");
		graph->cuda_select = (char *)lmalloc_with_check(sizeof(char) * graph->nvtxs,"cuda_select");
		graph->cuda_moved = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_moved");
		graph->cuda_to = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_to");
		graph->cuda_gain = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_gain");
		graph->cuda_kway_bin = (int *)lmalloc_with_check(sizeof(int) * (nparts + 1), "Mallocinit_refineinfo: cuda_kway_bin");
		graph->cuda_kway_idx = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_kway_idx");
		graph->cuda_kway_loss = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_kway_loss");
		// graph->cuda_csr = (int *)lmalloc_with_check(sizeof(int) * 2, "Mallocinit_refineinfo: cuda_csr");
		// graph->cuda_que = (int *)lmalloc_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2, "Mallocinit_refineinfo: cuda_que");
	}
	else
	{
		cudaMalloc((void **)&graph->cuda_pwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_tpwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_maxwgt, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_poverload, sizeof(int) * nparts);
		// cudaMalloc((void **)&graph->cuda_bndnum, sizeof(int));
		cudaMalloc((void **)&graph->cuda_balance, sizeof(int));
		cudaMalloc((void **)&graph->cuda_select, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_moved, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_to, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_gain, sizeof(int) * graph->nvtxs);

		cudaMalloc((void **)&graph->cuda_kway_bin, sizeof(int) * (nparts + 1));
		cudaMalloc((void **)&graph->cuda_kway_idx, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_kway_loss, sizeof(int) * graph->nvtxs);
	}
	graph->h_kway_bin = (int *)malloc(sizeof(int) * (nparts + 1));
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_gpu_malloc += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

	// cudaMemcpy(graph->cuda_where, graph->where, nvtxs * sizeof(int), cudaMemcpyHostToDevice);
	// cudaMemcpy(graph->cuda_bndnum, &num, sizeof(int), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
	initpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_pwgts, nparts);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_initpwgts += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
	calculateSum<<<(nvtxs + 127) / 128, 128, nparts * sizeof(int)>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_where, graph->cuda_vwgt);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_calculateSum += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

	// inittpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_tpwgts, hunyuangraph_admin->tpwgts[0], nparts);
	cudaMemcpy(graph->cuda_tpwgts,hunyuangraph_admin->tpwgts,nparts * sizeof(float),cudaMemcpyHostToDevice);
}

/*Malloc refine params*/
void hunyuangraph_malloc_refineinfo(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;
	// int num = 0;

	// printf("hunyuangraph_malloc_refineinfo nvtxs=%d\n", nvtxs);

	// cudaMalloc((void**)&graph->cuda_bnd,nvtxs * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_bndnum,sizeof(int));
	// cudaMalloc((void**)&graph->cuda_pwgts,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_tpwgts,nparts * sizeof(float));
	// cudaMalloc((void**)&graph->cuda_maxwgt,nparts * sizeof(int));
	// cudaMalloc((void**)&graph->cuda_minwgt,nparts * sizeof(int));

	if(GPU_Memory_Pool)
	{
		// graph->cuda_bnd = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_refineinfo: bnd");
		graph->cuda_pwgts = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_malloc_refineinfo: pwgts");
		graph->cuda_tpwgts = (float *)lmalloc_with_check(sizeof(float) * nparts, "hunyuangraph_malloc_refineinfo: tpwgts");
		graph->cuda_maxwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_malloc_refineinfo: maxwgt");
		graph->cuda_poverload = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_malloc_refineinfo: cuda_poverload");
		// graph->cuda_bndnum = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_malloc_refineinfo: bndnum");

		graph->cuda_balance = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_malloc_refineinfo: cuda_balance");
		// graph->cuda_bn = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_malloc_refineinfo: cuda_bn");
		graph->cuda_select = (char *)lmalloc_with_check(sizeof(char) * graph->nvtxs,"cuda_select");
		graph->cuda_to = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_to");
		graph->cuda_gain = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs, "Mallocinit_refineinfo: cuda_gain");
		// graph->cuda_csr = (int *)lmalloc_with_check(sizeof(int) * 2, "hunyuangraph_malloc_refineinfo: cuda_csr");
		// graph->cuda_que = (int *)lmalloc_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2, "hunyuangraph_malloc_refineinfo: cuda_que");
	}
	else
	{
		cudaMalloc((void **)&graph->cuda_pwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_tpwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_maxwgt, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_poverload, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_balance, sizeof(int));
		cudaMalloc((void **)&graph->cuda_select, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_to, sizeof(int) * graph->nvtxs);
		cudaMalloc((void **)&graph->cuda_gain, sizeof(int) * graph->nvtxs);
	}
	// cudaMemcpy(graph->cuda_bndnum, &num, sizeof(int), cudaMemcpyHostToDevice);

	initpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_pwgts, nparts);

	calculateSum<<<(nvtxs + 127) / 128, 128, nparts * sizeof(int)>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_where, graph->cuda_vwgt);

	// inittpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_tpwgts, hunyuangraph_admin->tpwgts[0], nparts);
	cudaMemcpy(graph->cuda_tpwgts,hunyuangraph_admin->tpwgts,nparts * sizeof(float),cudaMemcpyHostToDevice);
}

/*CUDA-kway parjection*/
__global__ void projectback(int *where, int *cwhere, int *cmap, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
	{
		int t = cmap[ii];
		where[ii] = cwhere[t];
	}
}

/*Kway parjection*/
void hunyuangraph_kway_project(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	hunyuangraph_graph_t *cgraph = graph->coarser;

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	projectback<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_where, cgraph->cuda_where, graph->cuda_cmap, nvtxs);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_projectback += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif
}

/*Free graph uncoarsening phase params*/
void hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
	if(GPU_Memory_Pool)
	{
		lfree_with_check((void *)graph->cuda_kway_loss, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_gain");			// cuda_kway_loss
		lfree_with_check((void *)graph->cuda_kway_idx, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_kway_idx");					// cuda_kway_idx
		lfree_with_check((void *)graph->cuda_kway_bin, sizeof(int) * (hunyuangraph_admin->nparts + 1), "hunyuangraph_uncoarsen_free_krefine: cuda_kway_bin");					// cuda_kway_bin
		// printf("hunyuangraph_uncoarsen_free_krefine nvtxs=%d\n", graph->nvtxs);
		// lfree_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2, "hunyuangraph_uncoarsen_free_krefine: cuda_que");	// cuda_que
		// lfree_with_check(sizeof(int) * 2, "cuda_csr");																	 	// cuda_csr
		lfree_with_check((void *)graph->cuda_gain, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_gain");					// cuda_gain
		lfree_with_check((void *)graph->cuda_to, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_to");						// cuda_to
		lfree_with_check((void *)graph->cuda_moved, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_moved");					// cuda_moved
		lfree_with_check((void *)graph->cuda_select, sizeof(char) * graph->nvtxs, "cuda_select");													// cuda_select
		// lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: cuda_bn");					 	// cuda_bn
		lfree_with_check((void *)graph->cuda_balance, sizeof(int), "hunyuangraph_uncoarsen_free_krefine: cuda_balance");							// cuda_balance
		// lfree_with_check((void *)graph->cuda_bndnum, sizeof(int), "hunyuangraph_uncoarsen_free_krefine: bndnum");									 	// bndnum
		lfree_with_check((void *)graph->cuda_poverload, sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: cuda_poverload");		// cuda_poverload
		lfree_with_check((void *)graph->cuda_maxwgt, sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: maxwgt");		// maxwgt
		lfree_with_check((void *)graph->cuda_tpwgts, sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: tpwgts");		// tpwgts
		lfree_with_check((void *)graph->cuda_pwgts, sizeof(int) * hunyuangraph_admin->nparts, "hunyuangraph_uncoarsen_free_krefine: pwgts");		// pwgts
		// lfree_with_check(sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_krefine: bnd");						 	// bnd
	}
	else
	{
		cudaFree(graph->cuda_kway_loss);
		cudaFree(graph->cuda_kway_idx);
		cudaFree(graph->cuda_kway_bin);
		cudaFree(graph->cuda_gain);
		cudaFree(graph->cuda_to);
		cudaFree(graph->cuda_moved);
		cudaFree(graph->cuda_select);
		cudaFree(graph->cuda_balance);
		cudaFree(graph->cuda_poverload);
		cudaFree(graph->cuda_maxwgt);
		cudaFree(graph->cuda_tpwgts);
		cudaFree(graph->cuda_pwgts);
	}
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_gpu_free += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
}

void hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	// printf("hunyuangraph_uncoarsen_free_coarsen nvtxs=%d\n", graph->nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
	if(GPU_Memory_Pool)
	{
		lfree_with_check((void *)graph->cuda_where, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: where");						// where
		if (graph->cuda_cmap != NULL)
			lfree_with_check((void *)graph->cuda_cmap, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: cmap");					// cmap;
		if (graph->bin_idx != NULL)
			lfree_with_check((void *)graph->bin_idx, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: bin_idx");					//	bin_idx
		if (graph->bin_offset != NULL)
			lfree_with_check((void *)graph->bin_offset, sizeof(int) * 15, "hunyuangraph_uncoarsen_free_coarsen: bin_offset");						//	bin_offset
		if (graph->length_vertex != NULL)
			lfree_with_check((void *)graph->length_vertex, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: length_vertex");		//	length_vertex
		lfree_with_check((void *)graph->cuda_adjwgt, sizeof(int) * graph->nedges, "hunyuangraph_uncoarsen_free_coarsen: adjwgt");					// adjwgt
		lfree_with_check((void *)graph->cuda_adjncy, sizeof(int) * graph->nedges, "hunyuangraph_uncoarsen_free_coarsen: adjncy");					// adjncy
		lfree_with_check((void *)graph->cuda_xadj, sizeof(int) * (graph->nvtxs + 1), "hunyuangraph_uncoarsen_free_coarsen: xadj");					// xadj
		lfree_with_check((void *)graph->cuda_vwgt, sizeof(int) * graph->nvtxs, "hunyuangraph_uncoarsen_free_coarsen: vwgt");						// vwgt
	}
	else
	{
		cudaFree(graph->cuda_where);
		if (graph->cuda_cmap != NULL)
			cudaFree(graph->cuda_cmap);
		cudaFree(graph->bin_idx);
		cudaFree(graph->bin_offset);
		cudaFree(graph->length_vertex);
		cudaFree(graph->cuda_adjwgt);
		cudaFree(graph->cuda_adjncy);
		cudaFree(graph->cuda_xadj);
		cudaFree(graph->cuda_vwgt);
	}

	if(graph->h_bin_offset != NULL)
		free(graph->h_bin_offset);
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_gpu_free += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

}

void hunyuangraph_GPU_uncoarsen(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, hunyuangraph_graph_t *cgraph)
{
	Mallocinit_refineinfo(hunyuangraph_admin, cgraph);

	hunyuangraph_k_refinement(hunyuangraph_admin, cgraph);
	// hunyuangraph_k_refinement_me(hunyuangraph_admin,cgraph);

	hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin, cgraph);

	for (int i = 0;; i++)
	{
		if (cgraph != graph)
		{
			cgraph = cgraph->finer;

			// cudaMalloc((void**)&cgraph->cuda_where, cgraph->nvtxs * sizeof(int));

			hunyuangraph_kway_project(hunyuangraph_admin, cgraph);

			hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin, cgraph->coarser);

			hunyuangraph_malloc_refineinfo(hunyuangraph_admin, cgraph);

			hunyuangraph_k_refinement(hunyuangraph_admin, cgraph);
			// hunyuangraph_k_refinement_me(hunyuangraph_admin,cgraph);

			hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin, cgraph);

			// hunyuangraph_free_uncoarsen(hunyuangraph_admin, cgraph->coarser);
		}
		else
			break;
	}
}

void hunyuangraph_GPU_uncoarsen_SC25(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, hunyuangraph_graph_t *cgraph, int *level)
{
	Mallocinit_refineinfo(hunyuangraph_admin, cgraph);

	// int edgecut;
	// cudaDeviceSynchronize();
	compute_edgecut_gpu(cgraph->nvtxs, &cgraph->mincut, cgraph->cuda_xadj, cgraph->cuda_adjncy, cgraph->cuda_adjwgt, cgraph->cuda_where);
	// cudaDeviceSynchronize();
	// printf("edgecut:%10d\n", edgecut);

	// printf("level=%d nvtxs=%d nedges=%d \n", level[0], cgraph->nvtxs, cgraph->nedges);
	// int h_flag, *d_flag;
	// h_flag = 0;
	// d_flag = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_GPU_coarsen: d_flag");
	// cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);
	// exam_cvwgt<<<(cgraph->nvtxs + 127) / 128, 128>>>(cgraph->nvtxs, hunyuangraph_admin->nparts, cgraph->tvwgt[0], cgraph->cuda_vwgt, d_flag);
	// cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
	// lfree_with_check(d_flag, sizeof(int), "hunyuangraph_GPU_coarsen: d_flag");
	// if(h_flag == 1)
	// {
	//     printf("hunyuangraph_GPU_coarsen: warning: coarsen graph has a vertex with weight > nvtxs / nparts * 1.03\n");
	//     // break;
	// }
	hunyuangraph_k_refinement_SC25(hunyuangraph_admin, cgraph, level);
	// hunyuangraph_k_refinement_me(hunyuangraph_admin,cgraph);
	// float imbalance = hunyuangraph_compute_imbalance_cpu(cgraph, cgraph->cuda_where, hunyuangraph_admin->nparts);
	// exit(0);

	hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin, cgraph);

	for (int i = 0;; i++)
	{
		if (cgraph != graph)
		{
			cgraph = cgraph->finer;
			level[0]--;

			// int h_flag, *d_flag;
			// h_flag = 0;
			// d_flag = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_GPU_coarsen: d_flag");
			// cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);
			// exam_cvwgt<<<(graph->nvtxs + 127) / 128, 128>>>(graph->nvtxs, hunyuangraph_admin->nparts, graph->tvwgt[0], graph->cuda_vwgt, d_flag);
			// cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
			// lfree_with_check(d_flag, sizeof(int), "hunyuangraph_GPU_coarsen: d_flag");
			// if(h_flag == 1)
			// {
			//     printf("hunyuangraph_GPU_coarsen: warning: coarsen graph has a vertex with weight > nvtxs / nparts * 1.03\n");
			//     // break;
			// }

			// cudaMalloc((void**)&cgraph->cuda_where, cgraph->nvtxs * sizeof(int));

			hunyuangraph_kway_project(hunyuangraph_admin, cgraph);

			hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin, cgraph->coarser);

			// hunyuangraph_malloc_refineinfo(hunyuangraph_admin, cgraph);
			Mallocinit_refineinfo(hunyuangraph_admin, cgraph);

			// if(cgraph->nvtxs >= 1000)
			// 	exit(0);

			// printf("level=%d nvtxs=%d nedges=%d \n", level[0], cgraph->nvtxs, cgraph->nedges);
			hunyuangraph_k_refinement_SC25(hunyuangraph_admin, cgraph, level);

			// if(level[0] == 20)
			// exit(0);
			// hunyuangraph_k_refinement(hunyuangraph_admin, cgraph);
			// hunyuangraph_k_refinement_me(hunyuangraph_admin,cgraph);

			hunyuangraph_uncoarsen_free_krefine(hunyuangraph_admin, cgraph);

			// hunyuangraph_free_uncoarsen(hunyuangraph_admin, cgraph->coarser);
		}
		else
			break;
	}
}

__global__ void compute_opt_max_pwgts(int nparts, int tvwgt, int *opt_pwgts, int *maxwgt)
{
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii >= nparts)
		return ;
	
	int val = (tvwgt / nparts);
	opt_pwgts[ii] = val;
	maxwgt[ii] = val * IMB;
}

void hunyuangraph_malloc_krefine(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nparts = hunyuangraph_admin->nparts;
	int nvtxs = graph->nvtxs;

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		graph->cuda_opt_pwgts = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_opt_pwgts");
		graph->cuda_poverload = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_poverload");
		graph->cuda_pwgts = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_pwgts");
		graph->cuda_maxwgt = (int *)lmalloc_with_check(sizeof(int) * nparts, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_maxwgt");
		graph->gain_offset = (int *)lmalloc_with_check(sizeof(int) * (nvtxs + 1), "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->gain_offset");
		graph->cuda_balance = (int *)lmalloc_with_check(sizeof(int), "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_balance");
		graph->dest_part = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->dest_part");
		graph->cuda_gain = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_gain");
		graph->cuda_select = (char *)lmalloc_with_check(sizeof(char) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_select");
		graph->lock = (char *)lmalloc_with_check(sizeof(char) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->lock");
		graph->pos_move = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->pos_move");
	}
	else
	{
		cudaMalloc((void **)&graph->cuda_opt_pwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_poverload, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_pwgts, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->cuda_maxwgt, sizeof(int) * nparts);
		cudaMalloc((void **)&graph->gain_offset, sizeof(int) * (nvtxs + 1));
		cudaMalloc((void **)&graph->cuda_balance, sizeof(int));
		cudaMalloc((void **)&graph->dest_part, sizeof(int) * nvtxs);
		cudaMalloc((void **)&graph->cuda_gain, sizeof(int) * nvtxs);
		cudaMalloc((void **)&graph->cuda_select, sizeof(char) * nvtxs);
		cudaMalloc((void **)&graph->lock, sizeof(char) * nvtxs);
		cudaMalloc((void **)&graph->pos_move, sizeof(int) * nvtxs);
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_gpu_malloc += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;

	// graph->h_gain_bin = (int *)malloc(sizeof(int) * 13);

	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	init_val<<<(nparts + 31) / 32, 32>>>(nparts, 0, graph->cuda_pwgts);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_initpwgts += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif

	compute_opt_max_pwgts<<<(nparts + 31) / 32, 32>>>(nparts, graph->tvwgt[0], graph->cuda_opt_pwgts, graph->cuda_maxwgt);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	calculateSum<<<(nvtxs + 127) / 128, 128, nparts * sizeof(int)>>>(nvtxs, nparts, graph->cuda_pwgts, graph->cuda_where, graph->cuda_vwgt);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_calculateSum += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif

	// inittpwgts<<<nparts / 32 + 1, 32>>>(graph->cuda_tpwgts, hunyuangraph_admin->tpwgts[0], nparts);
	// cudaMemcpy(graph->cuda_tpwgts, hunyuangraph_admin->tpwgts, nparts * sizeof(float), cudaMemcpyHostToDevice);
}

void hunyuangraph_free_krefine(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	int nparts = hunyuangraph_admin->nparts;

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_gpu_kway, NULL);
#endif
	if(GPU_Memory_Pool)
	{
		lfree_with_check(graph->pos_move, sizeof(int) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->pos_move");
		lfree_with_check(graph->lock, sizeof(char) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->lock");
		lfree_with_check(graph->cuda_select, sizeof(char) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_select");
		lfree_with_check(graph->cuda_gain, sizeof(int) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_gain");
		lfree_with_check(graph->dest_part, sizeof(int) * nvtxs, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->dest_part");
		lfree_with_check(graph->cuda_balance, sizeof(int), "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_balance");
		lfree_with_check(graph->gain_offset, sizeof(int) * (nvtxs + 1), "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->gain_offset");
		lfree_with_check(graph->cuda_maxwgt, sizeof(int) * nparts, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_maxwgt");
		lfree_with_check(graph->cuda_pwgts, sizeof(int) * nparts, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_pwgts");
		lfree_with_check(graph->cuda_poverload, sizeof(int) * nparts, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_poverload");
		lfree_with_check(graph->cuda_opt_pwgts, sizeof(int) * nparts, "hunyuangraph_GPU_uncoarsen_SC25_copy: graph->cuda_opt_pwgts");
	}
	else
	{
		cudaFree(graph->pos_move);
		cudaFree(graph->lock);
		cudaFree(graph->cuda_select);
		cudaFree(graph->cuda_gain);
		cudaFree(graph->dest_part);
		cudaFree(graph->cuda_balance);
		cudaFree(graph->gain_offset);
		cudaFree(graph->cuda_maxwgt);
		cudaFree(graph->cuda_pwgts);
		cudaFree(graph->cuda_poverload);
		cudaFree(graph->cuda_opt_pwgts);
	}
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_gpu_kway, NULL);
	uncoarsen_gpu_free += (end_gpu_kway.tv_sec - begin_gpu_kway.tv_sec) * 1000.0 + (end_gpu_kway.tv_usec - begin_gpu_kway.tv_usec) / 1000.0;
#endif
	// free(graph->h_gain_bin);
}

void hunyuangraph_GPU_uncoarsen_SC25_copy(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, hunyuangraph_graph_t *cgraph, int *level)
{
	int ori = level[0];

	compute_edgecut_gpu(cgraph->nvtxs, &cgraph->mincut, cgraph->cuda_xadj, cgraph->cuda_adjncy, cgraph->cuda_adjwgt, cgraph->cuda_where);

	while(level[0] >= 0)
	{
		hunyuangraph_malloc_krefine(hunyuangraph_admin, cgraph);

		// printf("k_refine begin\n");
		k_refine(hunyuangraph_admin, cgraph, level);
		// printf("k_refine end\n");

		hunyuangraph_free_krefine(hunyuangraph_admin, cgraph);

		// if(level[0] < ori - 1)
			// exit(0);

		// printf("level=%10d\n", level[0]);
		if(level[0] == 0)
			break;
		
		cgraph = cgraph->finer;
		cgraph->mincut = cgraph->coarser->mincut;
		
		hunyuangraph_kway_project(hunyuangraph_admin, cgraph);

		hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin, cgraph->coarser);
		
		level[0]--;
	}
}

#endif