#ifndef _H_GRAPH
#define _H_GRAPH

#include "hunyuangraph_struct.h"
#include "hunyuangraph_common.h"
#include "hunyuangraph_GPU_memory.h"

/*Set graph params*/
void hunyuangraph_init_cpu_graph(hunyuangraph_graph_t *graph)
{
	memset((void *)graph, 0, sizeof(hunyuangraph_graph_t));
	graph->nvtxs = -1;
	graph->nedges = -1;
	graph->xadj = NULL;
	graph->vwgt = NULL;
	graph->adjncy = NULL;
	graph->adjwgt = NULL;
	graph->label = NULL;
	graph->cmap = NULL;
	graph->tvwgt = NULL;
	graph->tvwgt_reverse = NULL;
	graph->where = NULL;
	graph->pwgts = NULL;
	graph->mincut = -1;
	graph->nbnd = -1;
	graph->id = NULL;
	graph->ed = NULL;
	graph->bndptr = NULL;
	graph->bndlist = NULL;
	graph->coarser = NULL;
	graph->finer = NULL;
	graph->h_bin_offset = NULL;

	//GPU
	graph->cuda_vwgt = NULL;
	graph->cuda_xadj = NULL;
	graph->cuda_adjncy = NULL;
	graph->cuda_adjwgt = NULL;
	graph->cuda_match = NULL;
	graph->length_vertex = NULL;
	graph->bin_offset = NULL;
	graph->bin_idx = NULL;
	graph->cuda_cmap = NULL;
	graph->cuda_where = NULL;

	graph->cuda_pwgts = NULL;
	graph->cuda_tpwgts = NULL;
	graph->cuda_maxwgt = NULL;
	graph->cuda_poverload = NULL;
	graph->cuda_balance = NULL;
	graph->cuda_select = NULL;
	graph->cuda_to = NULL;
	graph->cuda_gain = NULL;
}

/*Malloc graph*/
hunyuangraph_graph_t *hunyuangraph_create_cpu_graph(void)
{
  hunyuangraph_graph_t *graph = (hunyuangraph_graph_t *)malloc(sizeof(hunyuangraph_graph_t));
  hunyuangraph_init_cpu_graph(graph);
  return graph;
}

/*Set graph tvwgt value*/
void hunyuangraph_set_graph_tvwgt(hunyuangraph_graph_t *graph)
{
  if (graph->tvwgt == NULL)
  {
    graph->tvwgt = (int *)malloc(sizeof(int));
  }

  if (graph->tvwgt_reverse == NULL)
  {
    graph->tvwgt_reverse = (float *)malloc(sizeof(float));
  }

  graph->tvwgt[0] = hunyuangraph_int_sum(graph->nvtxs, graph->vwgt);
  graph->tvwgt_reverse[0] = 1.0 / (graph->tvwgt[0] > 0 ? graph->tvwgt[0] : 1);
}

/*Set graph vertex label*/
void hunyuangraph_set_graph_label(hunyuangraph_graph_t *graph)
{
  if (graph->label == NULL)
  {
    graph->label = (int *)malloc(sizeof(int) * (graph->nvtxs));
  }

  for (int i = 0; i < graph->nvtxs; i++)
    graph->label[i] = i;
}

/*Set graph information*/
hunyuangraph_graph_t *hunyuangraph_set_graph(hunyuangraph_admin_t *hunyuangraph_admin, int nvtxs, int *xadj, int *adjncy, int *vwgt, int *adjwgt, int *tvwgt)
{
  hunyuangraph_graph_t *graph = hunyuangraph_create_cpu_graph();

  graph->nvtxs = nvtxs;
  graph->nedges = xadj[nvtxs];
  graph->xadj = xadj;
  graph->adjncy = adjncy;

  graph->vwgt = vwgt;
  graph->adjwgt = adjwgt;

  graph->tvwgt = (int *)malloc(sizeof(int));
  graph->tvwgt_reverse = (float *)malloc(sizeof(float));

  graph->tvwgt[0] = tvwgt[0];
  graph->tvwgt_reverse[0] = 1.0 / (graph->tvwgt[0] > 0 ? graph->tvwgt[0] : 1);

  // init label spend much time
  //  graph->label = (int *)malloc(sizeof(int) * (graph->nvtxs));
  //  for(int i = 0;i < graph->nvtxs;i++)
  //      graph->label[i] = i;

  return graph;
}

hunyuangraph_graph_t *hunyuangraph_set_first_level_graph(int nvtxs, int *xadj, int *adjncy, int *vwgt, int *adjwgt)
{
  int i;
  hunyuangraph_graph_t *graph;

  graph = hunyuangraph_create_cpu_graph();
  graph->nvtxs = nvtxs;
  graph->nedges = xadj[nvtxs];
  graph->xadj = xadj;
  graph->adjncy = adjncy;

  graph->vwgt = vwgt;
  graph->adjwgt = adjwgt;

  graph->tvwgt = (int *)malloc(sizeof(int));
  graph->tvwgt_reverse = (float *)malloc(sizeof(float));
  graph->tvwgt[0] = nvtxs;
  graph->tvwgt[0] = hunyuangraph_int_sum(nvtxs, graph->vwgt);
  graph->tvwgt_reverse[0] = 1.0 / (graph->tvwgt[0] > 0 ? graph->tvwgt[0] : 1);

  return graph;
}

/*Compute Partition result edge-cut*/
int hunyuangraph_computecut_cpu(hunyuangraph_graph_t *graph, int *where)
{
  int i, j, cut = 0;
  for (i = 0; i < graph->nvtxs; i++)
  {
    // printf("i=%d\n",i);
    for (j = graph->xadj[i]; j < graph->xadj[i + 1]; j++)
      if (where[i] != where[graph->adjncy[j]])
        cut += graph->adjwgt[j];
  }
  return cut / 2;
}

float hunyuangraph_compute_imbalance_cpu(hunyuangraph_graph_t *graph, int *where, int nparts)
{
	int i, j, cut = 0;
	float imbalance = 0.0;
	int *pwgts = (int *)malloc(sizeof(int) * nparts);
	memset(pwgts, 0, sizeof(int) * nparts);

	for(i = 0; i < graph->nvtxs; i++)
		pwgts[where[i]] += graph->vwgt[i];

	for(i = 0;i < nparts;i++)
		imbalance = max(imbalance, (float)pwgts[i] / (float)((float)graph->nvtxs / (float)nparts));
	imbalance -= (IMB - 1.03);
	return imbalance;
}

__device__ int reduction_warp(int val)
{
	val += __shfl_down_sync(0xffffffff, val, 16);
	val += __shfl_down_sync(0xffffffff, val, 8);
	val += __shfl_down_sync(0xffffffff, val, 4);
	val += __shfl_down_sync(0xffffffff, val, 2);
	val += __shfl_down_sync(0xffffffff, val, 1);

	return val;
}

__global__ void compute_edgecut(int nvtxs, int *edgecut, int *xadj, int *adjncy, int *adjwgt, int *where)
{
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int ii =  blockIdx.x * 4 + warp_id;

	__shared__ int cache_sum[4];
	if(threadIdx.x < 4)
		cache_sum[threadIdx.x] = 0;
	__syncthreads();

	if(ii < nvtxs)
	{
		int sum = 0;
		int begin = xadj[ii];
		int end   = xadj[ii + 1];
		int me    = where[ii];

		for(int j = begin + lane_id; j < end; j += 32)
		{
			int neighbor = adjncy[j];
            int nbr_part = where[neighbor];
			// sum += adjwgt[j] * (nbr_part != me);
            if(nbr_part != me)
                sum += adjwgt[j];
		}
		// __syncwarp();
		sum = reduction_warp(sum);
		
		if(lane_id == 0)
			cache_sum[warp_id] = sum;
			// atomicAdd(edgecut, sum);
		__syncthreads();

		if(threadIdx.x < 4)
		{
			int temp = cache_sum[lane_id];
			temp += __shfl_down_sync(0xffffffff, temp, 2);
			temp += __shfl_down_sync(0xffffffff, temp, 1);

			if(lane_id == 0)
				atomicAdd(edgecut, temp);
		}
	}
}

int compute_edgecut_gpu(int nvtxs, int *edgecut, int *xadj, int *adjncy, int *adjwgt, int *where)
{
	int *edgecut_d;
	// cudaMalloc((void **)edgecut_d, sizeof(int));
	if(GPU_Memory_Pool)
		edgecut_d = (int *)lmalloc_with_check(sizeof(int), "compute_edgecut_gpu: edgecut_d");
	else	
		cudaMalloc((void **)&edgecut_d, sizeof(int));

	// edgecut[0] = 0;
	// cudaMemcpy(edgecut_d, edgecut, sizeof(int), cudaMemcpyHostToDevice);
	// compute_edgecut_pusu<<<1, 1>>>(nvtxs, edgecut_d, xadj, adjncy, adjwgt, where);
	// cudaDeviceSynchronize();

	edgecut[0] = 0;
	cudaMemcpy(edgecut_d, edgecut, sizeof(int), cudaMemcpyHostToDevice);

#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&begin_general, NULL);
#endif
	compute_edgecut<<<(nvtxs + 3) / 4, 128>>>(nvtxs, edgecut_d, xadj, adjncy, adjwgt, where);
#ifdef TIMER
	cudaDeviceSynchronize();
	gettimeofday(&end_general, NULL);
	uncoarsen_compute_edgecut += (end_general.tv_sec - begin_general.tv_sec) * 1000.0 + (end_general.tv_usec - begin_general.tv_usec) / 1000.0;
#endif
	// printf("Uncoarsen uncoarsen_compute_edgecut    %10.3lf\n", uncoarsen_compute_edgecut);

	cudaMemcpy(edgecut, edgecut_d, sizeof(int), cudaMemcpyDeviceToHost);

	if(GPU_Memory_Pool)
		lfree_with_check((void *)edgecut_d, sizeof(int), "compute_edgecut_gpu: edgecut_d");	//	edgecut_d
	else
		cudaFree(edgecut_d);
	
	edgecut[0] /= 2;
	return edgecut[0];
}

__global__ void compute_select(int nvtxs, int *sum_block, char *select)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int lane = ii & 31;

	if(ii < nvtxs)
	{
		// if((int)select[ii] != 0 && (int)select[ii] != 1)
		// 	printf("ii: %d select: %d\n", ii, (int)select[ii]);
		int val = (int)select[ii];
		__syncwarp();
		val = reduction_warp(val);

		if(lane == 0)
			atomicAdd(sum_block, val);
	}
}

int compute_graph_select_gpu(hunyuangraph_graph_t *graph)
{
	int *sum_block_d = 0;
	// cudaMalloc((void **)edgecut_d, sizeof(int));
	if(GPU_Memory_Pool)
		sum_block_d = (int *)lmalloc_with_check(sizeof(int), "compute_graph_select_gpu: sum_block_d");
	else	
		cudaMalloc((void **)&sum_block_d, sizeof(int));

	// edgecut[0] = 0;
	// cudaMemcpy(edgecut_d, edgecut, sizeof(int), cudaMemcpyHostToDevice);
	// compute_edgecut_pusu<<<1, 1>>>(nvtxs, edgecut_d, xadj, adjncy, adjwgt, where);
	// cudaDeviceSynchronize();

	int sum_block = 0;
	cudaMemcpy(sum_block_d, &sum_block, sizeof(int), cudaMemcpyHostToDevice);

	compute_select<<<(graph->nvtxs + 127) / 128, 128>>>(graph->nvtxs, sum_block_d, graph->cuda_select);
	// cudaDeviceSynchronize();

	cudaMemcpy(&sum_block, sum_block_d, sizeof(int), cudaMemcpyDeviceToHost);

	if(GPU_Memory_Pool)
		lfree_with_check((void *)sum_block_d, sizeof(int), "compute_graph_select_gpu: sum_block_d");	//	sum_block_d
	else
		cudaFree(sum_block_d);
	
	return sum_block;
}

int compute_graph_adjwgtsum_cpu(hunyuangraph_graph_t *graph)
{
  int sum = 0;
  for (int i = 0; i < graph->nedges; i++)
    sum += graph->adjwgt[i];
  return sum;
}

int compute_graph_adjwgtsum_gpu(hunyuangraph_graph_t *graph)
{
  int sum = thrust::reduce(thrust::device, graph->cuda_adjwgt, graph->cuda_adjwgt + graph->nedges);

  return sum;
}

/*Malloc cpu coarsen graph params*/
hunyuangraph_graph_t *hunyuangraph_set_cpu_cgraph(hunyuangraph_graph_t *graph, int cnvtxs)
{
  hunyuangraph_graph_t *cgraph;
  cgraph = hunyuangraph_create_cpu_graph();

  cgraph->nvtxs = cnvtxs;
  cgraph->xadj = (int *)malloc(sizeof(int) * (cnvtxs + 1));
  cgraph->adjncy = (int *)malloc(sizeof(int) * (graph->nedges));
  cgraph->adjwgt = (int *)malloc(sizeof(int) * (graph->nedges));
  cgraph->vwgt = (int *)malloc(sizeof(int) * cnvtxs);
  cgraph->tvwgt = (int *)malloc(sizeof(int));
  cgraph->tvwgt_reverse = (float *)malloc(sizeof(float));

  cgraph->finer = graph;
  graph->coarser = cgraph;

  return cgraph;
}

/*Malloc gpu coarsen graph params*/
hunyuangraph_graph_t *hunyuangraph_set_gpu_cgraph(hunyuangraph_graph_t *graph, int cnvtxs)
{
  hunyuangraph_graph_t *cgraph = hunyuangraph_create_cpu_graph();

  cgraph->nvtxs = cnvtxs;
  //   cgraph->xadj=(int*)malloc(sizeof(int)*(cnvtxs+1));
  cgraph->tvwgt = (int *)malloc(sizeof(int));
  cgraph->tvwgt_reverse = (float *)malloc(sizeof(float));

  cgraph->finer = graph;
  graph->coarser = cgraph;

  return cgraph;
}

/*Set split graph params*/
hunyuangraph_graph_t *hunyuangraph_set_splitgraph(hunyuangraph_graph_t *graph, int snvtxs, int snedges)
{
  hunyuangraph_graph_t *sgraph;
  sgraph = hunyuangraph_create_cpu_graph();

  sgraph->nvtxs = snvtxs;
  sgraph->nedges = snedges;

  sgraph->xadj = (int *)malloc(sizeof(int) * (snvtxs + 1));
  sgraph->vwgt = (int *)malloc(sizeof(int) * (snvtxs + 1));
  sgraph->adjncy = (int *)malloc(sizeof(int) * (snedges));
  sgraph->adjwgt = (int *)malloc(sizeof(int) * (snedges));
  sgraph->label = (int *)malloc(sizeof(int) * (snvtxs));
  sgraph->tvwgt = (int *)malloc(sizeof(int));
  sgraph->tvwgt_reverse = (float *)malloc(sizeof(float));

  return sgraph;
}

/*Free graph params*/
void hunyuangraph_free_graph(hunyuangraph_graph_t **r_graph)
{
  hunyuangraph_graph_t *graph;
  graph = *r_graph;

  free(graph->xadj);
  free(graph->vwgt);
  free(graph->adjncy);
  free(graph->adjwgt);
  free(graph->where);
  free(graph->pwgts);
  free(graph->id);
  free(graph->ed);
  free(graph->bndptr);
  free(graph->bndlist);
  free(graph->tvwgt);
  free(graph->tvwgt_reverse);
  free(graph->label);
  free(graph->cmap);
  free(graph);
  *r_graph = NULL;
}

__global__ void exam_csr(int nvtxs, int *xadj, int *adjncy, int *adjwgt)
{
	// for (int i = 0; i <= nvtxs && i < 200; i++)
	// 	printf("%7d ", xadj[i]);

	// printf("\nadjncy/adjwgt:\n");
	// for (int i = 0; i < nvtxs && i < 200; i++)
	// {
	// 	for (int j = xadj[i]; j < xadj[i + 1]; j++)
	// 		printf("%7d ", adjncy[j]);
	// 	printf("\n");
	// 	for (int j = xadj[i]; j < xadj[i + 1]; j++)
	// 		printf("%7d ", adjwgt[j]);
	// 	printf("\n");
	// }
  printf("xadj:\n");
	for (int i = 0; i <= nvtxs; i++)
		printf("%7d ", xadj[i]);

	// printf("\nadjncy/adjwgt:\n");
	// for (int i = 0; i < nvtxs; i++)
	// {
	// 	for (int j = xadj[i]; j < xadj[i + 1]; j++)
	// 		printf("%7d ", adjncy[j]);
	// 	printf("\n");
	// 	for (int j = xadj[i]; j < xadj[i + 1]; j++)
	// 		printf("%7d ", adjwgt[j]);
	// 	printf("\n");
	// }
}

__global__ void exam_csr_vwgt_label(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *vwgt, int *label)
{
	for (int i = 0; i < nvtxs; i++)
		printf("%7d ", i);
	printf("\n");
	for (int i = 0; i < nvtxs; i++)
		printf("%7d ", vwgt[i]);
	printf("\n");
	for (int i = 0; i < nvtxs; i++)
		printf("%7d ", label[i]);
	printf("\n");
	for (int i = 0; i <= nvtxs; i++)
		printf("%7d ", xadj[i]);
	printf("\n");
	printf("adjncy/adjwgt:\n");
	for (int i = 0; i < nvtxs; i++)
	{
		for (int j = xadj[i]; j < xadj[i + 1]; j++)
			printf("%7d ", adjncy[j]);
		printf("\n");
		for (int j = xadj[i]; j < xadj[i + 1]; j++)
			printf("%7d ", adjwgt[j]);
		printf("\n");
	}
}

__global__ void exam_csr_where(int nvtxs, int *xadj, int *adjncy, int *adjwgt, int *where)
{
	for (int i = 0; i <= nvtxs && i < 200; i++)
		printf("%7d ", i);
	printf("\n");
	for (int i = 0; i <= nvtxs && i < 200; i++)
		printf("%7d ", xadj[i]);
	printf("\n");
	for (int i = 0; i < nvtxs && i < 200; i++)
		printf("%7d ", where[i]);
	printf("\n");
	printf("adjncy/adjwgt/where:\n");
	for (int i = 0; i < nvtxs && i < 200; i++)
	{
		for (int j = xadj[i]; j < xadj[i + 1]; j++)
			printf("%7d ", adjncy[j]);
		printf("\n");
		for (int j = xadj[i]; j < xadj[i + 1]; j++)
			printf("%7d ", adjwgt[j]);
		printf("\n");
		for (int j = xadj[i]; j < xadj[i + 1]; j++)
			printf("%7d ", where[adjncy[j]]);
		printf("\n");
	}
}

__global__ void exam_map(int nvtxs, int *map, int *where)
{
	for(int i = 0;i < nvtxs;i++)
		printf("%7d ", i);
	printf("\n");
	for(int i = 0;i < nvtxs;i++)
		printf("%7d ", map[i]);
	printf("\n");
	for(int i = 0;i < nvtxs;i++)
		printf("%7d ", where[i]);
	printf("\n");
}

#endif