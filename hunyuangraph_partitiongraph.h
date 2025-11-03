#ifndef _H_PARTITIONGRAPH
#define _H_PARTITIONGRAPH

#include "hunyuangraph_struct.h"
#include "hunyuangraph_timer.h"
#include "hunyuangraph_GPU_memory.h"
#include "hunyuangraph_GPU_coarsen.h"
#include "hunyuangraph_CPU_initialpartition.h"
#include "hunyuangraph_GPU_initialpartition.h"
#include "hunyuangraph_GPU_uncoarsen.h"

/*Graph kway-partition algorithm*/
void hunyuangraph_kway_partition(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int *part)
{
	hunyuangraph_graph_t *cgraph;
	int level = 0;

	// printf("Coarsen begin\n");

#ifdef FIGURE9_TIME
	hunyuangraph_admin->time_coarsen = (double *)malloc(sizeof(double) * 100);
#endif

#ifdef FIGURE10_EXHAUSTIVE
	goto figure10_exhaustive;
#endif
#ifdef DFIGURE10_SAMPLING
	goto figure10_sampling;
#endif

	cudaDeviceSynchronize();
	gettimeofday(&begin_part_coarsen, NULL);
	cgraph = hunyuangarph_coarsen(hunyuangraph_admin, graph, &level);
	cudaDeviceSynchronize();
	gettimeofday(&end_part_coarsen, NULL);
	part_coarsen += (end_part_coarsen.tv_sec - begin_part_coarsen.tv_sec) * 1000 + (end_part_coarsen.tv_usec - begin_part_coarsen.tv_usec) / 1000.0;

	// printf("Coarsen end: level=%d cnvtxs=%d cnedges=%d\n", level, cgraph->nvtxs, cgraph->nedges);
	// printf("Coarsen end: level=%d cnvtxs=%d cnedges=%d adjwgtsum=%d\n", level, cgraph->nvtxs, cgraph->nedges, compute_graph_adjwgtsum_gpu(graph));
	if(cgraph->nvtxs >= 100000)
	{
		printf("coarsen graph error\n");
		exit(0);
	}

#ifdef FIGURE9_TIME
	for(int a = 0;a < level;a++)
		printf("level=%3d time=%10.3lf ms\n", a+1, hunyuangraph_admin->time_coarsen[a]);
#endif

	// print_time_coarsen();
	// print_time_topkfour_match();

#ifdef FIGURE10_CGRAPH
	FILE *fp = fopen("graph.txt","w");
    fprintf(fp, "%d %d 011\n",cgraph->nvtxs,cgraph->nedges / 2);
    for(int a = 0; a < cgraph->nvtxs; a++)
    {
    	fprintf(fp, "%d ",cgraph->vwgt[a]);
        for(int b = cgraph->xadj[a]; b < cgraph->xadj[a + 1]; b++)
        	fprintf(fp, "%d %d ",cgraph->adjncy[b] + 1, cgraph->adjwgt[b]);
        fprintf(fp, "\n");
    }
    fclose(fp);

#endif

#ifdef FIGURE9_SUM
	exit(0);
#endif
#ifdef FIGURE9_TIME
	exit(0);
#endif
#ifdef FIGURE10_CGRAPH
	exit(0);
#endif

	cudaDeviceSynchronize();
	gettimeofday(&begin_part_init, NULL);
	// hunyuangarph_initialpartition(hunyuangraph_admin, cgraph);
	// hunyuangraph_gpu_initialpartition(hunyuangraph_admin, cgraph);
#ifdef FIGURE10_EXHAUSTIVE
figure10_exhaustive:
	cgraph = graph;
#endif
#ifdef FIGURE10_SAMPLING
figure10_sampling:
	cgraph = graph;
#endif
	hunyuangraph_gpu_initialpartition(hunyuangraph_admin, cgraph);
	cudaDeviceSynchronize();
	gettimeofday(&end_part_init, NULL);
	part_init += (end_part_init.tv_sec - begin_part_init.tv_sec) * 1000 + (end_part_init.tv_usec - begin_part_init.tv_usec) / 1000.0;

#ifdef FIGURE10_EXHAUSTIVE
	exit(0);
#endif
#ifdef FIGURE10_SAMPLING
	exit(0);
#endif
	// cudaMemcpy(cgraph->where, cgraph->cuda_where, cgraph->nvtxs * sizeof(int), cudaMemcpyDeviceToHost);
	// for(int i = 0;i < cgraph->nvtxs;i++)
	// 	printf("%10d %10d\n", i, cgraph->where[i]);
	// FILE *fp = fopen("graph.txt","w");
    // fprintf(fp, "%d %d 011\n",cgraph->nvtxs,cgraph->nedges / 2);
    // for(int a = 0; a < cgraph->nvtxs; a++)
    // {
    // 	fprintf(fp, "%d %d %d %d\n",cgraph->vwgt[a], cgraph->where[a], cgraph->xadj[a], cgraph->xadj[a + 1]);
    //     for(int b = cgraph->xadj[a]; b < cgraph->xadj[a + 1]; b++)
    //     	fprintf(fp, "%10d ",cgraph->adjncy[b]);
    //     fprintf(fp, "\n");
	// 	for(int b = cgraph->xadj[a]; b < cgraph->xadj[a + 1]; b++)
    //     	fprintf(fp, "%10d ",cgraph->where[cgraph->adjncy[b]]);
    //     fprintf(fp, "\n");
	// 	for(int b = cgraph->xadj[a]; b < cgraph->xadj[a + 1]; b++)
    //     	fprintf(fp, "%10d ",cgraph->adjwgt[b]);
    //     fprintf(fp, "\n");
    // }
    // fclose(fp);
	// cudaDeviceSynchronize();
    // gettimeofday(&begin_gpu_bisection, NULL);
    // hunyuangraph_computecut_gpu<<<1, 1>>>(cgraph->nvtxs, cgraph->cuda_xadj, cgraph->cuda_adjncy, cgraph->cuda_adjwgt, cgraph->cuda_where);
    // cudaDeviceSynchronize();
    // gettimeofday(&end_gpu_bisection, NULL);
    // computecut_time += (end_gpu_bisection.tv_sec - begin_gpu_bisection.tv_sec) * 1000 + (end_gpu_bisection.tv_usec - begin_gpu_bisection.tv_usec) / 1000.0;

	// print_time_init();

	// hunyuangraph_memcpy_coarsentoinit(cgraph);
	// int edgecut;
	// compute_edgecut_gpu(cgraph->nvtxs, &edgecut, cgraph->cuda_xadj, cgraph->cuda_adjncy, cgraph->cuda_adjwgt, cgraph->cuda_where);
	// printf("edgecut=%d\n",edgecut);
	// printf("Init partition end\n");

	// exit(0);

	// cudaDeviceSynchronize();
	gettimeofday(&begin_part_uncoarsen, NULL);
	// hunyuangraph_GPU_uncoarsen(hunyuangraph_admin, graph, cgraph);
	// hunyuangraph_GPU_uncoarsen_SC25(hunyuangraph_admin, graph, cgraph, &level);
	hunyuangraph_GPU_uncoarsen_SC25_copy(hunyuangraph_admin, graph, cgraph, &level);
	cudaDeviceSynchronize();
	gettimeofday(&end_part_uncoarsen, NULL);
	part_uncoarsen += (end_part_uncoarsen.tv_sec - begin_part_uncoarsen.tv_sec) * 1000 + (end_part_uncoarsen.tv_usec - begin_part_uncoarsen.tv_usec) / 1000.0;
	
	// print_time_uncoarsen();
	// printf("Uncoarsen end\n");
	// exit(0);
}

/*Set kway balance params*/
void hunyuangraph_set_kway_bal(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	for (int i = 0, j = 0; i < hunyuangraph_admin->nparts; i++)
		hunyuangraph_admin->part_balance[i + j] = graph->tvwgt_reverse[j] / hunyuangraph_admin->tpwgts[i + j];
}

__global__ void init_vwgt(int *vwgt, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nvtxs)
		vwgt[ii] = 1;
}

__global__ void init_adjwgt(int *adjwgt, int nedges)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if (ii < nedges)
		adjwgt[ii] = 1;
}

/*Malloc and memcpy original graph from cpu to gpu*/
void hunyuangraph_malloc_original_coarseninfo(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int nvtxs = graph->nvtxs;
	int nedges = graph->nedges;

	if(GPU_Memory_Pool)
	{
		graph->cuda_vwgt = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_original_coarseninfo: vwgt");
		graph->cuda_xadj = (int *)lmalloc_with_check(sizeof(int) * (nvtxs + 1), "hunyuangraph_malloc_original_coarseninfo: xadj");
		graph->cuda_adjncy = (int *)lmalloc_with_check(sizeof(int) * nedges, "hunyuangraph_malloc_original_coarseninfo: adjncy");
		graph->cuda_adjwgt = (int *)lmalloc_with_check(sizeof(int) * nedges, "hunyuangraph_malloc_original_coarseninfo: adjwgt");
		graph->length_vertex = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_original_coarseninfo: graph->length_vertex");
        graph->bin_offset = (int *)lmalloc_with_check(sizeof(int) * 15, "hunyuangraph_malloc_original_coarseninfo: graph->bin_offset");
        graph->bin_idx    = (int *)lmalloc_with_check(sizeof(int) * nvtxs, "hunyuangraph_malloc_original_coarseninfo: graph->bin_idx");
	}
	else
	{
		cudaMalloc((void**)&graph->cuda_vwgt, sizeof(int) * nvtxs);
		cudaMalloc((void**)&graph->cuda_xadj, sizeof(int) * (nvtxs + 1));
		cudaMalloc((void**)&graph->cuda_adjncy, sizeof(int) * nedges);
		cudaMalloc((void**)&graph->cuda_adjwgt, sizeof(int) * nedges);
		cudaMalloc((void**)&graph->length_vertex, sizeof(int) * nvtxs);
        cudaMalloc((void**)&graph->bin_offset, sizeof(int) * 15);
        cudaMalloc((void**)&graph->bin_idx, sizeof(int) * nvtxs);
	}

	graph->h_bin_offset = (int *)malloc(sizeof(int) * 15);
	// cudaMalloc((void**)&graph->cuda_xadj,(nvtxs+1)*sizeof(int));
	// cudaMalloc((void**)&graph->cuda_vwgt,nvtxs*sizeof(int));
	// cudaMalloc((void**)&graph->cuda_adjncy,nedges*sizeof(int));
	// cudaMalloc((void**)&graph->cuda_adjwgt,nedges*sizeof(int));

	cudaMemcpy(graph->cuda_xadj, graph->xadj, (nvtxs + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(graph->cuda_adjncy, graph->adjncy, nedges * sizeof(int), cudaMemcpyHostToDevice);

	// ����CUDA��
	// cudaStream_t stream;
	// cudaStreamCreate(&stream);

	// // ���õ�һ���˺���
	// init_vwgt<<<(nvtxs + 127) / 128, 128, 0, stream>>>(graph->cuda_vwgt, nvtxs);
	// init_vwgt<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_vwgt, nvtxs);
	// // ���õڶ����˺���
	// init_adjwgt<<<(nedges + 127) / 128, 128, 0, stream>>>(graph->cuda_adjwgt, nedges);

	init_vwgt<<<(nvtxs + 127) / 128, 128>>>(graph->cuda_vwgt, nvtxs);
	init_adjwgt<<<(nedges + 127) / 128, 128>>>(graph->cuda_adjwgt, nedges);

#ifdef FIGURE10_EXHAUSTIVE
	cudaMemcpy(graph->cuda_vwgt,graph->vwgt,nvtxs*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(graph->cuda_adjwgt,graph->adjwgt,nedges*sizeof(int),cudaMemcpyHostToDevice);
#endif
#ifdef FIGURE10_SAMPLING
	cudaMemcpy(graph->cuda_vwgt,graph->vwgt,nvtxs*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(graph->cuda_adjwgt,graph->adjwgt,nedges*sizeof(int),cudaMemcpyHostToDevice);
#endif

	int *length_bin, *bin_size;
	if(GPU_Memory_Pool)
	{
		length_bin = (int *)rmalloc_with_check(sizeof(int) * 14, "hunyuangraph_malloc_original_coarseninfo: length_bin");
		bin_size   = (int *)rmalloc_with_check(sizeof(int) * 14, "hunyuangraph_malloc_original_coarseninfo: bin_size");
	}
	else 
	{
		cudaMalloc((void**)&length_bin, sizeof(int) * 14);
		cudaMalloc((void**)&bin_size, sizeof(int) * 14);
	}

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

	cudaMemcpy(graph->h_bin_offset, graph->bin_offset, sizeof(int) * 15, cudaMemcpyDeviceToHost);

	if(GPU_Memory_Pool)
	{
		rfree_with_check(bin_size, sizeof(int) * 14, "hunyuangraph_malloc_original_coarseninfo: bin_size");
		rfree_with_check(length_bin, sizeof(int) * 14, "hunyuangraph_malloc_original_coarseninfo: length_bin");
	}
	else 
	{
		cudaFree(length_bin);
		cudaFree(bin_size);
	}

	// // �ȴ����������??
	// cudaStreamSynchronize(stream);

	// // ����CUDA��
	// cudaStreamDestroy(stream);
}

/*Graph partition algorithm*/
void hunyuangraph_PartitionGraph(int *nvtxs, int *xadj, int *adjncy, int *vwgt, int *adjwgt, int *nparts, float *tpwgts, float *ubvec, int *part)
{
	hunyuangraph_graph_t *graph;
	hunyuangraph_admin_t *hunyuangraph_admin;

	hunyuangraph_admin = hunyuangraph_set_graph_admin(*nparts, tpwgts, ubvec);

	graph = hunyuangraph_set_first_level_graph(*nvtxs, xadj, adjncy, vwgt, adjwgt);

	hunyuangraph_set_kway_bal(hunyuangraph_admin, graph);

	hunyuangraph_admin->Coarsen_threshold = hunyuangraph_max((*nvtxs) / (20 * (hunyuangraph_compute_log2(*nparts))), 30 * (*nparts));
	hunyuangraph_admin->nIparts = (hunyuangraph_admin->Coarsen_threshold == 30 * (*nparts) ? 4 : 5);

	hunyuangraph_admin->Coarsen_threshold = (*nparts) * 8;
	if (hunyuangraph_admin->Coarsen_threshold > 1024)
	{
		hunyuangraph_admin->Coarsen_threshold = (*nparts) * 2;
		hunyuangraph_admin->Coarsen_threshold = hunyuangraph_max(1024, hunyuangraph_admin->Coarsen_threshold);
	}
	printf("hunyuangraph_admin->Coarsen_threshold=%10d\n", hunyuangraph_admin->Coarsen_threshold);

	if(GPU_Memory_Pool)
		Malloc_GPU_Memory(graph->nvtxs, graph->nedges);

	// cudaMalloc((void**)&cu_bn, graph->nvtxs * sizeof(int));
	// cudaMalloc((void**)&cu_bt, graph->nvtxs * sizeof(int));
	// cudaMalloc((void**)&cu_g,  graph->nvtxs * sizeof(int));
	// cudaMalloc((void**)&cu_csr, 2 * sizeof(int));
	// cudaMalloc((void**)&cu_que, 2 * hunyuangraph_admin->nparts * sizeof(int));

	// cu_bn  = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs,"cu_bn");
	// cu_bt  = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs,"cu_bt");
	// cu_g   = (int *)lmalloc_with_check(sizeof(int) * graph->nvtxs,"cu_g");
	// cu_csr = (int *)lmalloc_with_check(sizeof(int) * 2,"cu_csr");
	// cu_que = (int *)lmalloc_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2,"cu_que");

	hunyuangraph_malloc_original_coarseninfo(hunyuangraph_admin, graph);
	// cudaDeviceSynchronize();
	// exam_csr<<<1,1>>>(graph->nvtxs, graph->cuda_xadj, graph->cuda_adjncy, graph->cuda_adjwgt);
	// cudaDeviceSynchronize();

	printf("begin partition\n");
	// printf("nedges / nvtxs: %10.2lf\n", (double)graph->nedges / (double)graph->nvtxs);
	cudaDeviceSynchronize();
	gettimeofday(&begin_part_all, NULL);
	hunyuangraph_kway_partition(hunyuangraph_admin, graph, part);
	cudaDeviceSynchronize();
	gettimeofday(&end_part_all, NULL);
	part_all += (end_part_all.tv_sec - begin_part_all.tv_sec) * 1000 + (end_part_all.tv_usec - begin_part_all.tv_usec) / 1000.0;
	printf("end partition\n");

	cudaMemcpy(part, graph->cuda_where, graph->nvtxs * sizeof(int), cudaMemcpyDeviceToHost);

	hunyuangraph_uncoarsen_free_coarsen(hunyuangraph_admin, graph);

	// lfree_with_check(sizeof(int) * hunyuangraph_admin->nparts * 2,"cu_que");	//cu_que
	// lfree_with_check(sizeof(int) * 2,"cu_csr");									//cu_csr
	// lfree_with_check(sizeof(int) * graph->nvtxs,"cu_g");						//cu_g
	// lfree_with_check(sizeof(int) * graph->nvtxs,"cu_bt");						//cu_bt
	// lfree_with_check(sizeof(int) * graph->nvtxs,"cu_bn");						//cu_bn
	if(GPU_Memory_Pool)
		Free_GPU_Memory();
}

#endif