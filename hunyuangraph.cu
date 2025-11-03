
#include "HunyuanGRAPH.h"

/*Main function*/
int main(int argc, char **argv)
{
	cudaSetDevice(0);

	char *filename = (argv[1]);
	int nparts = atoi(argv[2]);
	GPU_Memory_Pool = atoi(argv[3]);

	hunyuangraph_graph_t *graph = hunyuangraph_readgraph(filename);

	printf("graph:%s %d %d %d %d\n", filename, graph->nvtxs, graph->nedges, nparts, GPU_Memory_Pool);
	// for(int i = 0;i <= graph->nvtxs; i++)
	// 	printf("%d ", graph->xadj[i]);
	// printf("\n");
	// for(int i = 0;i < graph->nvtxs; i++)
	// 	printf("%d ", graph->vwgt[i]);
	// printf("\n");
	// for(int i = 0;i < graph->nvtxs;i++)
	// {
	// 	for(int j = graph->xadj[i]; j < graph->xadj[i + 1]; j++)
	// 		printf("%d ", graph->adjncy[j]);
	// 	printf("\n");
	// 	for(int j = graph->xadj[i]; j < graph->xadj[i + 1]; j++)
	// 		printf("%d ", graph->adjwgt[j]);
	// 	printf("\n");
	// }

	int *part = (int *)malloc(sizeof(int) * graph->nvtxs);

	float tpwgts[nparts];
	for (int i = 0; i < nparts; i++)
		tpwgts[i] = 1.0 / nparts;
	float ubvec = 1.03;

	int best_edgecut = graph->nedges;
	int *best_partition = (int *)malloc(sizeof(int) * graph->nvtxs);

	double best_alltime, best_coarsentime, best_inittime, best_uncoarsentime;
	best_alltime = 0x3f3f3f3f;
	best_coarsentime = 0x3f3f3f3f;
	best_inittime = 0x3f3f3f3f;
	best_uncoarsentime = 0x3f3f3f3f;
	for (int iter = 0; iter < 2; iter++)
	{
		init_timer();

		hunyuangraph_PartitionGraph(&graph->nvtxs, graph->xadj, graph->adjncy, graph->vwgt, graph->adjwgt, &nparts, tpwgts, &ubvec, part);

		int edgecut = hunyuangraph_computecut_cpu(graph, part);
		float imbalance = hunyuangraph_compute_imbalance_cpu(graph, part, nparts);

		print_time_all(graph, part, edgecut, imbalance);

		if(best_alltime > part_all)
			best_alltime = part_all;
		if(best_coarsentime > part_coarsen)
			best_coarsentime = part_coarsen;
		if(best_inittime > part_init)
			best_inittime = part_init;
		if(best_uncoarsentime > part_uncoarsen)
			best_uncoarsentime = part_uncoarsen;
#ifdef TIMER
		print_time_coarsen();
		print_time_init();
		print_time_uncoarsen();
#endif
		// print_time_topkfour_match();

		if (edgecut < best_edgecut)
		{
			memcpy(best_partition, part, sizeof(int) * graph->nvtxs);
			best_edgecut = edgecut;
		}

		// if (iter == 0)
		// 	print_time_coarsen();
		// if(iter == 0)
		// 	print_time_init();
		// if (iter == 0)
		// 	print_time_uncoarsen();
	}

	if(part_uncoarsen < best_alltime - best_coarsentime - best_inittime);
		part_uncoarsen = best_alltime - best_coarsentime - best_inittime;
	printf("best_alltime=         %10.3lf\n", best_alltime);
	printf("best_coarsentime=     %10.3lf\n", best_coarsentime);
	printf("best_inittime=        %10.3lf\n", best_inittime);
	printf("best_uncoarsentime=   %10.3lf\n", best_uncoarsentime);
	
#ifdef FIGURE14_EDGECUT
	printf("best_edgecut=         %10d\n", best_edgecut);
#endif

	// hunyuangraph_writetofile(filename, part, graph->nvtxs, nparts);

	// double twoway_else = gpu_2way - (initmoveto + updatemoveto + computepwgts + thrustreduce + computegain + thrustsort + computegainv + inclusive + re_balance);

	// printf("initmoveto=    %10.3lf %7.3lf%\n",initmoveto, initmoveto / gpu_2way * 100);
	// printf("updatemoveto=  %10.3lf %7.3lf%\n",updatemoveto, updatemoveto / gpu_2way * 100);
	// printf("computepwgts=  %10.3lf %7.3lf%\n",computepwgts, computepwgts / gpu_2way * 100);
	// printf("thrustreduce=  %10.3lf %7.3lf%\n",thrustreduce, thrustreduce / gpu_2way * 100);
	// printf("computegain=   %10.3lf %7.3lf%\n",computegain, computegain / gpu_2way * 100);
	// printf("thrustsort=    %10.3lf %7.3lf%\n",thrustsort, thrustsort / gpu_2way * 100);
	// printf("computegainv=  %10.3lf %7.3lf%\n",computegainv, computegainv / gpu_2way * 100);
	// printf("inclusive=     %10.3lf %7.3lf%\n",inclusive, inclusive / gpu_2way * 100);
	// printf("re_balance=    %10.3lf %7.3lf%\n",re_balance, re_balance / gpu_2way * 100);
	// printf("malloc_2way=   %10.3lf %7.3lf%\n",malloc_2way, malloc_2way / gpu_2way * 100);
	// printf("else=          %10.3lf %7.3lf%\n",twoway_else, twoway_else / gpu_2way * 100);
	// printf("\n");

	// double split_else = part_slipt - (malloc_split + memcpy_split + free_split);
	// printf("malloc_split=    %10.3lf %7.3lf%\n",malloc_split, malloc_split / part_slipt * 100);
	// printf("memcpy_split=    %10.3lf %7.3lf%\n",memcpy_split, memcpy_split / part_slipt * 100);
	// printf("free_split=      %10.3lf %7.3lf%\n",free_split, free_split / part_slipt * 100);
	// printf("else=            %10.3lf %7.3lf%\n",split_else, split_else / part_slipt * 100);
	// printf("\n");

	// save_init = save_init + memcpy_split;
	// printf("could_save=      %10.2lf %7.3lf%\n",save_init, save_init / part_init * 100);
}

// repair cuda_nvtxs, cuda_nparts, refine_pass, cuda_tvwgt, Mallocinit_refineinfo, hunyuangraph_malloc_refineinfo, Sum_maxmin_pwgts,
//     hunyuangraph_findgraphbndinfo: Find_real_bnd_info, find_kayparams, findcsr,
// repair Exnode_part1 Exnode_part2
// repair graph->cuda_real_edge, cuda_cnvtxs, cxadj, ccc, cuda_s, findc1, prefix_sum(cmap), findc22_53, finc4,
// repair cuda_js, find_cnvtxsedge_original? Sort_2Sort_2_5, Sort_cnedges2_part1, Sort_cnedges2_part3, setcvwgt, cuda_maxvwgt/maxvwgt, cuda_hem
// merge findc1 findc2, gpu_match 32->128, divide_group, merge findc4 set_cvwgt, cuda_scan_nedges_original, cuda_real_nvtxs,
// error: Bump_2911(X), soc-Livejournal1�????

// 相较于cuMetis_10，本版本优化点如下：
// 		1、将cu_bn,cu_bt,cu_g,cu_csr,cu_que该五个变量的空间申请挪至k-way refinement阶段；
// 		2、粗化阶段中bb_segsort库函数中的cudaMalloc,cudaFree及cudaMemcpy；
// 		3、申请显存总空间进一步修改为“freeMem - usableMem - 2 * nedges * sizeof(int)”；

/*****************************		Attention about cudaMalloc!!!				*****************************
**********由于多级算法的对内存的要求与栈的性质完美的重合，因此本算法中涉及到的显存管理是以栈为基础的	   **********
**********由于算法中使用到了一些临时数组，仅使用单端栈的话会导致显存的浪费以及显存碎片化，			      **********
**********		因此本算法使用了双端栈的概念，左端存放需要保留的图的信息，右端存放临时数组				  **********
**********若利用本算法特定的显存管理方式，请注意“先申请后释放，后申请先释放”的特性						 **********
*/
// 左端栈先压入了cu_bn,cu_bt,cu_g,cu_csr,cu_que(cuMetis_10 yes/now no)
// 左端栈存放的图的信息从左至右依次为vwgt,xadj,adjncy,adjwgt,cmap,where,bnd,pwgts,tpwgts,maxwgt,minwgt,bndnum
// 粗化阶段右端栈存放的图的信息从右至左依次为match,txadj.tadjncy,tadjwgt,temp_scan

/**********     2024.11.21     **********
 * 1、粗化限制条件与Jet对齐(边切效果变差-已注释)
 * 2、kernel时间只记录第一次，其他20次只记录总时间，阶段时间和边切值
 * 3、添加内存泄露检查的python代码
 * 4、修改了uncoarsen阶段free的不一致问题
 ****************************************/

/**********     2024.12.07     **********
 * 1、粗化限制条件与Jet对齐(靠后的粗化不能收敛)
 * 2、写GPU上的初始分割代码
 * 3、prefixsum可以优化同步
 ****************************************/