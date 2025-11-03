#include "mygpmetis.h"
#include <metis.h>

/*************************************************************************/
/*! This version of the reordering function only supports:
		*1 graph vertices numbered from 0
		*2 without pruning and compression
		*3 does not use MMD to sort small graphs
*/
/*************************************************************************/
bool compare_num(Hunyuan_int_t *reflect, Hunyuan_int_t *iperm, Hunyuan_int_t nvtxs)
{
	bool flag = true;
	for(Hunyuan_int_t i = 0;i < nvtxs;i++)
		if(reflect[i] != iperm[i])
		{
			flag = false;
			break;
		}
	
	return flag;
}

bool exam_correct_nd(Hunyuan_int_t *reflect, Hunyuan_int_t nvtxs)
{
	Hunyuan_int_t *temp = (Hunyuan_int_t *)malloc(sizeof(Hunyuan_int_t) * nvtxs);

	memset(temp, -1, sizeof(Hunyuan_int_t) * nvtxs);

	for(Hunyuan_int_t i = 0;i < nvtxs;i++)
	{
		temp[reflect[i]] = i;
	}

	Hunyuan_int_t cnt = 0;
	for(Hunyuan_int_t i = 0;i < nvtxs;i++)
	{
		if(temp[i] == -1)
			cnt++;
	}

	if(cnt == 0) 
	{
		// printf("cnt=%"PRIDX"\n",cnt);
		return true;
	}
	else 
	{
		// printf("cnt=%"PRIDX"\n",cnt);
		return false;
	}
}
/*
Hunyuan_int_t compute_edgecut(Hunyuan_int_t *result, Hunyuan_int_t nvtxs, Hunyuan_int_t *xadj, Hunyuan_int_t *adjncy, Hunyuan_int_t *adjwgt)
{
	Hunyuan_int_t edgecut = 0;
	for(Hunyuan_int_t i = 0;i < nvtxs;i++)
	{
		Hunyuan_int_t partition = result[i];
		for(Hunyuan_int_t j = xadj[i];j < xadj[i+1];j++)
		{
			Hunyuan_int_t k = adjncy[j];
			if(partition != result[k])
				edgecut += adjwgt[j];
		}
	}

	return edgecut / 2;
}

/*bool exam_correct_gp(Hunyuan_int_t *result, Hunyuan_int_t nvtxs, Hunyuan_int_t nparts)
{
	Hunyuan_int_t cnt = 0;
	for(Hunyuan_int_t i = 0;i < nvtxs;i++)
	{
		Hunyuan_int_t t = result[i];
		if(t < 0 || t >= nparts)
			cnt++;
	}

	if(cnt == 0) 
	{
		printf("cnt=%"PRIDX"\n",cnt);
		return true;
	}
	else 
	{
		printf("cnt=%"PRIDX"\n",cnt);
		return false;
	}
}
*/

Hunyuan_int_t main(Hunyuan_int_t argc, char **argv)
{
	char *filename = (argv[1]);
	Hunyuan_int_t nparts = atoi(argv[2]);
	// Hunyuan_int_t vertex = atoi(argv[3]);
	// printf("filename=%s nparts=%"PRIDX" vertex=%"PRIDX" \n",filename, nparts, vertex);
	control = 1;
	printf("filename=%s nparts=%"PRIDX"\n",filename, nparts);
	// control = atoi(argv[3]);

	//	init memory manage
	bool is_memery_manage_before = false;
	if(init_memery_manage(filename) == 0)
	{
		return 0;
	}
	is_memery_manage_before = true;

	//	input graph data
	Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt;
    graph_t *graph = ReadGraph(filename, &xadj, &vwgt, &adjncy, &adjwgt);
	// printf("filename=%s nparts=%"PRIDX"\n",filename, nparts);
	graph->xadj   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (graph->nvtxs + 1), "main: xadj");
	graph->vwgt   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "main: vwgt");
	graph->adjncy = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "main: adjncy");
	graph->adjwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges,"main: adjwgt");
	copy_int(graph->nedges, adjwgt, graph->adjwgt);
	copy_int(graph->nedges, adjncy, graph->adjncy);
	copy_int(graph->nvtxs, vwgt, graph->vwgt);
	copy_int(graph->nvtxs + 1, xadj, graph->xadj);
	
	// printf("nvtxs=%"PRIDX" nedges=%"PRIDX"\n",graph->nvtxs,graph->nedges);
	// printf("%"PRIDX" %"PRIDX" 011\n",graph->nvtxs,graph->nedges / 2);
    // for(Hunyuan_int_t a = 0; a < graph->nvtxs; a++)
    // {
    //   printf("%"PRIDX" ",graph->vwgt[a]);
    //   for(Hunyuan_int_t b = graph->xadj[a]; b < graph->xadj[a + 1]; b++)
    //     printf("%"PRIDX" %"PRIDX" ",graph->adjncy[b] + 1, graph->adjwgt[b]);
    //   printf("\n");
    // }

	// metis_match = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "main: metis_match");
	// ReadNum(metismatchname, graph->nvtxs, metis_match);

	// printf("ReadNum end\n");

	//	set options for PartGraph
	// Hunyuan_int_t options[NUM_OPTIONS];
	// PartGraph_options(options, params);

	// store the result of reorder
	Hunyuan_int_t *result = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "main: result");
	Hunyuan_real_t *tpwgts = (Hunyuan_real_t *)check_malloc(sizeof(Hunyuan_real_t) * nparts, "main: tpwgts");
	Hunyuan_real_t balance_factor = 1.0300499;
	set_value_double(nparts, 1.0 / nparts, tpwgts);

	// printf("main 0\n");

	// PartitionGraph(&graph->nvtxs,&graph->nedges,graph->xadj,graph->vwgt,graph->adjncy,graph->adjwgt,\
		&nparts, tpwgts, &balance_factor, result, is_memery_manage_before);
	// printf("main 1\n");

	CONTROL_COMMAND(control, INITIALPARTITION_Time, gettimebegin(&start_initialpartition, &end_initialpartition, &time_initialpartition));
	// InitialPartition_multi(graph, nparts, tpwgts, &balance_factor);
	InitialPartition_NestedBisection(graph, nparts, tpwgts, &balance_factor);
	CONTROL_COMMAND(control, INITIALPARTITION_Time, gettimeend(&start_initialpartition, &end_initialpartition, &time_initialpartition));    

	memcpy(result, graph->where, sizeof(Hunyuan_int_t) * graph->nvtxs);
	// check_free(graph->where, sizeof(Hunyuan_int_t) * graph->nvtxs, "main: graph->where");
	// check_free(graph->xadj, sizeof(Hunyuan_int_t) * (graph->nvtxs + 1), "main: graph->xadj");
	// check_free(graph->vwgt, sizeof(Hunyuan_int_t) * graph->nvtxs, "main: graph->vwgt");
	// check_free(graph->adjncy, sizeof(Hunyuan_int_t) * graph->nedges, "main: graph->adjncy");
	// check_free(graph->adjwgt, sizeof(Hunyuan_int_t) * graph->nedges, "main: graph->adjwgt");
	
	if(exam_correct_gp(result, graph->nvtxs, nparts))
	{
		// printf("The Answer is Right\n");
		// printf("edgecut=%"PRIDX" \n", compute_edgecut(result, graph->nvtxs, xadj, adjncy, adjwgt));
		// printf("%"PRIDX" \n", compute_edgecut(result, graph->nvtxs, xadj, adjncy, adjwgt));
	}
	else 
		printf("The Answer is Error\n");

	check_free(result, sizeof(Hunyuan_int_t) * graph->nvtxs, "main: result");
	check_free(tpwgts, sizeof(Hunyuan_int_t) * nparts, "main: tpwgts");
	check_free(graph->tvwgt, sizeof(Hunyuan_real_t), "main: graph->tvwgt");
	check_free(graph, sizeof(graph_t), "main: graph");

	free(xadj);
	free(vwgt);
	free(adjncy);
	free(adjwgt);

	// PrintTime(control);
	// PrintMemory();
	// exam_memory();
	// printf("reflect size:               %10zu Bytes\n",sizeof(Hunyuan_int_t) * graph->nvtxs);
	// exam_num(reflect,graph->nvtxs);

	// printf("main 2\n");
	// graph = ReadGraph(filename);

	if(is_memery_manage_before)
		free_memory_block();

    return 0;
}