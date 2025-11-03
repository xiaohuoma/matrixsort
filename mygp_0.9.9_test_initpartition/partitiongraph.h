#ifndef PARTITIONGRAPH_H
#define PARTITIONGRAPH_H

#include <stdbool.h>
#include "memory.h"
#include "define.h"
#include "control.h"
#include "timer.h"
#include "coarsen.h"
#include "initialpartition.h"
#include "refine.h"
#include "kwayrefine.h"
#include "splitgraph.h"

void MultiLevelPartition(graph_t *graph, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor, Hunyuan_int_t *result)
{
    graph_t *cgraph;

    // Hunyuan_int_t Coarsen_Threshold = lyj_max(100, graph->nvtxs / 30);
    Hunyuan_int_t Coarsen_Threshold = lyj_max((graph->nvtxs) / (20 * lyj_log2(nparts)), 30 * (nparts));

    // printf("MultiLevelPartition 0\n");

    CONTROL_COMMAND(control, COARSEN_Time, gettimebegin(&start_coarsen, &end_coarsen, &time_coarsen));
	cgraph = CoarsenGraph(graph, Coarsen_Threshold);
	CONTROL_COMMAND(control, COARSEN_Time, gettimeend(&start_coarsen, &end_coarsen, &time_coarsen));
    
    // exam_nvtxs_nedges(cgraph);
    // exam_xadj(cgraph);
    // exam_vwgt(cgraph);
    // exam_adjncy_adjwgt(cgraph);

    printf("MultiLevelPartition 1\n");

    // exam_tpwgts(tpwgts, nparts);
    CONTROL_COMMAND(control, INITIALPARTITION_Time, gettimebegin(&start_initialpartition, &end_initialpartition, &time_initialpartition));
	InitialPartition(cgraph, nparts, tpwgts, balance_factor);
	CONTROL_COMMAND(control, INITIALPARTITION_Time, gettimeend(&start_initialpartition, &end_initialpartition, &time_initialpartition));    

    // printf("MultiLevelPartition 2\n");

    // exam_xadj(cgraph);
    // exam_vwgt(cgraph);
    // exam_adjncy_adjwgt(cgraph);
    // if(exam_correct_gp(cgraph->where, cgraph->nvtxs, nparts))
	// {
	// 	printf("The Answer is Right\n");
    //     printf("edgecut=%"PRIDX" \n", compute_edgecut(cgraph->where, cgraph->nvtxs, cgraph->xadj, cgraph->adjncy, cgraph->adjwgt));
    // }
	// else 
	// 	printf("The Answer is Error\n");

    // printf("InitialPartition end rand_count=%"PRIDX"\n", rand_count());

    // Hunyuan_int_t t = cgraph->adjncy[1];
    // for(Hunyuan_int_t i = 0;i < cgraph->nedges;i++)
    //     printf("%"PRIDX" \n",cgraph->adjncy[i]);

    cgraph->cnbr_size    = 2 * cgraph->nedges;
    cgraph->cnbr_length  = 0;
    cgraph->cnb_reallocs = 0;
    cgraph->cnbr = (cnbr_t *)check_malloc(cgraph->cnbr_size * sizeof(cnbr_t), "MultiLevelPartition: cgraph->cnbr");
    // exam_tpwgts(tpwgts, nparts);
    RefineKWayPartition(graph, cgraph, nparts, tpwgts, balance_factor);

    // printf("MultiLevelPartition 3\n");
    memcpy(result, graph->where, graph->nvtxs * sizeof(Hunyuan_int_t));
    // if(exam_correct_gp(graph->where, graph->nvtxs, nparts))
	// {
	// 	printf("The Answer is Right\n");
    //     printf("edgecut=%"PRIDX" \n", compute_edgecut(graph->where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt));
    // }
	// else 
	// 	printf("The Answer is Error\n");

    FreeGraph(&graph, nparts);
}

void PartitionGraph(Hunyuan_int_t *nvtxs, Hunyuan_int_t *nedges, Hunyuan_int_t *xadj, Hunyuan_int_t *vwgt, Hunyuan_int_t *adjncy, Hunyuan_int_t *adjwgt, 
    Hunyuan_int_t *nparts, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor, Hunyuan_int_t *result, bool is_memery_manage_before)
{
    CONTROL_COMMAND(control, ALL_Time, gettimebegin(&start_all, &end_all, &time_all));

	// printf("ReorderGraph 0\n");
    //  init memery manage
    if(!is_memery_manage_before)
        if(init_memery_manage(NULL) == 0)
        {
            char *error_message = (char *)check_malloc(sizeof(char) * 1024, "check_fopen: error_message");
            sprintf(error_message, "init memery manage failed.");
            error_exit(error_message);
            return ;
        }
    
    //  init
    Hunyuan_int_t reordernum = nvtxs[0];
    InitRandom(-1);

    graph_t *graph = SetupGraph(nvtxs[0], xadj, adjncy, vwgt, adjwgt); 

    // printf("PartitionGraph 1\n");

    CONTROL_COMMAND(control, PARTITIONGRAPH_Time, gettimebegin(&start_partitiongraph, &end_partitiongraph, &time_partitiongraph));
    MultiLevelPartition(graph, nparts[0], tpwgts, balance_factor, result);
	CONTROL_COMMAND(control, PARTITIONGRAPH_Time, gettimeend(&start_partitiongraph, &end_partitiongraph, &time_partitiongraph));

    // printf("PartitionGraph 2\n");

    //  free memery manage
    if(!is_memery_manage_before)
        free_memory_block();

    CONTROL_COMMAND(control, ALL_Time, gettimeend(&start_all, &end_all, &time_all));
}

#endif