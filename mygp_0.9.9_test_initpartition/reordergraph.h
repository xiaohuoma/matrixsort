#ifndef REORDERGRAPH_H
#define REORDERGRAPH_H

#include <stdbool.h>
#include "memory.h"
#include "define.h"
#include "control.h"
#include "timer.h"
#include "coarsen.h"
#include "initialpartition.h"
#include "refine.h"
#include "splitgraph.h"
#include "mmdorder.h"

Hunyuan_int_t num = 0;

void Reorderpartition(graph_t *graph, Hunyuan_int_t niparts)
{
	graph_t *cgraph;

	Hunyuan_int_t Coarsen_Threshold = graph->nvtxs / 8;
	if (Coarsen_Threshold > 100)
		Coarsen_Threshold = 100;
	else if (Coarsen_Threshold < 40)
		Coarsen_Threshold = 40;

    // printf("Reorderpartition 0\n");
    CONTROL_COMMAND(control, COARSEN_Time, gettimebegin(&start_coarsen, &end_coarsen, &time_coarsen));
	cgraph = CoarsenGraph(graph, Coarsen_Threshold);
	CONTROL_COMMAND(control, COARSEN_Time, gettimeend(&start_coarsen, &end_coarsen, &time_coarsen));    
    // printf("Reorderpartition 1\n");
    // printf("CoarsenGraph end ccnt=%"PRIDX"\n",rand_count());
    // printf("cgraph:\n");
    // exam_nvtxs_nedges(cgraph);
	// exam_xadj(cgraph);
	// exam_vwgt(cgraph);
	// exam_adjncy_adjwgt(cgraph);

	niparts = lyj_max(1, (cgraph->nvtxs <= Coarsen_Threshold ? niparts / 2: niparts));
    CONTROL_COMMAND(control, REORDERBISECTION_Time, gettimebegin(&start_reorderbisection, &end_reorderbisection, &time_reorderbisection));
	ReorderBisection(cgraph, niparts);
	CONTROL_COMMAND(control, REORDERBISECTION_Time, gettimeend(&start_reorderbisection, &end_reorderbisection, &time_reorderbisection));    

    // printf("Reorderpartition 2\n");
    // printf("ReorderBisection end ccnt=%"PRIDX"\n",rand_count());
    // printf("InitSeparator\n");
    // exam_where(cgraph);

    CONTROL_COMMAND(control, REFINE2WAYNODE_Time, gettimebegin(&start_refine2waynode, &end_refine2waynode, &time_refine2waynode));
	Refine2WayNode(cgraph, graph);
	CONTROL_COMMAND(control, REFINE2WAYNODE_Time, gettimeend(&start_refine2waynode, &end_refine2waynode, &time_refine2waynode));    

    // printf("Reorderpartition 3\n");
    // printf("L1 Refine2WayNode end ccnt=%"PRIDX"\n",rand_count());
    // printf("L1\n");
    // exam_where(graph);

    /*// choose the best one later
    Hunyuan_int_t Coarsen_Threshold = lyj_max(graph->nvtxs / 8, 128);
    printf("Reorderpartition 0\n");
    printf("Coarsen_Threshold=%d\n",Coarsen_Threshold);
    graph_t *cgraph = graph;
    cgraph = CoarsenGraph(graph,Coarsen_Threshold);
    printf("Reorderpartition 1\n");
    ReorderBisection(cgraph);
    printf("Reorderpartition 2\n");
    ReorderRefinement(cgraph, graph);*/
}

/*************************************************************************/
/*! This version of the main idea of the Bisection function
		*1 the coarsening is divided Hunyuan_int_to two times
		*2 the graph is small in second coarsening, 
            so can execute a few times to select the best one
		*3 the vertex weight constraHunyuan_int_t in coarsening is removed
*/
/*************************************************************************/
void Bisection(graph_t *graph, Hunyuan_int_t niparts)
{
	Hunyuan_int_t i, mincut, nruns = 5;
	graph_t *cgraph; 
	Hunyuan_int_t *bestwhere;

	/* if the graph is small, just find a single vertex separator */
	if (graph->nvtxs < 5000) 
	{
        // printf("Bisection 0\n");
		Reorderpartition(graph, niparts);
        // printf("Bisection 1\n");
		return;
	}

	Hunyuan_int_t Coarsen_Threshold = lyj_max(100, graph->nvtxs / 30);
	
    // printf("Bisection 00\n");
    CONTROL_COMMAND(control, COARSEN_Time, gettimebegin(&start_coarsen, &end_coarsen, &time_coarsen));
	cgraph = CoarsenGraphNlevels_metis(graph, Coarsen_Threshold, 4);
	CONTROL_COMMAND(control, COARSEN_Time, gettimeend(&start_coarsen, &end_coarsen, &time_coarsen));    

    // printf("Bisection 11\n");
    // printf("cgraph:\n");
    // exam_nvtxs_nedges(cgraph);
	// exam_xadj(cgraph);
	// exam_vwgt(cgraph);
	// exam_adjncy_adjwgt(cgraph);

	bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * cgraph->nvtxs, "Bisection: bestwhere");
    
	mincut = graph->tvwgt[0];
	for (i = 0; i < nruns; i++) 
	{
        // printf("Bisection 2\n");
		Reorderpartition(cgraph, 0.7 * niparts);
        // printf("Bisection 3\n");

		if (i == 0 || cgraph->mincut < mincut) 
		{
			mincut = cgraph->mincut;
			if (i < nruns - 1)
				copy_int(cgraph->nvtxs, cgraph->where, bestwhere);
		}

		if (mincut == 0)
			break;

		if (i < nruns - 1) 
			FreeRefineData(cgraph);
        // printf("Bisection 4\n");
	}

	if (mincut != cgraph->mincut) 
		copy_int(cgraph->nvtxs, bestwhere, cgraph->where);
    check_free(bestwhere, sizeof(Hunyuan_int_t) * cgraph->nvtxs, "Bisection: bestwhere");

    // printf("Bisection 5\n");
    CONTROL_COMMAND(control, REFINE2WAYNODE_Time, gettimebegin(&start_refine2waynode, &end_refine2waynode, &time_refine2waynode));
	Refine2WayNode(cgraph, graph);
	CONTROL_COMMAND(control, REFINE2WAYNODE_Time, gettimeend(&start_refine2waynode, &end_refine2waynode, &time_refine2waynode));    

    // printf("Bisection 6\n");
    // printf("L2 Refine2WayNode end ccnt=%"PRIDX"\n",rand_count());
    // exam_where(graph);

    /*// choose the best one later
    Hunyuan_int_t Coarsen_Threshold = lyj_max(graph->nvtxs / 30, 5000);
    printf("Bisection 0\n");
    //  yu han can do it
    graph_t *cgraph = graph;
    printf("Coarsen_Threshold=%d\n",Coarsen_Threshold);
    if(graph->nvtxs >= Coarsen_Threshold)
        cgraph = CoarsenGraph(graph,Coarsen_Threshold);

    //  Write first The second roughening is done only once
    // Hunyuan_int_t *bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "Bisection: bestwhere");
    
    printf("Bisection 1\n");
    Reorderpartition(cgraph);
    printf("Bisection 2\n");
    // ReorderRefinement(cgraph, graph);*/
}

void BisectionBest(graph_t *graph)
{
	/* if the graph is small, just find a single vertex separator */
	if (1 || graph->nvtxs < (0 ? 1000 : 2000)) 
	{
        // printf("BisectionBest 0\n");
		Bisection(graph, 7);
        // printf("BisectionBest 1\n");
		return;
	}
}

void NestedBisection(graph_t *graph, Hunyuan_int_t *reflect, Hunyuan_int_t reordernum, Hunyuan_int_t level)
{
    Hunyuan_int_t nbnd, *bndind, *label;
    graph_t *lgraph, *rgraph;

    // printf("NestedBisection 0\n");
    // printf("BisectionBest begin ccnt=%"PRIDX"\n",rand_count());
    //  Bisection
    CONTROL_COMMAND(control, BISECTIONBEST_Time, gettimebegin(&start_bisectionbest, &end_bisectionbest, &time_bisectionbest));
	BisectionBest(graph);
	CONTROL_COMMAND(control, BISECTIONBEST_Time, gettimeend(&start_bisectionbest, &end_bisectionbest, &time_bisectionbest));

    // printf("BisectionBest end ccnt=%"PRIDX"\n",rand_count());

    //  check function SplitGraphoRerder
    // graph->where = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "NestedBisection: where");
    // for(Hunyuan_int_t i = 0;i < graph->nvtxs / 3;i++)
    //    graph->where[i] = 0;
    // for(Hunyuan_int_t i = graph->nvtxs / 3;i < graph->nvtxs / 3 * 2;i++)
    //    graph->where[i] = 2;
    // for(Hunyuan_int_t i = graph->nvtxs / 3 * 2;i < graph->nvtxs;i++)
    //    graph->where[i] = 1;
    // exam_where(graph);
    // printf("NestedBisection 1\n");
    //  set up reflect
    nbnd   = graph->nbnd;
    bndind = graph->bndind;
    label  = graph->label;
    for(Hunyuan_int_t i = 0;i < graph->nbnd;i++)
        reflect[label[bndind[i]]] = --reordernum;
    // exam_num(reflect,graph->nvtxs);
    // printf("NestedBisection 2\n");
    //  set up subgraph
    CONTROL_COMMAND(control, SPLITGRAPHREORDER_Time, gettimebegin(&start_splitgraphreorder, &end_splitgraphreorder, &time_splitgraphreorder));
	SplitGraphReorder(graph, &lgraph, &rgraph);
	CONTROL_COMMAND(control, SPLITGRAPHREORDER_Time, gettimeend(&start_splitgraphreorder, &end_splitgraphreorder, &time_splitgraphreorder));    
    // printf("level=%"PRIDX"\n",level++);
    // printf("NestedBisection 3\n");
    FreeGraph(&graph);

    // if(level == 4)
    //     return ;
    
    //  Nest
    if(lgraph->nvtxs > 120 && lgraph->nedges > 0)
    {
        // printf("lgraph 0\n");
        NestedBisection(lgraph, reflect, reordernum - rgraph->nvtxs, level);
        // printf("lgraph 1\n");
    }
    else
    {
        // printf("MMD_Order lgraph 0\n");
        CONTROL_COMMAND(control, MMDORDER_Time, gettimebegin(&start_mmdorder, &end_mmdorder, &time_mmdorder));
        // MMD_Order_line(lgraph, reflect, reordernum - rgraph->nvtxs);
        MMD_Order(lgraph, reflect,reordernum - rgraph->nvtxs);
        CONTROL_COMMAND(control, MMDORDER_Time, gettimeend(&start_mmdorder, &end_mmdorder, &time_mmdorder));
        // printf("MMD_Order lgraph 1\n");
        FreeGraph(&lgraph);
        // printf("MMD_Order lgraph 2\n");
    }
    if(rgraph->nvtxs > 120 && rgraph->nedges > 0)
    {
        // printf("lgraph 0\n");
        NestedBisection(rgraph, reflect, reordernum, level);
        // printf("lgraph 1\n");
    }
    else
    {
        // printf("MMD_Order rgraph 0\n");
        CONTROL_COMMAND(control, MMDORDER_Time, gettimebegin(&start_mmdorder, &end_mmdorder, &time_mmdorder));
        // MMD_Order_line(rgraph, reflect, reordernum);
        MMD_Order(rgraph, reflect,reordernum);
        CONTROL_COMMAND(control, MMDORDER_Time, gettimeend(&start_mmdorder, &end_mmdorder, &time_mmdorder));
        // printf("MMD_Order rgraph 1\n");
        FreeGraph(&rgraph);
        // printf("MMD_Order rgraph 2\n");
    }
}

void ReorderGraph(Hunyuan_int_t *nvtxs, Hunyuan_int_t *nedges, Hunyuan_int_t *xadj, Hunyuan_int_t *vwgt, Hunyuan_int_t *adjncy, Hunyuan_int_t *adjwgt, 
    Hunyuan_int_t *reflect, bool is_memery_manage_before)
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

    // printf("irandInRange(nvtxs)=%"PRIDX"\n",irandInRange(nvtxs[0]));

    // printf("ReorderGraph 1\n");

    //  set up graph
    graph_t *graph = SetupGraph(nvtxs[0], xadj, adjncy, vwgt, adjwgt); 
    // printf("graph->tvwgt[0]=%"PRIDX" cgraph->invtvwgt[0]=%lf\n",\
        graph->tvwgt[0],graph->invtvwgt[0]);

    // Hunyuan_int_t *reflect = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs[0], "ReorderGraph: reflect");
    set_value_int(nvtxs[0],-1,reflect);
    // printf("ReorderGraph 2\n");

    CONTROL_COMMAND(control, NESTEDBISECTION_Time, gettimebegin(&start_nestedbisection, &end_nestedbisection, &time_nestedbisection));
	NestedBisection(graph, reflect, reordernum, 0);
	CONTROL_COMMAND(control, NESTEDBISECTION_Time, gettimeend(&start_nestedbisection, &end_nestedbisection, &time_nestedbisection));

    // printf("ReorderGraph 3\n");
    // printf("NestedBisection end ccnt=%"PRIDX"\n",rand_count());
    // exam_num(reflect, nvtxs[0]);

    // for (Hunyuan_int_t i = 0; i < nvtxs[0]; i++)
    //     ans[reflect[i]] = i;

    // exam_num(ans, nvtxs[0]);

    //  free memery manage
    if(!is_memery_manage_before)
        free_memory_block();

    CONTROL_COMMAND(control, ALL_Time, gettimeend(&start_all, &end_all, &time_all));
}

#endif