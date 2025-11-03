#ifndef _H_CPU_COARSEN
#define _H_CPU_COARSEN

#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
#include "hunyuangraph_timer.h"
#include "hunyuangraph_CPU_match.h"
#include "hunyuangraph_CPU_contraction.h"

/*Cpu multilevel coarsen*/
hunyuangraph_graph_t *hunyuangraph_cpu_coarsen(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int i, eqewgts, level=1;

	/* determine if the weights on the edges are all the same */
	for (eqewgts=1, i=1; i<graph->nedges; i++) {
		if (graph->adjwgt[0] != graph->adjwgt[i]) {
			eqewgts = 0;
			break;
		}
	}

	hunyuangraph_admin->maxvwgt = 1.5*graph->tvwgt[0]/hunyuangraph_admin->Coarsen_threshold;
 
	do{
		if(graph->cmap==NULL){
			graph->cmap=(int*)malloc(sizeof(int)*(graph->nvtxs));
		}

		// hunyuangraph_cpu_match_HEM(hunyuangraph_admin,graph,level);
		if (eqewgts || graph->nedges == 0)
          	hunyuangraph_cpu_match_RM(hunyuangraph_admin, graph);
        else
          	hunyuangraph_cpu_match_HEM(hunyuangraph_admin, graph);

		graph = graph->coarser;
		// eqewgts = 0;

		level++;

	}while(graph->nvtxs > hunyuangraph_admin->Coarsen_threshold && 
		graph->nvtxs < 0.85*graph->finer->nvtxs && 
		graph->nedges > graph->nvtxs/2);

  	return graph;
}


#endif