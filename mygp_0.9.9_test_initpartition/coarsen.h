#ifndef COARSEN_H
#define COARSEN_H

#include "struct.h"
#include "memory.h"
#include "match.h"
#include "createcoarsegraph.h"
#include "timer.h"

graph_t *CoarsenGraph(graph_t *graph, Hunyuan_int_t Coarsen_Threshold)
{
	Hunyuan_int_t i, eqewgts, level, maxvwgt, cnvtxs;

	/* determine if the weights on the edges are all the same */
	for (eqewgts = 1, i = 1; i < graph->nedges; i++) 
	{
		if (graph->adjwgt[0] != graph->adjwgt[i]) 
		{
			eqewgts = 0;
			break;
		}
	}

	/* set the maximum allowed coarsest vertex weight */
	for (i = 0; i < 1; i++)
		maxvwgt = 1.5 * graph->tvwgt[i] / Coarsen_Threshold;

	level = 0;

	do 
	{
		// printf("level=%"PRIDX" rand_count=%"PRIDX"\n",level, rand_count());
		/* allocate memory for cmap, if it has not already been done due to
			multiple cuts */
		if (graph->match == NULL)
			graph->match = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "CoarsenGraph: graph->match");
		if (graph->cmap == NULL)
			graph->cmap = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "CoarsenGraph: graph->cmap");
		// if (graph->where == NULL)
		// 	graph->where  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "CoarsenGraph: graph->where");
		/* determine which matching scheme you will use */

		// printf("CoarsenGraph 0\n");
		CONTROL_COMMAND(control, MATCH_Time, gettimebegin(&start_match, &end_match, &time_match));
		if (eqewgts || graph->nedges == 0)
			cnvtxs = Match_RM(graph, maxvwgt);
		else
		{
			cnvtxs = Match_SHEM_topk(graph, maxvwgt);
			// cnvtxs = Match_SHEM(graph, maxvwgt);
		}
		CONTROL_COMMAND(control, MATCH_Time, gettimeend(&start_match, &end_match, &time_match));
		// printf("CoarsenGraph 1\n");
		// printf("cnvtxs=%"PRIDX"\n",cnvtxs);
		// exam_num(graph->match,graph->nvtxs);

		CONTROL_COMMAND(control, CREATCOARSENGRAPH_Time, gettimebegin(&start_createcoarsengraph, &end_createcoarsengraph, &time_createcoarsengraph));
		// CreateCoarseGraph(graph, cnvtxs);
		// CreateCoarseGraph_S(graph, cnvtxs);
		// CreateCoarseGraph_BST(graph, cnvtxs);
		// CreateCoarseGraph_BST_2(graph, cnvtxs);
		// CreateCoarseGraph_HT(graph, cnvtxs);
		CreateCoarseGraph_HT_2(graph, cnvtxs);
		CONTROL_COMMAND(control, CREATCOARSENGRAPH_Time, gettimeend(&start_createcoarsengraph, &end_createcoarsengraph, &time_createcoarsengraph));
		// printf("CoarsenGraph 2\n");
		// printf("CreateCoarseGraph=%10.3"PRREAL"\n",time_createcoarsengraph);

		graph = graph->coarser;
		eqewgts = 0;
		level++;

		//  sort for adjncy adjwgt
		// for(Hunyuan_int_t z = 0;z < graph->nvtxs;z++)
		// {
		// 	for(Hunyuan_int_t y = graph->xadj[z];y < graph->xadj[z + 1];y++)
		// 	{
		// 		Hunyuan_int_t t = y;
		// 		for(Hunyuan_int_t x = y + 1;x < graph->xadj[z + 1];x++)
		// 			if(graph->adjncy[x] < graph->adjncy[t]) t = x;
		// 		Hunyuan_int_t temp;
		// 		temp = graph->adjncy[t],graph->adjncy[t] = graph->adjncy[y], graph->adjncy[y] = temp;
		// 		temp = graph->adjwgt[t],graph->adjwgt[t] = graph->adjwgt[y], graph->adjwgt[y] = temp;
		// 	}
		// }
		if(Coarsen_Threshold != 20)
			printf("level=%"PRIDX" nvtxs=%"PRIDX" nedges=%"PRIDX" adjwgtsum=%"PRIDX" \n",level, graph->nvtxs, graph->nedges, compute_adjwgtsum(graph));
		// if(level == 2)
		// 	exit(0);

		// exam_nvtxs_nedges(graph);
        // exam_xadj(graph);
        // exam_vwgt(graph);
        // exam_adjncy_adjwgt(graph);

	} while (graph->nvtxs > Coarsen_Threshold && 
			graph->nvtxs < 0.85 * graph->finer->nvtxs && 
			graph->nedges > graph->nvtxs / 2);

	return graph;
}

/*************************************************************************/
/*! This function takes a graph and creates a sequence of nlevels coarser 
    graphs, where nlevels is an input parameter.
 */
/*************************************************************************/
graph_t *CoarsenGraphNlevels_metis(graph_t *graph, Hunyuan_int_t Coarsen_Threshold, Hunyuan_int_t nlevels)
{
	Hunyuan_int_t i, eqewgts, level, maxvwgt, cnvtxs;

	/* determine if the weights on the edges are all the same */
	for (eqewgts = 1, i = 1; i < graph->nedges; i++) 
	{
		if (graph->adjwgt[0] != graph->adjwgt[i]) 
		{
			eqewgts = 0;
			break;
		}
	}

	/* set the maximum allowed coarsest vertex weight */
	for (i = 0; i < 1; i++)
		maxvwgt = 1.5 * graph->tvwgt[i] / Coarsen_Threshold;

	for (level = 0; level < nlevels; level++) 
	{

		/* allocate memory for cmap, if it has not already been done due to
			multiple cuts */
		if (graph->match == NULL)
			graph->match = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "CoarsenGraph: graph->match");
		if (graph->cmap == NULL)
			graph->cmap = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "CoarsenGraph: graph->cmap");
		if (graph->where == NULL)
			graph->where  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "CoarsenGraph: graph->where");
		
		// printf("CoarsenGraphNlevels 0\n");
    	/* determine which matching scheme you will use */
		CONTROL_COMMAND(control, MATCH_Time, gettimebegin(&start_match, &end_match, &time_match));
        if (eqewgts || graph->nedges == 0)
			cnvtxs = Match_RM(graph, maxvwgt);
        else
			// cnvtxs = Match_SHEM_topk(graph, maxvwgt);
			cnvtxs = Match_SHEM(graph, maxvwgt);
		CONTROL_COMMAND(control, MATCH_Time, gettimeend(&start_match, &end_match, &time_match));
		// printf("CoarsenGraphNlevels 1\n");
		// printf("cnvtxs=%"PRIDX"\n",cnvtxs);

		CONTROL_COMMAND(control, CREATCOARSENGRAPH_Time, gettimebegin(&start_createcoarsengraph, &end_createcoarsengraph, &time_createcoarsengraph));
		// CreateCoarseGraph(graph, cnvtxs);
		// CreateCoarseGraph_S(graph, cnvtxs);
		// CreateCoarseGraph_BST(graph, cnvtxs);
		// CreateCoarseGraph_BST_2(graph, cnvtxs);
		// CreateCoarseGraph_HT(graph, cnvtxs);
		CreateCoarseGraph_HT_2(graph, cnvtxs);
		CONTROL_COMMAND(control, CREATCOARSENGRAPH_Time, gettimeend(&start_createcoarsengraph, &end_createcoarsengraph, &time_createcoarsengraph));

		// printf("CoarsenGraphNlevels 2\n");
		// printf("CreateCoarseGraph=%10.3"PRREAL"\n",time_createcoarsengraph);

		graph = graph->coarser;
		eqewgts = 0;

		//  sort for adjncy adjwgt
		// for(Hunyuan_int_t z = 0;z < graph->nvtxs;z++)
		// {
		// 	for(Hunyuan_int_t y = graph->xadj[z];y < graph->xadj[z + 1];y++)
		// 	{
		// 		Hunyuan_int_t t = y;
		// 		for(Hunyuan_int_t x = y + 1;x < graph->xadj[z + 1];x++)
		// 			if(graph->adjncy[x] < graph->adjncy[t]) t = x;
		// 		Hunyuan_int_t temp;
		// 		temp = graph->adjncy[t],graph->adjncy[t] = graph->adjncy[y], graph->adjncy[y] = temp;
		// 		temp = graph->adjwgt[t],graph->adjwgt[t] = graph->adjwgt[y], graph->adjwgt[y] = temp;
		// 	}
		// }

		if (graph->nvtxs < Coarsen_Threshold || 
			graph->nvtxs > 0.85 * graph->finer->nvtxs || 
			graph->nedges < graph->nvtxs / 2)
		break; 
	}

	return graph;
}


#endif