#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>
#include "struct.h"
#include "memory.h"

/*************************************************************************/
/*! This function initializes a graph_t data structure */
/*************************************************************************/
void InitGraph(graph_t *graph) 
{
	memset((void *)graph, 0, sizeof(graph_t));

	/* graph size constants */
	graph->nvtxs     = -1;
	graph->nedges    = -1;
	// graph->ncon      = -1;
	graph->mincut    = -1;
	graph->minvol    = -1;
	graph->nbnd      = -1;
	graph->cnbr_size = 0;

	/* memory for the graph structure */
	graph->xadj      = NULL;
	graph->vwgt      = NULL;
	// graph->vsize     = NULL;
	graph->adjncy    = NULL;
	graph->adjwgt    = NULL;
	graph->label     = NULL;
	graph->cmap      = NULL;
	graph->match     = NULL;
	graph->tvwgt     = NULL;
	graph->invtvwgt  = NULL;

	/* memory for the partition/refinement structure */
	graph->where     = NULL;
	graph->pwgts     = NULL;
	graph->id        = NULL;
	graph->ed        = NULL;
	graph->bndptr    = NULL;
	graph->bndind    = NULL;
	graph->nrinfo    = NULL;
	graph->ckrinfo   = NULL;
	graph->cnbr      = NULL;
	// graph->vkrinfo   = NULL;

	/* linked-list structure */
	graph->coarser   = NULL;
	graph->finer     = NULL;
}


/*************************************************************************/
/*! This function creates and initializes a graph_t data structure */
/*************************************************************************/
graph_t *CreateGraph(void)
{
	graph_t *graph;

	graph = (graph_t *)check_malloc(sizeof(graph_t), "CreateGraph: graph");

	InitGraph(graph);

	return graph;
}

/*************************************************************************/
/*! Set's up the tvwgt/invtvwgt info */
/*************************************************************************/
void SetupGraph_tvwgt(graph_t *graph)
{
	if (graph->tvwgt == NULL) 
		graph->tvwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 1, "SetupGraph_tvwgt: tvwgt");
	if (graph->invtvwgt == NULL) 
		graph->invtvwgt = (double *)check_malloc(sizeof(double) * 1, "SetupGraph_tvwgt: invtvwgt");

	for (Hunyuan_int_t i = 0; i < 1; i++) 
	{
		graph->tvwgt[i]    = sum_int(graph->nvtxs, graph->vwgt + i, 1);
		graph->invtvwgt[i] = 1.0 / (graph->tvwgt[i] > 0 ? graph->tvwgt[i] : 1);
	}
}


/*************************************************************************/
/*! Set's up the label info */
/*************************************************************************/
void SetupGraph_label(graph_t *graph)
{
	if (graph->label == NULL)
		graph->label = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "SetupGraph_label: label");

	for (Hunyuan_int_t i = 0; i < graph->nvtxs; i++)
		graph->label[i] = i;
}

/*************************************************************************/
/*! This function sets up the graph from the user input */
/*************************************************************************/
graph_t *SetupGraph(Hunyuan_int_t nvtxs, Hunyuan_int_t *xadj, Hunyuan_int_t *adjncy, Hunyuan_int_t *vwgt, Hunyuan_int_t *adjwgt) 
{
	Hunyuan_int_t i, j, k, sum;
	double *nvwgt;
	graph_t *graph;

	/* allocate the graph and fill in the fields */
	graph = CreateGraph();

	graph->nvtxs  = nvtxs;
	graph->nedges = xadj[nvtxs];

	graph->xadj      = xadj;
	graph->adjncy    = adjncy;

	/* setup the vertex weights */
	if (vwgt) 
		graph->vwgt      = vwgt;
	else 
	{
		vwgt = graph->vwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "SetupGraph: vwgt");
		set_value_int(nvtxs, 1, vwgt);
	}

	graph->tvwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t), "SetupGraph: tvwgts");
	graph->invtvwgt = (double *)check_malloc(sizeof(double), "SetupGraph: invtvwgts");
	for (i = 0; i < 1; i++) 
	{
		graph->tvwgt[i]    = sum_int(nvtxs, vwgt + i, 1);
		graph->invtvwgt[i] = 1.0 / (graph->tvwgt[i] > 0 ? graph->tvwgt[i] : 1);
	}

	// if (ctrl->objtype == OBJTYPE_VOL) 
	// { 
	// 	/* Setup the vsize */
	// 	if (vsize) 
	// 	{
	// 		graph->vsize      = vsize;
	// 		graph->free_vsize = 0;
	// 	}
	// 	else 
	// 	{
	// 		vsize = graph->vsize = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "SetupGraph: vsize");
	// 		set_value_int(nvtxs, 1, vsize);
	// 	}

	// 	/* Allocate memory for edge weights and initialize them to the sum of the vsize */
	// 	adjwgt = graph->adjwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "SetupGraph: adjwgt");
	// 	for (i = 0; i < nvtxs; i++) 
	// 	{
	// 		for (j = xadj[i]; j < xadj[i + 1]; j++)
	// 			adjwgt[j] = 1 + vsize[i] + vsize[adjncy[j]];
	// 	}
	// }
	// else 
	// { /* For edgecut minimization */
		/* setup the edge weights */
		if (adjwgt) 
			graph->adjwgt = adjwgt;
		else 
		{
			adjwgt = graph->adjwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "SetupGraph: adjwgt");
			set_value_int(graph->nedges, 1, adjwgt);
		}
	// }


	/* setup various derived info */
	SetupGraph_tvwgt(graph);

	// if (ctrl->optype == OP_PMETIS || ctrl->optype == OP_OMETIS) 
		SetupGraph_label(graph);

	return graph;
}

/*************************************************************************/
/*! Setup the various arrays for the coarse graph 
 */
/*************************************************************************/
graph_t *SetupCoarseGraph(graph_t *graph, Hunyuan_int_t cnvtxs)
{
	graph_t *cgraph = CreateGraph();

	cgraph->nvtxs = cnvtxs;

	cgraph->finer  = graph;
	graph->coarser = cgraph;

	/* Allocate memory for the coarser graph */
	cgraph->xadj     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (cnvtxs + 1), "SetupCoarseGraph: xadj");
	// cgraph->adjncy   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "SetupCoarseGraph: adjncy");
	// cgraph->adjwgt   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "SetupCoarseGraph: adjwgt");
	cgraph->vwgt     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * cnvtxs, "SetupCoarseGraph: vwgt");
	cgraph->tvwgt    = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t), "SetupCoarseGraph: tvwgt");
	cgraph->invtvwgt = (double *)check_malloc(sizeof(double), "SetupCoarseGraph: invtvwgt");

	return cgraph;
}

/*************************************************************************/
/*! Setup the various arrays for the splitted graph */
/*************************************************************************/
graph_t *SetupSplitGraph(graph_t *graph, Hunyuan_int_t subnvtxs, Hunyuan_int_t subnedges)
{
	graph_t *subgraph = CreateGraph();

	subgraph->nvtxs  = subnvtxs;
	subgraph->nedges = subnedges;

	/* Allocate memory for the splitted graph */
	subgraph->xadj     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (subnvtxs + 1), "SetupSplitGraph: xadj");
	subgraph->vwgt     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * subnvtxs, "SetupSplitGraph: vwgt");
	subgraph->adjncy   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * subnedges,  "SetupSplitGraph: adjncy");
	subgraph->adjwgt   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * subnedges,  "SetupSplitGraph: adjwgt");
	subgraph->label	   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * subnvtxs,   "SetupSplitGraph: label");
	subgraph->tvwgt    = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t), "SetupSplitGraph: tvwgt");
	subgraph->invtvwgt = (double *)check_malloc(sizeof(double), "SetupSplitGraph: invtvwgt");

	return subgraph;
}

/*************************************************************************/
/*! This function frees the refinement/partition memory stored in a graph */
/*************************************************************************/
void FreeRefineData(graph_t *graph, Hunyuan_int_t nparts) 
{
	/* The following is for the -minconn and -contig to work properly in
		the vol-refinement routines */
	// if ((void *)graph->ckrinfo == (void *)graph->vkrinfo)
	// 	graph->ckrinfo = NULL;

	/* free partition/refinement structure */
	if(graph->where != NULL)
		check_free(graph->where, sizeof(Hunyuan_int_t) * graph->nvtxs, "FreeRefineData: graph->where");
	if(graph->nrinfo != NULL) 
		check_free(graph->nrinfo, sizeof(nrinfo_t) * graph->nvtxs, "FreeRefineData: graph->nrinfo");
	if(graph->ed != NULL) 
		check_free(graph->ed, sizeof(Hunyuan_int_t) * graph->nvtxs, "FreeRefineData: graph->ed");
	if(graph->id != NULL) 
		check_free(graph->id, sizeof(Hunyuan_int_t) * graph->nvtxs, "FreeRefineData: graph->id");
	if(graph->bndind != NULL) 
		check_free(graph->bndind, sizeof(Hunyuan_int_t) * graph->nvtxs, "FreeRefineData: graph->bndind");
	if(graph->bndptr != NULL) 
		check_free(graph->bndptr, sizeof(Hunyuan_int_t) * graph->nvtxs, "FreeRefineData: graph->bndptr");
	if(graph->pwgts != NULL) 
		check_free(graph->pwgts, sizeof(Hunyuan_int_t) * nparts, "FreeRefineData: graph->pwgts");
	if(graph->ckrinfo != NULL) 
		check_free(graph->ckrinfo, sizeof(ckrinfo_t) * graph->nvtxs, "FreeRefineData: graph->ckrinfo");
	if(graph->cnbr != NULL)
		check_free(graph->cnbr, sizeof(cnbr_t) * graph->cnbr_size, "FreeRefineData: graph->cnbr");
	// check_free(graph->vkrinfo);
}

/*************************************************************************/
/*! This function deallocates any memory stored in a graph */
/*************************************************************************/
void FreeGraph(graph_t **r_graph, Hunyuan_int_t nparts) 
{
	graph_t *graph = *r_graph;

	/* free partition/refinement structure */
	FreeRefineData(graph, nparts);
	
	/* free graph structure */
	if(graph->cmap != NULL) 
		check_free(graph->cmap, sizeof(Hunyuan_int_t) * graph->nvtxs, "FreeGraph: graph->cmap");
	if(graph->match != NULL) 
		check_free(graph->match, sizeof(Hunyuan_int_t) * graph->nvtxs, "FreeGraph:graph->match");
	if(graph->adjwgt != NULL) 
		check_free(graph->adjwgt, sizeof(Hunyuan_int_t) * graph->nedges, "FreeGraph: graph->adjwgt");
	if(graph->adjncy != NULL) 
		check_free(graph->adjncy, sizeof(Hunyuan_int_t) * graph->nedges, "FreeGraph: graph->adjncy");
	if(graph->invtvwgt != NULL) 
		check_free(graph->invtvwgt, sizeof(double), "FreeGraph: graph->invtvwgt");
	if(graph->tvwgt != NULL) 
		check_free(graph->tvwgt, sizeof(Hunyuan_int_t), "FreeGraph: graph->tvwgt");
	if(graph->vwgt != NULL) 
		check_free(graph->vwgt, sizeof(Hunyuan_int_t) * graph->nvtxs, "FreeGraph: graph->vwgt");
	if(graph->xadj != NULL) 
		check_free(graph->xadj, sizeof(Hunyuan_int_t) * (graph->nvtxs + 1), "FreeGraph: graph->xadj");
	
	if(graph->label != NULL) 
		check_free(graph->label, sizeof(Hunyuan_int_t) * graph->nvtxs, "FreeGraph: graph->label");

	check_free(graph, sizeof(graph_t), "FreeGraph: graph");
	
	*r_graph = NULL;
}

/*************************************************************************/
/*! This function changes the numbering to start from 0 instead of 1 */
/*************************************************************************/
void Change2CNumbering(Hunyuan_int_t nvtxs, Hunyuan_int_t *xadj, Hunyuan_int_t *adjncy)
{
	for (Hunyuan_int_t i = 0; i <= nvtxs; i++)
		xadj[i]--;
	for (Hunyuan_int_t i = 0; i < xadj[nvtxs]; i++)
		adjncy[i]--;
}

/*************************************************************************/
/*! This function changes the numbering to start from 1 instead of 0 */
/*************************************************************************/
void Change2FNumbering(Hunyuan_int_t nvtxs, Hunyuan_int_t *xadj, Hunyuan_int_t *adjncy, Hunyuan_int_t *vector)
{
	for (Hunyuan_int_t i = 0; i < nvtxs; i++)
		vector[i]++;

	for (Hunyuan_int_t i = 0; i < xadj[nvtxs]; i++)
		adjncy[i]++;

	for (Hunyuan_int_t i = 0; i <= nvtxs; i++)
		xadj[i]++;
}

/*************************************************************************/
/*! This function resets the cnbrpool */
/*************************************************************************/
void cnbrReset(graph_t *graph)
{
	graph->cnbr_length = 0;
}

/*************************************************************************/
/*! This function gets the next free index from cnbrpool */
/*************************************************************************/
Hunyuan_int_t cnbrGetNext(graph_t *graph, Hunyuan_int_t add_size)
{
  	graph->cnbr_length += add_size;

	if (graph->cnbr_length > graph->cnbr_size) 
	{
		Hunyuan_int_t old_size = graph->cnbr_size;
		graph->cnbr_size += lyj_max(10 * add_size, graph->cnbr_size / 2);

		graph->cnbr = (cnbr_t *)check_realloc(graph->cnbr,  old_size * sizeof(cnbr_t),
							graph->cnbr_size * sizeof(cnbr_t), "cnbrGetNext: graph->cnbr");
		graph->cnb_reallocs++;
	}

	return graph->cnbr_length - add_size;
}

void exam_nvtxs_nedges(graph_t *graph)
{
    printf("nvtxs:%"PRIDX" nedges:%"PRIDX"\n",graph->nvtxs,graph->nedges);
}

void exam_xadj(graph_t *graph)
{
    printf("xadj:\n");
    for(Hunyuan_int_t i = 0;i <= graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->xadj[i]);
	printf("\n");
}

void exam_vwgt(graph_t *graph)
{
    printf("vwgt:\n");
    for(Hunyuan_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->vwgt[i]);
	printf("\n");
}

void exam_adjncy_adjwgt(graph_t *graph)
{
    printf("adjncy adjwgt:\n");
    for(Hunyuan_int_t i = 0;i < graph->nvtxs;i++)
	{
		printf("ncy:");
		for(Hunyuan_int_t j = graph->xadj[i];j < graph->xadj[i + 1];j++)
			printf("%"PRIDX" ",graph->adjncy[j]);
		printf("\nwgt:");
		for(Hunyuan_int_t j = graph->xadj[i];j < graph->xadj[i + 1];j++)
			printf("%"PRIDX" ",graph->adjwgt[j]);
		printf("\n");
	}
}

void exam_label(graph_t *graph)
{
    printf("label:\n");
    for(Hunyuan_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->label[i]);
	printf("\n");
}

void exam_where(graph_t *graph)
{
    printf("where:\n");
    for(Hunyuan_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->where[i]);
	printf("\n");
}

void exam_pwgts(graph_t *graph, Hunyuan_int_t nparts)
{
    printf("pwgts:");
    for(Hunyuan_int_t i = 0;i < nparts;i++)
		printf("%"PRIDX" ",graph->pwgts[i]);
	printf("\n");
}

void exam_edid(graph_t *graph)
{
    printf("edid:\n");
	printf("ed:");
    for(Hunyuan_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->ed[i]);
	printf("\n");
	printf("id:");
    for(Hunyuan_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->id[i]);
	printf("\n");
}

void exam_bnd(graph_t *graph)
{
    printf("bnd:\n");
	printf("nbnd=%"PRIDX"\n",graph->nbnd);
	printf("bndind:\n");
    for(Hunyuan_int_t i = 0;i < graph->nbnd;i++)
		printf("%"PRIDX" ",graph->bndind[i]);
	printf("\n");
	printf("bndptr:\n");
    for(Hunyuan_int_t i = 0;i < graph->nvtxs;i++)
		printf("%"PRIDX" ",graph->bndptr[i]);
	printf("\n");
}

void exam_nrinfo(graph_t *graph)
{
	printf("nrinfo.edegrees[0]:\n");
    for(Hunyuan_int_t i = 0;i < graph->nbnd;i++)
		printf("%"PRIDX" ",graph->nrinfo[i].edegrees[0]);
	printf("\n");
	printf("nrinfo.edegrees[1]:\n");
    for(Hunyuan_int_t i = 0;i < graph->nbnd;i++)
		printf("%"PRIDX" ",graph->nrinfo[i].edegrees[1]);
	printf("\n");
}

void exam_num(Hunyuan_int_t *num, Hunyuan_int_t n)
{
    printf("num:\n");
    for(Hunyuan_int_t i = 0;i < n;i++)
		printf("%"PRIDX" ",num[i]);
	printf("\n");
}

void exam_tpwgts(Hunyuan_real_t *num, Hunyuan_int_t n)
{
    printf("tpwgts:\n");
    for(Hunyuan_int_t i = 0;i < n;i++)
		printf("%.10"PRREAL" ",num[i]);
	printf("\n");
}

bool exam_correct_gp_where(Hunyuan_int_t *where, Hunyuan_int_t nvtxs, Hunyuan_int_t nparts)
{
	Hunyuan_int_t cnt = 0;
	for(Hunyuan_int_t i = 0;i < nvtxs;i++)
	{
		Hunyuan_int_t t = where[i];
		if(t < 0 || t >= nparts)	
		{
			// printf("t=%"PRIDX" t=%"PRIDX"\n",i,t);
			cnt++;
		}
	}

	if(cnt == 0) 
	{
		// printf("cnt=%"PRIDX"\n",cnt);
		return true;
	}
	else 
	{
		printf("cnt=%"PRIDX"\n",cnt);
		return false;
	}
}

bool exam_is_balance(Hunyuan_int_t *pwgts, Hunyuan_int_t *tvwgt, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor)
{
	Hunyuan_int_t *max_wgts, *min_wgts;

	max_wgts = (Hunyuan_int_t *)check_malloc(nparts * sizeof(Hunyuan_int_t), "exam_is_balance: max_wgts");
	min_wgts = (Hunyuan_int_t *)check_malloc(nparts * sizeof(Hunyuan_int_t), "exam_is_balance: min_wgts");

	for(Hunyuan_int_t i = 0;i < nparts;i++)
	{
		max_wgts[i] = ceilUp((Hunyuan_real_t)tvwgt[0] * tpwgts[i] * balance_factor[0]);
		min_wgts[i] = floorDown((Hunyuan_real_t)tvwgt[0] * tpwgts[i] / balance_factor[0]);
	}

	for(Hunyuan_int_t i = 0;i < nparts;i++)
	{
		if(pwgts[i] > max_wgts[i] || pwgts[i] < min_wgts[i])
		{
			printf("i=%"PRIDX" pwgts=%"PRIDX" max=%"PRIDX" min=%"PRIDX"\n", i, pwgts[i], max_wgts[i], min_wgts[i]);
			return false;
		}
	}

	check_free(max_wgts, nparts * sizeof(Hunyuan_int_t), "exam_is_balance: max_wgts");
	check_free(min_wgts, nparts * sizeof(Hunyuan_int_t), "exam_is_balance: min_wgts");

	return true;
}

bool exam_correct_gp(Hunyuan_int_t *result, Hunyuan_int_t nvtxs, Hunyuan_int_t nparts)
{
	Hunyuan_int_t cnt = 0;
	for(Hunyuan_int_t i = 0;i < nvtxs;i++)
	{
		Hunyuan_int_t t = result[i];
		if(t < 0 || t >= nparts)	
		{
			// printf("t=%"PRIDX" t=%"PRIDX"\n",i,t);
			cnt++;
		}
	}

	if(cnt == 0) 
	{
		// printf("cnt=%"PRIDX"\n",cnt);
		return true;
	}
	else 
	{
		printf("cnt=%"PRIDX"\n",cnt);
		return false;
	}
}

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

Hunyuan_int_t compute_adjwgtsum(graph_t *graph)
{
	Hunyuan_int_t sum_adjwgt = 0;

	sum_adjwgt = sum_int(graph->nedges, graph->adjwgt, 1);

	return sum_adjwgt;
}

#endif