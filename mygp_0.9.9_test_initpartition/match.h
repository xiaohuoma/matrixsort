#ifndef MATCH_H
#define MATCH_H

#include "struct.h"
#include "memory.h"
#include "define.h"
#include "graph.h"
#include "timer.h"

/*************************************************************************/
/*! This function matches the unmatched vertices whose degree is less than
    maxdegree using a 2-hop matching that involves vertices that are two 
    hops away from each other. 
    The requirement of the 2-hop matching is a simple non-empty overlap
    between the adjancency lists of the vertices. */
/**************************************************************************/
Hunyuan_int_t Match_2HopAny(graph_t *graph, Hunyuan_int_t *perm, Hunyuan_int_t *match, Hunyuan_int_t cnvtxs, size_t *r_nunmatched, size_t maxdegree)
{
	Hunyuan_int_t i, pi, ii, j, jj, k, nvtxs;
	Hunyuan_int_t *xadj, *adjncy, *colptr, *rowind;
	Hunyuan_int_t *cmap;
	size_t nunmatched;

	nvtxs  = graph->nvtxs;
	xadj   = graph->xadj;
	adjncy = graph->adjncy;
	cmap   = graph->cmap;

	nunmatched = *r_nunmatched;

	/* create the inverted index */
	colptr = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (nvtxs + 1), "Match_2HopAny: colptr");
	set_value_int(nvtxs + 1, 0, colptr);

	for (i = 0; i < nvtxs; i++) 
	{
		if (match[i] == -1 && xadj[i + 1] - xadj[i] < maxdegree) 
		{
			for (j = xadj[i]; j < xadj[i + 1]; j++)
				colptr[adjncy[j]]++;
		}
	}
	MAKECSR(i, nvtxs, colptr);

	Hunyuan_int_t rowind_size = colptr[nvtxs];
	rowind = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * colptr[nvtxs], "Match_2HopAny: rowind");
	for (pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];
		if (match[i] == -1 && xadj[i + 1] - xadj[i] < maxdegree) 
		{
			for (j = xadj[i]; j < xadj[i + 1]; j++)
				rowind[colptr[adjncy[j]]++] = i;
		}
	}
	SHIFTCSR(i, nvtxs, colptr);

	// printf("Match_2HopAny 0\n");

	/* compute matchings by going down the inverted index */
	for (pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];
		if (colptr[i + 1] - colptr[i] < 2)
			continue;

		for (jj = colptr[i + 1], j = colptr[i]; j < jj; j++) 
		{
			if (match[rowind[j]] == -1) 
			{
				for (jj--; jj > j; jj--) 
				{
					if (match[rowind[jj]] == -1) 
					{
						cmap[rowind[j]] = cmap[rowind[jj]] = cnvtxs++;
						match[rowind[j]]  = rowind[jj];
						match[rowind[jj]] = rowind[j];
						nunmatched -= 2;
						break;
					}
				}
			}
		}
	}

	// printf("Match_2HopAny 1\n");
	check_free(rowind, sizeof(Hunyuan_int_t) * rowind_size, "Match_2HopAny: rowind");
	check_free(colptr, sizeof(Hunyuan_int_t) * (nvtxs + 1), "Match_2HopAny: colptr");

	*r_nunmatched = nunmatched;
	return cnvtxs;
}

/*************************************************************************/
/*! This function matches the unmatched vertices whose degree is less than
    maxdegree using a 2-hop matching that involves vertices that are two 
    hops away from each other. 
    The requirement of the 2-hop matching is that of identical adjacency
    lists.
 */
/**************************************************************************/
Hunyuan_int_t Match_2HopAll(graph_t *graph, Hunyuan_int_t *perm, Hunyuan_int_t *match, Hunyuan_int_t cnvtxs, size_t *r_nunmatched, size_t maxdegree)
{
	Hunyuan_int_t i, pi, pk, ii, j, jj, k, nvtxs, mask, idegree;
	Hunyuan_int_t *xadj, *adjncy;
	Hunyuan_int_t *cmap, *mark;
	ikv_t *keys;
	size_t nunmatched, ncand;

	nvtxs  = graph->nvtxs;
	xadj   = graph->xadj;
	adjncy = graph->adjncy;
	cmap   = graph->cmap;

	nunmatched = *r_nunmatched;
	mask = IDX_MAX / maxdegree;

	/* collapse vertices with identical adjancency lists */
	keys = (ikv_t *)check_malloc(sizeof(ikv_t) * nunmatched, "Match_2HopAll: keys");
	for (ncand = 0, pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];
		idegree = xadj[i + 1] - xadj[i];
		if (match[i] == -1 && idegree > 1 && idegree < maxdegree) 
		{
			for (k = 0, j = xadj[i]; j < xadj[i + 1]; j++) 
				k += adjncy[j] % mask;
			keys[ncand].val = i;
			keys[ncand].key = (k % mask) * maxdegree + idegree;
			ncand++;
		}
	}
	ikvsorti(ncand, keys);

	mark = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_2HopAll: mark");
	set_value_int(nvtxs, 0, mark);
	for (pi = 0; pi < ncand; pi++) 
	{
		i = keys[pi].val;
		if (match[i] != -1)
			continue;
		
		for (j = xadj[i]; j < xadj[i + 1]; j++)
			mark[adjncy[j]] = i;

		for (pk = pi + 1; pk < ncand; pk++) 
		{
			k = keys[pk].val;
			if (match[k] != -1)
				continue;

			if (keys[pi].key != keys[pk].key)
				break;
			if (xadj[i + 1] - xadj[i] != xadj[k + 1] - xadj[k])
				break;
			
			for (jj = xadj[k]; jj < xadj[k + 1]; jj++) 
			{
				if (mark[adjncy[jj]] != i)
					break;
			}
			if (jj == xadj[k+1]) 
			{
				cmap[i] = cmap[k] = cnvtxs++;
				match[i] = k;
				match[k] = i;
				nunmatched -= 2;
				break;
			}
		}
	}

	check_free(mark, sizeof(Hunyuan_int_t) * nvtxs, "Match_2HopAll: mark");
	check_free(keys, sizeof(ikv_t) * (*r_nunmatched), "Match_2HopAll: keys");

	*r_nunmatched = nunmatched;
	return cnvtxs;
}

/*************************************************************************/
/*! This function matches the unmatched vertices using a 2-hop matching 
    that involves vertices that are two hops away from each other. */
/**************************************************************************/
Hunyuan_int_t Match_2Hop(graph_t *graph, Hunyuan_int_t *perm, Hunyuan_int_t *match, Hunyuan_int_t cnvtxs, size_t nunmatched)
{
	Hunyuan_int_t i, me, jj, j, k, nvtxs;
	Hunyuan_int_t cnt, compare, ourless, metis_less;
	Hunyuan_int_t *xadj, *adjncy, *adjwgt;
	
	nvtxs = graph->nvtxs;
	xadj   = graph->xadj;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	
	// printf("Match_2Hop 0\n");
	// leaf match
	cnvtxs = Match_2HopAny(graph, perm, match, cnvtxs, &nunmatched, 2);
	
	// printf("Match_2HopAny match\n");
	// exam_num(match,graph->nvtxs);
	// printf("Match_2Hop 1\n");
	// twin match
	cnvtxs = Match_2HopAll(graph, perm, match, cnvtxs, &nunmatched, 64);

	// printf("Match_2HopAll match\n");
	// exam_num(match,graph->nvtxs);
	// printf("Match_2Hop 2\n");
	if (nunmatched > 1.5 * 0.10 * graph->nvtxs) 
	{
		// printf("Match_2Hop 3\n");
		cnvtxs = Match_2HopAny(graph, perm, match, cnvtxs, &nunmatched, 3);
		// printf("Match_2Hop 4\n");
	}
	if (nunmatched > 2.0 * 0.10 * graph->nvtxs) 
	{
		// printf("Match_2Hop 5\n");
		// relative match
		cnvtxs = Match_2HopAny(graph, perm, match, cnvtxs, &nunmatched, graph->nvtxs);
		// printf("Match_2Hop 6\n");
	}

	return cnvtxs;
}

/*************************************************************************/
/*! This function finds a matching by randomly selecting one of the 
    unmatched adjacent vertices. 
 */
/**************************************************************************/
Hunyuan_int_t Match_RM(graph_t *graph, Hunyuan_int_t maxvwgt)
{
	Hunyuan_int_t i, pi, ii, j, jj, jjinc, k, nvtxs, ncon, cnvtxs, maxidx, last_unmatched;
	Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt;
	Hunyuan_int_t *match, *cmap, *perm;
	size_t nunmatched = 0;

	nvtxs  = graph->nvtxs;
	// ncon   = graph->ncon;
	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;

	// graph->match = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_RM: graph->match");
	match = graph->match;
	set_value_int(nvtxs, -1, match);

	perm  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_RM: perm");

  	irandArrayPermute(nvtxs, perm, nvtxs/8, 1);
	// printf("perm\n");
	// exam_num(perm,nvtxs);

	// printf("Match_RM 0\n");

	for (cnvtxs = 0, last_unmatched = 0, pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];

		if (match[i] == -1) 
		{  
			/* Unmatched */
			maxidx = i;

			if(vwgt[i] < maxvwgt)
			{
				/* Deal with island vertices. Find a non-island and match it with. 
					The matching ignores ctrl->maxvwgt requirements */
				if (xadj[i] == xadj[i + 1]) 
				{
					last_unmatched = lyj_max(pi, last_unmatched) + 1;
					for (; last_unmatched<nvtxs; last_unmatched++) 
					{
						j = perm[last_unmatched];
						if (match[j] == -1) 
						{
							maxidx = j;
							break;
						}
					}
				}
				else
				{
					/* single constraHunyuan_int_t version */
					for (j = xadj[i]; j < xadj[i + 1]; j++) 
					{
						k = adjncy[j];
						if (match[k] == -1 && vwgt[i] + vwgt[k] <= maxvwgt) 
						{
							maxidx = k;
							break;
						}
					}

					/* If it did not match, record for a 2-hop matching. */
					if (maxidx == i && 3 * vwgt[i] < maxvwgt) 
					{
						nunmatched++;
						maxidx = -1;
					}
				}
			}

			if (maxidx != -1) 
			{
				cmap[i]  = cmap[maxidx] = cnvtxs++;
				match[i] = maxidx;
				match[maxidx] = i;
			}
		}
	}

	// printf("Match_RM 1\n");
	// exam_num(match,nvtxs);

  //printf("nunmatched: %zu\n", nunmatched);

	/* see if a 2-hop matching is required/allowed */
	if (nunmatched > 0.1 * nvtxs) 
		cnvtxs = Match_2Hop(graph, perm, match, cnvtxs, nunmatched);

	// printf("Match_RM 2\n");
	// exam_num(match,nvtxs);

	/* match the final unmatched vertices with themselves and reorder the vertices 
		of the coarse graph for memory-friendly contraction */
	for (cnvtxs = 0, i = 0; i < nvtxs; i++) 
	{
		if (match[i] == -1) 
		{
			match[i] = i;
			cmap[i]  = cnvtxs++;
		}
		else 
		{
			if (i <= match[i]) 
				cmap[i] = cmap[match[i]] = cnvtxs++;
		}
	}

	// printf("Match_RM 3\n");
	check_free(perm, sizeof(Hunyuan_int_t) * nvtxs, "Match_RM: perm");
	// exam_num(match,nvtxs);

	return cnvtxs;
}

/*************************************************************************
* This function uses simple counting sort to return a permutation array
* corresponding to the sorted order. The keys are arsumed to start from
* 0 and they are positive.  This sorting is used during matching.
**************************************************************************/
void BucketSortKeysInc(Hunyuan_int_t n, Hunyuan_int_t max, Hunyuan_int_t *keys, Hunyuan_int_t *tperm, Hunyuan_int_t *perm)
{
	Hunyuan_int_t i, ii;
	Hunyuan_int_t *counts;

	counts = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (max + 2), "BucketSortKeysInc: counts");
	set_value_int(max + 2, 0, counts);

	for (i = 0; i < n; i++)
		counts[keys[i]]++;
	MAKECSR(i, max + 1, counts);

	for (ii = 0; ii < n; ii++) 
	{
		i = tperm[ii];
		perm[counts[keys[i]]++] = i;
	}

	check_free(counts, sizeof(Hunyuan_int_t) * (max + 2), "BucketSortKeysInc: counts");
}

/**************************************************************************/
/*! This function finds a matching using the HEM heuristic. The vertices 
    are visited based on increasing degree to ensure that all vertices are 
    given a chance to match with something. 
 */
/**************************************************************************/
Hunyuan_int_t Match_SHEM(graph_t *graph, Hunyuan_int_t maxvwgt)
{
	Hunyuan_int_t i, pi, ii, j, jj, jjinc, k, nvtxs, ncon, cnvtxs, maxidx, maxwgt, 
			last_unmatched, avgdegree;
	Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt;
	Hunyuan_int_t *match, *cmap, *degrees, *perm, *tperm;
	// Hunyuan_int_t *receive, *send, *tmatch;
	Hunyuan_int_t nunmatched = 0;
	// Hunyuan_int_t cnt, compare, ourless, metis_less;

	nvtxs  = graph->nvtxs;
	// ncon   = graph->ncon;
	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;

	match = graph->match;
	set_value_int(nvtxs, -1, match);
	perm = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: perm");
	tperm = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: tperm");
	degrees = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: degrees");

	irandArrayPermute(nvtxs, tperm, nvtxs/8, 1);

	avgdegree = 0.7 * (xadj[nvtxs] / nvtxs);
	for (i = 0; i < nvtxs; i++) 
		degrees[i] = (xadj[i + 1] - xadj[i] > avgdegree ? avgdegree : xadj[i + 1] - xadj[i]);
	BucketSortKeysInc(nvtxs, avgdegree, degrees, tperm, perm);

	// printf("Match_SHEM 0\n");
	for (cnvtxs = 0, last_unmatched = 0, pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];

		if (match[i] == -1) 
		{  /* Unmatched */
			maxidx = i;
			maxwgt = -1;

			if (vwgt[i] < maxvwgt) 
			{
				/* Deal with island vertices. Find a non-island and match it with. 
					The matching ignores ctrl->maxvwgt requirements */
				if (xadj[i] == xadj[i + 1]) 
				{ 
					last_unmatched = lyj_max(pi, last_unmatched) + 1;
					for (; last_unmatched < nvtxs; last_unmatched++) 
					{
						j = perm[last_unmatched];
						if (match[j] == -1) 
						{
							maxidx = j;
							break;
						}
					}
				}
				else 
				{
					/* Find a heavy-edge matching, subject to maxvwgt constraHunyuan_int_ts */
					/* single constraHunyuan_int_t version */
					for (j = xadj[i]; j < xadj[i + 1]; j++) 
					{
						k = adjncy[j];
						if (match[k] == -1 && maxwgt < adjwgt[j] && vwgt[i] + vwgt[k] <= maxvwgt) 
						{
							maxidx = k;
							maxwgt = adjwgt[j];
						}
					}

					/* If it did not match, record for a 2-hop matching. */
					if (maxidx == i && 3 * vwgt[i] < maxvwgt) 
					{
						nunmatched++;
						maxidx = -1;
					}
				}
			}

			if (maxidx != -1) 
			{
				cmap[i]  = cmap[maxidx] = cnvtxs++;
				match[i] = maxidx;
				match[maxidx] = i;
			}
    	}
  	}
	// printf("Match_SHEM 1\n");
	// exam_num(match,nvtxs);
	//printf("nunmatched: %zu\n", nunmatched);

	/* see if a 2-hop matching is required/allowed */
	if (nunmatched > 0.1 * nvtxs) 
		cnvtxs = Match_2Hop(graph, perm, match, cnvtxs, nunmatched);
	// printf("Match_SHEM 2\n");
	// exam_num(match,nvtxs);

	/* match the final unmatched vertices with themselves and reorder the vertices 
		of the coarse graph for memory-friendly contraction */
	for (cnvtxs=0, i=0; i<nvtxs; i++) 
	{
		if (match[i] == -1) 
		{
			match[i] = i;
			cmap[i] = cnvtxs++;
		}
		else 
		{
			if (i <= match[i]) 
				cmap[i] = cmap[match[i]] = cnvtxs++;
		}
	}
	// printf("metis cnvtxs=%"PRIDX"\n", cnvtxs);
	// printf("Match_SHEM 3\n");
	check_free(degrees, sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: degrees");
	check_free(tperm, sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: tperm");
	check_free(perm, sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: perm");
	// exam_num(match,nvtxs);
	return cnvtxs;
}

Hunyuan_int_t Match_SHEM_topk(graph_t *graph, Hunyuan_int_t maxvwgt)
{
	Hunyuan_int_t i, pi, ii, j, jj, jjinc, k, nvtxs, ncon, cnvtxs, maxidx, maxwgt, 
			last_unmatched, avgdegree;
	Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt;
	Hunyuan_int_t *match, *cmap, *degrees, *perm, *tperm;
	Hunyuan_int_t *receive, *send, *tmatch;
	Hunyuan_int_t nunmatched = 0;
	Hunyuan_int_t cnt, compare, ourless, metis_less;

	nvtxs  = graph->nvtxs;
	// ncon   = graph->ncon;
	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;

	match = graph->match;
	set_value_int(nvtxs, -1, match);
	perm = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: perm");
	// tperm = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: tperm");
	// degrees = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: degrees");

	receive = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs * 5, "Match_SHEM: receive");
	send    = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs * 5, "Match_SHEM: send");
	// tmatch  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: tmatch");
	set_value_int(nvtxs * 5, -1, receive);
	set_value_int(nvtxs * 5, -1, send);
	// set_value_int(nvtxs, -1, tmatch);
	// check_free(tmatch, sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: tmatch");
	tmatch = match;
	// printf("begin %"PRIDX"\n",nvtxs);
	Hunyuan_int_t offset = 1;
	Hunyuan_int_t iter = 1;

	// sort
	for(i = 0;i < nvtxs;i++)
	{
		// printf("%"PRIDX"\n",i);
		for(j = xadj[i];j < xadj[i + 1];j++)
		{
			// printf("j=%"PRIDX"\n",j);
			Hunyuan_int_t t = j;
			for(k = j + 1;k < xadj[i + 1];k++)
			{
				// printf("k=%"PRIDX"\n",k);
				if(adjwgt[k] > adjwgt[t]) 
					t = k;
			}
			jj = adjncy[j], adjncy[j] = adjncy[t], adjncy[t] = jj;
			jj = adjwgt[j], adjwgt[j] = adjwgt[t], adjwgt[t] = jj;
			// printf("%d %d\n",adjncy[j],adjwgt[j]);
		}
	}

	for(iter = 0;iter < 5;iter++)
	{
		offset = iter + 1;
		// offset = 1;
		//set array send and receive
		for(i = 0;i < nvtxs;i++)
		{
			if(tmatch[i] != -1)
				continue;
			for(j = xadj[i], k = 0;j < xadj[i + 1] && k < offset;)
			{
				if(tmatch[adjncy[j]] != -1)
				{
					j++;
					continue;
				}
				ii = adjncy[j];
				if(send[i * offset + k] == -1)
				{
					send[i * offset + k] = ii;
					k++;
					for(jj = 0;jj < offset;jj++)
					{
						if(receive[ii * offset + jj] == -1)
						{
							receive[ii * offset + jj] = i;
							break;
						}
					}
					j++;
				}
			}

		}
		// set match
		for(i = 0;i < nvtxs;i++)
		{
			if(tmatch[i] != -1)
				continue;
			// find
			Hunyuan_int_t flag = i;
			for(j = 0;j < offset;j++)
			{
				k = send[i * offset + j];
				if(k == -1)
					break;
				for(jj = 0;jj < offset;jj++)
				{
					if(receive[i * offset + jj] == k)
					{
						flag = k;
						break;
					}
				}
				if(flag != i)
					break;
			}
			// printf("i=%"PRIDX" flag=%"PRIDX"\n",i,flag);
			if(flag != i)
			{
				tmatch[i] = flag;
				tmatch[flag] = i;
			}
		}

		// resolve the conflict
		cnt = 0;
		for(i = 0;i < nvtxs;i++)
		{
			if(tmatch[i] != -1)
			{
				if(tmatch[i] == i || (tmatch[i] != i && tmatch[tmatch[i]] != i))
					tmatch[i] = -1;
			}
		}
		// reset array send and receive
		set_value_int(nvtxs * offset, -1, receive);
		set_value_int(nvtxs * offset, -1, send);

		// printf("    %"PRIDX"     |   %"PRIDX"   |     %02.2"PRREAL"%%     |  %10"PRIDX"   |   %10"PRIDX"   |     %02.2"PRREAL"%%    | %10"PRIDX"  |    %10"PRIDX"     |     %10"PRIDX"     |\n", \
		// 	iter, offset, (Hunyuan_real_t)(cnt / (Hunyuan_real_t)nvtxs) * 100.0, cnt, nvtxs - cnt / 2, \
		// 	(Hunyuan_real_t)(compare / (Hunyuan_real_t)ourless) * 100.0,compare, ourless, metis_less);
	}

	for(i = 0;i < nvtxs;i++)
		perm[i] = i;
	nunmatched = nvtxs - cnt;
	if (nunmatched > 0.1 * nvtxs) 
		cnvtxs = Match_2Hop(graph, perm, tmatch, nvtxs - cnt / 2, nunmatched);

	for (cnvtxs = 0, i = 0; i < nvtxs; i++) 
	{
		if (tmatch[i] == -1) 
		{
			tmatch[i] = i;
			cmap[i] = cnvtxs++;
		}
		else 
		{
			if (i <= tmatch[i]) 
			{
				cmap[i] = cmap[tmatch[i]] = cnvtxs++;
			}
		}
	}

	// printf("   iter   |   k   |  success_rate  |  success_num  |     cnvtxs     |  great_rate  |  great_num  |  topk_lessweight  |  metis_lessweight  |\n");
	// printf("    %"PRIDX"     |  end  |     %02.2"PRREAL"%%     |  %10"PRIDX"   |   %10"PRIDX"   |     %02.2"PRREAL"%%    | %10"PRIDX"  |    %10"PRIDX"     |     %10"PRIDX"     |\n", \
	// 		iter + 3, (Hunyuan_real_t)(cnt / (Hunyuan_real_t)nvtxs) * 100.0, cnt, nvtxs - cnt / 2, \
	// 		(Hunyuan_real_t)(compare / (Hunyuan_real_t)ourless) * 100.0, compare, ourless, metis_less);

	/*irandArrayPermute(nvtxs, tperm, nvtxs/8, 1);

	avgdegree = 0.7 * (xadj[nvtxs] / nvtxs);
	for (i = 0; i < nvtxs; i++) 
		degrees[i] = (xadj[i + 1] - xadj[i] > avgdegree ? avgdegree : xadj[i + 1] - xadj[i]);
	BucketSortKeysInc(nvtxs, avgdegree, degrees, tperm, perm);

	// printf("Match_SHEM 0\n");
	for (cnvtxs = 0, last_unmatched = 0, pi = 0; pi < nvtxs; pi++) 
	{
		i = perm[pi];

		if (match[i] == -1) 
		{  /* Unmatched *
			maxidx = i;
			maxwgt = -1;

			if (vwgt[i] < maxvwgt) 
			{
				/* Deal with island vertices. Find a non-island and match it with. 
					The matching ignores ctrl->maxvwgt requirements *
				if (xadj[i] == xadj[i + 1]) 
				{ 
					last_unmatched = lyj_max(pi, last_unmatched) + 1;
					for (; last_unmatched < nvtxs; last_unmatched++) 
					{
						j = perm[last_unmatched];
						if (match[j] == -1) 
						{
							maxidx = j;
							break;
						}
					}
				}
				else 
				{
					/* Find a heavy-edge matching, subject to maxvwgt constraHunyuan_int_ts */
					/* single constraHunyuan_int_t version *
					for (j = xadj[i]; j < xadj[i + 1]; j++) 
					{
						k = adjncy[j];
						if (match[k] == -1 && maxwgt < adjwgt[j] && vwgt[i] + vwgt[k] <= maxvwgt) 
						{
							maxidx = k;
							maxwgt = adjwgt[j];
						}
					}

					/* If it did not match, record for a 2-hop matching. *
					if (maxidx == i && 3 * vwgt[i] < maxvwgt) 
					{
						nunmatched++;
						maxidx = -1;
					}
				}
			}

			if (maxidx != -1) 
			{
				cmap[i]  = cmap[maxidx] = cnvtxs++;
				match[i] = maxidx;
				match[maxidx] = i;
			}
    	}
  	}
	// printf("Match_SHEM 1\n");
	// exam_num(match,nvtxs);
	//printf("nunmatched: %zu\n", nunmatched);

	/* see if a 2-hop matching is required/allowed *
	if (nunmatched > 0.1 * nvtxs) 
		cnvtxs = Match_2Hop(graph, perm, match, cnvtxs, nunmatched);
	// printf("Match_SHEM 2\n");
	// exam_num(match,nvtxs);

	/* match the final unmatched vertices with themselves and reorder the vertices 
		of the coarse graph for memory-friendly contraction *
	for (cnvtxs=0, i=0; i<nvtxs; i++) 
	{
		if (match[i] == -1) 
		{
			match[i] = i;
			cmap[i] = cnvtxs++;
		}
		else 
		{
			if (i <= match[i]) 
				cmap[i] = cmap[match[i]] = cnvtxs++;
		}
	}
	printf("metis cnvtxs=%"PRIDX"\n", cnvtxs);*/
	// printf("Match_SHEM 3\n");
	// check_free(degrees, sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: degrees");
	// check_free(tperm, sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: tperm");
	check_free(perm, sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: perm");
	check_free(receive, sizeof(Hunyuan_int_t) * nvtxs * 5, "Match_SHEM: receive");
	check_free(send, sizeof(Hunyuan_int_t) * nvtxs * 5, "Match_SHEM: send");
	// check_free(tmatch, sizeof(Hunyuan_int_t) * nvtxs, "Match_SHEM: tmatch");
	// exam_num(match,nvtxs);
	return cnvtxs;
}


#endif