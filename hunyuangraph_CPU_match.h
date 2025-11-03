#ifndef _H_CPU_MATCH
#define _H_CPU_MATCH

#include "hunyuangraph_define.h"
#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
#include "hunyuangraph_timer.h"
#include "hunyuangraph_CPU_contraction.h"

/*Get permutation array*/
void hunyuangraph_matching_sort(hunyuangraph_admin_t *hunyuangraph_admin, int n, \
int max, int *keys, int *tperm, int *perm)
{
  int i,ii;
  int *counts;
  counts=hunyuangraph_int_set_value(max+2,0,hunyuangraph_int_malloc_space(hunyuangraph_admin,max+2));
  
  for(i=0; i<n; i++){
    counts[keys[i]]++;
  }
  
  hunyuangraph_tocsr(i,max+1,counts);
  
  for(ii=0;ii<n;ii++){
    i=tperm[ii];
    perm[counts[keys[i]]++]=i;
  }
}

/*************************************************************************/
/*! This function matches the unmatched vertices whose degree is less than
    maxdegree using a 2-hop matching that involves vertices that are two 
    hops away from each other. 
    The requirement of the 2-hop matching is a simple non-empty overlap
    between the adjancency lists of the vertices. */
/**************************************************************************/
int Match_2HopAny(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int *perm, int *match, 
          int cnvtxs, size_t *r_nunmatched, size_t maxdegree)
{
  int i, pi, ii, j, jj, k, nvtxs;
  int *xadj, *adjncy, *colptr, *rowind;
  int *cmap;
  size_t nunmatched;

  nvtxs  = graph->nvtxs;
  xadj   = graph->xadj;
  adjncy = graph->adjncy;
  cmap   = graph->cmap;

  nunmatched = *r_nunmatched;

  /* create the inverted index */
  colptr = hunyuangraph_int_set_value(nvtxs, 0, hunyuangraph_int_malloc_space(hunyuangraph_admin, nvtxs+1));
  for (i=0; i<nvtxs; i++) {
    if (match[i] == -1 && xadj[i+1]-xadj[i] < maxdegree) {
      for (j=xadj[i]; j<xadj[i+1]; j++)
        colptr[adjncy[j]]++;
    }
  }
  hunyuangraph_tocsr(i, nvtxs, colptr);

  rowind = hunyuangraph_int_malloc_space(hunyuangraph_admin, colptr[nvtxs]);
  for (pi=0; pi<nvtxs; pi++) {
    i = perm[pi];
    if (match[i] == -1 && xadj[i+1]-xadj[i] < maxdegree) {
      for (j=xadj[i]; j<xadj[i+1]; j++)
        rowind[colptr[adjncy[j]]++] = i;
    }
  }
  SHIFTCSR(i, nvtxs, colptr);

  /* compute matchings by going down the inverted index */
  for (pi=0; pi<nvtxs; pi++) {
    i = perm[pi];
    if (colptr[i+1]-colptr[i] < 2)
      continue;

    for (jj=colptr[i+1], j=colptr[i]; j<jj; j++) {
      if (match[rowind[j]] == -1) {
        for (jj--; jj>j; jj--) {
          if (match[rowind[jj]] == -1) {
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
int Match_2HopAll(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int *perm, int *match, 
          int cnvtxs, size_t *r_nunmatched, size_t maxdegree)
{
  int i, pi, pk, ii, j, jj, k, nvtxs, mask, idegree;
  int *xadj, *adjncy;
  int *cmap, *mark;
  ikv_t *keys;
  size_t nunmatched, ncand;

  nvtxs  = graph->nvtxs;
  xadj   = graph->xadj;
  adjncy = graph->adjncy;
  cmap   = graph->cmap;

  nunmatched = *r_nunmatched;
  mask = IDX_MAX/maxdegree;

  /* collapse vertices with identical adjancency lists */
  keys = ikvwspacemalloc(hunyuangraph_admin, nunmatched);
  for (ncand=0, pi=0; pi<nvtxs; pi++) {
    i = perm[pi];
    idegree = xadj[i+1]-xadj[i];
    if (match[i] == -1 && idegree > 1 && idegree < maxdegree) {
      for (k=0, j=xadj[i]; j<xadj[i+1]; j++) 
        k += adjncy[j]%mask;
      keys[ncand].val = i;
      keys[ncand].key = (k%mask)*maxdegree + idegree;
      ncand++;
    }
  }
  ikvsorti(ncand, keys);

  match=hunyuangraph_int_set_value(nvtxs,0, hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs));
  for (pi=0; pi<ncand; pi++) {
    i = keys[pi].val;
    if (match[i] != -1)
      continue;

    for (j=xadj[i]; j<xadj[i+1]; j++)
      mark[adjncy[j]] = i;

    for (pk=pi+1; pk<ncand; pk++) {
      k = keys[pk].val;
      if (match[k] != -1)
        continue;

      if (keys[pi].key != keys[pk].key)
        break;
      if (xadj[i+1]-xadj[i] != xadj[k+1]-xadj[k])
        break;

      for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
        if (mark[adjncy[jj]] != i)
          break;
      }
      if (jj == xadj[k+1]) {
        cmap[i] = cmap[k] = cnvtxs++;
        match[i] = k;
        match[k] = i;
        nunmatched -= 2;
        break;
      }
    }
  }

  *r_nunmatched = nunmatched;

  return cnvtxs;
}

/*************************************************************************/
/*! This function matches the unmatched vertices using a 2-hop matching 
    that involves vertices that are two hops away from each other. */
/**************************************************************************/
int Match_2Hop(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int *perm, int *match, 
          int cnvtxs, size_t nunmatched)
{

	cnvtxs = Match_2HopAny(hunyuangraph_admin, graph, perm, match, cnvtxs, &nunmatched, 2);
	cnvtxs = Match_2HopAll(hunyuangraph_admin, graph, perm, match, cnvtxs, &nunmatched, 64);
	if (nunmatched > 1.5*0.1*graph->nvtxs) 
		cnvtxs = Match_2HopAny(hunyuangraph_admin, graph, perm, match, cnvtxs, &nunmatched, 3);
	if (nunmatched > 2.0*0.1*graph->nvtxs) 
		cnvtxs = Match_2HopAny(hunyuangraph_admin, graph, perm, match, cnvtxs, &nunmatched, graph->nvtxs);

 	return cnvtxs;
}

int hunyuangraph_cpu_match_RM(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
	int i, pi, ii, j, jj, jjinc, k, nvtxs, cnvtxs, maxidx, last_unmatched;
	int *xadj, *vwgt, *adjncy, *adjwgt, *maxvwgt;
	int *match, *cmap, *perm;
	size_t nunmatched=0;

	nvtxs  = graph->nvtxs;
	xadj   = graph->xadj;
	vwgt   = graph->vwgt;
	adjncy = graph->adjncy;
	adjwgt = graph->adjwgt;
	cmap   = graph->cmap;

	maxvwgt = (int *)malloc(sizeof(int));
	maxvwgt[0] = hunyuangraph_admin->maxvwgt;

	match=hunyuangraph_int_set_value(nvtxs,-1, hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs));
  	perm=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);

	hunyuangraph_int_randarrayofp(nvtxs,perm,nvtxs/8,1);   

	for (cnvtxs=0, last_unmatched=0, pi=0; pi<nvtxs; pi++) 
	{
		i = perm[pi];

		if (match[i] == -1) {  /* Unmatched */
			maxidx = i;

			if (vwgt[i] < maxvwgt[0]) {
				/* Deal with island vertices. Find a non-island and match it with. 
				The matching ignores ctrl->maxvwgt requirements */
				if (xadj[i] == xadj[i+1]) {
					last_unmatched = hunyuangraph_max(pi, last_unmatched)+1;
					for (; last_unmatched<nvtxs; last_unmatched++) {
						j = perm[last_unmatched];
						if (match[j] == -1) {
							maxidx = j;
							break;
						}
					}
				}
				else {
				/* Find a random matching, subject to maxvwgt constraints */
					/* single constraint version */
					for (j=xadj[i]; j<xadj[i+1]; j++) {
						k = adjncy[j];
						if (match[k] == -1 && vwgt[i]+vwgt[k] <= maxvwgt[0]) {
							maxidx = k;
							break;
						}
					}

					/* If it did not match, record for a 2-hop matching. */
					if (maxidx == i && 3*vwgt[i] < maxvwgt[0]) {
						nunmatched++;
						maxidx = -1;
					}
				}
			}

			if (maxidx != -1) {
				cmap[i]  = cmap[maxidx] = cnvtxs++;
				match[i] = maxidx;
				match[maxidx] = i;
			}
		}
	}

	/* see if a 2-hop matching is required/allowed */
	if (!hunyuangraph_admin->no2hop && nunmatched > 0.1*nvtxs) {
		cnvtxs = Match_2Hop(hunyuangraph_admin, graph, perm, match, cnvtxs, nunmatched);
	}

	/* match the final unmatched vertices with themselves and reorder the vertices 
     of the coarse graph for memory-friendly contraction */
	for (cnvtxs=0, i=0; i<nvtxs; i++) {
		if (match[i] == -1) {
			match[i] = i;
			cmap[i]  = cnvtxs++;
		}
		else {
			if (i <= match[i]) 
				cmap[i] = cmap[match[i]] = cnvtxs++;
		}
	}

	hunyuangraph_cpu_create_cgraph(hunyuangraph_admin, graph, cnvtxs, match);
}

/*Get cpu graph matching params by hem*/
int hunyuangraph_cpu_match_HEM(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
  int i,j,pi,k,nvtxs,cnvtxs,maxidx,maxwgt,last_unmatched,aved;
  int *xadj,*vwgt,*adjncy,*adjwgt,maxvwgt;
  int *match,*cmap,*d,*perm,*tperm;
  size_t nunmatched=0;

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  cmap=graph->cmap;
  maxvwgt=hunyuangraph_admin->maxvwgt;
  
  cnvtxs=0;
  match=hunyuangraph_int_set_value(nvtxs,-1, hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs));
  perm=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);
  tperm=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);
  d=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);         
  hunyuangraph_int_randarrayofp(nvtxs,tperm,nvtxs/8,1);   
  aved=0.7*(xadj[nvtxs]/nvtxs);

  for(i=0;i<nvtxs;i++){ 
    d[i]=(xadj[i+1]-xadj[i]>aved?aved:xadj[i+1]-xadj[i]);
  }

  hunyuangraph_matching_sort(hunyuangraph_admin,nvtxs,aved,d,tperm,perm);         
  
  last_unmatched=0;
  for(pi=0;pi<nvtxs;pi++) 
  {
    i=perm[pi];  

    if(match[i]==-1){  
      maxidx=i;                                                                               
      maxwgt=-1;           

	  if(vwgt[i] < maxvwgt)
	  {
		/* Deal with island vertices. Find a non-island and match it with. 
           The matching ignores ctrl->maxvwgt requirements */
        if (xadj[i] == xadj[i+1]) { 
			last_unmatched = hunyuangraph_max(pi, last_unmatched)+1;
			for (; last_unmatched<nvtxs; last_unmatched++) {
				j = perm[last_unmatched];
				if (match[j] == -1) {
					maxidx = j;
					break;
				}
			}
        }
		else
		{
			/* Find a heavy-edge matching, subject to maxvwgt constraints */
			/* single constraint version */
			for(j=xadj[i];j<xadj[i+1];j++){
				k=adjncy[j];

				if(match[k]==-1&&maxwgt<adjwgt[j]&&vwgt[i]+vwgt[k]<=maxvwgt){
					maxidx=k;
					maxwgt=adjwgt[j];
				}   
			}

			if(maxidx==i&&3*vwgt[i]<maxvwgt){ 
				maxidx=-1;
			}
	    }
	  }

      if(maxidx!=-1){
        cmap[i]=cmap[maxidx]=cnvtxs++;              
        match[i]=maxidx;                                        
        match[maxidx]=i; 
      }
    }
  }

	/* see if a 2-hop matching is required/allowed */
	if (!hunyuangraph_admin->no2hop && nunmatched > 0.1*nvtxs) 
	{
		cnvtxs = Match_2Hop(hunyuangraph_admin, graph, perm, match, cnvtxs, nunmatched);
	}

  for(cnvtxs=0,i=0;i<nvtxs;i++){
    if(match[i]==-1){
      match[i]=i;
      cmap[i]=cnvtxs++;                                                    
    }
    else{
      if(i<=match[i]){ 
        cmap[i]=cmap[match[i]]=cnvtxs++;
      }
    }
  }

  /*for(i = 0;i < nvtxs;i += 2)
  {
	if(i + 1 < nvtxs) 
	{
		match[i] = i + 1;
		match[i + 1] = i;
		cmap[i] = i / 2;
		cmap[i + 1] = (i + 1) / 2;
	}
	else
	{
		match[i] = i;
		cmap[i] = i / 2;
	}
  }
  cnvtxs = (nvtxs - 1) / 2 + 1;*/

  hunyuangraph_cpu_create_cgraph(hunyuangraph_admin, graph, cnvtxs, match);

  return cnvtxs;
}


#endif