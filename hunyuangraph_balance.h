#ifndef _H_BALANCE
#define _H_BALANCE

#include "hunyuangraph_struct.h"
#include "hunyuangraph_priorityqueue.h"

/*Compute cpu imbalance params*/
float hunyuangraph_compute_cpu_imbal(hunyuangraph_graph_t *graph, int nparts, float *part_balance, float *ubvec)
{
  int j,*pwgts;
  float max,cur;
  pwgts=graph->pwgts;
  max=-1.0;

  for(j=0;j<nparts;j++){
    cur=pwgts[j]*part_balance[j]-ubvec[0];

    if(cur>max){
      max=cur;
    }
  }

  return max;
}

/*Balance two partition by moving boundary vertex*/
void hunyuangraph_bndvertex_2way_bal(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, float *ntpwgts)
{
  int i,ii,j,k,kwgt,nvtxs,nbnd,nswaps,from,to,temp;
  int *xadj,*vwgt,*adjncy,*adjwgt,*where,*id,*ed,*bndptr,*bndlist,*pwgts;
  int *moved,*perm;

  hunyuangraph_queue_t *queue;
  int higain,mincut,mindiff;
  int tpwgts[2];

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  where=graph->where;
  id=graph->id;
  ed=graph->ed;
  pwgts=graph->pwgts;
  bndptr=graph->bndptr;
  bndlist=graph->bndlist;

  moved=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);
  perm=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);

  tpwgts[0]=graph->tvwgt[0]*ntpwgts[0];
  tpwgts[1]=graph->tvwgt[0]-tpwgts[0];
  mindiff=abs(tpwgts[0]-pwgts[0]);
  from=(pwgts[0]<tpwgts[0]?1:0);
  to=(from+1)%2;

  queue=hunyuangraph_queue_create(nvtxs);
  hunyuangraph_int_set_value(nvtxs,-1,moved);
  nbnd=graph->nbnd;
  hunyuangraph_int_randarrayofp(nbnd,perm,nbnd/5,1);

  for(ii=0;ii<nbnd;ii++){
    i=perm[ii];

    if(where[bndlist[i]]==from&&vwgt[bndlist[i]]<=mindiff){
      hunyuangraph_queue_insert(queue,bndlist[i],ed[bndlist[i]]-id[bndlist[i]]);
    }
  }

  mincut=graph->mincut;

  for(nswaps=0;nswaps<nvtxs;nswaps++) 
  {
    if((higain=hunyuangraph_queue_top(queue))==-1)
      break;
    if(pwgts[to]+vwgt[higain]>tpwgts[to])
      break;

    mincut-=(ed[higain]-id[higain]);
    hunyuangraph_add_sub(pwgts[to],pwgts[from],vwgt[higain]);

    where[higain]=to;
    moved[higain]=nswaps;
    hunyuangraph_swap(id[higain],ed[higain],temp);

    if(ed[higain]==0&&xadj[higain]<xadj[higain+1]){ 
      hunyuangraph_listdelete(nbnd,bndlist,bndptr,higain);
    }

    for(j=xadj[higain];j<xadj[higain+1];j++){
      k=adjncy[j];
      kwgt=(to==where[k]?adjwgt[j]:-adjwgt[j]);
      hunyuangraph_add_sub(id[k],ed[k],kwgt);

      if(bndptr[k]!=-1){ 
        if(ed[k]==0){ 
          hunyuangraph_listdelete(nbnd,bndlist,bndptr,k);

          if(moved[k]==-1&&where[k]==from&&vwgt[k]<=mindiff){ 
            hunyuangraph_queue_delete(queue,k);
          }
        }
        else{ 
          if(moved[k]==-1&&where[k]==from&&vwgt[k]<=mindiff){
            hunyuangraph_queue_update(queue,k,ed[k]-id[k]);
          }
        }
      }
      else{
        if(ed[k]>0){  
          hunyuangraph_listinsert(nbnd,bndlist,bndptr,k);

          if(moved[k]==-1&&where[k]==from&&vwgt[k]<=mindiff){ 
            hunyuangraph_queue_insert(queue,k,ed[k]-id[k]);
          }
        }
      }
    }
  }

  graph->mincut=mincut;
  graph->nbnd=nbnd;
  hunyuangraph_queue_free(queue);

}

/*Balance 2-way partition*/
void hunyuangraph_2way_bal(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, float *ntpwgts)
{
  if(hunyuangraph_compute_cpu_imbal(graph,2,hunyuangraph_admin->part_balance,hunyuangraph_admin->ubfactors)<=0){ 
    return;
  }

  if(abs(ntpwgts[0]*graph->tvwgt[0]-graph->pwgts[0])<3*graph->tvwgt[0]/graph->nvtxs){
    return;
  }

  hunyuangraph_bndvertex_2way_bal(hunyuangraph_admin,graph,ntpwgts);
}

/*Compute 2way balance params*/
void hunyuangraph_compute_2way_balance(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, float *tpwgts)
{
  int i;
  for(i=0;i<2;i++){
      hunyuangraph_admin->part_balance[i]=graph->tvwgt_reverse[0]/tpwgts[i];
  }
}

#endif
