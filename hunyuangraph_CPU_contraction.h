#ifndef _H_CPU_CONSTRUCTION
#define _H_CPU_CONSTRUCTION

#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
#include "hunyuangraph_timer.h"

/*Create cpu coarsen graph by contract*/
void hunyuangraph_cpu_create_cgraph(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, int cnvtxs, int *match)
{
  int j,k,m,istart,iend,nvtxs,nedges,cnedges,v,u;
  int *xadj,*vwgt,*adjncy,*adjwgt;
  int *cmap,*htable;
  int *cxadj,*cvwgt,*cadjncy,*cadjwgt;
  hunyuangraph_graph_t *cgraph;
  
  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  cmap=graph->cmap;                  
  
  cgraph=hunyuangraph_set_cpu_cgraph(graph,cnvtxs);            
  cxadj=cgraph->xadj;
  cvwgt=cgraph->vwgt;
  cadjncy=cgraph->adjncy;
  cadjwgt=cgraph->adjwgt;                               
  htable=hunyuangraph_int_set_value(cnvtxs,-1,hunyuangraph_int_malloc_space(hunyuangraph_admin,cnvtxs));      
  cxadj[0] = cnvtxs = cnedges = 0; 
  nedges=graph->nedges;
   
  for(v=0;v<nvtxs;v++){

    if((u=match[v])<v)         
      continue;   

    cvwgt[cnvtxs]=vwgt[v];                 
    nedges=0;                                                    
    istart=xadj[v];
    iend=xadj[v+1];    

    for(j=istart;j<iend;j++){

      k=cmap[adjncy[j]];     

      if((m=htable[k])==-1){
        cadjncy[nedges]=k;                           
        cadjwgt[nedges] = adjwgt[j];                      
        htable[k] = nedges++;  
      }
      else{
        cadjwgt[m] += adjwgt[j];                                 
      }
    }

    if(v!=u){ 
      cvwgt[cnvtxs]+=vwgt[u];                   
      istart=xadj[u];                                    
      iend=xadj[u+1];      

      for(j=istart;j<iend;j++){
        k=cmap[adjncy[j]];

        if((m=htable[k])==-1){
          cadjncy[nedges]=k;
          cadjwgt[nedges]=adjwgt[j];
          htable[k]=nedges++;
        }
        else{
          cadjwgt[m] += adjwgt[j];
        }
      }

      if((j=htable[cnvtxs])!=-1){
        cadjncy[j]=cadjncy[--nedges];
        cadjwgt[j]=cadjwgt[nedges];
        htable[cnvtxs] = -1;
      }
    }

    for(j=0;j<nedges;j++){
       htable[cadjncy[j]] = -1;  
    }

    cnedges+=nedges;
    cxadj[++cnvtxs]=cnedges;
    cadjncy+=nedges;                                                                 
    cadjwgt+=nedges;
  }

  cgraph->nedges=cnedges;
  cgraph->tvwgt[0]=hunyuangraph_int_sum(cgraph->nvtxs,cgraph->vwgt); 
  cgraph->tvwgt_reverse[0]=1.0/(cgraph->tvwgt[0]>0?cgraph->tvwgt[0]:1);    

//   printf("cnvtxs=%d cnedges=%d\n",cgraph->nvtxs,cgraph->nedges);

}


#endif