#ifndef _H_CPU_SPLITGRAPH
#define _H_CPU_SPLITGRAPH

#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"

/*Split graph to lgraph and rgraph*/
void hunyuangraph_splitgraph(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, \
    hunyuangraph_graph_t **r_lgraph, hunyuangraph_graph_t **r_rgraph)
{
  int i,j,k,l,istart,iend,mypart,nvtxs,snvtxs[2],snedges[2];
  int *xadj,*vwgt,*adjncy,*adjwgt,*label,*where,*bndptr;
  int *sxadj[2],*svwgt[2],*sadjncy[2],*sadjwgt[2],*slabel[2];
  int *rename;
  int *temp_adjncy,*temp_adjwgt;

  hunyuangraph_graph_t *lgraph,*rgraph;

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  label=graph->label;
  where=graph->where;
  bndptr=graph->bndptr;

  rename=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);
  snvtxs[0]=snvtxs[1]=snedges[0]=snedges[1]=0;

  for(i=0;i<nvtxs;i++){
    k=where[i];
    rename[i]=snvtxs[k]++;
    snedges[k]+=xadj[i+1]-xadj[i];
  }

  lgraph=hunyuangraph_set_splitgraph(graph,snvtxs[0],snedges[0]);
  sxadj[0]=lgraph->xadj;
  svwgt[0]=lgraph->vwgt;
  sadjncy[0]=lgraph->adjncy; 	
  sadjwgt[0]=lgraph->adjwgt; 
  slabel[0]=lgraph->label;

  rgraph=hunyuangraph_set_splitgraph(graph,snvtxs[1],snedges[1]);
  sxadj[1]=rgraph->xadj;
  svwgt[1]=rgraph->vwgt;
  sadjncy[1]=rgraph->adjncy; 	
  sadjwgt[1]=rgraph->adjwgt; 
  slabel[1]=rgraph->label;

  snvtxs[0]=snvtxs[1]=snedges[0]=snedges[1]=0;
  sxadj[0][0]=sxadj[1][0]=0;

  for(i=0;i<nvtxs;i++){
    mypart=where[i];
    istart=xadj[i];
    iend=xadj[i+1];

    if(bndptr[i]==-1){ 
      temp_adjncy=sadjncy[mypart]+snedges[mypart]-istart;
      temp_adjwgt=sadjwgt[mypart]+snedges[mypart]-istart;

      for(j=istart;j<iend;j++){
        temp_adjncy[j]=adjncy[j];
        temp_adjwgt[j]=adjwgt[j]; 
      }

      snedges[mypart]+=iend-istart;
    }
    else{
      temp_adjncy=sadjncy[mypart];
      temp_adjwgt=sadjwgt[mypart];
      l=snedges[mypart];

      for(j=istart;j<iend;j++){
        k=adjncy[j];
        
        if(where[k]==mypart){
          temp_adjncy[l]=k;
          temp_adjwgt[l++]=adjwgt[j]; 
        }
      }
      snedges[mypart]=l;
    }

    svwgt[mypart][snvtxs[mypart]]=vwgt[i];
    // printf("i=%d label[i]=%d\n",i,label[i]);
    slabel[mypart][snvtxs[mypart]]=label[i];
    sxadj[mypart][++snvtxs[mypart]]=snedges[mypart];
  }

  for(mypart=0;mypart<2;mypart++){
    iend=sxadj[mypart][snvtxs[mypart]];
    temp_adjncy=sadjncy[mypart];

    for(i=0;i<iend;i++){ 
      temp_adjncy[i]=rename[temp_adjncy[i]];
    }
  }

  lgraph->nedges=snedges[0];
  rgraph->nedges=snedges[1];

//   printf("CPU:lnedges=%d rnedges=%d nedges=%d +:%d\n",lgraph->nedges,rgraph->nedges,graph->nedges,lgraph->nedges + rgraph->nedges + graph->mincut);
	/*printf("CPU_lgraph:\n");
  	printf("lgraph->nvtxs:%d lgraph->nedges:%d\n",lgraph->nvtxs,lgraph->nedges);
	printf("lgraph->vwgt:\n");
  	for(i = 0;i < lgraph->nvtxs;i++)
	{
		printf("%d ",lgraph->vwgt[i]);
	}
	printf("\nlgraph->label:\n");
  	for(i = 0;i < lgraph->nvtxs;i++)
	{
		printf("%d ",lgraph->label[i]);
	}
	printf("\nlgraph->xadj:\n");
	for(i = 0;i <= lgraph->nvtxs;i++)
	{
		printf("%d ",lgraph->xadj[i]);
	}
	printf("\nlgraph->adjncy:\n");
	for(i = 0;i < lgraph->nvtxs;i++)
	{
		for(int j = lgraph->xadj[i];j < lgraph->xadj[i + 1];j++)
		{
			printf("%d ",lgraph->adjncy[j]);
		}
		printf("\n");
	}
	printf("CPU_rgraph:\n");
	printf("rgraph->nvtxs:%d rgraph->nedges:%d\n",rgraph->nvtxs,rgraph->nedges);
	printf("rgraph->vwgt:\n");
  	for(i = 0;i < rgraph->nvtxs;i++)
	{
		printf("%d ",rgraph->vwgt[i]);
	}
	printf("\ngraph->label:\n");
  	for(i = 0;i < rgraph->nvtxs;i++)
	{
		printf("%d ",rgraph->label[i]);
	}
	printf("\nrgraph->xadj:\n");
	for(i = 0;i <= rgraph->nvtxs;i++)
	{
		printf("%d ",rgraph->xadj[i]);
	}
	printf("\nrgraph->adjncy:\n");
	for(i = 0;i < rgraph->nvtxs;i++)
	{
		for(int j = rgraph->xadj[i];j < rgraph->xadj[i + 1];j++)
		{
			printf("%d ",rgraph->adjncy[j]);
		}
		printf("\n");
	}*/

  hunyuangraph_set_graph_tvwgt(lgraph);
  hunyuangraph_set_graph_tvwgt(rgraph);

	// printf("lgraph->tvwgt=%d lgraph->tvwgt_reverse=%f rgraph->tvwgt=%d rgraph->tvwgt_reverse=%f\n",lgraph->tvwgt[0],lgraph->tvwgt_reverse[0],rgraph->tvwgt[0],rgraph->tvwgt_reverse[0]);


  *r_lgraph=lgraph;
  *r_rgraph=rgraph;

}

void hunyuangraph_splitgraph_first(hunyuangraph_admin_t *hunyuangraph_admin, \
	hunyuangraph_graph_t *graph, hunyuangraph_graph_t **r_lgraph, hunyuangraph_graph_t **r_rgraph)
{
  int i,j,k,l,istart,iend,mypart,nvtxs,snvtxs[2],snedges[2];
  int *xadj,*vwgt,*adjncy,*adjwgt,*label,*where,*bndptr;
  int *sxadj[2],*svwgt[2],*sadjncy[2],*sadjwgt[2],*slabel[2];
  int *rename;
  int *temp_adjncy,*temp_adjwgt;

  hunyuangraph_graph_t *lgraph,*rgraph;

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
//   label=graph->label;
  where=graph->where;
  bndptr=graph->bndptr;

  rename=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);
  snvtxs[0]=snvtxs[1]=snedges[0]=snedges[1]=0;

  for(i=0;i<nvtxs;i++){
    k=where[i];
    rename[i]=snvtxs[k]++;
    snedges[k]+=xadj[i+1]-xadj[i];
  }

  lgraph=hunyuangraph_set_splitgraph(graph,snvtxs[0],snedges[0]);
  sxadj[0]=lgraph->xadj;
  svwgt[0]=lgraph->vwgt;
  sadjncy[0]=lgraph->adjncy; 	
  sadjwgt[0]=lgraph->adjwgt; 
  slabel[0]=lgraph->label;

  rgraph=hunyuangraph_set_splitgraph(graph,snvtxs[1],snedges[1]);
  sxadj[1]=rgraph->xadj;
  svwgt[1]=rgraph->vwgt;
  sadjncy[1]=rgraph->adjncy; 	
  sadjwgt[1]=rgraph->adjwgt; 
  slabel[1]=rgraph->label;

  snvtxs[0]=snvtxs[1]=snedges[0]=snedges[1]=0;
  sxadj[0][0]=sxadj[1][0]=0;

  for(i=0;i<nvtxs;i++){
    mypart=where[i];
    istart=xadj[i];
    iend=xadj[i+1];

    if(bndptr[i]==-1){ 
      temp_adjncy=sadjncy[mypart]+snedges[mypart]-istart;
      temp_adjwgt=sadjwgt[mypart]+snedges[mypart]-istart;

      for(j=istart;j<iend;j++){
        temp_adjncy[j]=adjncy[j];
        temp_adjwgt[j]=adjwgt[j]; 
      }

      snedges[mypart]+=iend-istart;
    }
    else{
      temp_adjncy=sadjncy[mypart];
      temp_adjwgt=sadjwgt[mypart];
      l=snedges[mypart];

      for(j=istart;j<iend;j++){
        k=adjncy[j];
        
        if(where[k]==mypart){
          temp_adjncy[l]=k;
          temp_adjwgt[l++]=adjwgt[j]; 
        }
      }
      snedges[mypart]=l;
    }

    svwgt[mypart][snvtxs[mypart]]=vwgt[i];
    slabel[mypart][snvtxs[mypart]]=i;
    sxadj[mypart][++snvtxs[mypart]]=snedges[mypart];
  }

  for(mypart=0;mypart<2;mypart++){
    iend=sxadj[mypart][snvtxs[mypart]];
    temp_adjncy=sadjncy[mypart];

    for(i=0;i<iend;i++){ 
      temp_adjncy[i]=rename[temp_adjncy[i]];
    }
  }

  lgraph->nedges=snedges[0];
  rgraph->nedges=snedges[1];

  	/*printf("CPU_graph:\n");
  	printf("graph->nvtxs:%d graph->nedges:%d\n",graph->nvtxs,graph->nedges);
	printf("\ngraph->xadj:\n");
	for(i = 0;i < graph->nvtxs;i++)
	{
		printf("i=%d where[i]=%d rename=%d\n",i,graph->where[i],rename[i]);
	}
	for(i = 0;i <= graph->nvtxs;i++)
	{
		printf("%d ",graph->xadj[i]);
	}
	printf("\ngraph->adjncy:\n");
	for(i = 0;i < graph->nvtxs;i++)
	{
		for(int j = graph->xadj[i];j < graph->xadj[i + 1];j++)
		{
			printf("i=%d j=%d where[i]=%d where[j]=%d\n",i,graph->adjncy[j],graph->where[i],graph->where[graph->adjncy[j]]);
		}
	}*/

	/*printf("CPU_lgraph:\n");
  	printf("lgraph->nvtxs:%d lgraph->nedges:%d\n",lgraph->nvtxs,lgraph->nedges);
	printf("lgraph->vwgt:\n");
  	for(i = 0;i < lgraph->nvtxs;i++)
	{
		printf("%d ",lgraph->vwgt[i]);
	}
	printf("\nlgraph->label:\n");
  	for(i = 0;i < lgraph->nvtxs;i++)
	{
		printf("%d ",lgraph->label[i]);
	}
	printf("\nlgraph->xadj:\n");
	for(i = 0;i <= lgraph->nvtxs;i++)
	{
		printf("%d ",lgraph->xadj[i]);
	}
	printf("\nlgraph->adjncy:\n");
	for(i = 0;i < lgraph->nvtxs;i++)
	{
		for(int j = lgraph->xadj[i];j < lgraph->xadj[i + 1];j++)
		{
			printf("%d ",lgraph->adjncy[j]);
		}
		printf("\n");
	}
	printf("CPU_rgraph:\n");
	printf("rgraph->nvtxs:%d rgraph->nedges:%d\n",rgraph->nvtxs,rgraph->nedges);
	printf("rgraph->vwgt:\n");
  	for(i = 0;i < rgraph->nvtxs;i++)
	{
		printf("%d ",rgraph->vwgt[i]);
	}
	printf("\ngraph->label:\n");
  	for(i = 0;i < rgraph->nvtxs;i++)
	{
		printf("%d ",rgraph->label[i]);
	}
	printf("\nrgraph->xadj:\n");
	for(i = 0;i <= rgraph->nvtxs;i++)
	{
		printf("%d ",rgraph->xadj[i]);
	}
	printf("\nrgraph->adjncy:\n");
	for(i = 0;i < rgraph->nvtxs;i++)
	{
		for(int j = rgraph->xadj[i];j < rgraph->xadj[i + 1];j++)
		{
			printf("%d ",rgraph->adjncy[j]);
		}
		printf("\n");
	}*/

  hunyuangraph_set_graph_tvwgt(lgraph);
  hunyuangraph_set_graph_tvwgt(rgraph);

//   printf("lgraph->tvwgt=%d lgraph->tvwgt_reverse=%f rgraph->tvwgt=%d rgraph->tvwgt_reverse=%f\n",lgraph->tvwgt[0],lgraph->tvwgt_reverse[0],rgraph->tvwgt[0],rgraph->tvwgt_reverse[0]);


  *r_lgraph=lgraph;
  *r_rgraph=rgraph;

}


#endif