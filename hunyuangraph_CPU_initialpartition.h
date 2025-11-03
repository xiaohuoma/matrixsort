#ifndef _H_CPU_INITIALPARTITION
#define _H_CPU_INITIALPARTITION

#include "hunyuangraph_struct.h"
#include "hunyuangraph_common.h"
#include "hunyuangraph_admin.h"
#include "hunyuangraph_CPU_coarsen.h"
#include "hunyuangraph_priorityqueue.h"
#include "hunyuangraph_balance.h"
#include "hunyuangraph_CPU_2wayrefine.h"
#include "hunyuangraph_CPU_splitgraph.h"

/*************************************************************************/
/*! Computes the maximum load imbalance difference of a partitioning 
    solution over all the constraints. 
    The difference is defined with respect to the allowed maximum 
    unbalance for the respective constraint. 
 */
/**************************************************************************/ 
float ComputeLoadImbalanceDiff(hunyuangraph_graph_t *graph, int nparts, float *pijbm,float *ubvec)
{
	int  j, *pwgts;
	float max, cur;
	pwgts = graph->pwgts;
	max = -1.0;
	for (j=0; j<nparts; j++) 
	{
		cur = pwgts[j]*pijbm[j] - ubvec[0];
		if (cur > max)
			max = cur;
	}
	return max;
}

/*Cpu growbisection algorithm*/
void huyuangraph_cpu_growbisection(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, float *ntpwgts, int niparts)
{
  int i,j,k,nvtxs,dd,nleft,first,last,pwgts[2],oneminpwgt,onemaxpwgt, 
      bestcut=0,iter;

  int *xadj,*vwgt,*adjncy,*where;
  int *queue,*tra,*bestwhere;

  nvtxs=graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;

  hunyuangraph_allocate_cpu_2waymem(hunyuangraph_admin,graph);

  where=graph->where;

  bestwhere=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);
  queue=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);
  tra=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);

  onemaxpwgt=hunyuangraph_admin->ubfactors[0]*graph->tvwgt[0]*ntpwgts[1];
  oneminpwgt=(1.0/hunyuangraph_admin->ubfactors[0])*graph->tvwgt[0]*ntpwgts[1]; 
  
  for (iter=0; iter<niparts; iter++){
    hunyuangraph_int_set_value(nvtxs,1,where);
    hunyuangraph_int_set_value(nvtxs,0,tra);

    pwgts[1]=graph->tvwgt[0];
    pwgts[0]=0;
    queue[0]=hunyuangraph_int_randinrange(nvtxs);
    tra[queue[0]]=1;
    first=0; 
    last=1;
    nleft=nvtxs-1;
    dd=0;

    for(;;){
      if(first==last){ 
        if(nleft==0||dd){
          break;
        }

        k=hunyuangraph_int_randinrange(nleft);

        for(i=0;i<nvtxs;i++){
          if(tra[i]==0){
            if(k==0){
              break;
            }
            else{
              k--;
            }
          }
        }

        queue[0]=i;
        tra[i]=1;
        first=0; 
        last=1;
        nleft--;
      }

      i=queue[first++];

      if(pwgts[0]>0&&pwgts[1]-vwgt[i]<oneminpwgt){
        dd=1;
        continue;
      }

      where[i]=0;

      hunyuangraph_add_sub(pwgts[0],pwgts[1],vwgt[i]);

      if(pwgts[1]<=onemaxpwgt){
        break;
      }

      dd=0;

      for(j=xadj[i];j<xadj[i+1];j++){
        k=adjncy[j];

        if(tra[k]==0){
          queue[last++]=k;
          tra[k]=1;
          nleft--;
        }
      }
    }

    hunyuangraph_compute_cpu_2wayparam(hunyuangraph_admin,graph);
    hunyuangraph_2way_bal(hunyuangraph_admin,graph,ntpwgts);

    hunyuangraph_cpu_2way_refine(hunyuangraph_admin,graph,ntpwgts,hunyuangraph_admin->iteration_num);

    if(iter==0||bestcut>graph->mincut){
      bestcut=graph->mincut;
      hunyuangraph_int_copy(nvtxs,where,bestwhere);
      
      if(bestcut==0){
        break;
      }
    }
  }

  graph->mincut=bestcut;
  hunyuangraph_int_copy(nvtxs,bestwhere,where);

}

/*Cpu multilevel bisection algorithm*/
int hunyuangraph_cpu_mlevelbisect(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, float *tpwgts)
{
	int niparts,bestobj=0,curobj=0,*bestwhere=NULL;
	hunyuangraph_graph_t *tgraph = graph;
	hunyuangraph_graph_t *cgraph;
	double bestbal=0.0, curbal=0.0;

	hunyuangraph_compute_2way_balance(hunyuangraph_admin,graph,tpwgts);

	// printf("cnvtxs=%d cnedges=%d\n",graph->nvtxs,graph->nedges);
	/*if(graph->nvtxs > 10000) 
	{
		int nvtxs  = graph->nvtxs;
		int nedges = graph->nedges;

		cudaMalloc((void**)&graph->cuda_xadj,(nvtxs+1)*sizeof(int));
		cudaMalloc((void**)&graph->cuda_vwgt,nvtxs*sizeof(int));
		cudaMalloc((void**)&graph->cuda_adjncy,nedges*sizeof(int));
		cudaMalloc((void**)&graph->cuda_adjwgt,nedges*sizeof(int));

		cudaMemcpy(graph->cuda_xadj,graph->xadj,(nvtxs + 1) * sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(graph->cuda_adjncy,graph->adjncy,nedges*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(graph->cuda_adjwgt,graph->adjwgt,nedges*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(graph->cuda_vwgt,graph->vwgt,nvtxs*sizeof(int),cudaMemcpyHostToDevice);

		int level = 0;
		hunyuangraph_admin->maxvwgt = 1.5 * graph->tvwgt[0] / hunyuangraph_admin->Coarsen_threshold; 
		do
		{
			hunyuangraph_malloc_coarseninfo(hunyuangraph_admin,graph);
		
			hunyuangraph_gpu_match(hunyuangraph_admin,graph);

			graph->cmap=(int*)malloc(sizeof(int)*(graph->nvtxs));
			cudaMemcpy(graph->cmap, graph->cuda_cmap, graph->nvtxs * sizeof(int), cudaMemcpyDeviceToHost);

			// cudaFree(graph->cuda_xadj);
			// cudaFree(graph->cuda_vwgt);
			// cudaFree(graph->cuda_adjncy);
			// cudaFree(graph->cuda_adjwgt);
			// cudaFree(graph->cuda_cmap);

			graph = graph->coarser;

			cudaDeviceSynchronize();
			gettimeofday(&begin_save_init, NULL);
			graph->xadj   = (int *)malloc(sizeof(int) * (graph->nvtxs + 1)); 
			graph->vwgt   = (int *)malloc(sizeof(int) * graph->nvtxs); 
			graph->adjncy = (int *)malloc(sizeof(int) * graph->nedges);
			graph->adjwgt = (int *)malloc(sizeof(int) * graph->nedges);
			cudaMemcpy(graph->xadj, graph->cuda_xadj, (graph->nvtxs + 1) * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(graph->vwgt, graph->cuda_vwgt, graph->nvtxs * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(graph->adjncy, graph->cuda_adjncy, graph->nedges * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(graph->adjwgt, graph->cuda_adjwgt, graph->nedges * sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			gettimeofday(&end_save_init, NULL);
			save_init += (end_save_init.tv_sec - begin_save_init.tv_sec) * 1000 + (end_save_init.tv_usec - begin_save_init.tv_usec) / 1000.0;

			level++;
			// printf("level=%d\n",level);
		}while(
			graph->nvtxs > 10000 && \
			graph->nvtxs > hunyuangraph_admin->Coarsen_threshold && \
			graph->nvtxs < 0.85 * graph->finer->nvtxs && \
			graph->nedges > graph->nvtxs / 2); 

		// graph->xadj   = (int *)malloc(sizeof(int) * (graph->nvtxs + 1)); 
		// graph->vwgt   = (int *)malloc(sizeof(int) * graph->nvtxs); 
		// graph->adjncy = (int *)malloc(sizeof(int) * graph->nedges);
		// graph->adjwgt = (int *)malloc(sizeof(int) * graph->nedges);
		// cudaMemcpy(graph->xadj, graph->cuda_xadj, (graph->nvtxs + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		// cudaMemcpy(graph->vwgt, graph->cuda_vwgt, graph->nvtxs * sizeof(int), cudaMemcpyDeviceToHost);
		// cudaMemcpy(graph->adjncy, graph->cuda_adjncy, graph->nedges * sizeof(int), cudaMemcpyDeviceToHost);
		// cudaMemcpy(graph->adjwgt, graph->cuda_adjwgt, graph->nedges * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(graph->cuda_xadj);
		cudaFree(graph->cuda_vwgt);
		cudaFree(graph->cuda_adjncy);
		cudaFree(graph->cuda_adjwgt);
	}*/
	/*cgraph=hunyuangraph_cpu_coarsen(hunyuangraph_admin,graph);
	// printf("cgraph->nvtxs=%d\n",cgraph->nvtxs);
	niparts=5;

	huyuangraph_cpu_growbisection(hunyuangraph_admin,cgraph,tpwgts,niparts);

	graph = tgraph;
	hunyuangraph_cpu_refinement(hunyuangraph_admin,graph,cgraph,tpwgts);

	curobj=graph->mincut;
	bestobj=curobj;

	if(bestobj!=curobj){
		hunyuangraph_int_copy(graph->nvtxs,bestwhere,graph->where);
		hunyuangraph_compute_cpu_2wayparam(hunyuangraph_admin,graph);
	}*/

	if (hunyuangraph_admin->ncuts > 1)
    	bestwhere = (int *)malloc(sizeof(int) * graph->nvtxs);

  	for (int i = 0; i < hunyuangraph_admin->ncuts; i++) 
	{
		cgraph=hunyuangraph_cpu_coarsen(hunyuangraph_admin,graph);

		niparts = (cgraph->nvtxs <= hunyuangraph_admin->Coarsen_threshold ? 5 : 7);
		huyuangraph_cpu_growbisection(hunyuangraph_admin,cgraph,tpwgts,niparts);

		hunyuangraph_cpu_refinement(hunyuangraph_admin,graph,cgraph,tpwgts);

		curobj = graph->mincut;
		curbal = ComputeLoadImbalanceDiff(graph, 2, hunyuangraph_admin->part_balance, hunyuangraph_admin->ubfactors);

		if (i == 0  || (curbal <= 0.0005 && bestobj > curobj) || (bestbal > 0.0005 && curbal < bestbal)) 
		{
			bestobj = curobj;
			bestbal = curbal;
			if (i < hunyuangraph_admin->ncuts-1)
				hunyuangraph_int_copy(graph->nvtxs, graph->where, bestwhere);
    	}

		if (bestobj == 0)
			break;

		if (i < hunyuangraph_admin->ncuts-1)
			hunyuangraph_free_graph(&graph);
  	}

	if (bestobj != curobj) {
		hunyuangraph_int_copy(graph->nvtxs, bestwhere, graph->where);
		hunyuangraph_compute_cpu_2wayparam(hunyuangraph_admin,graph);
	}

  	return bestobj;
}

/*Cpu Multilevel resursive bisection*/
int hunyuangraph_mlevel_rbbisection(hunyuangraph_admin_t *hunyuangraph_admin, \
	hunyuangraph_graph_t *graph, int nparts, int *part, float *tpwgts, int fpart, int level)
{
	int i,nvtxs,objval;
	int *label,*where;

	hunyuangraph_graph_t *lgraph,*rgraph;
	float wsum,*tpwgts2;

	if(graph->nvtxs == 0){
		printf("****You are trying to partition too many parts!****\n");
		return 0;
	}

	nvtxs=graph->nvtxs;

	tpwgts2=hunyuangraph_float_malloc_space(hunyuangraph_admin);
	tpwgts2[0]=hunyuangraph_float_sum((nparts>>1),tpwgts);
	tpwgts2[1]=1.0-tpwgts2[0];

  	objval=hunyuangraph_cpu_mlevelbisect(hunyuangraph_admin,graph,tpwgts2);

	// printf("hunyuangraph_cpu_mlevelbisect\n");

    level++;

	if(level == 1)
	{
		where = graph->where;
		for(i = 0;i < nvtxs;i++)
			part[i] = where[i] + fpart;
	}
	else
	{
		label = graph->label;
		where = graph->where;

		for(i = 0;i < nvtxs;i++){
			part[label[i]] = where[i] + fpart;
		}
	}

	// printf("label\n");

	if(nparts>2)
	{
		// if(graph->nvtxs > 10000) splitgraph_GPU(hunyuangraph_admin,graph,&lgraph,&rgraph,level);
		// else
		// {
			if(level != 1) hunyuangraph_splitgraph(hunyuangraph_admin,graph,&lgraph,&rgraph);
			else hunyuangraph_splitgraph_first(hunyuangraph_admin,graph,&lgraph,&rgraph);
		// }
		// printf("hunyuangraph_cpu_mlevelbisect\n");
	}
	
	hunyuangraph_free_graph(&graph);

	wsum=hunyuangraph_float_sum((nparts>>1),tpwgts);
	
	hunyuangraph_tpwgts_rescale((nparts>>1),1.0/wsum,tpwgts);
	hunyuangraph_tpwgts_rescale(nparts-(nparts>>1),1.0/(1.0-wsum),tpwgts+(nparts>>1));

	if(nparts>3)
	{
		objval+=hunyuangraph_mlevel_rbbisection(hunyuangraph_admin,lgraph,(nparts>>1),part,tpwgts,fpart,level);
		objval+=hunyuangraph_mlevel_rbbisection(hunyuangraph_admin,rgraph,nparts-(nparts>>1),part,tpwgts+(nparts>>1),fpart+(nparts>>1),level);
	}
	else if(nparts==3)
	{
		hunyuangraph_free_graph(&lgraph);
		objval+=hunyuangraph_mlevel_rbbisection(hunyuangraph_admin,rgraph,nparts-(nparts>>1),part,tpwgts+(nparts>>1),fpart+(nparts>>1),level);
	}
	
	return objval;
}

/*Cpu graph partition algorithm*/
int hunyuangraph_rbbisection(int *nvtxs, int *xadj, int *adjncy, int *vwgt,int *adjwgt, int *nparts, float *tpwgts, float *ubvec, int *objval, int *part, int *tvwgt)
{
	hunyuangraph_graph_t *graph;
	hunyuangraph_admin_t *hunyuangraph_admin;

	hunyuangraph_admin = hunyuangraph_set_graph_admin(*nparts, tpwgts, ubvec);

    graph = hunyuangraph_set_graph(hunyuangraph_admin, *nvtxs, xadj, adjncy, vwgt, adjwgt, tvwgt);
	hunyuangraph_allocatespace(hunyuangraph_admin, graph);           
	
	*objval = hunyuangraph_mlevel_rbbisection(hunyuangraph_admin, graph, *nparts, part, hunyuangraph_admin->tpwgts, 0, 0);
  
  	return 1;
}

/*Graph initial partition algorithm*/
void hunyuangarph_initialpartition(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
  int objval=0;
  int *bestwhere=NULL;
  float *ubvec=NULL;

  graph->where=(int *)malloc(sizeof(int)*graph->nvtxs);
  hunyuangraph_admin->ncuts = hunyuangraph_admin->nIparts;

  ubvec=(float*)malloc(sizeof(float));
  ubvec[0]=(float)pow(hunyuangraph_admin->ubfactors[0],1.0/log(hunyuangraph_admin->nparts));
  
  hunyuangraph_rbbisection(&graph->nvtxs,graph->xadj,graph->adjncy,graph->vwgt,graph->adjwgt, \
    &hunyuangraph_admin->nparts,hunyuangraph_admin->tpwgts,ubvec,&objval,graph->where,graph->tvwgt);
  
  free(ubvec);
  free(bestwhere);
}

#endif