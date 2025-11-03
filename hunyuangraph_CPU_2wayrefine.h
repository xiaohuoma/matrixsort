#ifndef _H_CPU_2WAYREFINE
#define _H_CPU_2WAYREFINE

#include "hunyuangraph_struct.h"
#include "hunyuangraph_balance.h"
#include "hunyuangraph_priorityqueue.h"

/*Malloc cpu 2way-refine params*/
void hunyuangraph_allocate_cpu_2waymem(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
  int nvtxs;
  nvtxs = graph->nvtxs;

  graph->pwgts=(int*)malloc(2*sizeof(int));
  graph->where=(int*)malloc(nvtxs*sizeof(int));
  graph->bndptr=(int*)malloc(nvtxs*sizeof(int));
  graph->bndlist=(int*)malloc(nvtxs*sizeof(int));
  graph->id=(int*)malloc(nvtxs*sizeof(int));
  graph->ed=(int*)malloc(nvtxs*sizeof(int));
}

/*Compute cpu 2way-refine params*/
void hunyuangraph_compute_cpu_2wayparam(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
  int i,j,nvtxs,nbnd,mincut,istart,iend,tid,ted,me;
  int *xadj,*vwgt,*adjncy,*adjwgt,*pwgts;
  int *where,*bndptr,*bndlist,*id,*ed;

  nvtxs= graph->nvtxs;
  xadj=graph->xadj;
  vwgt=graph->vwgt;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  where=graph->where;
  id=graph->id;
  ed=graph->ed;

  pwgts=hunyuangraph_int_set_value(2,0,graph->pwgts);
  bndptr=hunyuangraph_int_set_value(nvtxs,-1,graph->bndptr);

  bndlist=graph->bndlist;

  for(i=0;i<nvtxs;i++){
    pwgts[where[i]] += vwgt[i];
  }

  for(nbnd=0,mincut=0,i=0;i<nvtxs;i++){
    istart=xadj[i];
    iend=xadj[i+1];
    me=where[i];
    tid=ted=0;

    for(j=istart;j<iend;j++){
      if(me==where[adjncy[j]]){
        tid+=adjwgt[j];
      }
      else{
        ted+=adjwgt[j];
      }
    }

    id[i]=tid;
    ed[i]=ted;

    if(ted>0||istart==iend){
      hunyuangraph_listinsert(nbnd,bndlist,bndptr,i);
      mincut+=ted;
    }
  }

  graph->mincut=mincut/2;
  graph->nbnd=nbnd; 

}

/*Cpu graph refine two partitions*/
void hunyuangraph_cpu_2way_refine(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph, float *ntpwgts, int iteration_num)
{
  int i,ii,j,k,kwgt,nvtxs,nbnd,nswaps,from,to,pass,limit,temp;
  int *xadj,*vwgt,*adjncy,*adjwgt,*where,*id,*ed,*bndptr,*bndlist,*pwgts;
  int *moved,*swaps,*perm;

  hunyuangraph_queue_t *queues[2];
  int higain,mincut, mindiff,origdiff,initcut,newcut,mincutorder,avgvwgt;
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
  swaps=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);
  perm=hunyuangraph_int_malloc_space(hunyuangraph_admin,nvtxs);

  tpwgts[0]=graph->tvwgt[0]*ntpwgts[0];
  tpwgts[1]=graph->tvwgt[0]-tpwgts[0];

  limit=hunyuangraph_min(hunyuangraph_max(0.01*nvtxs,15),100);
  avgvwgt=hunyuangraph_min((pwgts[0]+pwgts[1])/20,2*(pwgts[0]+pwgts[1])/nvtxs);

  queues[0]=hunyuangraph_queue_create(nvtxs);
  queues[1]=hunyuangraph_queue_create(nvtxs);

  origdiff=abs(tpwgts[0]-pwgts[0]);
  hunyuangraph_int_set_value(nvtxs,-1,moved);

  for(pass=0;pass<iteration_num;pass++){ 
    hunyuangraph_queue_reset(queues[0]);
    hunyuangraph_queue_reset(queues[1]);

    mincutorder=-1;
    newcut=mincut=initcut=graph->mincut;
    mindiff=abs(tpwgts[0]-pwgts[0]);
    nbnd=graph->nbnd;
    hunyuangraph_int_randarrayofp(nbnd,perm,nbnd,1); 

    for(ii=0;ii<nbnd;ii++){
      i=perm[ii];
      hunyuangraph_queue_insert(queues[where[bndlist[i]]],bndlist[i],ed[bndlist[i]]-id[bndlist[i]]);
    }       

    for(nswaps=0;nswaps<nvtxs;nswaps++){
      from=(tpwgts[0]-pwgts[0]<tpwgts[1]-pwgts[1]?0:1);
      to=(from+1)%2;

      if((higain=hunyuangraph_queue_top(queues[from]))==-1){
        break;
      }

      newcut-=(ed[higain]-id[higain]);
      hunyuangraph_add_sub(pwgts[to],pwgts[from],vwgt[higain]);

      if((newcut<mincut&&abs(tpwgts[0]-pwgts[0])<=origdiff+avgvwgt)|| 
          (newcut==mincut&&abs(tpwgts[0]-pwgts[0])<mindiff)){
        mincut=newcut;
        mindiff=abs(tpwgts[0]-pwgts[0]);
        mincutorder=nswaps;
      }
      else if(nswaps-mincutorder>limit){ 
        newcut+=(ed[higain]-id[higain]);
        hunyuangraph_add_sub(pwgts[from],pwgts[to],vwgt[higain]);
        break;
      }

      where[higain]=to;
      moved[higain]=nswaps;
      swaps[nswaps]=higain;

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
            
            if(moved[k]==-1){  
              hunyuangraph_queue_delete(queues[where[k]],k);
            }
          }
          else{ 
            if(moved[k]==-1){ 
              hunyuangraph_queue_update(queues[where[k]],k,ed[k]-id[k]);
            }
          }
        }
        else{
          if(ed[k]>0){  
            hunyuangraph_listinsert(nbnd,bndlist,bndptr,k);
            
            if(moved[k]==-1){ 
              hunyuangraph_queue_insert(queues[where[k]],k,ed[k]-id[k]);
            }
          }
        }
      }
    }

	// printf("2way_CPU:nvtxs=%d moved=%d\n",nvtxs,nswaps);

    for(i=0;i<nswaps;i++){
      moved[swaps[i]]=-1;  
    }

    for(nswaps--;nswaps>mincutorder;nswaps--){
      higain=swaps[nswaps];
      to=where[higain]=(where[higain]+1)%2;
      hunyuangraph_swap(id[higain],ed[higain],temp);

      if(ed[higain]==0&&bndptr[higain]!=-1&&xadj[higain]<xadj[higain+1]){
        hunyuangraph_listdelete(nbnd,bndlist,bndptr,higain);
      }
      else if(ed[higain]>0&&bndptr[higain]==-1){
        hunyuangraph_listinsert(nbnd,bndlist,bndptr,higain);
      }

      hunyuangraph_add_sub(pwgts[to],pwgts[(to+1)%2],vwgt[higain]);

      for(j=xadj[higain];j<xadj[higain+1];j++){
        k=adjncy[j];
        kwgt=(to==where[k]?adjwgt[j]:-adjwgt[j]);
        hunyuangraph_add_sub(id[k],ed[k],kwgt);

        if(bndptr[k]!=-1&&ed[k]==0){
          hunyuangraph_listdelete(nbnd,bndlist,bndptr,k);
        }
        if(bndptr[k]==-1&&ed[k]>0){
          hunyuangraph_listinsert(nbnd,bndlist,bndptr,k);
        }
      }
    }

    graph->mincut=mincut;
    graph->nbnd=nbnd;

    // printf("pass=%d nvtxs=%d\n",pass,nvtxs);
    // printf("graph->mincut=%d\n\n",graph->mincut);

    if(mincutorder<=0||mincut==initcut){
      break;
    }
  }

  hunyuangraph_queue_free(queues[0]);
  hunyuangraph_queue_free(queues[1]);

}

/*Cpu graph 2-way projection*/
void hunyuangraph_2way_project(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
  int i,j,istart,iend,nvtxs,nbnd,me,tid,ted;
  int *xadj,*adjncy,*adjwgt;
  int *cmap,*where,*bndptr,*bndlist;
  int *cwhere,*cbndptr;
  int *id,*ed;

  hunyuangraph_graph_t *cgraph;
  hunyuangraph_allocate_cpu_2waymem(hunyuangraph_admin,graph);

  cgraph=graph->coarser;
  cwhere=cgraph->where;
  cbndptr=cgraph->bndptr;
  nvtxs=graph->nvtxs;
  cmap=graph->cmap;
  xadj=graph->xadj;
  adjncy=graph->adjncy;
  adjwgt=graph->adjwgt;
  where=graph->where;
  id=graph->id;
  ed=graph->ed;

  bndptr=hunyuangraph_int_set_value(nvtxs,-1,graph->bndptr);
  bndlist=graph->bndlist;

  for(i=0;i<nvtxs;i++){
    j=cmap[i];
    where[i]=cwhere[j];
    cmap[i]=cbndptr[j];
  }

	cudaDeviceSynchronize();
	gettimeofday(&begin_save_init, NULL);
  for(nbnd=0,i=0;i<nvtxs;i++){
    istart=xadj[i];
    iend=xadj[i+1];
    tid=ted=0;

    if(cmap[i]==-1){ 
      for(j=istart;j<iend;j++){
        tid+=adjwgt[j];
      }
    }
    else{ 
      me=where[i];

      for(j=istart;j<iend;j++){
        if(me==where[adjncy[j]]){
          tid += adjwgt[j];
        }
        else{
          ted+=adjwgt[j];
        }
      }
    }

    id[i]=tid;
    ed[i]=ted;

    if(ted>0||istart==iend){ 
      hunyuangraph_listinsert(nbnd,bndlist,bndptr,i);
    }

  }
  	cudaDeviceSynchronize();
	gettimeofday(&end_save_init, NULL);
	save_init += (end_save_init.tv_sec - begin_save_init.tv_sec) * 1000 + (end_save_init.tv_usec - begin_save_init.tv_usec) / 1000.0;

  graph->mincut=cgraph->mincut;
  graph->nbnd=nbnd;

  hunyuangraph_int_copy(2,cgraph->pwgts,graph->pwgts);
  hunyuangraph_free_graph(&graph->coarser);
  graph->coarser=NULL;

}

__global__ void projectback_init(int *where, int *cwhere, int *cmap, int nvtxs)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;

	if(ii < nvtxs)
	{
		int t = cmap[ii];
		where[ii] = cwhere[t];
	}
}

/*Cpu refinement algorithm*/
void hunyuangraph_cpu_refinement(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *orggraph, hunyuangraph_graph_t *graph, float *tpwgts)
{
  hunyuangraph_compute_cpu_2wayparam(hunyuangraph_admin,graph);

  for(;;){
	hunyuangraph_2way_bal(hunyuangraph_admin,graph,tpwgts);
    // if(graph->nvtxs <= 10000) hunyuangraph_2way_bal(hunyuangraph_admin,graph,tpwgts);

	hunyuangraph_cpu_2way_refine(hunyuangraph_admin,graph,tpwgts,hunyuangraph_admin->iteration_num);
    /*if(graph->nvtxs <= 10000) hunyuangraph_cpu_2way_refine(hunyuangraph_admin,graph,tpwgts,hunyuangraph_admin->iteration_num); 
	else
	{
		FM_2WayCutRefine_GPU(hunyuangraph_admin, graph, tpwgts);
		n++;
	}*/
    
    if(graph==orggraph){
      	break;
    }

	// cudaFree(graph->cuda_xadj);
	// cudaFree(graph->cuda_adjncy);
	// cudaFree(graph->cuda_adjwgt);
	// cudaFree(graph->cuda_vwgt);
	// cudaFree(graph->cuda_where);

    graph=graph->finer;
	// printf("nvtxs=%d\n",graph->nvtxs);

	hunyuangraph_2way_project(hunyuangraph_admin,graph);
    /*if(graph->nvtxs <= 10000) hunyuangraph_2way_project(hunyuangraph_admin,graph);
	else
	{
		int *cwhere;
		cudaMalloc((void**)&graph->cuda_where,sizeof(int) * graph->nvtxs);
		cudaMalloc((void**)&cwhere,sizeof(int) * graph->coarser->nvtxs);
		printf("a\n");
		cudaMemcpy(cwhere,graph->coarser->where,sizeof(int) * graph->nvtxs, cudaMemcpyDeviceToHost);
		printf("b\n");
		projectback_init<<<(graph->nvtxs + 127) / 128,128>>>(graph->cuda_where,cwhere,graph->cuda_cmap,graph->nvtxs);
		printf("c\n");
		graph->where = (int *)malloc(sizeof(int) * graph->nvtxs);
		graph->pwgts = (int *)malloc(sizeof(int) * 2);
		cudaMemcpy(graph->where,graph->cuda_where,sizeof(int) * graph->nvtxs, cudaMemcpyDeviceToHost);
		cudaFree(graph->cuda_where);
		cudaFree(graph->cuda_cmap);
		cudaFree(cwhere);
		printf("d\n");
	}*/
  }

}

#endif