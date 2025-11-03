#ifndef _H_PRIORITYQUEUE
#define _H_PRIORITYQUEUE

#include "hunyuangraph_struct.h"

/*Init queue */
 void hunyuangraph_queue_init(hunyuangraph_queue_t *queue, size_t maxnodes)
{
  int i;
  queue->nnodes=0;
  queue->maxnodes=maxnodes;
  queue->heap=(hunyuangraph_rkv_t*)malloc(sizeof(hunyuangraph_rkv_t)*maxnodes);
  queue->locator=(ssize_t*)malloc(sizeof(ssize_t)*maxnodes);

  for(i=0;i<maxnodes;i++){
    queue->locator[i]=-1;
  }

}

/*Create queue*/
hunyuangraph_queue_t *hunyuangraph_queue_create(size_t maxnodes)
{
  hunyuangraph_queue_t *queue; 
  queue = (hunyuangraph_queue_t *)malloc(sizeof(hunyuangraph_queue_t));

  hunyuangraph_queue_init(queue, maxnodes);

  return queue;
}

/*Insert node to queue*/
int hunyuangraph_queue_insert(hunyuangraph_queue_t *queue, int node, int key)
{
  ssize_t i,j;
  ssize_t *locator=queue->locator;
  hunyuangraph_rkv_t *heap=queue->heap;
  i = queue->nnodes++;

  while(i>0){
    j=(i-1)>>1;

    if(M_GT_N(key,heap[j].key)){
      heap[i]=heap[j];
      locator[heap[i].val]=i;
      i=j;
    }
    else
      break;
  }

  heap[i].key=key;
  heap[i].val=node;
  locator[node]=i;

  return 0;

}

/*Get top of queue*/
int hunyuangraph_queue_top(hunyuangraph_queue_t *queue)
{
  ssize_t i, j;
  ssize_t *locator;
  hunyuangraph_rkv_t *heap;

  int vtx, node;
  float key;

  if (queue->nnodes==0){
    return -1;
  }

  queue->nnodes--;
  heap=queue->heap;
  locator=queue->locator;
  vtx=heap[0].val;
  locator[vtx]=-1;

  if ((i=queue->nnodes)>0){
    key=heap[i].key;
    node=heap[i].val;
    i=0;

    while((j=2*i+1)<queue->nnodes){
      if(M_GT_N(heap[j].key,key)){
        if(j+1 < queue->nnodes&&M_GT_N(heap[j+1].key,heap[j].key)){
          j=j+1;
        }

        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else if(j+1<queue->nnodes&&M_GT_N(heap[j+1].key,key)){
        j=j+1;
        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else
        break;
    }

    heap[i].key=key;
    heap[i].val=node;
    locator[node]=i;

  }

  return vtx;

}

/*Delete node of queue*/
int hunyuangraph_queue_delete(hunyuangraph_queue_t *queue, int node)
{
  ssize_t i, j, nnodes;
  float newkey, oldkey;
  ssize_t *locator=queue->locator;

  hunyuangraph_rkv_t *heap=queue->heap;

  i=locator[node];
  locator[node]=-1;

  if(--queue->nnodes>0&&heap[queue->nnodes].val!=node) {
    node=heap[queue->nnodes].val;
    newkey=heap[queue->nnodes].key;
    oldkey=heap[i].key;

    if(M_GT_N(newkey,oldkey)){ 
      while(i>0){
        j=(i-1)>>1;

        if(M_GT_N(newkey,heap[j].key)){
          heap[i]=heap[j];
          locator[heap[i].val]=i;
          i=j;
        }
        else
          break;
      }
    }
    else{ 
      nnodes=queue->nnodes;

      while((j=(i<<1)+1)<nnodes){
        if(M_GT_N(heap[j].key,newkey)){
          if(j+1<nnodes&&M_GT_N(heap[j+1].key,heap[j].key)){
            j++;
          }

          heap[i]=heap[j];
          locator[heap[i].val]=i;
          i=j;
        }
        else if(j+1<nnodes&&M_GT_N(heap[j+1].key,newkey)){
          j++;
          heap[i]=heap[j];
          locator[heap[i].val]=i;
          i=j;
        }
        else
          break;
      }
    }

    heap[i].key=newkey;
    heap[i].val=node;
    locator[node]=i;

  }

  return 0;
}

/*Update queue node key*/
void hunyuangraph_queue_update(hunyuangraph_queue_t *queue, int node, int newkey)
{
  ssize_t i, j, nnodes;
  float oldkey;
  ssize_t *locator=queue->locator;

  hunyuangraph_rkv_t *heap=queue->heap;
  oldkey=heap[locator[node]].key;
  i=locator[node];

  if(M_GT_N(newkey,oldkey)){ 
    while(i>0){
      j=(i-1)>>1;

      if(M_GT_N(newkey,heap[j].key)){
        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else
        break;
    }
  }
  else{ 
    nnodes = queue->nnodes;

    while((j=(i<<1)+1)<nnodes){
      if(M_GT_N(heap[j].key,newkey)){
        if(j+1<nnodes&&M_GT_N(heap[j+1].key,heap[j].key)){
          j++;
        }

        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else if(j+1<nnodes&&M_GT_N(heap[j+1].key,newkey)){
        j++;
        heap[i]=heap[j];
        locator[heap[i].val]=i;
        i=j;
      }
      else
        break;
    }
  }

  heap[i].key=newkey;
  heap[i].val=node;
  locator[node]=i;
  return;

}

/*Free queue*/
void hunyuangraph_queue_free(hunyuangraph_queue_t *queue)
{
  if(queue == NULL) return;

  free(queue->heap);
  free(queue->locator);

  queue->maxnodes = 0;

  free(queue);
}

/*Reset queue*/
void hunyuangraph_queue_reset(hunyuangraph_queue_t *queue)
{
  ssize_t i;
  ssize_t *locator=queue->locator;

  hunyuangraph_rkv_t *heap=queue->heap;

  for(i=queue->nnodes-1;i>=0;i--){
    locator[heap[i].val]=-1;
  }

  queue->nnodes=0;

}


#endif