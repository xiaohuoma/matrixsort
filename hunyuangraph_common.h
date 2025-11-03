#ifndef _H_COMMON
#define _H_COMMON

#include "hunyuangraph_struct.h"
#include "hunyuangraph_define.h"

/*Error exit*/
void hunyuangraph_error_exit(char *f_str,...)
{
  va_list a;
  va_start(a,f_str);
  vfprintf(stderr,f_str,a);
  va_end(a);

  if (strlen(f_str)==0||f_str[strlen(f_str)-1]!='\n'){
    fprintf(stderr,"\n");
  }

  fflush(stderr);

  if(1)
    exit(-2);
}

/*Compute log2 algorithm*/
int hunyuangraph_compute_log2(int a)
{
  int i;
  for(i=1;a>1;i++,a=a>>1);
  return i-1;
}

/*Get int rand number*/
int hunyuangraph_int_rand() 
{
  if(sizeof(int)<=sizeof(int32_t)) 
    return (int)(uint32_t)rand();
  else  
    return (int)(uint64_t)rand(); 
}

/*Get int rand number between (0,max)*/
int hunyuangraph_int_randinrange(int max) 
{
  return (int)((hunyuangraph_int_rand())%max); 
}

/*Compute sum of int array*/
int hunyuangraph_int_sum(size_t n, int *a)
{
  size_t i;
  int sum=0;
  for(i=0;i<n;i++,a+=1){
    sum+=(*a);
  }
  return sum;
}

/*Copy int array a to b*/
int  *hunyuangraph_int_copy(size_t n, int *a, int *b)
{
  return (int *)memmove((void *)b, (void *)a, sizeof(int)*n);
}

/*Set int array value*/
int *hunyuangraph_int_set_value(size_t n, int val, int *a)
{
  size_t i;
  for(i=0;i<n;i++){
    a[i]=val;
  }
  return a;
}

/*Compute sum of float array*/
float hunyuangraph_float_sum(size_t n, float *a)
{
  size_t i;
  float sum=0;
  for(i=0;i<n;i++,a+=1){
    sum+=(*a);
  }
  return sum;
}

double hunyuangraph_double_sum(size_t n, double *src)
{
	double sum = 0;
	for(int i = 0;i < n;i++)
		sum += src[i];
	return sum;
}

/*Rescale tpwgts array*/
float *hunyuangraph_tpwgts_rescale(size_t n, float wsum, float *a)
{
  size_t i;
  for(i=0;i<n;i++,a+=1){
    (*a)*=wsum;
  }
  return a;
}

/*Get random permute of p*/
void hunyuangraph_int_randarrayofp(int n, int *p, int m, int flag)
{
  int i,u,v;
  int temp;
  if(flag==1){
    for(i=0;i<n;i++)
      p[i] = (int)i;
  }

  if(n<10){
    for(i=0;i<n;i++){

      v=hunyuangraph_int_randinrange(n);
      u=hunyuangraph_int_randinrange(n);
     
      hunyuangraph_swap(p[v],p[u],temp);

    }
  }
  else{
    for(i=0;i<m;i++){

      v=hunyuangraph_int_randinrange(n-3);
      u=hunyuangraph_int_randinrange(n-3);
      
      hunyuangraph_swap(p[v+0],p[u+2],temp);
      hunyuangraph_swap(p[v+1],p[u+3],temp);
      hunyuangraph_swap(p[v+2],p[u+0],temp);
      hunyuangraph_swap(p[v+3],p[u+1],temp);

    }
  }
}

/*Creates mcore*/
hunyuangraph_mcore_t *hunyuangraph_create_mcore(size_t coresize)
{
  hunyuangraph_mcore_t *mcore;
  mcore=(hunyuangraph_mcore_t *)malloc(sizeof(hunyuangraph_mcore_t));
  memset(mcore,0,sizeof(hunyuangraph_mcore_t));

  mcore->coresize=coresize;
  mcore->corecpos=0;
  mcore->core=(coresize==0?NULL:(size_t*)malloc(sizeof(size_t)*(mcore->coresize)));
  mcore->nmops=2048;
  mcore->cmop=0;
  mcore->mops=(hunyuangraph_mop_t *)malloc((mcore->nmops)*sizeof(hunyuangraph_mop_t));

  return mcore;
}

/*Allocate work space*/
void hunyuangraph_allocatespace(hunyuangraph_admin_t *hunyuangraph_admin, hunyuangraph_graph_t *graph)
{
  size_t coresize;
  coresize=3*(graph->nvtxs+1)*sizeof(int)+5*(hunyuangraph_admin->nparts+1)*sizeof(int)\
  +5*(hunyuangraph_admin->nparts+1)*sizeof(float);

  hunyuangraph_admin->mcore=hunyuangraph_create_mcore(coresize);
  hunyuangraph_admin->nbrpoolsize=0;
  hunyuangraph_admin->nbrpoolcpos=0;
}

/*Add memory allocation*/
void hunyuangraph_add_mcore(hunyuangraph_mcore_t *mcore, int type, size_t nbytes, void *ptr)
{
  if(mcore->cmop==mcore->nmops){
    mcore->nmops*=2;
    mcore->mops=(hunyuangraph_mop_t*)realloc(mcore->mops, mcore->nmops*sizeof(hunyuangraph_mop_t));
    if(mcore->mops==NULL){
      exit(0);
    }
  }

  mcore->mops[mcore->cmop].type=type;
  mcore->mops[mcore->cmop].nbytes=nbytes;
  mcore->mops[mcore->cmop].ptr=ptr;
  mcore->cmop++;

  switch(type){
    case 1:
      break;
    
    case 2:
      mcore->num_callocs++;
      mcore->size_callocs+=nbytes;
      mcore->cur_callocs+=nbytes;
      if(mcore->max_callocs<mcore->cur_callocs){
        mcore->max_callocs=mcore->cur_callocs;
      }
      break;
    
    case 3:
      mcore->num_hallocs++;
      mcore->size_hallocs+=nbytes;
      mcore->cur_hallocs+=nbytes;
      if(mcore->max_hallocs<mcore->cur_hallocs){
        mcore->max_hallocs=mcore->cur_hallocs;
      }
      break;
    
    default:
      exit(0);
  }
}

/*Malloc mcore*/
void *hunyuangraph_malloc_mcore(hunyuangraph_mcore_t *mcore, size_t nbytes)
{
  void *ptr;
  nbytes+=(nbytes%8==0?0:8-nbytes%8);

  if(mcore->corecpos+nbytes<mcore->coresize){
    ptr=((char *)mcore->core)+mcore->corecpos;
    mcore->corecpos+=nbytes;
    hunyuangraph_add_mcore(mcore,2,nbytes,ptr);
  }
  else{
    ptr=(size_t*)malloc(nbytes);
    hunyuangraph_add_mcore(mcore,3,nbytes,ptr);
  }

  return ptr;
}

/*Malloc mcore space*/
void *hunyuangraph_malloc_space(hunyuangraph_admin_t *hunyuangraph_admin, size_t nbytes)
{
  return hunyuangraph_malloc_mcore(hunyuangraph_admin->mcore,nbytes);
}

/*Malloc int mcore space*/
int *hunyuangraph_int_malloc_space(hunyuangraph_admin_t *hunyuangraph_admin, size_t n)
{
  return (int *)hunyuangraph_malloc_space(hunyuangraph_admin, n*sizeof(int));
}

ikv_t *ikvwspacemalloc(hunyuangraph_admin_t *hunyuangraph_admin, size_t nunmatched)
{
	return (ikv_t *)hunyuangraph_malloc_space(hunyuangraph_admin, nunmatched*sizeof(ikv_t));
}

/*Malloc float mcore space*/
float *hunyuangraph_float_malloc_space(hunyuangraph_admin_t *hunyuangraph_admin)
{
  return (float *)hunyuangraph_malloc_space(hunyuangraph_admin,2*sizeof(float));
}

void ikvsorti(size_t n, ikv_t *base)
{
  #define ikey_lt(a, b) ((a)->key < (b)->key)
    GK_MKQSORT(ikv_t, base, n, ikey_lt);
  #undef ikey_lt
}

#endif