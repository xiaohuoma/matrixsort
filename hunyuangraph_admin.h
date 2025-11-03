#ifndef _H_ADMIN
#define _H_ADMIN

#include "hunyuangraph_struct.h"

/*Set graph admin params*/
hunyuangraph_admin_t *hunyuangraph_set_graph_admin(int nparts, float *tpwgts, float *ubvec)
{
  int i;
  hunyuangraph_admin_t *hunyuangraph_admin;
  hunyuangraph_admin=(hunyuangraph_admin_t *)malloc(sizeof(hunyuangraph_admin_t));
  memset((void *)hunyuangraph_admin,0,sizeof(hunyuangraph_admin_t));

  hunyuangraph_admin->iteration_num=10;
  hunyuangraph_admin->Coarsen_threshold=200;
  hunyuangraph_admin->nparts=nparts; 

  hunyuangraph_admin->maxvwgt=0;  
  hunyuangraph_admin->ncuts=1; 

  hunyuangraph_admin->tpwgts=(float*)malloc(sizeof(float)*nparts);
  for(i=0;i<nparts;i++){
    hunyuangraph_admin->tpwgts[i]=1.0/nparts;
  }

  hunyuangraph_admin->ubfactors=(float*)malloc(sizeof(float));
  hunyuangraph_admin->ubfactors[0] =1.03;

  hunyuangraph_admin->part_balance =(float*) malloc(sizeof(float)*nparts);
  return hunyuangraph_admin;  
}


#endif