#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<stdint.h>
#include<stdarg.h>
#include<time.h>
#include<cuda_runtime.h>
#include<sys/time.h>
#include <sys/types.h>
#include<thrust/scan.h>
#include <thrust/reduce.h>
#include<thrust/sort.h>
#include<thrust/execution_policy.h>
#include<thrust/device_ptr.h>
#include "bb_segsort.h"
#include "hunyuangraph_io.h"
#include "hunyuangraph_bb_segsort.h"
#include "hunyuangraph_define.h"
#include "hunyuangraph_GPU_memory.h"
#include "hunyuangraph_GPU_prefixsum.h"
#include "hunyuangraph_timer.h"
#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"
#include "hunyuangraph_admin.h"
#include "hunyuangraph_common.h"
#include "hunyuangraph_GPU_common.h"
#include "hunyuangraph_partitiongraph.h"
#include "hunyuangraph_GPU_coarsen.h"
#include "hunyuangraph_GPU_match.h"
#include "hunyuangraph_GPU_contraction.h"
#include "hunyuangraph_CPU_initialpartition.h"
#include "hunyuangraph_GPU_initialpartition.h"
#include "hunyuangraph_CPU_coarsen.h"
#include "hunyuangraph_CPU_match.h"
#include "hunyuangraph_CPU_contraction.h"
#include "hunyuangraph_priorityqueue.h"
#include "hunyuangraph_CPU_2wayrefine.h"
#include "hunyuangraph_balance.h"
#include "hunyuangraph_CPU_splitgraph.h"
#include "hunyuangraph_GPU_uncoarsen.h"
#include "hunyuangraph_GPU_krefine.h"
// #include "reduce_hem.h"
// #include "struct.h"
// #include "define.h"
// #include "graph.h"
// #include "admin.h"
// #include "common.h"
// #include "queue.h"
// #include "read.h"
// #include "GPU_coarsen.h"
// #include "CPU_coarsen.h"
// #include "initpartition.h"
// #include "GPU_uncoarsen.h"
// #include "nouse.h"

int hunyuangraph_PartGraph(int *nvtxs,  int *xadj, int *adjncy, int *vwgt,int *adjwgt, \
    int *nparts, float *tpwgts, float *ubvec, int *objval, int *part);