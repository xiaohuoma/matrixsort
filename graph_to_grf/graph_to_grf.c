#include<stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<stdint.h>
#include<stdarg.h>
#include<time.h>
#include<sys/time.h>

/*Graph data structure*/
typedef struct hunyuangraph_graph_t {
  /*graph cpu params*/
  int nvtxs;                            //Graph vertex
  int nedges;	                          //Graph edge
  int *xadj;                            //Graph vertex csr array (xadj[nvtxs+1])
  int *adjncy;                          //Graph adjacency list (adjncy[nedges])
  int *adjwgt;   		                    //Graph edge weight array (adjwgt[nedges])
  int *vwgt;			                      //Graph vertex weight array(vwgr[nvtxs])
  int *tvwgt;                           //The sum of graph vertex weight 
  float *tvwgt_reverse;                 //The reciprocal of tvwgt
  int *label;                           //Graph vertex label(label[nvtxs])
  int *cmap;                            //The Label of graph vertex in cgraph(cmap[nvtxs]) 
  int mincut;                           //The min edfe-cut of graph partition
  int *where;                           //The label of graph vertex in which part(where[nvtxs]) 
  int *pwgts;                           //The partition vertex weight(pwgts[nparts])
  int nbnd;                             //Boundary vertex number
  int *bndlist;                         //Boundary vertex list
  int *bndptr;                          //Boundary vertex pointer
  int *id;                              //The sum of edge weight in same part
  int *ed;                              //The sum of edge weight in different part
  struct hunyuangraph_graph_t *coarser; //The coarser graph
  struct hunyuangraph_graph_t *finer;   //The finer graph
  /*graph gpu params*/
  int *cuda_xadj;
  int *cuda_adjncy;
  int *cuda_adjwgt;
  int *cuda_vwgt;               
  int *cuda_match;                      //CUDA graph vertex match array(match[nvtxs])
  int *cuda_cmap;
  // int *cuda_maxvwgt;                    //CUDA graph constraint vertex weight 
  int *txadj;                  //CUDA graph vertex pairs csr edge array(txadj[cnvtxs+1])
//   int *cuda_real_nvtxs;                 //CUDA graph params (i<match[cmap[i]])
//   int *cuda_s;                          //CUDA support array (cuda_s[nvtxs])
  int *tadjwgt;       //CUDA support scan array (tadjwgt[nedges])
//   int *cuda_scan_nedges_original;       //CUDA support scan array (cuda_scan_nedges_original[nedges])
  int *tadjncy;      //CUDA support scan array (tadjncy[nedges])
  int *cuda_maxwgt;                     //CUDA part weight array (cuda_maxwgt[npart])
  int *cuda_minwgt;                     //CUDA part weight array (cuda_minwgt[npart])
  int *cuda_where;
  int *cuda_label;
  int *cuda_pwgts;
  int *cuda_bnd;
  int *cuda_bndnum;
  int *cpu_bndnum;
  int *cuda_info;                       //CUDA support array(cuda_info[bnd_num*nparts])
  int *cuda_real_bnd_num;
  int *cuda_real_bnd;
//   int *cuda_tvwgt;  // graph->tvwgt
  float *cuda_tpwgts;
} hunyuangraph_graph_t;

/*Set graph params*/
void hunyuangraph_init_cpu_graph(hunyuangraph_graph_t *graph) 
{
  memset((void *)graph,0,sizeof(hunyuangraph_graph_t));
  graph->nvtxs     = -1;
  graph->nedges    = -1;
  graph->xadj      = NULL;
  graph->vwgt      = NULL;
  graph->adjncy    = NULL;
  graph->adjwgt    = NULL;
  graph->label     = NULL;
  graph->cmap      = NULL;
  graph->tvwgt     = NULL;
  graph->tvwgt_reverse  = NULL;
  graph->where     = NULL;
  graph->pwgts     = NULL;
  graph->mincut    = -1;
  graph->nbnd      = -1;
  graph->id        = NULL;
  graph->ed        = NULL;
  graph->bndptr    = NULL;
  graph->bndlist   = NULL;
  graph->coarser   = NULL;
  graph->finer     = NULL;
}

/*Malloc graph*/
hunyuangraph_graph_t *hunyuangraph_create_cpu_graph(void)
{
    hunyuangraph_graph_t *graph = (hunyuangraph_graph_t *)malloc(sizeof(hunyuangraph_graph_t));
    hunyuangraph_init_cpu_graph(graph);
    return graph;
}


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

/*Open file*/
FILE *hunyuangraph_fopen(char *fname, char *mode, const char *msg)
{
  FILE *fp;
  char error_message[8192];
  fp=fopen(fname, mode);
  if(fp!=NULL){
    return fp;
  }
  sprintf(error_message,"file: %s, mode: %s, [%s]",fname,mode,msg);
  perror(error_message);
  hunyuangraph_error_exit("Failed on file fopen()\n");
  return NULL;
}

/*Read graph file*/
hunyuangraph_graph_t *hunyuangraph_readgraph(char *filename)
{
  int i,k,fmt,nfields,readew,readvw,readvs,edge,ewgt;
  int *xadj,*adjncy,*vwgt,*adjwgt;
  char *line=NULL,fmtstr[256],*curstr,*newstr;
  size_t lnlen=0;
  FILE *fpin;

  hunyuangraph_graph_t *graph;
  graph = hunyuangraph_create_cpu_graph();

  fpin = hunyuangraph_fopen(filename,"r","Readgraph: Graph");

  do{
    if(getline(&line,&lnlen,fpin)==-1){ 
      hunyuangraph_error_exit("Premature end of input file: file: %s\n", filename);
    }
  }while(line[0]=='%');

  fmt= 0;
  nfields = sscanf(line, "%d %d %d", &(graph->nvtxs), &(graph->nedges), &fmt);

  if(nfields<2){
    hunyuangraph_error_exit("The input file does not specify the number of vertices and edges.\n");
  }

  if(graph->nvtxs<=0||graph->nedges<=0){
   hunyuangraph_error_exit("The supplied nvtxs:%d and nedges:%d must be positive.\n",graph->nvtxs,graph->nedges);
  }

  if(fmt>111){ 
    hunyuangraph_error_exit("Cannot read this type of file format [fmt=%d]!\n",fmt);
  }

  sprintf(fmtstr,"%03d",fmt%1000);
  readvs=(fmtstr[0]=='1');
  readvw=(fmtstr[1]=='1');
  readew=(fmtstr[2]=='1');

  graph->nedges *=2;

  xadj=graph->xadj=(int*)malloc(sizeof(int)*(graph->nvtxs+1));
  for(i=0;i<graph->nvtxs+1;i++){
    xadj[i]=graph->xadj[i]=0;
  }

  adjncy=graph->adjncy=(int*)malloc(sizeof(int)*(graph->nedges));

  vwgt=graph->vwgt= (int*)malloc(sizeof(int)*(graph->nvtxs));

  for(i=0;i<graph->nvtxs;i++){
    vwgt[i]=graph->vwgt[i]=1;
  }

  adjwgt = graph->adjwgt=(int*)malloc(sizeof(int)*(graph->nedges));
  for(i=0;i<graph->nedges;i++){
    adjwgt[i]=graph->adjwgt[i]=1;
  }

  for(xadj[0]=0,k=0,i=0;i<graph->nvtxs;i++){
    do{
      if(getline(&line,&lnlen,fpin)==-1){
      hunyuangraph_error_exit("Premature end of input file while reading vertex %d.\n", i+1);
      } 
    }while(line[0]=='%');

    curstr=line;
    newstr=NULL;

    if(readvw){
      vwgt[i]=strtol(curstr, &newstr, 10);

      if(newstr==curstr){
        hunyuangraph_error_exit("The line for vertex %d does not have enough weights "
          "for the %d constraints.\n", i+1, 1);
      }
      if(vwgt[i]<0){
        hunyuangraph_error_exit("The weight vertex %d and constraint %d must be >= 0\n", i+1, 0);
      }
      curstr = newstr;
    }

    while(1){
      edge=strtol(curstr,&newstr,10);
      if(newstr==curstr){
        break; 
      }

      curstr=newstr;
      if (edge< 1||edge>graph->nvtxs){
        hunyuangraph_error_exit("Edge %d for vertex %d is out of bounds\n",edge,i+1);
      }

      ewgt=1;

      if(readew){
        ewgt=strtol(curstr,&newstr,10);

        if(newstr==curstr){
          hunyuangraph_error_exit("Premature end of line for vertex %d\n", i+1);
        }

        if(ewgt<=0){
          hunyuangraph_error_exit("The weight (%d) for edge (%d, %d) must be positive.\n",    ewgt, i+1, edge);
        }

        curstr=newstr;
      }

      if(k==graph->nedges){
        hunyuangraph_error_exit("There are more edges in the file than the %d specified.\n", graph->nedges/2);
      }

      adjncy[k]=edge-1;
      adjwgt[k]=ewgt;
      k++;

    } 
    xadj[i+1]=k;

  }
  fclose(fpin);

  if(k!=graph->nedges){
    printf("------------------------------------------------------------------------------\n");
    printf("***  I detected an error in your input file  ***\n\n");
    printf("In the first line of the file, you specified that the graph contained\n"
      "%d edges. However, I only found %d edges in the file.\n", graph->nedges/2,k/2);
    if(2*k==graph->nedges){
      printf("\n *> I detected that you specified twice the number of edges that you have in\n");
      printf("    the file. Remember that the number of edges specified in the first line\n");
      printf("    counts each edge between vertices v and u only once.\n\n");
    }
    printf("Please specify the correct number of edges in the first line of the file.\n");
    printf("------------------------------------------------------------------------------\n");
    exit(0);
  }
  free(line);
  return graph;
}


/*Main function*/
int main(int argc, char **argv)
{

    char *filename = (argv[1]);

    hunyuangraph_graph_t *graph = hunyuangraph_readgraph(filename);

    printf("0\n");   // 第一行
    printf("%d\t%d\n",graph->nvtxs,graph->nedges);
    printf("0\t000\n");   // 第三行
    for(int i = 0;i < graph->nvtxs;i++)
    {
        printf("%d\t",graph->xadj[i + 1] - graph->xadj[i]);
        for(int j = graph->xadj[i];j < graph->xadj[i + 1];j++)
            printf("%d\t",graph->adjncy[j]);
        printf("\n");
    }

    return 0;
}