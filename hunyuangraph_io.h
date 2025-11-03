#ifndef _H_IO
#define _H_IO

#include <stdio.h>
#include "hunyuangraph_struct.h"
#include "hunyuangraph_graph.h"

/*Open file*/
FILE *hunyuangraph_fopen(char *fname, char *mode, const char *msg)
{
	FILE *fp;
	char error_message[8192];
	fp = fopen(fname, mode);
	if (fp != NULL)
	{
		return fp;
	}
	sprintf(error_message, "file: %s, mode: %s, [%s]", fname, mode, msg);
	perror(error_message);
	hunyuangraph_error_exit("Failed on file fopen()\n");
	return NULL;
}

/*Read graph file*/
hunyuangraph_graph_t *hunyuangraph_readgraph(char *filename)
{
	int i, k, fmt, nfields, readew, readvw, readvs, edge, ewgt;
	int *xadj, *adjncy, *vwgt, *adjwgt;
	char *line = NULL, fmtstr[256], *curstr, *newstr;
	size_t lnlen = 0;
	FILE *fpin;

	hunyuangraph_graph_t *graph;
	graph = hunyuangraph_create_cpu_graph();

	fpin = hunyuangraph_fopen(filename, "r", "Readgraph: Graph");

	do
	{
		if (getline(&line, &lnlen, fpin) == -1)
		{
			hunyuangraph_error_exit("Premature end of input file: file: %s\n", filename);
		}
	} while (line[0] == '%');

	fmt = 0;
	nfields = sscanf(line, "%d %d %d", &(graph->nvtxs), &(graph->nedges), &fmt);

	if (nfields < 2)
	{
		hunyuangraph_error_exit("The input file does not specify the number of vertices and edges.\n");
	}

	if (graph->nvtxs <= 0 || graph->nedges <= 0)
	{
		hunyuangraph_error_exit("The supplied nvtxs:%d and nedges:%d must be positive.\n", graph->nvtxs, graph->nedges);
	}

	if (fmt > 111)
	{
		hunyuangraph_error_exit("Cannot read this type of file format [fmt=%d]!\n", fmt);
	}

	sprintf(fmtstr, "%03d", fmt % 1000);
	readvs = (fmtstr[0] == '1');
	readvw = (fmtstr[1] == '1');
	readew = (fmtstr[2] == '1');

	graph->nedges *= 2;

	xadj = graph->xadj = (int *)malloc(sizeof(int) * (graph->nvtxs + 1));
	for (i = 0; i < graph->nvtxs + 1; i++)
	{
		xadj[i] = graph->xadj[i] = 0;
	}

	adjncy = graph->adjncy = (int *)malloc(sizeof(int) * (graph->nedges));

	vwgt = graph->vwgt = (int *)malloc(sizeof(int) * (graph->nvtxs));

	for (i = 0; i < graph->nvtxs; i++)
	{
		vwgt[i] = graph->vwgt[i] = 1;
	}

	adjwgt = graph->adjwgt = (int *)malloc(sizeof(int) * (graph->nedges));
	for (i = 0; i < graph->nedges; i++)
	{
		adjwgt[i] = graph->adjwgt[i] = 1;
	}

	for (xadj[0] = 0, k = 0, i = 0; i < graph->nvtxs; i++)
	{
		do
		{
			if (getline(&line, &lnlen, fpin) == -1)
			{
				hunyuangraph_error_exit("Premature end of input file while reading vertex %d.\n", i + 1);
			}
		} while (line[0] == '%');

		curstr = line;
		newstr = NULL;

		if (readvw)
		{
			vwgt[i] = strtol(curstr, &newstr, 10);

			if (newstr == curstr)
			{
				hunyuangraph_error_exit("The line for vertex %d does not have enough weights "
										"for the %d constraints.\n",
										i + 1, 1);
			}
			if (vwgt[i] < 0)
			{
				hunyuangraph_error_exit("The weight vertex %d and constraint %d must be >= 0\n", i + 1, 0);
			}
			curstr = newstr;
		}

		while (1)
		{
			edge = strtol(curstr, &newstr, 10);
			if (newstr == curstr)
			{
				break;
			}

			curstr = newstr;
			if (edge < 1 || edge > graph->nvtxs)
			{
				hunyuangraph_error_exit("Edge %d for vertex %d is out of bounds\n", edge, i + 1);
			}

			ewgt = 1;

			if (readew)
			{
				ewgt = strtol(curstr, &newstr, 10);

				if (newstr == curstr)
				{
					hunyuangraph_error_exit("Premature end of line for vertex %d\n", i + 1);
				}

				if (ewgt <= 0)
				{
					hunyuangraph_error_exit("The weight (%d) for edge (%d, %d) must be positive.\n", ewgt, i + 1, edge);
				}

				curstr = newstr;
			}

			if (k == graph->nedges)
			{
				hunyuangraph_error_exit("There are more edges in the file than the %d specified.\n", graph->nedges / 2);
			}

			adjncy[k] = edge - 1;
			adjwgt[k] = ewgt;
			k++;
		}
		xadj[i + 1] = k;
	}
	fclose(fpin);

	if (k != graph->nedges)
	{
		printf("------------------------------------------------------------------------------\n");
		printf("***  I detected an error in your input file  ***\n\n");
		printf("In the first line of the file, you specified that the graph contained\n"
			   "%d edges. However, I only found %d edges in the file.\n",
			   graph->nedges / 2, k / 2);
		if (2 * k == graph->nedges)
		{
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

/*Write to file*/
void hunyuangraph_writetofile(char *fname, int *part, int n, int nparts)
{
	FILE *fpout;
	int i;
	char filename[1280000];
	sprintf(filename, "%s.part.%d", fname, nparts);

	fpout = hunyuangraph_fopen(filename, "w", __func__);

	for (i = 0; i < n; i++)
	{
		fprintf(fpout, "%d\n", part[i]);
	}

	fclose(fpout);
}

#endif