#ifndef TIMER_H
#define TIMER_H

#include <stdio.h>
#include <sys/time.h>

#include "typedef.h"
#include "control.h"

Hunyuan_real_t time_all = 0;
struct timeval start_all;
struct timeval end_all;

Hunyuan_real_t time_partitiongraph = 0;
struct timeval start_partitiongraph;
struct timeval end_partitiongraph;

Hunyuan_real_t time_bisectionbest = 0;
struct timeval start_bisectionbest;
struct timeval end_bisectionbest;

Hunyuan_real_t time_coarsen = 0;
struct timeval start_coarsen;
struct timeval end_coarsen;

Hunyuan_real_t time_initialpartition = 0;
struct timeval start_initialpartition;
struct timeval end_initialpartition;

Hunyuan_real_t time_refine2waynode = 0;
struct timeval start_refine2waynode;
struct timeval end_refine2waynode;

Hunyuan_real_t time_splitgraphreorder = 0;
struct timeval start_splitgraphreorder;
struct timeval end_splitgraphreorder;

Hunyuan_real_t time_mmdorder = 0;
struct timeval start_mmdorder;
struct timeval end_mmdorder;

Hunyuan_real_t time_match = 0;
struct timeval start_match;
struct timeval end_match;

Hunyuan_real_t time_createcoarsengraph = 0;
struct timeval start_createcoarsengraph;
struct timeval end_createcoarsengraph;

Hunyuan_real_t time_BFS = 0;
struct timeval start_BFS;
struct timeval end_BFS;

Hunyuan_real_t time_partitioninf2way = 0;
struct timeval start_partitioninf2way;
struct timeval end_partitioninf2way;

Hunyuan_real_t time_fm2waycutbalance = 0;
struct timeval start_fm2waycutbalance;
struct timeval end_fm2waycutbalance;

Hunyuan_real_t time_fm2waycutrefine = 0;
struct timeval start_fm2waycutrefine;
struct timeval end_fm2waycutrefine;

Hunyuan_real_t time_reorderinf2way = 0;
struct timeval start_reorderinf2way;
struct timeval end_reorderinf2way;

Hunyuan_real_t time_fmnodebalance = 0;
struct timeval start_fmnodebalance;
struct timeval end_fmnodebalance;

Hunyuan_real_t time_fm1sidenoderefine = 0;
struct timeval start_fm1sidenoderefine;
struct timeval end_fm1sidenoderefine;

Hunyuan_real_t time_fm2sidenoderefine = 0;
struct timeval start_fm2sidenoderefine;
struct timeval end_fm2sidenoderefine;

Hunyuan_real_t time_malloc = 0;
struct timeval start_malloc;
struct timeval end_malloc;

Hunyuan_real_t time_free = 0;
struct timeval start_free;
struct timeval end_free;

// Hunyuan_real_t time_

void gettimebegin(struct timeval *start, struct timeval *end, Hunyuan_real_t *time)
{
	gettimeofday(start,NULL);
}

void gettimeend(struct timeval *start, struct timeval *end, Hunyuan_real_t *time)
{
	gettimeofday(end,NULL);
	time[0] += (end[0].tv_sec - start[0].tv_sec) * 1000 + (end[0].tv_usec - start[0].tv_usec) / 1000.0;
}

void PrintTimeGeneral()
{
	printf("\nTiming Information -------------------------------------------------\n");
	printf("All:                 %10.3"PRREAL" ms\n", time_all);
	printf("-------------------------------------------------------------------\n");
}

void PrintTimePhases()
{
	printf("\nTiming Information -------------------------------------------------\n");
	printf("All:                 %10.3"PRREAL" ms\n", time_all);
	printf("    Multi-level Partition:      %10.3"PRREAL" ms\n", time_partitiongraph);
	printf("        Bisection-Best:          %10.3"PRREAL" ms\n", time_bisectionbest);
	printf("            Coarsen:                   %10.3"PRREAL" ms\n", time_coarsen);
	printf("            Reorder Bisection:         %10.3"PRREAL" ms\n", time_initialpartition);
	printf("            Refine 2way-Node:          %10.3"PRREAL" ms\n", time_refine2waynode);
	printf("        Reorder Split Graph:     %10.3"PRREAL" ms\n", time_splitgraphreorder);
	printf("        MMD Order Small Graph:   %10.3"PRREAL" ms\n", time_mmdorder);
	printf("-------------------------------------------------------------------\n");

}

void PrintTimeSteps()
{
	printf("\nTiming Information -------------------------------------------------\n");
	printf("All:                 %10.3"PRREAL" ms\n", time_all);
	// printf("    Multi-level Partition:      %10.3"PRREAL" ms\n", time_partitiongraph);
	// printf("        Bisection-Best:          %10.3"PRREAL" ms\n", time_bisectionbest);
	// printf("            Coarsen:                   %10.3"PRREAL" ms\n", time_coarsen);
	// printf("                Matching:                    %10.3"PRREAL" ms\n", time_match);
	// printf("                Create Coarsen Graph:        %10.3"PRREAL" ms\n", time_createcoarsengraph);
	printf("            Initial Partition:         %10.3"PRREAL" ms\n", time_initialpartition);
	printf("                BFS:                         %10.3"PRREAL" ms\n", time_BFS);
	printf("                Compute Partition Inf 2way:  %10.3"PRREAL" ms\n", time_partitioninf2way);
	printf("                FM 2way-Cut Balance:         %10.3"PRREAL" ms\n", time_fm2waycutbalance);
	printf("                FM 2way-Cut Refine:          %10.3"PRREAL" ms\n", time_fm2waycutrefine);
	// printf("            Refine 2way-Node:          %10.3"PRREAL" ms\n", time_refine2waynode);
	// printf("                Compute Reorder Inf 2way:    %10.3"PRREAL" ms\n", time_reorderinf2way);
	// printf("                FM 2way-Node Balance:        %10.3"PRREAL" ms\n", time_fmnodebalance);
	// printf("                FM 1Side-Node Refine:        %10.3"PRREAL" ms\n", time_fm1sidenoderefine);
	// printf("                FM 2Side-Node Refine:        %10.3"PRREAL" ms\n", time_fm2sidenoderefine);
	// printf("        Reorder Split Graph:     %10.3"PRREAL" ms\n", time_splitgraphreorder);
	// printf("        MMD Order Small Graph:   %10.3"PRREAL" ms\n", time_mmdorder);
	printf("-------------------------------------------------------------------\n");

}

void PrintTime(Hunyuan_int_t control)
{
	//	001
	if(control & PRINTTIMESTEPS) 
		PrintTimeSteps();
	//	010
	else if(control & PRINTTIMEPHASES) 
		PrintTimePhases();
	//	100
	else if(control & PRINTTIMEGENERAL) 
		PrintTimeGeneral();
}

#endif