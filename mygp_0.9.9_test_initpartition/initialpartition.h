#ifndef INITIALPARTITION_H
#define INITIALPARTITION_H

// #include <math.h>
#include "struct.h"
#include "memory.h"
#include "queue.h"
#include "balance.h"
#include "splitgraph.h"
#include "refine.h"
#include <math.h>

void Init2WayPartition(graph_t *graph, Hunyuan_real_t *ntpwgts, Hunyuan_real_t *balance_factor, Hunyuan_int_t niparts)
{
    Hunyuan_int_t i, j, k, nvtxs, drain, nleft, first, last, pwgts[2], \
        oneminpwgt, onemaxpwgt, from, me, bestcut=0, icut, mincut, inbfs;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndind;
    Hunyuan_int_t *queue, *touched, *gain, *bestwhere;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;

    onemaxpwgt = balance_factor[0] * graph->tvwgt[0] * ntpwgts[0];
    oneminpwgt = (1.0 / balance_factor[0]) * graph->tvwgt[0] * ntpwgts[0];
    // printf("ctrl->ubfactors[0]=%.10lf ntpwgts[1]=%.10lf\n",ctrl->ubfactors[0],ntpwgts[1]);

    /* Allocate refinement memory. Allocate sufficient memory for both edge and node */
    graph->where  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->where");
    graph->pwgts  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: graph->pwgts");
    graph->bndptr = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->bndptr");
    graph->bndind = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->bndind");
    graph->id     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->id");
    graph->ed     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->ed");
    // graph->nrinfo = (nrinfo_t *)check_malloc(sizeof(nrinfo_t) * nvtxs, "Init2WayPartition: graph->nrinfo");

    bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: bestwhere");
    queue     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: queue");
    touched   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: touched");
  
    bndind = graph->bndind;
    where = graph->where;

    for (inbfs=0; inbfs<niparts; inbfs++) 
    {
        set_value_int(nvtxs, 1, where);
        set_value_int(nvtxs, 0, touched);

        pwgts[1] = graph->tvwgt[0];
        pwgts[0] = 0;

        queue[0] = irandInRange(nvtxs);
        // printf("inbfs=%"PRIDX" queue[0]=%"PRIDX"\n",inbfs,queue[0]);
        touched[queue[0]] = 1;
        first = 0;
        last = 1;
        nleft = nvtxs - 1;
        drain = 0;

        /* Start the BFS from queue to get a partition */
        for (;;) 
        {
            if (first == last) 
            { /* Empty. Disconnected graph! */
                if (nleft == 0 || drain)
                    break;

                k = irandInRange(nleft);
                // printf("inbfs=%"PRIDX" k=%"PRIDX"\n",inbfs,k);
                for (i=0; i<nvtxs; i++) 
                {
                    if (touched[i] == 0) 
                    {
                        if (k == 0)
                            break;
                        else
                            k--;
                    }
                }

                queue[0]   = i;
                touched[i] = 1;
                first      = 0; 
                last       = 1;
                nleft--;
            }

            i = queue[first++];
            if (pwgts[0] > 0 && pwgts[1]-vwgt[i] < oneminpwgt) 
            {
                drain = 1;
                continue;
            }

            where[i] = 0;
            pwgts[0] += vwgt[i];
            pwgts[1] -= vwgt[i];
            if (pwgts[1] <= onemaxpwgt)
                break;

            drain = 0;
            for (j = xadj[i]; j < xadj[i + 1]; j++) 
            {
                k = adjncy[j];
                if (touched[k] == 0) 
                {
                    queue[last++] = k;
                    touched[k] = 1;
                    nleft--;
                }
            }
        }

        // for (i = 0; i < nvtxs; i++)
        //     printf("%"PRIDX" ", graph->where[i]);
        // printf("\n");
        printf("%"PRIDX" %"PRIDX" \n", pwgts[0], pwgts[1]);

        /* Check to see if we hit any bad limiting cases */
        if (pwgts[1] == 0) 
            where[irandInRange(nvtxs)] = 1;
        if (pwgts[0] == 0) 
            where[irandInRange(nvtxs)] = 0;

        // printf("grow end rand_count=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        /*************************************************************
        * Do some partition refinement 
        **************************************************************/
        CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimebegin(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
        Compute_Partition_Informetion_2way(graph);
        CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimeend(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
        // printf("%"PRIDX" %"PRIDX" \n", pwgts[0], pwgts[1]);
        // printf("ReorderBisection 2\n");
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, gettimebegin(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
        Balance2Way_partition(graph, ntpwgts, balance_factor);
        CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, gettimeend(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
        // printf("balance end rand_count=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        // printf("ReorderBisection 3\n");
        // printf("ReorderBisection 3 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimebegin(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));
        FM_2WayCutRefine(graph, ntpwgts, 10);
        CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimeend(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));
        // printf("FM_2WayCutRefine end rand_count=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        // printf("ReorderBisection 4\n");
        // printf("ReorderBisection 4 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        if (inbfs == 0 || bestcut > graph->mincut) 
        {
            bestcut = graph->mincut;
            copy_int(nvtxs, where, bestwhere);
            
            if (bestcut == 0)
                break;
        }
    }

    graph->mincut = bestcut;
    copy_int(nvtxs, bestwhere, where);

    check_free(touched, sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: touched");
    check_free(queue, sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: queue");
    check_free(bestwhere, sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: bestwhere");
}

void MultiLevelBisection(graph_t *graph, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor)
{
    graph_t *cgraph;
    Hunyuan_int_t Coarsen_Threshold = 20;
    Hunyuan_int_t *bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "MultiLevelBisection: bestwhere");
    Hunyuan_real_t bestbal = 0.0, curbal = 0.0;
    Hunyuan_int_t bestobj = 0, curobj = 0;

    for(Hunyuan_int_t i = 0;i < 5;i++)
    {
        // printf("rand_count=%"PRIDX"\n",rand_count());
        /*if(graph->nvtxs == 8186)
        {
            printf("rand_count=%"PRIDX"\n",rand_count());
            for(Hunyuan_int_t j = 0;j < graph->nvtxs * 2;j++)
            {
                Hunyuan_int_t k = irand();
                printf("j=%"PRIDX" irand=%"PRIDX"\n",j,k);
                if(k == 4978487952656886808)
                    break;
            }
        }*/

        CONTROL_COMMAND(control, COARSEN_Time, gettimebegin(&start_coarsen, &end_coarsen, &time_coarsen));
        cgraph = CoarsenGraph(graph, Coarsen_Threshold);
        CONTROL_COMMAND(control, COARSEN_Time, gettimeend(&start_coarsen, &end_coarsen, &time_coarsen));

        // printf("coarsen end rand_count=%"PRIDX"\n",rand_count());
        // exam_nvtxs_nedges(cgraph);
        // exam_xadj(cgraph);
        // exam_vwgt(cgraph);
        // exam_adjncy_adjwgt(cgraph);

        Hunyuan_int_t niparts = (cgraph->nvtxs <= Coarsen_Threshold ? 5 : 7);
        Init2WayPartition(cgraph, tpwgts, balance_factor, niparts);
        // printf("Init end rand_count=%"PRIDX"\n",rand_count());
        // exam_where(cgraph);
        // if(exam_correct_gp(cgraph->where, cgraph->nvtxs, 2))
        // {
        //     printf("The Answer is Right\n");
        //     printf("edgecut=%"PRIDX" \n", compute_edgecut(cgraph->where, cgraph->nvtxs, cgraph->xadj, cgraph->adjncy, cgraph->adjwgt));
        // }
        // else 
        //     printf("The Answer is Error\n");

        Refine2WayPartition(graph, cgraph, tpwgts, balance_factor);
        // printf("2refine end rand_count=%"PRIDX"\n",rand_count());

        curobj = graph->mincut;
        curbal = ComputeLoadImbalanceDiff(graph, 2, tpwgts,  balance_factor[0]);

        // printf("i=%"PRIDX" edgecut=%"PRIDX" \n",i, compute_edgecut(graph->where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt));

        if (i == 0  
            || (curbal <= 0.0005 && bestobj > curobj) 
            || (bestbal > 0.0005 && curbal < bestbal)) 
        {
            bestobj = curobj;
            bestbal = curbal;
            if (i < 4)
                copy_int(graph->nvtxs, graph->where, bestwhere);
        }

        if (bestobj == 0)
            break;

        if (i < 4)
            FreeRefineData(graph, 2);
    }

    if (bestobj != curobj) 
    {
        copy_int(graph->nvtxs, bestwhere, graph->where);
        Compute_Partition_Informetion_2way(graph);
    }
    
    check_free(bestwhere, sizeof(Hunyuan_int_t) * graph->nvtxs, "MultiLevelBisection: bestwhere");
}

void MultiLevelRecursivePartition(graph_t *graph, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor, Hunyuan_int_t *part, Hunyuan_int_t fpart, Hunyuan_int_t level)
{
    // printf("level=%"PRIDX" nvtxs=%"PRIDX" \n",level, graph->nvtxs);
    // exam_tpwgts(tpwgts, nparts);
    Hunyuan_int_t i, j, nvtxs, ncon;
    Hunyuan_int_t *label, *where;
    graph_t *lgraph, *rgraph;
    Hunyuan_real_t wsum, *tpwgts2;

    if ((nvtxs = graph->nvtxs) == 0) 
	{
        printf("\t***Cannot bisect a graph with 0 vertices!\n"
            "\t***You are trying to partition a graph into too many parts!\n");
        return ;
    }

    /* determine the weights of the two partitions as a function of the weight of the
        target partition weights */
    tpwgts2 = (Hunyuan_real_t *)check_malloc(sizeof(Hunyuan_real_t) * 2, "MultiLevelRecursivePartition: tpwgts2");
    for (i = 0; i < 1; i++) 
	{
        tpwgts2[i]      = sum_real((nparts >> 1), tpwgts, 1);
        tpwgts2[i + 1] = 1.0 - tpwgts2[i];
    }

    /* perform the bisection */
    MultiLevelBisection(graph, tpwgts2, balance_factor);

    check_free(tpwgts2, sizeof(Hunyuan_real_t) * 2, "MultiLevelRecursivePartition: tpwgts2");

    label = graph->label;
    where = graph->where;
    // for (i = 0; i < nvtxs; i++)
    //     printf("%"PRIDX" ",where[i]);
    // printf("\n");
    for (i = 0; i < nvtxs; i++)
        part[label[i]] = where[i] + fpart;

    if (nparts > 2) 
        SplitGraphPartition(graph, &lgraph, &rgraph);

    /* Free the memory of the top level graph */
    FreeGraph(&graph, 2);

    /* Scale the fractions in the tpwgts according to the true weight */
    for (i = 0; i < 1; i++) 
	{
        wsum = sum_real((nparts >> 1), tpwgts, 1);
        rscale_real((nparts >> 1), 1.0 / wsum, tpwgts);
        rscale_real(nparts - (nparts >> 1), 1.0 / (1.0 - wsum), tpwgts + (nparts >> 1));
    }

    /* Do the recursive call */
    if (nparts > 3) {
        MultiLevelRecursivePartition(lgraph, (nparts>>1), tpwgts, balance_factor, part, fpart, level * 2 + 1);
        MultiLevelRecursivePartition(rgraph, nparts - (nparts>>1), tpwgts + (nparts>>1), balance_factor, part, fpart + (nparts>>1), level * 2 + 2);
    }
    else if (nparts == 3) {
        FreeGraph(&lgraph, 0);
        MultiLevelRecursivePartition(rgraph, nparts - (nparts>>1), tpwgts + (nparts>>1), balance_factor, part, fpart + (nparts>>1), level * 2 + 2);
    }
}

void InitialPartition(graph_t *graph, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts_ori, Hunyuan_real_t *balance_factor)
{
    Hunyuan_int_t *vwgt, *xadj, *adjncy, *adjwgt;
    Hunyuan_real_t *tpwgts;

    vwgt   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "InitialPartition: vwgt");
    xadj   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (graph->nvtxs + 1), "InitialPartition: xadj");
    adjncy = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "InitialPartition: adjncy");
    adjwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "InitialPartition: adjwgt");
    tpwgts = (Hunyuan_real_t *)check_malloc(sizeof(Hunyuan_real_t) * nparts, "InitialPartition: tpwgts");

    memcpy(xadj, graph->xadj, sizeof(Hunyuan_int_t) * (graph->nvtxs + 1));
    memcpy(vwgt, graph->vwgt, sizeof(Hunyuan_int_t) * graph->nvtxs);
    memcpy(adjncy, graph->adjncy, sizeof(Hunyuan_int_t) * graph->nedges);
    memcpy(adjwgt, graph->adjwgt, sizeof(Hunyuan_int_t) * graph->nedges);
    memcpy(tpwgts, tpwgts_ori, sizeof(Hunyuan_real_t) * nparts);

    InitRandom(-1);

    Hunyuan_real_t ubvec = (Hunyuan_real_t)pow(balance_factor[0], 1.0 / log(nparts));

	graph_t *t_graph = SetupGraph(graph->nvtxs, xadj, adjncy, vwgt, adjwgt);
    Hunyuan_int_t *part = graph->where = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "InitialPartition: graph->where");
    // exam_tpwgts(tpwgts, nparts);
    MultiLevelRecursivePartition(t_graph, nparts, tpwgts, &ubvec, part, 0, 0);
    // exam_tpwgts(tpwgts, nparts);

    check_free(tpwgts, sizeof(Hunyuan_real_t) * nparts, "InitialPartition: tpwgts");
}

void InitialPartition_multi(graph_t *graph, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts_ori, Hunyuan_real_t *balance_factor)
{
    Hunyuan_int_t *vwgt, *xadj, *adjncy, *adjwgt;
    Hunyuan_real_t *tpwgts;

    xadj = graph->xadj;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    vwgt = graph->vwgt;
    tpwgts = tpwgts_ori;

    InitRandom(-1);

    Hunyuan_real_t ubvec = (Hunyuan_real_t)pow(balance_factor[0], 1.0 / log(nparts));

    Hunyuan_int_t *part = graph->where = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "InitialPartition_multi: graph->where");
    Hunyuan_int_t *bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "InitialPartition_multi: bestwhere");
    graph->pwgts  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "InitialPartition_multi: graph->pwgts");

    Hunyuan_int_t nvtxs;
    Hunyuan_int_t *pwgts, *where;
    nvtxs = graph->nvtxs;
    pwgts = graph->pwgts;
    where = graph->where;

    Hunyuan_int_t *onemaxpwgt, *oneminpwgt, *oneidealpwgt;
    onemaxpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "exam_is_balance: onemaxpwgt");
	oneminpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "exam_is_balance: oneminpwgt");
    oneidealpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "exam_is_balance: oneidealpwgt");

    for(Hunyuan_int_t i = 0;i < nparts;i++)
	{
		onemaxpwgt[i] = ceilUp((Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i] * ubvec);
		oneminpwgt[i] = floorDown((Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i] / ubvec);
        oneidealpwgt[i] = ceilUp((Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i]);
        printf("onemaxpwgt[%"PRIDX"]=%"PRIDX" oneminpwgt[%"PRIDX"]=%"PRIDX" oneidealpwgt[%"PRIDX"]=%"PRIDX" %"PRREAL" %"PRREAL"\n", i, onemaxpwgt[i], i, oneminpwgt[i], i, oneidealpwgt[i],\
            (Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i] * ubvec, (Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i] / ubvec);
	}

    Hunyuan_int_t i, j, k, p, edgecut, best_edgecut;
    best_edgecut = graph->nedges;

    // random
    Hunyuan_int_t imbalance_number = 0;
    for(Hunyuan_int_t iter = 0;iter < 4000;iter++)
    {
        memset(where, -1, sizeof(Hunyuan_int_t) * graph->nvtxs);
        memset(pwgts, 0, sizeof(Hunyuan_int_t) * nparts);

        for(i = 0;i < nvtxs;i++)
        {
            if(where[i] != -1)
                continue;
            
            p = irandInRange(nparts);
            while(pwgts[p] + vwgt[i] > oneidealpwgt[p])
            {
                p++;
                p = p % nparts;

                // p = irandInRange(nparts);
                // printf("iter=%"PRIDX" i=%"PRIDX" p=%"PRIDX" \n",iter, i, p);
            }
            // printf("iter=%"PRIDX" i=%"PRIDX"\n",iter, i);
            where[i] = p;
            pwgts[p] += vwgt[i];
            for(j = xadj[i];j < xadj[i + 1];j++)
            {
                // printf("iter=%"PRIDX" i=%"PRIDX" j=%"PRIDX"\n",iter, i, j);
                k = adjncy[j];
                if(where[k] != -1)
                    continue;
                
                if(pwgts[p] + vwgt[k] <= onemaxpwgt[p])
                {
                    where[k] = p;
                    pwgts[p] += vwgt[k];
                }

                if(pwgts[p] >= oneminpwgt[p] && pwgts[p] <= onemaxpwgt[p])
                    break;
            }
        }

        if(exam_is_balance(pwgts, graph->tvwgt, nparts, tpwgts, balance_factor))
        {   

        }
        else
        {
            imbalance_number ++;
            // printf("The where is not balance\n");
            // for(i = 0;i < nparts;i++)
            //     printf("pwgts[%"PRIDX"]=%"PRIDX" ", i, pwgts[i]);
            // printf("\n");
            // continue;
        }

        if(exam_correct_gp_where(where, graph->nvtxs, nparts))
        {
            // printf("The where is Right\n");
        }
        else 
        {
            printf("The where is Error\n");
            continue;
        }
        
        edgecut = compute_edgecut(where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt);

        if(iter == 0 || best_edgecut > edgecut)
        {
            best_edgecut = edgecut;
            memcpy(bestwhere, where, sizeof(Hunyuan_int_t) * graph->nvtxs);
        }
        printf("iter=%"PRIDX" edgecut=%"PRIDX" best_edgecut=%"PRIDX"\n", iter, edgecut, best_edgecut);
    }

    memcpy(part, bestwhere, sizeof(Hunyuan_int_t) * graph->nvtxs);

    printf("imbalance_number=%"PRIDX" \n",imbalance_number);

    memset(pwgts, 0, sizeof(Hunyuan_int_t) * nparts);
    for(i = 0;i < nvtxs;i++)
    {
        pwgts[where[i]] += vwgt[i];
    }
    if(exam_is_balance(pwgts, graph->tvwgt, nparts, tpwgts, balance_factor))
    {   
        printf("The where is balance\n");
    }
    else
    {
        printf("The where is not balance\n");
        for(i = 0;i < nparts;i++)
            printf("pwgts[%"PRIDX"]=%"PRIDX" ", i, pwgts[i]);
        printf("\n");
    }

    if(exam_correct_gp(where, graph->nvtxs, nparts))
	{
		printf("The Answer is Right\n");
		printf("edgecut=%"PRIDX" \n", compute_edgecut(where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt));
	}
	else 
		printf("The Answer is Error\n");

    check_free(bestwhere, sizeof(Hunyuan_int_t) * graph->nvtxs, "InitialPartition_multi: bestwhere");
    check_free(pwgts, sizeof(Hunyuan_int_t) * nparts, "InitialPartition_multi: pwgts");
    check_free(onemaxpwgt, nparts * sizeof(Hunyuan_int_t), "exam_is_balance: onemaxpwgt");
	check_free(oneminpwgt, nparts * sizeof(Hunyuan_int_t), "exam_is_balance: oneminpwgt");
}

Hunyuan_int_t select_queue_begin(Hunyuan_int_t nvtxs, Hunyuan_int_t undistributed, Hunyuan_int_t *where)
{
    if(undistributed == 0)
        return -1;

    Hunyuan_int_t select = irandInRange(nvtxs);

    while (where[select] != -1)
    {
        select++;
        select = select % nvtxs;

        // select = select_queue_begin(nvtxs, undistributed, where);
    }

    return select;
}

void IterationBisection(graph_t *graph, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor)
{
    Hunyuan_int_t i, j, k, v, undistributed, queue_first, queue_begin, queue_end;
    Hunyuan_int_t nvtxs, best_edgecut, edgecut, imbalance_number;
    Hunyuan_int_t *xadj, *adjncy, *adjwgt, *vwgt, *where, *pwgts, *bestwhere, *moved, *onemaxpwgt, *oneminpwgt, *oneidealpwgt;
    Hunyuan_int_t *queue;
    // priority_queue_t *queue;

    nvtxs = graph->nvtxs;
    best_edgecut = graph->nedges;
    best_edgecut = 0;
    imbalance_number = 0;

    xadj = graph->xadj;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    vwgt = graph->vwgt;
    where = graph->where  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->where");
    pwgts = graph->pwgts  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: graph->pwgts");
    bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: bestwhere");
    queue = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: queue");
    moved = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: moved");
    onemaxpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: onemaxpwgt");
    oneminpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: oneminpwgt");
    oneidealpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: oneidealpwgt");

    for(i = 0;i < 2;i++)
    {
        onemaxpwgt[i] = ceilUp((Hunyuan_real_t)balance_factor[0] * graph->tvwgt[0] * tpwgts[i]);
        oneminpwgt[i] = floorDown((Hunyuan_real_t)(1.0 / balance_factor[0]) * graph->tvwgt[0] * tpwgts[i]);
        oneidealpwgt[i] = ceilUp((Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i]);

        printf("onemaxpwgt[%"PRIDX"]=%"PRIDX" oneminpwgt[%"PRIDX"]=%"PRIDX" oneidealpwgt[%"PRIDX"]=%"PRIDX" \n", \
            i, onemaxpwgt[i], i, oneminpwgt[i], i, oneidealpwgt[i]);
    }

    // queue = priority_queue_Create(nvtxs);

    for(Hunyuan_int_t iter = 0;iter < nvtxs;iter++)
    {
        memset(where, -1, sizeof(Hunyuan_int_t) * nvtxs);
        memset(moved, 0, sizeof(Hunyuan_int_t) * nvtxs);
        memset(pwgts, 0, sizeof(Hunyuan_int_t) * 2);
        undistributed = nvtxs;

        queue_first = select_queue_begin(nvtxs, undistributed, where);
        queue_first = iter;
        queue_begin = 0;
        queue_end = 1;
        queue[queue_begin] = queue_first;
        moved[queue_first]++;
        // priority_queue_Insert(queue, queue_first, moved[queue_first]);
        undistributed--;

        Hunyuan_int_t step = 0;
        while(pwgts[0] < oneidealpwgt[0])
        {
            v = queue[queue_begin];
            // v = priority_queue_GetTop(queue);
            moved[v] = -1;
            queue_begin++;

            if(pwgts[0] + vwgt[v] > onemaxpwgt[0])
                continue;
            
            where[v] = 0;
            pwgts[0] += vwgt[v];

            // printf("v=%"PRIDX" where[%"PRIDX"]=%"PRIDX" vwgt[%"PRIDX"]=%"PRIDX" pwgts[0]=%"PRIDX"\n", v, v, where[v], v, vwgt[v], pwgts[0]);

            for(j = xadj[v];j < xadj[v + 1];j++)
            {
                k = adjncy[j];

                if(where[k] == -1 && moved[k] == 0)
                {
                    if(moved[k] == 0)
                    {
                        moved[k]++;
                        // priority_queue_Insert(queue, k, moved[k]);
                        queue[queue_end] = k;
                        queue_end++;
                        undistributed--;
                    }
                    else if(moved[k] != -1)
                    {
                        moved[k]++;
                        // priority_queue_Update(queue, k, moved[k]);
                    }
                }
            }

            if(queue_begin == queue_end)
            {
                queue_first = select_queue_begin(nvtxs, undistributed, where);
                queue_begin = 0;
                queue_end = 1;
                queue[queue_begin] = queue_first;
                moved[queue_first]++;
                // priority_queue_Insert(queue, queue_first, moved[k]);
                undistributed--;
            }
        }

        Hunyuan_int_t sum1 = 0;
        for(i = 0;i < nvtxs;i++)
            if(where[i] == -1)
            {
                where[i] = 1;
                sum1 += vwgt[i];
            }
        
        pwgts[1] = sum1;

        edgecut = compute_edgecut(where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt);

        if(exam_is_balance(pwgts, graph->tvwgt, 2, tpwgts, balance_factor))
        {   

        }
        else
        {
            printf("iter %"PRIDX": The where is not balance\n", iter);
            for(i = 0;i < 2;i++)
                printf("pwgts[%"PRIDX"]=%"PRIDX" ", i, pwgts[i]);
            printf("\n");
            imbalance_number ++;
        }

        if(iter == 0 || best_edgecut > edgecut)
        {
            best_edgecut = edgecut;
            memcpy(bestwhere, where, sizeof(Hunyuan_int_t) * graph->nvtxs);
        }
        printf("iter=%"PRIDX" edgecut=%"PRIDX" best_edgecut=%"PRIDX"\n", iter, edgecut, best_edgecut);

        // priority_queue_Reset(queue);
    } 

    memcpy(where, bestwhere, sizeof(Hunyuan_int_t) * graph->nvtxs);

    printf("imbalance_number=%"PRIDX" \n",imbalance_number);

    memset(pwgts, 0, sizeof(Hunyuan_int_t) * 2);
    for(i = 0;i < nvtxs;i++)
    {
        pwgts[where[i]] += vwgt[i];
    }
    if(exam_is_balance(pwgts, graph->tvwgt, 2, tpwgts, balance_factor))
    {   
        printf("The where is balance\n");
    }
    else
    {
        printf("The where is not balance\n");
        for(i = 0;i < 2;i++)
            printf("pwgts[%"PRIDX"]=%"PRIDX" ", i, pwgts[i]);
        printf("\n");
    }

    if(exam_correct_gp(where, graph->nvtxs, 2))
	{
		printf("The Answer is Right\n");
		printf("edgecut=%"PRIDX" \n", compute_edgecut(where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt));
	}
	else 
		printf("The Answer is Error\n");

}

void RefinementBisection(graph_t *graph, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor)
{
    Hunyuan_int_t i, j, k, v;
    Hunyuan_int_t nvtxs, best_edgecut, edgecut, imbalance_number;
    Hunyuan_int_t *xadj, *adjncy, *adjwgt, *vwgt, *where, *pwgts, *ed, *id, *bestwhere, *moved, *onemaxpwgt, *oneminpwgt, *oneidealpwgt;
    priority_queue_t *queue;

    nvtxs = graph->nvtxs;
    best_edgecut = graph->nedges;
    best_edgecut = 0;
    imbalance_number = 0;

    xadj = graph->xadj;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    vwgt = graph->vwgt;
    where = graph->where  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "RefinementBisection: graph->where");
    pwgts = graph->pwgts  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "RefinementBisection: graph->pwgts");
    bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "RefinementBisection: bestwhere");
    moved = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "RefinementBisection: moved");
    ed = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "RefinementBisection: ed");
    id = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "RefinementBisection: id");
    onemaxpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "RefinementBisection: onemaxpwgt");
    oneminpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "RefinementBisection: oneminpwgt");
    oneidealpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "RefinementBisection: oneidealpwgt");

    for(i = 0;i < 2;i++)
    {
        onemaxpwgt[i] = ceilUp((Hunyuan_real_t)balance_factor[0] * graph->tvwgt[0] * tpwgts[i]);
        oneminpwgt[i] = floorDown((Hunyuan_real_t)(1.0 / balance_factor[0]) * graph->tvwgt[0] * tpwgts[i]);
        oneidealpwgt[i] = ceilUp((Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i]);

        printf("onemaxpwgt[%"PRIDX"]=%"PRIDX" oneminpwgt[%"PRIDX"]=%"PRIDX" oneidealpwgt[%"PRIDX"]=%"PRIDX" \n", \
            i, onemaxpwgt[i], i, oneminpwgt[i], i, oneidealpwgt[i]);
    }

    queue = priority_queue_Create(nvtxs);

    for(Hunyuan_int_t iter = 0;iter < nvtxs;iter++)
    {
        memset(where, 0, sizeof(Hunyuan_int_t) * nvtxs);

        for(i = 0;i < nvtxs;i++)
        {
            k = 0;
            for(j = xadj[i];j < xadj[i + 1];j++)
                k += adjwgt[j];
            ed[i] = 0;
            id[i] = k;
            priority_queue_Insert(queue, i, k);
        }

        v = iter;
        where[v] = 1;
        // priority_queue_Update(queue, v, );
    }
}

void DFS_1127(graph_t *graph, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor, Hunyuan_int_t *onemaxpwgt, Hunyuan_int_t *oneminpwgt,\
    Hunyuan_int_t v, Hunyuan_int_t pwgts0, Hunyuan_int_t pwgts1, Hunyuan_int_t *bestwhere, Hunyuan_int_t *best_edgecut, Hunyuan_int_t *cnt)
{
    if(v == graph->nvtxs)
    {
        // exam_where(graph);
        // exam_num(bestwhere, graph->nvtxs);
        printf("pwgts0=%"PRIDX" pwgts1=%"PRIDX"\n", pwgts0, pwgts1);
        // for(Hunyuan_int_t i = 0;i < 2;i++)
        // {
        //     printf("oneminpwgt[%"PRIDX"]=%"PRIDX" onemaxpwgt[%"PRIDX"]=%"PRIDX"\n", i, oneminpwgt[i], i, onemaxpwgt[i]);
        // }
        if(pwgts0 >= oneminpwgt[0] && pwgts0 <= onemaxpwgt[0] && pwgts1 >= oneminpwgt[1] && pwgts1 <= onemaxpwgt[1])
        {
            Hunyuan_int_t edgecut = compute_edgecut(graph->where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt);
            // printf("edgecut=%"PRIDX" best_edgecut=%"PRIDX"\n", edgecut, best_edgecut[0]);
            if(edgecut < best_edgecut[0])
            {
                printf("edgecut=%"PRIDX" best_edgecut=%"PRIDX"\n", edgecut, best_edgecut[0]);
                best_edgecut[0] = edgecut;
                memcpy(bestwhere, graph->where, sizeof(Hunyuan_int_t) * graph->nvtxs);
            }

            cnt[0]++;
        }
        return ;
    }

    if(pwgts0 > onemaxpwgt[0] || pwgts1 > onemaxpwgt[1])
    {
        printf("v=%"PRIDX" imbalance pwgts0=%"PRIDX" pwgts1=%"PRIDX"\n", v, pwgts0, pwgts1);
        return ;
    }

    // 0
    graph->where[v] = 0;
    DFS_1127(graph, tpwgts, balance_factor, onemaxpwgt, oneminpwgt, v + 1, pwgts0 + graph->vwgt[v], pwgts1, bestwhere, best_edgecut, cnt);

    if(v == 0)
        return ;
    // 1
    graph->where[v] = 1;
    DFS_1127(graph, tpwgts, balance_factor, onemaxpwgt, oneminpwgt, v + 1, pwgts0, pwgts1 + graph->vwgt[v], bestwhere, best_edgecut, cnt);
}

void DFS_Bisection(graph_t *graph, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor)
{
    Hunyuan_int_t i, j, k, v;
    Hunyuan_int_t nvtxs, best_edgecut, edgecut, imbalance_number, cnt;
    Hunyuan_int_t *xadj, *adjncy, *adjwgt, *vwgt, *where, *pwgts, *bestwhere, *moved, *onemaxpwgt, *oneminpwgt, *oneidealpwgt;

    nvtxs = graph->nvtxs;
    best_edgecut = graph->nedges;
    imbalance_number = 0;
    cnt = 0;

    xadj = graph->xadj;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    vwgt = graph->vwgt;
    where = graph->where  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->where");
    pwgts = graph->pwgts  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: graph->pwgts");
    bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: bestwhere");
    onemaxpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: onemaxpwgt");
    oneminpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: oneminpwgt");
    oneidealpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: oneidealpwgt");

    for(i = 0;i < 2;i++)
    {
        onemaxpwgt[i] = ceilUp((Hunyuan_real_t)balance_factor[0] * graph->tvwgt[0] * tpwgts[i]);
        oneminpwgt[i] = floorDown((Hunyuan_real_t)(1.0 / balance_factor[0]) * graph->tvwgt[0] * tpwgts[i]);
        oneidealpwgt[i] = ceilUp((Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i]);

        printf("onemaxpwgt[%"PRIDX"]=%"PRIDX" oneminpwgt[%"PRIDX"]=%"PRIDX" oneidealpwgt[%"PRIDX"]=%"PRIDX" \n", \
            i, onemaxpwgt[i], i, oneminpwgt[i], i, oneidealpwgt[i]);
    }

    memset(pwgts, 0, sizeof(Hunyuan_int_t) * 2);
    memset(where, 0, sizeof(Hunyuan_int_t) * nvtxs);

    DFS_1127(graph, tpwgts, balance_factor, onemaxpwgt, oneminpwgt, 0, 0.0, 0.0, bestwhere, &best_edgecut, &cnt);

    memcpy(where, bestwhere, sizeof(Hunyuan_int_t) * graph->nvtxs);

    exam_where(graph);

    printf("cnt=%"PRIDX"\n",cnt);

    // exit(0);

    // printf("imbalance_number=%"PRIDX" \n",imbalance_number);

    memset(pwgts, 0, sizeof(Hunyuan_int_t) * 2);
    for(i = 0;i < nvtxs;i++)
    {
        pwgts[where[i]] += vwgt[i];
    }
    if(exam_is_balance(pwgts, graph->tvwgt, 2, tpwgts, balance_factor))
    {   
        printf("The where is balance\n");
    }
    else
    {
        printf("The where is not balance\n");
        for(i = 0;i < 2;i++)
            printf("pwgts[%"PRIDX"]=%"PRIDX" ", i, pwgts[i]);
        printf("\n");
    }

    if(exam_correct_gp(where, graph->nvtxs, 2))
	{
		printf("The Answer is Right\n");
		printf("edgecut=%"PRIDX" \n", compute_edgecut(where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt));
	}
	else 
		printf("The Answer is Error\n");
}

void GGGP_IterationBisection(graph_t *graph, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor)
{
    Hunyuan_int_t i, j, k, iter, nvtxs, drain, nleft, first, last, pwgts[2], \
        from, me, bestcut=0, icut, mincut;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndind;
    Hunyuan_int_t *queue, *touched, *gain, *bestwhere;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;

    // printf("ctrl->ubfactors[0]=%.10lf ntpwgts[1]=%.10lf\n",ctrl->ubfactors[0],ntpwgts[1]);

    /* Allocate refinement memory. Allocate sufficient memory for both edge and node */
    graph->where  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->where");
    graph->pwgts  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Init2WayPartition: graph->pwgts");
    graph->bndptr = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->bndptr");
    graph->bndind = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->bndind");
    graph->id     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->id");
    graph->ed     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: graph->ed");
    // graph->nrinfo = (nrinfo_t *)check_malloc(sizeof(nrinfo_t) * nvtxs, "Init2WayPartition: graph->nrinfo");

    Hunyuan_int_t *onemaxpwgt, *oneminpwgt, *oneidealpwgt;
    onemaxpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "exam_is_balance: onemaxpwgt");
	oneminpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "exam_is_balance: oneminpwgt");
    oneidealpwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "exam_is_balance: oneidealpwgt");

    for(Hunyuan_int_t i = 0;i < 2;i++)
	{
		// onemaxpwgt[i] = ceilUp((Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i] * balance_factor[0]);
        onemaxpwgt[i] = balance_factor[0] * graph->tvwgt[0] * tpwgts[i];
		// oneminpwgt[i] = floorDown((Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i] / balance_factor[0]);
        oneminpwgt[i] = (1.0 / balance_factor[0]) * graph->tvwgt[0] * tpwgts[i];
        oneidealpwgt[i] = ceilUp((Hunyuan_real_t)graph->tvwgt[0] * tpwgts[i]);
        // printf("onemaxpwgt[%"PRIDX"]=%"PRIDX" oneminpwgt[%"PRIDX"]=%"PRIDX" oneidealpwgt[%"PRIDX"]=%"PRIDX" %"PRREAL"\n", i, onemaxpwgt[i], i, oneminpwgt[i], i, oneidealpwgt[i],\
        //     balance_factor[0]);
	}

    bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: bestwhere");
    queue     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: queue");
    touched   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: touched");
  
    bndind = graph->bndind;
    where = graph->where;

    for (iter = 0; iter < 5; iter++) 
    {
        // if(iter != 0)
        // {
        //     // PrintTimeSteps();
        //     exit(0);
        // }
        //     exit(0);
        CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimebegin(&start_BFS, &end_BFS, &time_BFS));
        set_value_int(nvtxs, 1, where);
        set_value_int(nvtxs, 0, touched);

        pwgts[1] = graph->tvwgt[0];
        pwgts[0] = 0;

        queue[0] = irandInRange(nvtxs);
        // queue[0] = iter;
        // queue[0] = vertex;
        // printf("queue[0]=%"PRIDX"\n", queue[0]);
        // queue[0] = vertex;
        // printf("inbfs=%"PRIDX" queue[0]=%"PRIDX"\n",inbfs,queue[0]);
        touched[queue[0]] = 1;
        first = 0;
        last = 1;
        nleft = nvtxs - 1;
        drain = 0;

        /* Start the BFS from queue to get a partition */
        for (;;) 
        {
            if (first == last) 
            { /* Empty. Disconnected graph! */
                if (nleft == 0 || drain)
                    break;

                k = irandInRange(nleft);
                // printf("inbfs=%"PRIDX" k=%"PRIDX"\n",inbfs,k);
                for (i=0; i<nvtxs; i++) 
                {
                    if (touched[i] == 0) 
                    {
                        if (k == 0)
                            break;
                        else
                            k--;
                    }
                }

                queue[0]   = i;
                touched[i] = 1;
                first      = 0; 
                last       = 1;
                nleft--;
            }

            i = queue[first++];
            if (pwgts[0] > 0 && pwgts[1]-vwgt[i] < oneminpwgt[1]) 
            {
                drain = 1;
                continue;
            }

            where[i] = 0;
            pwgts[0] += vwgt[i];
            pwgts[1] -= vwgt[i];
            if (pwgts[1] <= onemaxpwgt[1])
                break;

            drain = 0;
            for (j = xadj[i]; j < xadj[i + 1]; j++) 
            {
                k = adjncy[j];
                if (touched[k] == 0) 
                {
                    queue[last++] = k;
                    touched[k] = 1;
                    nleft--;
                }
            }
        }

        // for (i = 0; i < nvtxs; i++)
        //     printf("%"PRIDX" ", graph->where[i]);
        // printf("\n");
        // printf("%"PRIDX" %"PRIDX" \n", pwgts[0], pwgts[1]);

        /* Check to see if we hit any bad limiting cases */
        if (pwgts[1] == 0) 
            where[irandInRange(nvtxs)] = 1;
        if (pwgts[0] == 0) 
            where[irandInRange(nvtxs)] = 0;
        CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimeend(&start_BFS, &end_BFS, &time_BFS));
        // printf("grow end rand_count=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        /*************************************************************
        * Do some partition refinement 
        **************************************************************/
        CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimebegin(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
        Compute_Partition_Informetion_2way(graph);
        CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimeend(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
        // printf("ReorderBisection 2\n");
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, gettimebegin(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
        Balance2Way_partition(graph, tpwgts, balance_factor);
        CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, gettimeend(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
        // printf("balance end rand_count=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        // printf("ReorderBisection 3\n");
        // printf("ReorderBisection 3 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimebegin(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));
        FM_2WayCutRefine(graph, tpwgts, 10);
        CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimeend(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));

        // if(pwgts[0] >= oneminpwgt[0] && pwgts[1] >= oneminpwgt[1]
        //     && pwgts[0] <= onemaxpwgt[0] && pwgts[1] <= onemaxpwgt[1])
        // if(ComputeLoadImbalanceDiff(graph, 2, tpwgts, balance_factor[0]) <= 0)
        //     printf("%7"PRIDX" Yes\n", iter);
        // else    
        //     printf("%7"PRIDX" No\n", iter);
        // printf("FM_2WayCutRefine end rand_count=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        // printf("ReorderBisection 4\n");
        // printf("ReorderBisection 4 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        printf("%7"PRIDX" edgecut=%"PRIDX"\n", iter, graph->mincut);
        // printf("%7"PRIDX" %"PRIDX"\n", vertex, graph->mincut);
        // printf("%"PRIDX"\n", graph->mincut);
        if (iter == 0 || bestcut > graph->mincut) 
        {
            bestcut = graph->mincut;
            copy_int(nvtxs, where, bestwhere);
            
            if (bestcut == 0)
                break;
        }

        // printf("iter=%"PRIDX" edgecut=%"PRIDX" bestcut=%"PRIDX" compute_edgecut_where=%"PRIDX" compute_edgecut_bestwhere=%"PRIDX"\n", iter, graph->mincut, bestcut, compute_edgecut(where, nvtxs, xadj, adjncy, adjwgt), compute_edgecut(bestwhere, nvtxs, xadj, adjncy, adjwgt));
        // printf("%"PRIDX" \n", graph->mincut);
        // exam_num(where, nvtxs);
        // exam_num(bestwhere, nvtxs);
    }

    graph->mincut = bestcut;
    copy_int(nvtxs, bestwhere, where);

    check_free(touched, sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: touched");
    check_free(queue, sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: queue");
    check_free(bestwhere, sizeof(Hunyuan_int_t) * nvtxs, "Init2WayPartition: bestwhere");
}

void RecursiveBisection(graph_t *graph, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor, Hunyuan_int_t *part, Hunyuan_int_t fpart, Hunyuan_int_t level)
{
    Hunyuan_int_t i, j, nvtxs, ncon;
    Hunyuan_int_t *label, *where;
    graph_t *lgraph, *rgraph;
    Hunyuan_real_t wsum, *tpwgts2;

    if ((nvtxs = graph->nvtxs) == 0) 
	{
        printf("\t***Cannot bisect a graph with 0 vertices!\n"
            "\t***You are trying to partition a graph into too many parts!\n");
        return ;
    }

    /* determine the weights of the two partitions as a function of the weight of the
        target partition weights */
    tpwgts2 = (Hunyuan_real_t *)check_malloc(sizeof(Hunyuan_real_t) * 2, "MultiLevelRecursivePartition: tpwgts2");
    for (i = 0; i < 1; i++) 
	{
        tpwgts2[i]      = sum_real((nparts >> 1), tpwgts, 1);
        tpwgts2[i + 1] = 1.0 - tpwgts2[i];
    }

    /* perform the bisection */
    // IterationBisection(graph, tpwgts2, balance_factor);
    CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimebegin(&start_initialpartition, &end_initialpartition, &time_initialpartition));
    GGGP_IterationBisection(graph, tpwgts2, balance_factor);
    CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimeend(&start_initialpartition, &end_initialpartition, &time_initialpartition));
    // DFS_Bisection(graph, tpwgts2, balance_factor);

    check_free(tpwgts2, sizeof(Hunyuan_real_t) * 2, "MultiLevelRecursivePartition: tpwgts2");

    label = graph->label;
    where = graph->where;
    // for (i = 0; i < nvtxs; i++)
    //     printf("%"PRIDX" ",where[i]);
    // printf("\n");
    for (i = 0; i < nvtxs; i++)
        part[label[i]] = where[i] + fpart;

    if (nparts > 2) 
        SplitGraphPartition(graph, &lgraph, &rgraph);

    /* Free the memory of the top level graph */
    FreeGraph(&graph, 2);

    /* Scale the fractions in the tpwgts according to the true weight */
    for (i = 0; i < 1; i++) 
	{
        wsum = sum_real((nparts >> 1), tpwgts, 1);
        rscale_real((nparts >> 1), 1.0 / wsum, tpwgts);
        rscale_real(nparts - (nparts >> 1), 1.0 / (1.0 - wsum), tpwgts + (nparts >> 1));
    }

    /* Do the recursive call */
    if (nparts > 3) {
        RecursiveBisection(lgraph, (nparts>>1), tpwgts, balance_factor, part, fpart, level * 2 + 1);
        RecursiveBisection(rgraph, nparts - (nparts>>1), tpwgts + (nparts>>1), balance_factor, part, fpart + (nparts>>1), level * 2 + 2);
    }
    else if (nparts == 3) {
        FreeGraph(&lgraph, 0);
        RecursiveBisection(rgraph, nparts - (nparts>>1), tpwgts + (nparts>>1), balance_factor, part, fpart + (nparts>>1), level * 2 + 2);
    }
}

void InitialPartition_NestedBisection(graph_t *graph, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts_ori, Hunyuan_real_t *balance_factor)
{
    Hunyuan_int_t *vwgt, *xadj, *adjncy, *adjwgt;
    Hunyuan_real_t *tpwgts;

    vwgt   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "InitialPartition: vwgt");
    xadj   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (graph->nvtxs + 1), "InitialPartition: xadj");
    adjncy = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "InitialPartition: adjncy");
    adjwgt = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nedges, "InitialPartition: adjwgt");
    tpwgts = (Hunyuan_real_t *)check_malloc(sizeof(Hunyuan_real_t) * nparts, "InitialPartition: tpwgts");

    memcpy(xadj, graph->xadj, sizeof(Hunyuan_int_t) * (graph->nvtxs + 1));
    memcpy(vwgt, graph->vwgt, sizeof(Hunyuan_int_t) * graph->nvtxs);
    memcpy(adjncy, graph->adjncy, sizeof(Hunyuan_int_t) * graph->nedges);
    memcpy(adjwgt, graph->adjwgt, sizeof(Hunyuan_int_t) * graph->nedges);
    memcpy(tpwgts, tpwgts_ori, sizeof(Hunyuan_real_t) * nparts);

    InitRandom(-1);

    Hunyuan_real_t ubvec = (Hunyuan_real_t)pow(balance_factor[0], 1.0 / log(nparts));
    ubvec = 1.001050;

    graph_t *t_graph = SetupGraph(graph->nvtxs, xadj, adjncy, vwgt, adjwgt);
    Hunyuan_int_t *part = graph->where = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * graph->nvtxs, "InitialPartition: graph->where");
    // exam_tpwgts(tpwgts, nparts);
    RecursiveBisection(t_graph, nparts, tpwgts, &ubvec, part, 0, 0);
    // exam_tpwgts(tpwgts, nparts);

    check_free(tpwgts, sizeof(Hunyuan_real_t) * nparts, "InitialPartition: tpwgts");
}

//  InitSeparator + GrowBisectionNode
void ReorderBisection(graph_t *graph, Hunyuan_int_t niparts)
{
    double ntpwgts[2] = {0.5, 0.5};

    /* this is required for the cut-based part of the refinement */
    // Setup2WayBalMultipliers(graph, ntpwgts);

    // GrowBisectionNode(graph, ntpwgts, niparts);
    Hunyuan_int_t i, j, k, nvtxs, drain, nleft, first, last, pwgts[2], oneminpwgt, 
        onemaxpwgt, from, me, bestcut=0, icut, mincut, inbfs;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndind;
    Hunyuan_int_t *queue, *touched, *gain, *bestwhere;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;

    onemaxpwgt = 1.2000499 * graph->tvwgt[0] * 0.5;
    oneminpwgt = (1.0 / 1.2000499) * graph->tvwgt[0] * 0.5;

    /* Allocate refinement memory. Allocate sufficient memory for both edge and node */
    graph->where  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: graph->where");
    graph->pwgts  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 3, "ReorderBisection: graph->pwgts");
    graph->bndptr = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: graph->bndptr");
    graph->bndind = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: graph->bndind");
    graph->id     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: graph->id");
    graph->ed     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: graph->ed");
    graph->nrinfo = (nrinfo_t *)check_malloc(sizeof(nrinfo_t) * nvtxs, "ReorderBisection: graph->nrinfo");

    bestwhere = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: bestwhere");
    queue     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: queue");
    touched   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: touched");
  
    where  = graph->where;
    bndind = graph->bndind;
    // printf("ReorderBisection 0\n");
    for (inbfs = 0; inbfs < niparts; inbfs++) 
    {
        set_value_int(nvtxs, 1, where);
        set_value_int(nvtxs, 0, touched);

        pwgts[1] = graph->tvwgt[0];
        pwgts[0] = 0;

        queue[0] = irandInRange(nvtxs);
        touched[queue[0]] = 1;
        first = 0; last = 1;
        nleft = nvtxs-1;
        drain = 0;

        /* Start the BFS from queue to get a partition */
        for (;;) 
        {
            if (first == last) 
            { 
                /* Empty. Disconnected graph! */
                if (nleft == 0 || drain)
                    break;
  
                k = irandInRange(nleft);
                for (i = 0; i < nvtxs; i++) 
                { 
                    /* select the kth untouched vertex */
                    if (touched[i] == 0) 
                    {
                        if (k == 0)
                            break;
                        else
                            k--;
                    }
                }

                queue[0]   = i;
                touched[i] = 1;
                first      = 0; 
                last       = 1;
                nleft--;
            }

            i = queue[first++];
            if (pwgts[1] - vwgt[i] < oneminpwgt) 
            {
                drain = 1;
                continue;
            }

            where[i] = 0;
            pwgts[0] += vwgt[i];
            pwgts[1] -= vwgt[i];
            if (pwgts[1] <= onemaxpwgt)
                break;

            drain = 0;
            for (j = xadj[i]; j < xadj[i + 1]; j++) 
            {
                k = adjncy[j];
                if (touched[k] == 0) 
                {
                    queue[last++] = k;
                    touched[k] = 1;
                    nleft--;
                }
            }
        }
        // printf("ReorderBisection 1\n");
        // printf("ReorderBisection 1 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        /*************************************************************
        * Do some partition refinement 
        **************************************************************/
        CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimebegin(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
		Compute_Partition_Informetion_2way(graph);
		CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimeend(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
        // printf("ReorderBisection 2\n");
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, gettimebegin(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
		Balance2Way_node(graph, ntpwgts);
		CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, gettimeend(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
        // printf("ReorderBisection 3\n");
        // printf("ReorderBisection 3 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimebegin(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));
		FM_2WayCutRefine(graph, ntpwgts, 4);
		CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimeend(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));
        // printf("ReorderBisection 4\n");
        // printf("ReorderBisection 4 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        /* Construct and refine the vertex separator */
        for (i = 0; i < graph->nbnd; i++) 
        {
            j = bndind[i];
            if (xadj[j + 1] - xadj[j] > 0) /* ignore islands */
                where[j] = 2;
        }

        // printf("ReorderBisection 5\n");
        // exam_where(graph);

        CONTROL_COMMAND(control, REORDERINF2WAY_Time, gettimebegin(&start_reorderinf2way, &end_reorderinf2way, &time_reorderinf2way));
		Compute_Reorder_Informetion_2way(graph); 
		CONTROL_COMMAND(control, REORDERINF2WAY_Time, gettimeend(&start_reorderinf2way, &end_reorderinf2way, &time_reorderinf2way));
        // printf("ReorderBisection 6\n");
        // exam_where(graph);
        CONTROL_COMMAND(control, FM2SIDENODEREFINE_Time, gettimebegin(&start_fm2sidenoderefine, &end_fm2sidenoderefine, &time_fm2sidenoderefine));
		FM_2WayNodeRefine2Sided(graph, 1);
		CONTROL_COMMAND(control, FM2SIDENODEREFINE_Time, gettimeend(&start_fm2sidenoderefine, &end_fm2sidenoderefine, &time_fm2sidenoderefine));
        // printf("ReorderBisection 7\n");
        // printf("ReorderBisection 7 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);
        CONTROL_COMMAND(control, FM1SIDENODEREFINE_Time, gettimebegin(&start_fm1sidenoderefine, &end_fm1sidenoderefine, &time_fm1sidenoderefine));
		FM_2WayNodeRefine1Sided(graph, 4);
		CONTROL_COMMAND(control, FM1SIDENODEREFINE_Time, gettimeend(&start_fm1sidenoderefine, &end_fm1sidenoderefine, &time_fm1sidenoderefine));
        // printf("ReorderBisection 8\n");
        // printf("ReorderBisection 8 end ccnt=%"PRIDX"\n",rand_count());
        // exam_where(graph);

        /*
        printf("ISep: [%"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX"] %"PRIDX"\n", 
            inbfs, graph->pwgts[0], graph->pwgts[1], graph->pwgts[2], bestcut); 
        */
        // printf("inbfs=%"PRIDX"\n",inbfs);
        if (inbfs == 0 || bestcut > graph->mincut) 
        {
            bestcut = graph->mincut;
            copy_int(nvtxs, where, bestwhere);
        }
    }

    graph->mincut = bestcut;
    copy_int(nvtxs, bestwhere, where);
    // printf("ReorderBisection 9\n");
    check_free(touched, sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: touched");
    check_free(queue, sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: queue");
    check_free(bestwhere, sizeof(Hunyuan_int_t) * nvtxs, "ReorderBisection: bestwhere");
    //     exam_where(graph);
}

#endif