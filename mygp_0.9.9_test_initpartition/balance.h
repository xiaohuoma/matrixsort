#ifndef BALANCE_H
#define BALANCE_H

#include "struct.h"
#include "queue.h"

bool Is_Balance_2way(Hunyuan_int_t *pwgts, double ubvec, Hunyuan_int_t *tvwgt, Hunyuan_int_t *target_pwgts)
{
    bool flag = true;

    //  If the partition weight is not required, it is equally partitioned by default
    for(Hunyuan_int_t i = 0;i < 2;i++)
    {
        double now_pwgts = pwgts[i];
        double target = tvwgt[0] * 0.5;
        printf("now_pwgts=%lf target=%lf\n",now_pwgts,target);
        printf("%lf >= %lf %lf <= %lf\n",now_pwgts,target * (1 - ubvec),now_pwgts,target * (1 + ubvec));
        if(now_pwgts >= target * (1 - ubvec) && now_pwgts <= target * (1 + ubvec))
            continue ;
        
        flag = false;
        break;
    }

    // flag = true;
    printf("Is_Balance_2way: flag=%d\n",flag);

    return flag;
}

void Balance_Partition_2way(graph_t *graph)
{
    Hunyuan_int_t nvtxs, nbnd, mincut, from, to;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndptr, *bndind, *ed, *id, *pwgts;
    priority_queue_t *queue;

    nvtxs  = graph->nvtxs;
    nbnd   = graph->nbnd;
    mincut = graph->mincut;

    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    where  = graph->where;
    bndptr = graph->bndptr;
    bndind = graph->bndind;
    ed     = graph->ed;
    id     = graph->id;
    pwgts  = graph->pwgts;

    //  Determine the direction of vertex movement
    from = pwgts[0] > pwgts[1] ? 0 : 1;
    to   = (from + 1) % 2;

    // printf("from=%d to=%d\n",from,to);

    printf("Balance_Partition_2way 0\n");
    //  Check whether the current partition is balanced
    if(Is_Balance_2way(pwgts, 0.03, graph->tvwgt, NULL))
        return ;
    printf("Balance_Partition_2way 1\n");
    //  Set up and init the priority queue
    //  only from
    queue = priority_queue_Create(nvtxs);
    for(Hunyuan_int_t i = 0;i < nbnd;i++)
    {
        Hunyuan_int_t vertex = bndind[i];
        if(where[vertex] == from)
            priority_queue_Insert(queue,vertex, ed[vertex] - id[vertex]);
    }

    // priority_queue_exam(queue);

    while(priority_queue_Length(queue) != 0)
    {
        //  select the best vertex to move
        Hunyuan_int_t vertex = priority_queue_GetTop(queue);

        //  update the information of the vertex
        where[vertex] = to;
        pwgts[from] -= vwgt[vertex];
        pwgts[to]   += vwgt[vertex];
        Hunyuan_int_t z;
        lyj_swap(ed[vertex], id[vertex], z);
        mincut -= ed[vertex] - id[vertex];
        if(ed[vertex] == 0 && xadj[vertex + 1] < xadj[vertex])
            nbnd = delete_queue(nbnd, bndptr, bndind, vertex);
        
        //  update the vertex's adjacent vertices
        for(Hunyuan_int_t j = xadj[vertex];j < xadj[vertex + 1];j++)
        {
            Hunyuan_int_t k = adjncy[j];

            //  the array ed and id of k
            if(where[k] == to)
            {
                ed[k] -= adjwgt[vertex];
                id[k] += adjwgt[vertex];
            }
            else 
            {
                ed[k] += adjwgt[vertex];
                id[k] -= adjwgt[vertex];
            }

            //  if the vertex k is a boundary vertex
            if(bndptr[k] != -1)
            {
                //  if the vertex k isn't a boundary vertex now
                if(ed[k] == 0 && xadj[k + 1] - xadj[k] > 0)
                {
                    nbnd = delete_queue(nbnd, bndptr, bndind, k);
                    if(where[k] == from)
                        priority_queue_Delete(queue, k);
                }
                //  if the vertex k still is a boundary vertex now
                else
                {
                    //  update the queue
                    if(where[k] == from)
                        priority_queue_Update(queue, k, ed[k] - id[k]);
                }

            }
            //  if the vertex k isn't a boundary vertex
            else
            {
                //  if the vertex become a boundary vertex now
                if(ed[k] > 0 &&  xadj[k + 1] - xadj[k] > 0)
                {
                    nbnd = insert_queue(nbnd, bndptr, bndind, k);
                    if(where[k] == from)
                        priority_queue_Insert(queue,k,ed[k] - id[k]);
                }
            }
        }

        //  Check whether the current partition is balanced
        if(Is_Balance_2way(pwgts, 0.03, graph->tvwgt, NULL))
            break ;
        
        if(pwgts[from] < pwgts[to])
            break ;
    }

    priority_queue_Destroy(queue);

    // exam_where(graph);
    // exam_edid(graph);
    // exam_pwgts(graph);

    graph->mincut = mincut;
    graph->nbnd   = nbnd;

    // exam_bnd(graph);
}

/*************************************************************************/
/*! Computes the maximum load imbalance difference of a partitioning 
    solution over all the constraHunyuan_int_ts. 
    The difference is defined with respect to the allowed maximum 
    unbalance for the respective constraHunyuan_int_t. 
 */
/**************************************************************************/ 
Hunyuan_real_t ComputeLoadImbalanceDiff(graph_t *graph, Hunyuan_int_t nparts, Hunyuan_real_t *ntpwgts, Hunyuan_real_t ubvec)
{
    Hunyuan_int_t i, j, *pwgts;
    Hunyuan_real_t max, cur;

    pwgts = graph->pwgts;

    // printf("ComputeLoadImbalanceDiff 0\n");

    max = -1.0;
    for (j = 0; j < nparts; j++) 
    {
        cur = pwgts[j] * (Hunyuan_real_t)((Hunyuan_real_t)graph->invtvwgt[0] / ntpwgts[j]) - ubvec;

        if (cur > max)
            max = cur;
    }
    
    //  need exam
    return max;
}

/*************************************************************************
* This function balances two partitions by moving boundary nodes
* from the domain that is overweight to the one that is underweight.
**************************************************************************/
void Bnd2WayBalance(graph_t *graph, Hunyuan_real_t *ntpwgts)
{
    Hunyuan_int_t i, ii, j, k, kwgt, nvtxs, nbnd, nswaps, from, to, pass, me, tmp;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind, *pwgts;
    Hunyuan_int_t *moved, *perm;
    priority_queue_t *queue;
    Hunyuan_int_t higain, mincut, mindiff;
    Hunyuan_int_t tpwgts[2];

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    where  = graph->where;
    id     = graph->id;
    ed     = graph->ed;
    pwgts  = graph->pwgts;
    bndptr = graph->bndptr;
    bndind = graph->bndind;

    moved = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Bnd2WayBalance: moved");
    perm  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Bnd2WayBalance: perm");

    /* Determine from which domain you will be moving data */
    tpwgts[0] = graph->tvwgt[0] * ntpwgts[0];
    tpwgts[1] = graph->tvwgt[0] - tpwgts[0];
    mindiff   = lyj_abs(tpwgts[0] - pwgts[0]);
    from      = (pwgts[0] < tpwgts[0] ? 1 : 0);
    to        = (from + 1) % 2;

    // printf("Partitions: [%6"PRIDX" %6"PRIDX"] T[%6"PRIDX" %6"PRIDX"], Nv-Nb[%6"PRIDX" %6"PRIDX"]. ICut: %6"PRIDX" [B]\n", \
        pwgts[0], pwgts[1], tpwgts[0], tpwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    queue = priority_queue_Create(nvtxs);

    set_value_int(nvtxs, -1, moved);

    /* Insert the boundary nodes of the proper partition whose size is OK in the priority queue */
    nbnd = graph->nbnd;
    irandArrayPermute(nbnd, perm, nbnd/5, 1);
    for (ii = 0; ii < nbnd; ii++)
    {
        i = perm[ii];
        if (where[bndind[i]] == from && vwgt[bndind[i]] <= mindiff)
            priority_queue_Insert(queue, bndind[i], ed[bndind[i]] - id[bndind[i]]);
    }

    mincut = graph->mincut;
    for (nswaps = 0; nswaps < nvtxs; nswaps++) 
    {
        if ((higain = priority_queue_GetTop(queue)) == -1)
            break;

        if (pwgts[to] + vwgt[higain] > tpwgts[to])
            break;

        mincut -= (ed[higain] - id[higain]);
        pwgts[to] += vwgt[higain];
        pwgts[from] -= vwgt[higain];

        where[higain] = to;
        moved[higain] = nswaps;

        // printf("Moved %6"PRIDX" from %"PRIDX". [%3"PRIDX" %3"PRIDX"] %5"PRIDX" [%4"PRIDX" %4"PRIDX"]\n", higain, from, ed[higain]-id[higain], vwgt[higain], mincut, pwgts[0], pwgts[1]);

        /**************************************************************
        * Update the id[i]/ed[i] values of the affected nodes
        ***************************************************************/
        lyj_swap(id[higain], ed[higain], tmp);
        if (ed[higain] == 0 && xadj[higain] < xadj[higain + 1]) 
            nbnd = delete_queue(nbnd, bndptr,  bndind, higain);

        for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
        {
            k = adjncy[j];
            kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
            id[k] += kwgt;
            ed[k] -= kwgt;

            /* Update its boundary information and queue position */
            if (bndptr[k] != -1) 
            {
                /* If k was a boundary vertex */
                if (ed[k] == 0) 
                { 
                    /* Not a boundary vertex any more */
                    nbnd = delete_queue(nbnd, bndptr,  bndind, k);
                    if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)  /* Remove it if in the queues */
                        priority_queue_Delete(queue, k);
                }
                else 
                { 
                    /* If it has not been moved, update its position in the queue */
                    if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)
                        priority_queue_Update(queue, k, ed[k] - id[k]);
                }
            }
            else 
            {
                if (ed[k] > 0) 
                {  
                    /* It will now become a boundary vertex */
                    nbnd = insert_queue(nbnd, bndptr,  bndind, k);
                    if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff) 
                        priority_queue_Insert(queue, k, ed[k] - id[k]);
                }
            }
        }
    }

    // printf("\tMinimum cut: %6"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", mincut, pwgts[0], pwgts[1], nbnd);
    // printf("Bnd2WayBalance\n");
    graph->mincut = mincut;
    graph->nbnd   = nbnd;

    priority_queue_Destroy(queue);

    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "Bnd2WayBalance: moved");
    check_free(perm, sizeof(Hunyuan_int_t) * nvtxs, "Bnd2WayBalance: perm");
}

/*************************************************************************
* This function balances two partitions by moving the highest gain 
* (including negative gain) vertices to the other domain.
* It is used only when tha unbalance is due to non contigous
* subdomains. That is, the are no boundary vertices.
* It moves vertices from the domain that is overweight to the one that 
* is underweight.
**************************************************************************/
void General2WayBalance(graph_t *graph, Hunyuan_real_t *ntpwgts)
{
    Hunyuan_int_t i, ii, j, k, kwgt, nvtxs, nbnd, nswaps, from, to, pass, me, tmp;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind, *pwgts;
    Hunyuan_int_t *moved, *perm;
    priority_queue_t *queue;
    Hunyuan_int_t higain, mincut, mindiff;
    Hunyuan_int_t tpwgts[2];

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    where  = graph->where;
    id     = graph->id;
    ed     = graph->ed;
    pwgts  = graph->pwgts;
    bndptr = graph->bndptr;
    bndind = graph->bndind;

    moved = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Bnd2WayBalance: moved");
    perm  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Bnd2WayBalance: perm");

    /* Determine from which domain you will be moving data */
    tpwgts[0] = graph->tvwgt[0] * ntpwgts[0];
    tpwgts[1] = graph->tvwgt[0] - tpwgts[0];
    mindiff   = lyj_abs(tpwgts[0] - pwgts[0]);
    from      = (pwgts[0] < tpwgts[0] ? 1 : 0);
    to        = (from + 1) % 2;

    // printf("Partitions: [%6"PRIDX" %6"PRIDX"] T[%6"PRIDX" %6"PRIDX"], Nv-Nb[%6"PRIDX" %6"PRIDX"]. ICut: %6"PRIDX" [B]\n", \
        pwgts[0], pwgts[1], tpwgts[0], tpwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    queue = priority_queue_Create(nvtxs);

    set_value_int(nvtxs, -1, moved);

    /* Insert the nodes of the proper partition whose size is OK in the priority queue */
    irandArrayPermute(nvtxs, perm, nvtxs / 5, 1);
    for (ii = 0; ii < nvtxs; ii++) 
    {
        i = perm[ii];
        if (where[i] == from && vwgt[i] <= mindiff)
            priority_queue_Insert(queue, i, ed[i] - id[i]);
    }

    mincut = graph->mincut;
    nbnd = graph->nbnd;
    for (nswaps = 0; nswaps < nvtxs; nswaps++) 
    {
        if ((higain = priority_queue_GetTop(queue)) == -1)
            break;

        if (pwgts[to] + vwgt[higain] > tpwgts[to])
            break;

        mincut -= (ed[higain] - id[higain]);
        pwgts[to] += vwgt[higain];
        pwgts[from] -= vwgt[higain];

        where[higain] = to;
        moved[higain] = nswaps;

        // printf("Moved %6"PRIDX" from %"PRIDX". [%3"PRIDX" %3"PRIDX"] %5"PRIDX" [%4"PRIDX" %4"PRIDX"]\n", higain, from, ed[higain]-id[higain], vwgt[higain], mincut, pwgts[0], pwgts[1]);

        /**************************************************************
        * Update the id[i]/ed[i] values of the affected nodes
        ***************************************************************/
        lyj_swap(id[higain], ed[higain], tmp);
        if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain + 1]) 
            nbnd = delete_queue(nbnd, bndptr,  bndind, higain);
        if (ed[higain] > 0 && bndptr[higain] == -1)
            nbnd = insert_queue(nbnd, bndptr,  bndind, higain);

        for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
        {
            k = adjncy[j];

            kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
            id[k] += kwgt;
            ed[k] -= kwgt;

            /* Update the queue position */
            if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)
                priority_queue_Update(queue, k, ed[k] - id[k]);

            /* Update its boundary information */
            if (ed[k] == 0 && bndptr[k] != -1) 
                nbnd = delete_queue(nbnd, bndptr, bndind, k);
            else if (ed[k] > 0 && bndptr[k] == -1)  
                nbnd = insert_queue(nbnd, bndptr,  bndind, k);
        }
    }

    // printf("\tMinimum cut: %6"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", mincut, pwgts[0], pwgts[1], nbnd);

    graph->mincut = mincut;
    graph->nbnd   = nbnd;

    priority_queue_Destroy(queue);

    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "General2WayBalance: moved");
    check_free(perm, sizeof(Hunyuan_int_t) * nvtxs, "General2WayBalance: perm");
}

void Balance2Way_node(graph_t *graph, Hunyuan_real_t *ntpwgts)
{
    // printf("Balance2Way begin %lf\n",ComputeLoadImbalanceDiff(graph, 2, 1.200050));
    // printf("ubvec=%lf\n",ubvec);
    if (ComputeLoadImbalanceDiff(graph, 2, ntpwgts, 1.200050) <= 0) 
        return;

    /* return right away if the balance is OK */
    if (lyj_abs(ntpwgts[0] * graph->tvwgt[0] - graph->pwgts[0]) < 3 * graph->tvwgt[0] / graph->nvtxs)
      return;

    if (graph->nbnd > 0)
    {
        // printf("Bnd2WayBalance begin\n");
        Bnd2WayBalance(graph, ntpwgts);
        // printf("Bnd2WayBalance end\n");
    }
    else
        General2WayBalance(graph, ntpwgts);
}

/*************************************************************************/
/*! This function balances the left/right partitions of a separator 
    tri-section */
/*************************************************************************/
void FM_2WayNodeBalance(graph_t *graph)
{
    Hunyuan_int_t i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, gain;
    Hunyuan_int_t badmaxpwgt, higain, oldgain, pass, to, other;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
    Hunyuan_int_t *perm, *moved;
    priority_queue_t *queue; 
    nrinfo_t *rinfo;
    double mult;

    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    adjncy = graph->adjncy;
    vwgt   = graph->vwgt;

    bndind = graph->bndind;
    bndptr = graph->bndptr;
    where  = graph->where;
    pwgts  = graph->pwgts;
    rinfo  = graph->nrinfo;

    mult = 0.5 * 1.2000499;

    badmaxpwgt = (Hunyuan_int_t)(mult * (pwgts[0] + pwgts[1]));
    if (lyj_max(pwgts[0], pwgts[1]) < badmaxpwgt)
        return;
    if (lyj_abs(pwgts[0] - pwgts[1]) < 3 * graph->tvwgt[0] / nvtxs)
        return;

    to    = (pwgts[0] < pwgts[1] ? 0 : 1); 
    other = (to + 1) % 2;

    queue = priority_queue_Create(nvtxs);

    perm   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeBalance: perm");
    moved  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeBalance: moved");
    set_value_int(nvtxs,-1,moved);


    // printf("Partitions: [%6"PRIDX" %6"PRIDX"] Nv-Nb[%6"PRIDX" %6"PRIDX"]. ISep: %6"PRIDX" [B]\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    nbnd = graph->nbnd;
    // printf("FM_2WayNodeBalance ccnt=%"PRIDX"\n",rand_count());
    irandArrayPermute(nbnd, perm, nbnd, 1);
    // printf("FM_2WayNodeBalance ccnt=%"PRIDX"\n",rand_count());
    for (ii = 0; ii < nbnd; ii++) 
    {
        i = bndind[perm[ii]];
        priority_queue_Insert(queue, i, vwgt[i]-rinfo[i].edegrees[other]);
    }

    /******************************************************
     * Get Hunyuan_int_to the FM loop
     *******************************************************/
    for (nswaps = 0; nswaps < nvtxs; nswaps++) 
    {
        if ((higain = priority_queue_GetTop(queue)) == -1)
            break;

        moved[higain] = 1;

        gain = vwgt[higain] - rinfo[higain].edegrees[other];
        badmaxpwgt = (Hunyuan_int_t)(mult * (pwgts[0] + pwgts[1]));

        /* break if other is now underwight */
        if (pwgts[to] > pwgts[other])
            break;

        /* break if balance is achieved and no +ve or zero gain */
        if (gain < 0 && pwgts[other] < badmaxpwgt) 
            break;

        /* skip this vertex if it will violate balance on the other side */
        if (pwgts[to] + vwgt[higain] > badmaxpwgt) 
            continue;

        pwgts[2] -= gain;

        nbnd = delete_queue(nbnd,bndptr,bndind,higain);        
        pwgts[to] += vwgt[higain];
        where[higain] = to;

        // printf("Moved %6"PRIDX" to %3"PRIDX", Gain: %3"PRIDX", \t[%5"PRIDX" %5"PRIDX" %5"PRIDX"]\n", higain, to, vwgt[higain]-rinfo[higain].edegrees[other], pwgts[0], pwgts[1], pwgts[2]);

        /**********************************************************
        * Update the degrees of the affected nodes
        ***********************************************************/
        for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
        {
            k = adjncy[j];
            if (where[k] == 2) 
            { 
                /* For the in-separator vertices modify their edegree[to] */
                rinfo[k].edegrees[to] += vwgt[higain];
            }
            else if (where[k] == other) 
            { /* This vertex is pulled Hunyuan_int_to the separator */
                nbnd = insert_queue(nbnd, bndptr, bndind, k);

                where[k] = 2;
                pwgts[other] -= vwgt[k];

                edegrees = rinfo[k].edegrees;
                edegrees[0] = edegrees[1] = 0;
                for (jj = xadj[k]; jj < xadj[k + 1]; jj++) 
                {
                    kk = adjncy[jj];
                    if (where[kk] != 2) 
                        edegrees[where[kk]] += vwgt[kk];
                    else 
                    {
                        oldgain = vwgt[kk] - rinfo[kk].edegrees[other];
                        rinfo[kk].edegrees[other] -= vwgt[k];

                        if (moved[kk] == -1)
                            priority_queue_Update(queue, kk, oldgain + vwgt[k]);
                    }
                }

                /* Insert the new vertex Hunyuan_int_to the priority queue */
                priority_queue_Insert(queue, k, vwgt[k]-edegrees[other]);
            }
        }
    }

    // printf("\tBalanced sep: %6"PRIDX" at %4"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", pwgts[2], nswaps, pwgts[0], pwgts[1], nbnd);
    // printf("FM_2WayNodeBalance\n");
    graph->mincut = pwgts[2];
    graph->nbnd   = nbnd;

    priority_queue_Destroy(queue);

    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeBalance: moved");
    check_free(perm, sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeBalance: perm");
}

void Balance2Way_partition(graph_t *graph, Hunyuan_real_t *ntpwgts, Hunyuan_real_t *balance_factor)
{
    // printf("Balance2Way begin %lf\n",ComputeLoadImbalanceDiff(graph, 2, ntpwgts, balance_factor[0]));
    // printf("ubvec=%lf\n",ubvec);
    if (ComputeLoadImbalanceDiff(graph, 2, ntpwgts, balance_factor[0]) <= 0) 
        return;

    /* return right away if the balance is OK */
    if (lyj_abs(ntpwgts[0] * graph->tvwgt[0] - graph->pwgts[0]) < 3 * graph->tvwgt[0] / graph->nvtxs)
      return;

    // printf("execute 2WayBalance\n");
    if (graph->nbnd > 0)
    {
        // printf("Bnd2WayBalance begin\n");
        Bnd2WayBalance(graph, ntpwgts);
        // printf("Bnd2WayBalance end\n");
    }
    else
    {
        // printf("General2WayBalance begin\n");
        General2WayBalance(graph, ntpwgts);
    }
}

/*************************************************************************/
/*! This function checks if the partition weights are within the balance
contraints */
/*************************************************************************/
Hunyuan_int_t IsBalancedKway(graph_t *graph, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor, Hunyuan_real_t ffactor)
{
    return 
        (ComputeLoadImbalanceDiff(graph, nparts, tpwgts,  balance_factor[0]) <= ffactor);
}

#endif