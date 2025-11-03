#ifndef REFINE_H
#define REFINE_H

#include "struct.h"
#include "graph.h"
#include "memory.h"
#include "queue.h"

void Compute_Partition_Informetion_2way(graph_t *graph)
{
    Hunyuan_int_t nvtxs, nbnd, mincut;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndptr, *bndind, *ed, *id, *pwgts;

    nvtxs = graph->nvtxs;
    nbnd  = graph->nbnd;

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

    mincut = 0;

    //  init nbnd, bndptr and bndind
    nbnd = init_queue(nbnd, bndptr, nvtxs);

    //  init pwgts
    set_value_int(2,0,pwgts);

    //  compute array nbnd, bndptr, bndind, ed, id
    for(Hunyuan_int_t i = 0;i < nvtxs;i++)
    {
        Hunyuan_int_t partition = where[i];
        Hunyuan_int_t ted = 0;
        Hunyuan_int_t tid = 0;
        for(Hunyuan_int_t j = xadj[i];j < xadj[i + 1];j++)
        {
            Hunyuan_int_t k = adjncy[j];
            if(partition != where[k])
                ted += adjwgt[j];
            else 
                tid += adjwgt[j];
        }

        // printf("i=%d flag_boundary=%d\n",i,flag_boundary);

        if(ted > 0 || xadj[i] == xadj[i + 1])
        {
            // printf("i=%d flag_boundary=%d\n",i,flag_boundary);
            nbnd = insert_queue(nbnd, bndptr, bndind, i);
            mincut += ted;
        }

        ed[i] = ted;
        id[i] = tid;
        pwgts[partition] += vwgt[i];
    }

    // exam_where(graph);
    // exam_edid(graph);
    // exam_pwgts(graph);

    graph->nbnd   = nbnd;
    graph->mincut = mincut / 2;

    // exam_bnd(graph);
}

void Compute_Reorder_Informetion_2way(graph_t *graph)
{
    Hunyuan_int_t nvtxs, nbnd, mincut;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *where, *bndptr, *bndind, *pwgts;
    nrinfo_t *nrinfo;

    nvtxs = graph->nvtxs;
    nbnd  = graph->nbnd;

    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    
    where  = graph->where;
    bndptr = graph->bndptr;
    bndind = graph->bndind;
    nrinfo = graph->nrinfo;
    pwgts  = graph->pwgts;

    //  init nbnd, bndptr and bndind
    nbnd = init_queue(nbnd, bndptr, nvtxs);

    //  init pwgts
    set_value_int(3,0,pwgts);

    //  compute array nbnd, bndptr, bndind, ed, id
    for(Hunyuan_int_t i = 0;i < nvtxs;i++)
    {
        Hunyuan_int_t partition = where[i];

        pwgts[partition] += vwgt[i];

        if(partition == 2)
        {
            nbnd = insert_queue(nbnd, bndptr, bndind, i);
            nrinfo[i].edegrees[0] = nrinfo[i].edegrees[1] = 0;

            for(Hunyuan_int_t j = xadj[i];j < xadj[i + 1];j++)
            {
                Hunyuan_int_t k = adjncy[j];
                Hunyuan_int_t other = where[k];
                if(other != 2)
                    nrinfo[i].edegrees[other] += vwgt[k];
            }
        }
    }

    graph->nbnd   = nbnd;
    graph->mincut = pwgts[2];

    // exam_nrinfo(graph);
    // exam_pwgts(graph);
    // exam_bnd(graph);
}

/*************************************************************************/
/*! This version of the main idea of the Refine_Partition_2way
		*1 moving first, then rebalance
*/
/*************************************************************************/
void Refine_Partition_2way_new(graph_t *graph)
{
    Hunyuan_int_t nvtxs, nbnd, mincut, from, to, step, beststate, bestvalue;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndptr, *bndind, *ed, *id, *pwgts;
    Hunyuan_int_t *moved, *record;
    double *dst_pwgts;
    priority_queue_t *queue[2];

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

    moved  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: moved");
    record = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: record");
    set_value_int(nvtxs,-1,moved);

    //  Set up and init the priority queue
    queue[0] = priority_queue_Create(nvtxs);
    queue[1] = priority_queue_Create(nvtxs);
    for(Hunyuan_int_t i = 0;i < nbnd;i++)
    {
        Hunyuan_int_t vertex = bndind[i];
        priority_queue_Insert(queue[where[vertex]],vertex, ed[vertex] - id[vertex]);
    }

    step = -1;
    bestvalue = mincut;
    beststate = -1;
    while(1)
    {
        //  begin to refine
        //  determine the vertex to moving
        if(priority_queue_Length(queue[0]) != 0)
        {
            if(priority_queue_Length(queue[1]) != 0)
            {
                if(priority_queue_SeeTopKey(queue[0]) >= priority_queue_SeeTopKey(queue[1]))
                    from = 0, to = 1;
                else 
                    from = 1, to = 0;
            }
            else 
                from = 0, to = 1;
        }
        else 
        {
            if(priority_queue_Length(queue[1]) != 0)
                from = 1, to = 0;
            else 
                break ;
        }

        //  fetch vertex
        Hunyuan_int_t vertex = priority_queue_GetTop(queue[from]);

        //  end condition
        if(ed[vertex] - id[vertex] >= (double) (0.25 * ed[vertex]))
            break;

        //  update the information of the vertex
        step++;
        where[vertex] = to;
        pwgts[from] -= vwgt[vertex];
        pwgts[to]   += vwgt[vertex];
        Hunyuan_int_t z;
        lyj_swap(ed[vertex], id[vertex], z);
        mincut -= ed[vertex] - id[vertex];
        if(ed[vertex] == 0 && xadj[vertex + 1] < xadj[vertex])
            nbnd = delete_queue(nbnd, bndptr, bndind, vertex);
        moved[vertex] = step;
        record[step]  = vertex;
        if(mincut < bestvalue)
            beststate = step, bestvalue = mincut;

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
                    if(moved[k] == -1)
                        priority_queue_Delete(queue[where[k]], k);
                }
                //  if the vertex k still is a boundary vertex now
                else
                {
                    //  update the queue
                    if(moved[k] == -1)
                        priority_queue_Update(queue[where[k]], k, ed[k] - id[k]);
                }

            }
            //  if the vertex k isn't a boundary vertex
            else
            {
                //  if the vertex become a boundary vertex now
                if(ed[k] > 0 &&  xadj[k + 1] - xadj[k] > 0)
                {
                    nbnd = insert_queue(nbnd, bndptr, bndind, k);
                    if(moved[k] == -1)
                        priority_queue_Insert(queue[where[k]],k,ed[k] - id[k]);
                }
            }
        }
    }

    //  roll back
    if(bestvalue < step && beststate != -1)
    {
        while(step > beststate)
        {
            //  fetch vertex
            Hunyuan_int_t vertex = record[step];
            from = where[vertex];
            to = (from + 1) % 2;

            //  update the information of the vertex
            where[vertex] = to;
            pwgts[from] -= vwgt[vertex];
            pwgts[to]   += vwgt[vertex];
            Hunyuan_int_t z;
            lyj_swap(ed[vertex], id[vertex], z);
            mincut -= ed[vertex] - id[vertex];
            if(ed[vertex] == 0 && bndptr[vertex] != -1 && xadj[vertex + 1] < xadj[vertex])
                nbnd = delete_queue(nbnd, bndptr, bndind, vertex);
            else if(ed[vertex] > 0 && bndptr[vertex] == -1)
                nbnd = insert_queue(nbnd,bndptr,bndind,vertex);

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

                //  update boundary queue
                if(bndptr[k] != -1 && ed[k] == 0)
                    nbnd = delete_queue(nbnd, bndptr, bndind, k);
                if(bndptr[k] == -1 && ed[k] > 0)
                    nbnd = insert_queue(nbnd, bndptr, bndind, k);
            }

            step--;
        }
    }

    priority_queue_Destroy(queue[0]);
    priority_queue_Destroy(queue[1]);

    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way_new: moved");
    check_free(record, sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way_new: record");

    graph->mincut = mincut;
    graph->nbnd   = nbnd;
}

void Refine_Partition_2way(graph_t *graph)
{
    Hunyuan_int_t nvtxs, nbnd, mincut, from, to, step, beststate, bestvalue;
    Hunyuan_int_t limit, avgvwgt, origdiff, mindiff, mincutstep, newcut;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndptr, *bndind, *ed, *id, *pwgts;
    Hunyuan_int_t *moved, *record;
    double *dst_pwgts;
    priority_queue_t *queue[2];

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

    moved  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: moved");
    record = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: record");
    set_value_int(nvtxs,-1,moved);

    dst_pwgts = (double *)check_malloc(sizeof(double) * 2, "Refine_Partition_2way: dst_pwgts");
    set_value_double(2,(double)(0.5 * graph->tvwgt[0]),dst_pwgts);

    limit   = lyj_min(lyj_max(0.01 * nvtxs, 15), 100);
    avgvwgt = lyj_min((pwgts[0] + pwgts[1]) / 20, 2 * (pwgts[0] + pwgts[1]) / nvtxs);

    //  Set up and init the priority queue
    queue[0] = priority_queue_Create(nvtxs);
    queue[1] = priority_queue_Create(nvtxs);
    for(Hunyuan_int_t i = 0;i < nbnd;i++)
    {
        Hunyuan_int_t vertex = bndind[i];
        priority_queue_Insert(queue[where[vertex]],vertex, ed[vertex] - id[vertex]);
    }

    // priority_queue_exam(queue[0]);
    // priority_queue_exam(queue[1]);

    step = 0;
    mincutstep = -1;
    origdiff = lyj_abs(dst_pwgts[0] - pwgts[0]);
    mindiff = lyj_abs(dst_pwgts[0] - pwgts[0]);

    printf("Refine_Partition_2way 0\n");
    //  Perform one refinement sessions
    while(step < nvtxs)
    {
        from = (dst_pwgts[0] - pwgts[0] < dst_pwgts[1] - pwgts[1] ? 0 : 1);
        to   = (from + 1) % 2;

        // priority_queue_exam(queue[0]);
        // priority_queue_exam(queue[1]);

        if(priority_queue_Length(queue[from]) == 0)
            break;
        
        Hunyuan_int_t vertex = priority_queue_GetTop(queue[from]);
        newcut -= (ed[vertex] - id[vertex]);
        pwgts[from] -= vwgt[vertex];
        pwgts[to] += vwgt[vertex];

        // printf("from=%d to=%d vertex=%d\n",from,to,vertex);

        //  determine the vertex to moving
        if((newcut < mincut && lyj_abs(dst_pwgts[0] - pwgts[0]) <= origdiff + avgvwgt) ||
           (newcut == mincut && lyj_abs(dst_pwgts[0] - pwgts[0]) < mindiff))
        {
            mincut = newcut;
            mindiff = lyj_abs(dst_pwgts[0] - pwgts[0]);
            mincutstep = step;
        }
        else if(step - mincutstep > limit)
        {
            newcut += (ed[vertex] - id[vertex]);
            pwgts[from] += vwgt[vertex];
            pwgts[to] -= vwgt[vertex];
            break;
        }

        where[vertex] = to;
        moved[vertex] = step;
        record[step]  = vertex;
        Hunyuan_int_t z;
        lyj_swap(ed[vertex], id[vertex], z);
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
                    if(moved[k] == -1)
                        priority_queue_Delete(queue[where[k]], k);
                }
                //  if the vertex k still is a boundary vertex now
                else
                {
                    //  update the queue
                    if(moved[k] == -1)
                        priority_queue_Update(queue[where[k]], k, ed[k] - id[k]);
                }

            }
            //  if the vertex k isn't a boundary vertex
            else
            {
                //  if the vertex become a boundary vertex now
                if(ed[k] > 0 &&  xadj[k + 1] - xadj[k] > 0)
                {
                    nbnd = insert_queue(nbnd, bndptr, bndind, k);
                    if(moved[k] == -1)
                        priority_queue_Insert(queue[where[k]],k,ed[k] - id[k]);
                }
            }
        }
        step++;
    }

    printf("Refine_Partition_2way 1\n");
    //  roll back
    step--;
    if(bestvalue < step && beststate != -1)
    {
        while(step > beststate)
        {
            //  fetch vertex
            Hunyuan_int_t vertex = record[step];
            from = where[vertex];
            to = (from + 1) % 2;

            //  update the information of the vertex
            where[vertex] = to;
            pwgts[from] -= vwgt[vertex];
            pwgts[to]   += vwgt[vertex];
            Hunyuan_int_t z;
            lyj_swap(ed[vertex], id[vertex], z);
            if(ed[vertex] == 0 && bndptr[vertex] != -1 && xadj[vertex + 1] < xadj[vertex])
                nbnd = delete_queue(nbnd, bndptr, bndind, vertex);
            else if(ed[vertex] > 0 && bndptr[vertex] == -1)
                nbnd = insert_queue(nbnd,bndptr,bndind,vertex);

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

                //  update boundary queue
                if(bndptr[k] != -1 && ed[k] == 0)
                    nbnd = delete_queue(nbnd, bndptr, bndind, k);
                if(bndptr[k] == -1 && ed[k] > 0)
                    nbnd = insert_queue(nbnd, bndptr, bndind, k);
            }

            step--;
        }
    }

    printf("Refine_Partition_2way 2\n");
    priority_queue_Destroy(queue[0]);
    priority_queue_Destroy(queue[1]);

    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: moved");
    check_free(record, sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: record");
    check_free(dst_pwgts, sizeof(double) * 2, "Refine_Partition_2way: dst_pwgts");

    graph->mincut = mincut;
    graph->nbnd   = nbnd;

    // exam_where(graph);
    // exam_edid(graph);
    // exam_pwgts(graph);
    // exam_bnd(graph);
}

void Refine_Reorder_2way(graph_t *graph)
{
    Hunyuan_int_t nvtxs, nbnd, mincut, to, other, process, step, oldgain, beststate, bestvalue;
    Hunyuan_int_t limit, maxpwgt, origdiff, newdiff, mindiff, mincutstep, newcut;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndptr, *bndind, *ed, *id, *pwgts;
    nrinfo_t *nrinfo;
    Hunyuan_int_t *moved, *record, *mind, *mptr, nmind;
    priority_queue_t *queue[2];

    nvtxs  = graph->nvtxs;
    nbnd   = graph->nbnd;
    mincut = graph->mincut;

    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    // adjwgt = graph->adjwgt;
    where  = graph->where;
    bndptr = graph->bndptr;
    bndind = graph->bndind;
    // ed     = graph->ed;
    // id     = graph->id;
    pwgts  = graph->pwgts;
    nrinfo = graph->nrinfo;

    moved  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: moved");
    record = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: record");
    mind   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs * 2, "Refine_Partition_2way: mind");
    mptr   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (nvtxs + 1), "Refine_Partition_2way: mptr");
    set_value_int(nvtxs,-1,moved);

    maxpwgt = 0.6 * graph->tvwgt[0];
    process = 0;

    // limit   = (ctrl->compress ? gk_min(5*nbnd, 400) : gk_min(2*nbnd, 300));
    limit = lyj_min(5 * nbnd, 400);

    //  Set up and init the priority queue
    queue[0] = priority_queue_Create(nvtxs);
    queue[1] = priority_queue_Create(nvtxs);
    for(Hunyuan_int_t i = 0;i < nbnd;i++)
    {
        Hunyuan_int_t vertex = bndind[i];
        if(where[vertex] == 2)
        {
            priority_queue_Insert(queue[0],vertex, vwgt[vertex] - nrinfo[vertex].edegrees[1]);
            priority_queue_Insert(queue[1],vertex, vwgt[vertex] - nrinfo[vertex].edegrees[0]);
        }
    }

    // priority_queue_exam(queue[0]);
    // priority_queue_exam(queue[1]);

    step = 0;
    mincutstep = -1;
    mindiff = lyj_abs(pwgts[0] - pwgts[1]);
    mptr[0] = 0;
    nmind = 0;
    to = (pwgts[0] < pwgts[1] ? 0 : 1);

    printf("Refine_Partition_2way 0\n");
    while(step < nvtxs)
    {
        Hunyuan_int_t vertexs[2];
        vertexs[0] = priority_queue_SeeTopVal(queue[0]);
        vertexs[1] = priority_queue_SeeTopVal(queue[1]);
        if(vertexs[0] != -1 && vertexs[1] != -1)
        {
            Hunyuan_int_t gain[2];
            gain[0] = vwgt[vertexs[0]] - nrinfo[vertexs[0]].edegrees[1];
            gain[1] = vwgt[vertexs[1]] - nrinfo[vertexs[1]].edegrees[0];
            to = (gain[0] > gain[1] ? 0 : (gain[0] < gain[1] ? 1 : process % 2));
            
            if(pwgts[to] + vwgt[vertexs[to]] > maxpwgt)
                to = (to + 1) % 2;
        }
        else if(vertexs[0] == -1 && vertexs[1] == -1)
            break;
        else if(vertexs[0] != -1 && pwgts[0]+vwgt[vertexs[0]] <= maxpwgt)
            to = 0;
        else if(vertexs[1] != -1 && pwgts[1]+vwgt[vertexs[1]] <= maxpwgt)
            to = 1;
        else 
            break;
        
        other = (to + 1) % 2;
        Hunyuan_int_t vertex = priority_queue_GetTop(queue[to]);
        if(moved[vertex] == -1)
            priority_queue_Delete(queue[other],vertex);
        
        if (nmind + xadj[vertex + 1]-xadj[vertex] >= 2 * nvtxs - 1) 
            break;
        
        pwgts[2] -= (vwgt[vertex] - nrinfo[vertex].edegrees[other]);
        
        newdiff = lyj_abs(pwgts[to] + vwgt[vertex] - (pwgts[other] - nrinfo[vertex].edegrees[other]));
        if(pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff))
        {
            // printf("update mincutstep\n");
            mincut = pwgts[2];
            mincutstep = step;
            mindiff = newdiff;
        }
        else
        {
            if(step - mincutstep > 2 * limit || 
                (step - mincutstep > limit && pwgts[2] > 1.1 * mincut))
            {
                pwgts[2] += (vwgt[vertex] - nrinfo[vertex].edegrees[other]);
                break;
            }
        }

        nbnd = delete_queue(nbnd,bndptr,bndind,vertex);
        pwgts[to] += vwgt[vertex];
        where[vertex] = to;
        moved[vertex] = step;
        record[step] = vertex;

        //  update the vertex's adjacent vertices
        for(Hunyuan_int_t j = xadj[vertex];j < xadj[vertex + 1];j++)
        {
            Hunyuan_int_t k = adjncy[j];

            //  the array ed and id of k
            if(where[k] == 2)
            {
                oldgain = vwgt[k] - nrinfo[k].edegrees[to];
                nrinfo[k].edegrees[to] += vwgt[vertex];
                if(moved[k] == -1 || moved[k] == -(2 + other))
                    priority_queue_Update(queue[other],k,oldgain - vwgt[vertex]);
            }
            else if(where[k] == other)
            {
                nbnd = insert_queue(nbnd,bndptr,bndind,k);
                
                mind[nmind] = k;
                nmind++;
                where[k] = 2;
                pwgts[other] -= vwgt[k];

                nrinfo[k].edegrees[0] = nrinfo[k].edegrees[1] = 0;
                for(Hunyuan_int_t jj = xadj[k];jj < xadj[k + 1];jj++)
                {
                    Hunyuan_int_t kk = adjncy[jj];
                    if(where[kk] != 2)
                        nrinfo[k].edegrees[where[kk]] += vwgt[kk];
                    else
                    {
                        oldgain = vwgt[kk] - nrinfo[kk].edegrees[other];
                        nrinfo[kk].edegrees[other] -= vwgt[k];
                        if(moved[kk] == -1 || moved[k] == -(2 + to))
                            priority_queue_Update(queue[to],kk,oldgain + vwgt[k]);
                    }
                }

                if(moved[k] == -1)
                {
                    priority_queue_Insert(queue[to],k,vwgt[k] - nrinfo[k].edegrees[other]);
                    moved[k] = -(2 + to);
                }
            }
        }

        mptr[step + 1] = nmind;
        // printf("other=%d to=%d vertex=%d step=%d\n",other,to,vertex,step);
        // priority_queue_exam(queue[0]);
        // priority_queue_exam(queue[1]);
        // exam_where(graph);
        // exam_pwgts(graph);
        step++;
    }

    printf("Refine_Partition_2way 1\n");
    // printf("step=%d mincutstep=%d\n",step,mincutstep);
    //  roll back
    step--;
    while (step > mincutstep)
    {
        Hunyuan_int_t vertex = record[step];

        to  = where[vertex];
        other = (to + 1) % 2;
        pwgts[2] += vwgt[vertex];
        pwgts[to] -= vwgt[vertex];
        where[vertex] = 2;
        nbnd = insert_queue(nbnd,bndptr,bndind,vertex);
        nrinfo[vertex].edegrees[0] = nrinfo[vertex].edegrees[1] = 0;
        for(Hunyuan_int_t j = xadj[vertex];j < xadj[vertex + 1];j++)
        {
            Hunyuan_int_t k = adjncy[j];
            if(where[k] == 2)
                nrinfo[k].edegrees[to] -= vwgt[vertex];
            else 
                nrinfo[vertex].edegrees[where[k]] += vwgt[k];
        }

        for(Hunyuan_int_t j = mptr[step];j < mptr[step + 1];j++)
        {
            Hunyuan_int_t k = mind[j];
            where[k] = other;
            pwgts[other] += vwgt[k];
            pwgts[2] -= vwgt[k];
            nbnd = delete_queue(nbnd,bndptr,bndind,k);
            for(Hunyuan_int_t jj = xadj[k];jj < xadj[k + 1];jj++)
            {
                Hunyuan_int_t kk = adjncy[jj];
                if(where[kk] == 2)
                    nrinfo[kk].edegrees[other] += vwgt[kk];
            }
        }

        step--;
        // printf("other=%d to=%d vertex=%d step=%d\n",other,to,vertex,step);
        // exam_where(graph);
        // exam_pwgts(graph);
    }

    printf("Refine_Partition_2way 2\n");
    graph->mincut = mincut;
    graph->nbnd = nbnd;
    
    priority_queue_Destroy(queue[0]);
    priority_queue_Destroy(queue[1]);

    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "Refine_Reorder_2way: moved");
    check_free(record, sizeof(Hunyuan_int_t) * nvtxs, "Refine_Reorder_2way: record");
    check_free(mind, sizeof(Hunyuan_int_t) * nvtxs * 2, "Refine_Reorder_2way: mind");
    check_free(mptr, sizeof(Hunyuan_int_t) * (nvtxs + 1), "Refine_Reorder_2way: mptr");
}

void Refine_Reorder_1way(graph_t *graph)
{
    Hunyuan_int_t nvtxs, nbnd, mincut, to, other, process, step, oldgain, beststate, bestvalue;
    Hunyuan_int_t limit, maxpwgt, origdiff, newdiff, mindiff, mincutstep, newcut;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndptr, *bndind, *ed, *id, *pwgts;
    nrinfo_t *nrinfo;
    Hunyuan_int_t *moved, *record, *mind, *mptr, nmind;
    priority_queue_t *queue;

    nvtxs  = graph->nvtxs;
    nbnd   = graph->nbnd;
    mincut = graph->mincut;

    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    // adjwgt = graph->adjwgt;
    where  = graph->where;
    bndptr = graph->bndptr;
    bndind = graph->bndind;
    // ed     = graph->ed;
    // id     = graph->id;
    pwgts  = graph->pwgts;
    nrinfo = graph->nrinfo;

    moved  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: moved");
    record = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: record");
    mind   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs * 2, "Refine_Partition_2way: mind");
    mptr   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (nvtxs + 1), "Refine_Partition_2way: mptr");
    set_value_int(nvtxs,-1,moved);

    maxpwgt = 0.6 * graph->tvwgt[0];
    process = 0;
    to = (pwgts[0] < pwgts[1] ? 0 : 1);
    other = (to + 1) % 2;

    // limit   = (ctrl->compress ? gk_min(5*nbnd, 400) : gk_min(2*nbnd, 300));
    limit = lyj_min(5 * nbnd, 400);

    //  Set up and init the priority queue
    queue = priority_queue_Create(nvtxs);
    for(Hunyuan_int_t i = 0;i < nbnd;i++)
    {
        Hunyuan_int_t vertex = bndind[i];
        if(where[vertex] == 2)
            priority_queue_Insert(queue,vertex, vwgt[vertex] - nrinfo[vertex].edegrees[other]);
    }

    // priority_queue_exam(queue);

    step = 0;
    mincutstep = -1;
    mindiff = lyj_abs(pwgts[0] - pwgts[1]);
    mptr[0] = 0;
    nmind = 0;
    printf("Refine_Reorder_1way 0\n");
    while(step < nvtxs)
    {
        Hunyuan_int_t vertex = priority_queue_GetTop(queue);
        if(vertex == -1)
            break;
        
        if (nmind + xadj[vertex + 1]-xadj[vertex] >= 2 * nvtxs - 1) 
            break;
        
        pwgts[2] -= (vwgt[vertex] - nrinfo[vertex].edegrees[other]);
        
        newdiff = lyj_abs(pwgts[to] + vwgt[vertex] - (pwgts[other] - nrinfo[vertex].edegrees[other]));
        if(pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff))
        {
            // printf("update mincutstep\n");
            mincut = pwgts[2];
            mincutstep = step;
            mindiff = newdiff;
        }
        else
        {
            if(step - mincutstep > 3 * limit || 
                (step - mincutstep > limit && pwgts[2] > 1.1 * mincut))
            {
                pwgts[2] += (vwgt[vertex] - nrinfo[vertex].edegrees[other]);
                break;
            }
        }

        nbnd = delete_queue(nbnd,bndptr,bndind,vertex);
        pwgts[to] += vwgt[vertex];
        where[vertex] = to;
        moved[vertex] = step;
        record[step] = vertex;

        //  update the vertex's adjacent vertices
        for(Hunyuan_int_t j = xadj[vertex];j < xadj[vertex + 1];j++)
        {
            Hunyuan_int_t k = adjncy[j];

            //  the array ed and id of k
            if(where[k] == 2)
                nrinfo[k].edegrees[to] += vwgt[vertex];
            else if(where[k] == other)
            {
                nbnd = insert_queue(nbnd,bndptr,bndind,k);
                
                mind[nmind] = k;
                nmind++;
                where[k] = 2;
                pwgts[other] -= vwgt[k];

                nrinfo[k].edegrees[0] = nrinfo[k].edegrees[1] = 0;
                for(Hunyuan_int_t jj = xadj[k];jj < xadj[k + 1];jj++)
                {
                    Hunyuan_int_t kk = adjncy[jj];
                    if(where[kk] != 2)
                        nrinfo[k].edegrees[where[kk]] += vwgt[kk];
                    else
                    {
                        nrinfo[kk].edegrees[other] -= vwgt[k];
                        priority_queue_Update(queue,kk,vwgt[kk] -  nrinfo[kk].edegrees[other]);
                    }
                }
                priority_queue_Insert(queue,k,vwgt[k] - nrinfo[k].edegrees[other]);
            }
        }

        mptr[step + 1] = nmind;

        // printf("other=%d to=%d vertex=%d step=%d\n",other,to,vertex,step);
        // priority_queue_exam(queue);
        // priority_queue_exam(queue);
        // exam_where(graph);
        // exam_pwgts(graph);
        step++;
    }

    // printf("step=%d mincutstep=%d\n",step,mincutstep);
    printf("Refine_Reorder_1way 1\n");
    //  roll back
    step--;
    while (step > mincutstep)
    {
        Hunyuan_int_t vertex = record[step];

        pwgts[2] += vwgt[vertex];
        pwgts[to] -= vwgt[vertex];
        where[vertex] = 2;
        nbnd = insert_queue(nbnd,bndptr,bndind,vertex);
        
        nrinfo[vertex].edegrees[0] = nrinfo[vertex].edegrees[1] = 0;
        for(Hunyuan_int_t j = xadj[vertex];j < xadj[vertex + 1];j++)
        {
            Hunyuan_int_t k = adjncy[j];
            if(where[k] == 2)
                nrinfo[k].edegrees[to] -= vwgt[vertex];
            else 
                nrinfo[vertex].edegrees[where[k]] += vwgt[k];
        }

        for(Hunyuan_int_t j = mptr[step];j < mptr[step + 1];j++)
        {
            Hunyuan_int_t k = mind[j];
            where[k] = other;
            pwgts[other] += vwgt[k];
            pwgts[2] -= vwgt[k];
            nbnd = delete_queue(nbnd,bndptr,bndind,k);
            for(Hunyuan_int_t jj = xadj[k];jj < xadj[k + 1];jj++)
            {
                Hunyuan_int_t kk = adjncy[jj];
                if(where[kk] == 2)
                    nrinfo[kk].edegrees[other] += vwgt[kk];
            }
        }

        // printf("other=%d to=%d vertex=%d step=%d\n",other,to,vertex,step);
        // exam_where(graph);
        // exam_pwgts(graph);

        step--;
    }

    printf("Refine_Reorder_1way 2\n");
    graph->mincut = mincut;
    graph->nbnd = nbnd;
    
    priority_queue_Destroy(queue);

    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "Refine_Reorder_1way: moved");
    check_free(record, sizeof(Hunyuan_int_t) * nvtxs, "Refine_Reorder_1way: record");
    check_free(mind, sizeof(Hunyuan_int_t) * nvtxs * 2, "Refine_Reorder_1way: mind");
    check_free(mptr, sizeof(Hunyuan_int_t) * (nvtxs + 1), "Refine_Reorder_1way: mptr");
}

void project_Reorder(graph_t *graph)
{
    Hunyuan_int_t nvtxs, *cmap, *where, *cwhere;
    graph_t *cgraph = graph->coarser;

    nvtxs  = graph->nvtxs;
    cmap   = graph->cmap;
    cwhere = cgraph->where;

    // printf("project_Reorder 0\n");
    where  = graph->where;
    for(Hunyuan_int_t i = 0;i < nvtxs;i++)
        where[i] = cwhere[cmap[i]];
    
    // printf("project_Reorder 1\n");
    
    FreeGraph(&graph->coarser, 3);
    graph->coarser = NULL;

    graph->pwgts  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 3, "project_Reorder: pwgts");
    graph->bndptr = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "project_Reorder: bndptr");
    graph->bndind = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "project_Reorder: bndind");
    graph->nrinfo = (nrinfo_t *)check_malloc(sizeof(nrinfo_t) * nvtxs, "project_Reorder: nrinfo");


    // printf("project_Reorder 2\n");
    Compute_Reorder_Informetion_2way(graph);
    // printf("project_Reorder 3\n");
}

void Project_2WayPartition(graph_t *graph)
{
    Hunyuan_int_t nvtxs, *cmap, *where, *cwhere;
    graph_t *cgraph = graph->coarser;

    nvtxs  = graph->nvtxs;
    cmap   = graph->cmap;
    cwhere = cgraph->where;

    graph->where  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Project_2WayPartition: where");
    graph->pwgts  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * 2, "Project_2WayPartition: pwgts");
    graph->bndptr = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Project_2WayPartition: bndptr");
    graph->bndind = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Project_2WayPartition: bndind");
    graph->ed     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Project_2WayPartition: ed");
    graph->id     = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Project_2WayPartition: id");

    where  = graph->where;
    for(Hunyuan_int_t i = 0;i < nvtxs;i++)
        where[i] = cwhere[cmap[i]];

    FreeGraph(&graph->coarser, 2);
    graph->coarser = NULL;

    Compute_Partition_Informetion_2way(graph);
}

void Reorder_Balance(graph_t *graph)
{
    Hunyuan_int_t nvtxs, nbnd, mincut, to, other, process, step, oldgain, beststate, bestvalue;
    Hunyuan_int_t maxpwgt, gain;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndptr, *bndind, *ed, *id, *pwgts;
    nrinfo_t *nrinfo;
    Hunyuan_int_t *moved;
    priority_queue_t *queue;

    nvtxs  = graph->nvtxs;
    nbnd   = graph->nbnd;
    mincut = graph->mincut;

    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    // adjwgt = graph->adjwgt;
    where  = graph->where;
    bndptr = graph->bndptr;
    bndind = graph->bndind;
    // ed     = graph->ed;
    // id     = graph->id;
    pwgts  = graph->pwgts;
    nrinfo = graph->nrinfo;

    maxpwgt = 0.6 * (pwgts[0] + pwgts[1]);

    printf("Reorder_Balance 0\n");
    //  make sure it is balanced 
    if(lyj_max(pwgts[0], pwgts[1]) < maxpwgt)
        return ;
    if(lyj_abs(pwgts[0] - pwgts[1]) < 3 * graph->tvwgt[0] / nvtxs)
        return ;

    moved  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Refine_Partition_2way: moved");
    set_value_int(nvtxs,-1,moved);

    process = 0;
    to = (pwgts[0] < pwgts[1] ? 0 : 1);
    other = (to + 1) % 2;

    //  Set up and init the priority queue
    queue = priority_queue_Create(nvtxs);
    for(Hunyuan_int_t i = 0;i < nbnd;i++)
    {
        Hunyuan_int_t vertex = bndind[i];
        if(where[vertex] == 2)
            priority_queue_Insert(queue,vertex, vwgt[vertex] - nrinfo[vertex].edegrees[other]);
    }

    exam_priority_queue(queue);

    step = 0;
    printf("Reorder_Balance 1\n");
    while(step < nvtxs)
    {
        Hunyuan_int_t vertex = priority_queue_GetTop(queue);
        if(vertex == -1)
            break;
        
        moved[vertex] = 1;
        gain = vwgt[vertex] - nrinfo[vertex].edegrees[other];
        maxpwgt = 0.6 * (pwgts[0] + pwgts[1]);

        //  already balance
        if(pwgts[to] > pwgts[other])
            break;
        if(gain < 0 && pwgts[other] < maxpwgt)
            break;
        if(pwgts[to] + vwgt[vertex] > maxpwgt)
            continue;
        
        pwgts[2] -= gain;

        nbnd = delete_queue(nbnd,bndptr,bndind,vertex);
        pwgts[to] += vwgt[vertex];
        where[vertex] = to;
        moved[vertex] = step;

        //  update the vertex's adjacent vertices
        for(Hunyuan_int_t j = xadj[vertex];j < xadj[vertex + 1];j++)
        {
            Hunyuan_int_t k = adjncy[j];

            //  the array ed and id of k
            if(where[k] == 2)
                nrinfo[k].edegrees[to] += vwgt[vertex];
            else if(where[k] == other)
            {
                nbnd = insert_queue(nbnd,bndptr,bndind,k);
                
                where[k] = 2;
                pwgts[other] -= vwgt[k];

                nrinfo[k].edegrees[0] = nrinfo[k].edegrees[1] = 0;
                for(Hunyuan_int_t jj = xadj[k];jj < xadj[k + 1];jj++)
                {
                    Hunyuan_int_t kk = adjncy[jj];
                    if(where[kk] != 2)
                        nrinfo[k].edegrees[where[kk]] += vwgt[kk];
                    else
                    {
                        oldgain = vwgt[kk] - nrinfo[kk].edegrees[other];
                        nrinfo[kk].edegrees[other] -= vwgt[k];

                        if(moved[kk] == -1)
                            priority_queue_Update(queue,kk,oldgain + vwgt[k]);
                    }
                }
                priority_queue_Insert(queue,k,vwgt[k] - nrinfo[k].edegrees[other]);
            }
        }

        printf("other=%"PRIDX" to=%"PRIDX" vertex=%"PRIDX" step=%"PRIDX"\n",other,to,vertex,step);
        exam_priority_queue(queue);
        exam_where(graph);
        step++;
    }

    printf("Reorder_Balance 2\n");
    graph->mincut = pwgts[2];
    graph->nbnd   = nbnd;

    priority_queue_Destroy(queue);

    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "Reorder_Balance: moved");
}

void ReorderRefinement(graph_t *graph, graph_t *origraph)
{
    if(origraph == graph)
        Compute_Reorder_Informetion_2way(graph);
    else
    {
        do
        {
            graph = graph->finer;
            printf("ReorderRefinement 0\n");
            project_Reorder(graph);
            printf("ReorderRefinement 1\n");
            Reorder_Balance(graph);
            exam_where(graph);
            printf("ReorderRefinement 2\n");
            Refine_Reorder_2way(graph);
            exam_where(graph);
            printf("ReorderRefinement 3\n");

        } while (origraph != graph);
    }
}

/*************************************************************************/
/*! This function performs a cut-focused FM refinement */
/*************************************************************************/
void FM_2WayCutRefine(graph_t *graph, Hunyuan_real_t *ntpwgts, Hunyuan_int_t niter)
{
    Hunyuan_int_t i, ii, j, k, kwgt, nvtxs, nbnd, nswaps, from, to, pass, me, limit, tmp;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind, *pwgts;
    Hunyuan_int_t *moved, *swaps, *perm;
    priority_queue_t *queues[2];
    Hunyuan_int_t higain, mincut, mindiff, origdiff, initcut, newcut, mincutorder, avgvwgt;
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

    moved = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayCutRefine: moved");
    swaps = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayCutRefine: swaps");
    perm  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayCutRefine: perm");

    tpwgts[0] = graph->tvwgt[0] * ntpwgts[0];
    tpwgts[1] = graph->tvwgt[0] - tpwgts[0];
    
    limit   = lyj_min(lyj_max(0.01 * nvtxs, 15), 100);
    avgvwgt = lyj_min((pwgts[0] + pwgts[1]) / 20, 2 * (pwgts[0] + pwgts[1]) / nvtxs);

    queues[0] = priority_queue_Create(nvtxs);
    queues[1] = priority_queue_Create(nvtxs);

    origdiff = lyj_abs(tpwgts[0] - pwgts[0]);
    set_value_int(nvtxs, -1, moved);
    // exam_pwgts(graph);
    for (pass = 0; pass < niter; pass++) 
    { 
        // printf("pass=%"PRIDX"\n",pass);
        // printf("rollback\n");
        // exam_where(graph);
        // exam_bnd(graph);
        // exam_priority_queue(queues[0]);
        // exam_priority_queue(queues[1]);
        /* Do a number of passes */
        priority_queue_Reset(queues[0]);
        priority_queue_Reset(queues[1]);

        mincutorder = -1;
        newcut = mincut = initcut = graph->mincut;
        mindiff = lyj_abs(tpwgts[0] - pwgts[0]);

        /* Insert boundary nodes in the priority queues */
        nbnd = graph->nbnd;
        // printf("pass=%"PRIDX" begin rand_count=%"PRIDX"\n",pass,rand_count());
        irandArrayPermute(nbnd, perm, nbnd, 1);
        // printf("pass=%"PRIDX" end   rand_count=%"PRIDX"\n",pass,rand_count());
        for (ii = 0; ii < nbnd; ii++) 
        {
            i = perm[ii];
            priority_queue_Insert(queues[where[bndind[i]]], bndind[i], ed[bndind[i]] - id[bndind[i]]);
        }

        // printf("reset\n");
        // exam_priority_queue(queues[0]);
        // exam_priority_queue(queues[1]);

        for (nswaps = 0; nswaps < nvtxs; nswaps++) 
        {
            from = (tpwgts[0] - pwgts[0] < tpwgts[1] - pwgts[1] ? 0 : 1);
            to = (from + 1) % 2;

            if ((higain = priority_queue_GetTop(queues[from])) == -1)
                break;
            
            // printf("higain=%"PRIDX" from=%"PRIDX" to=%"PRIDX" \n",higain,from,to);
            // exam_pwgts(graph);

            newcut -= (ed[higain] - id[higain]);
            pwgts[to] += vwgt[higain];
            pwgts[from] -= vwgt[higain];

            if ((newcut < mincut && lyj_abs(tpwgts[0] - pwgts[0]) <= origdiff + avgvwgt) || 
                (newcut == mincut && lyj_abs(tpwgts[0] - pwgts[0]) < mindiff)) 
            {
                mincut  = newcut;
                mindiff = lyj_abs(tpwgts[0] - pwgts[0]);
                mincutorder = nswaps;
                // printf("tpwgts[0] - pwgts[0]=%"PRIDX" mindiff=%"PRIDX"\n",tpwgts[0] - pwgts[0], mindiff);
                // printf("nswaps=%"PRIDX" mincutorder=%"PRIDX"\n",nswaps,mincutorder);
            }
            else if (nswaps - mincutorder > limit) 
            { 
                /* We hit the limit, undo last move */
                newcut += (ed[higain] - id[higain]);
                pwgts[from] += vwgt[higain];
                pwgts[to] -= vwgt[higain];
                // printf("nswaps-mincutorder > limit\n");
                // printf("nswaps=%"PRIDX" mincutorder=%"PRIDX" limit=%"PRIDX"\n",nswaps,mincutorder,limit);
                break;
            }

            where[higain] = to;
            moved[higain] = nswaps;
            swaps[nswaps] = higain;

            // printf("Moved %6"PRIDX" from %"PRIDX". [%3"PRIDX" %3"PRIDX"] %5"PRIDX" [%4"PRIDX" %4"PRIDX"]\n", higain, from, ed[higain]-id[higain], vwgt[higain], newcut, pwgts[0], pwgts[1]);

            // printf("Moved %6"PRIDX" from %"PRIDX". [%3"PRIDX" %3"PRIDX"] %5"PRIDX" [%4"PRIDX" %4"PRIDX"]\n", higain, from, ed[higain]-id[higain], vwgt[higain], newcut, pwgts[0], pwgts[1]);

            /**************************************************************
             * Update the id[i]/ed[i] values of the affected nodes
             ***************************************************************/
            lyj_swap(id[higain], ed[higain], tmp);
            if (ed[higain] == 0 && xadj[higain] < xadj[higain + 1]) 
                nbnd = delete_queue(nbnd, bndptr, bndind, higain);

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
                        nbnd = delete_queue(nbnd, bndptr, bndind, k);
                        if (moved[k] == -1)  /* Remove it if in the queues */
                        {
                            priority_queue_Delete(queues[where[k]], k);
                        }
                    }
                    else 
                    { 
                        /* If it has not been moved, update its position in the queue */
                        if (moved[k] == -1) 
                        {
                            priority_queue_Update(queues[where[k]], k, ed[k]-id[k]);
                        }
                    }
                }
                else 
                {
                    if (ed[k] > 0) 
                    {  
                        /* It will now become a boundary vertex */
                        nbnd = insert_queue(nbnd, bndptr, bndind, k);
                        if (moved[k] == -1) 
                        {
                            priority_queue_Insert(queues[where[k]], k, ed[k] - id[k]);
                        }
                    }
                }
            }
        }
        // printf("moved\n");
        // exam_where(graph);
        // graph->nbnd = nbnd;
        // exam_bnd(graph);
        // exam_priority_queue(queues[0]);
        // exam_priority_queue(queues[1]);

        /****************************************************************
        * Roll back computations
        *****************************************************************/
        for (i = 0; i < nswaps; i++)
            moved[swaps[i]] = -1;  /* reset moved array */
        for (nswaps--; nswaps > mincutorder; nswaps--) 
        {
            higain = swaps[nswaps];

            to = where[higain] = (where[higain] + 1) % 2;
            lyj_swap(id[higain], ed[higain], tmp);
            if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain + 1])
                nbnd = delete_queue(nbnd, bndptr, bndind, higain);
            else if (ed[higain] > 0 && bndptr[higain] == -1)
                nbnd = insert_queue(nbnd, bndptr, bndind, higain);

            pwgts[to] += vwgt[higain];
            pwgts[(to + 1) % 2] -= vwgt[higain];
            // exam_pwgts(graph);
            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];

                kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
                id[k] += kwgt;
                ed[k] -= kwgt;

                if (bndptr[k] != -1 && ed[k] == 0)
                    nbnd = delete_queue(nbnd, bndptr, bndind, k);
                if (bndptr[k] == -1 && ed[k] > 0)
                    nbnd = insert_queue(nbnd, bndptr, bndind, k);
            }
        }

        graph->mincut = mincut;
        // graph->mincut = newcut;
        graph->nbnd   = nbnd;

        if (mincutorder <= 0 || mincut == initcut)
            break;
    }

    priority_queue_Destroy(queues[1]);
    priority_queue_Destroy(queues[0]);

    check_free(perm, sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayCutRefine: perm");
    check_free(swaps, sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayCutRefine: swaps");
    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayCutRefine: moved");
}

/*************************************************************************/
/*! This function performs a node-based FM refinement */
/**************************************************************************/
void FM_2WayNodeRefine2Sided(graph_t *graph, Hunyuan_int_t niter)
{
    Hunyuan_int_t i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, nmind;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
    Hunyuan_int_t *mptr, *mind, *moved, *swaps;
    priority_queue_t *queues[2]; 
    nrinfo_t *rinfo;
    Hunyuan_int_t higain, oldgain, mincut, initcut, mincutorder;	
    Hunyuan_int_t pass, to, other, limit;
    Hunyuan_int_t badmaxpwgt, mindiff, newdiff;
    Hunyuan_int_t u[2], g[2];
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

    queues[0] = priority_queue_Create(nvtxs);
    queues[1] = priority_queue_Create(nvtxs);

    moved = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeRefine2Sided: moved");
    swaps = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeRefine2Sided: swaps");
    mptr = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (nvtxs + 1), "FM_2WayNodeRefine2Sided: mptr");
    mind = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs * 2, "FM_2WayNodeRefine2Sided: mind");

    mult = 0.5 * 1.2000499;
    badmaxpwgt = (Hunyuan_int_t)(mult * (pwgts[0] + pwgts[1] + pwgts[2]));

    // printf("Partitions-N2: [%6"PRIDX" %6"PRIDX"] Nv-Nb[%6"PRIDX" %6"PRIDX"]. ISep: %6"PRIDX"\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    for (pass = 0; pass < niter; pass++) 
    {
        // printf("pass=%"PRIDX" \n",pass);
        set_value_int(nvtxs, -1, moved);
        priority_queue_Reset(queues[0]);
        priority_queue_Reset(queues[1]);

        mincutorder = -1;
        initcut = mincut = graph->mincut;
        nbnd = graph->nbnd;

        /* use the swaps array in place of the traditional perm array to save memory */
        irandArrayPermute(nbnd, swaps, nbnd, 1);
        for (ii = 0; ii < nbnd; ii++) 
        {
            i = bndind[swaps[ii]];
            priority_queue_Insert(queues[0], i, vwgt[i]-rinfo[i].edegrees[1]);
            priority_queue_Insert(queues[1], i, vwgt[i]-rinfo[i].edegrees[0]);
        }

        limit = (0 ? lyj_min(5*nbnd, 400) : lyj_min(2*nbnd, 300));

        /******************************************************
        * Get Hunyuan_int_to the FM loop
        *******************************************************/
        mptr[0] = nmind = 0;
        mindiff = lyj_abs(pwgts[0] - pwgts[1]);
        to = (pwgts[0] < pwgts[1] ? 0 : 1);
        for (nswaps = 0; nswaps < nvtxs; nswaps++) 
        {
            u[0] = priority_queue_SeeTopVal(queues[0]);  
            u[1] = priority_queue_SeeTopVal(queues[1]);
            // printf("u[0]=%"PRIDX" u[1]=%"PRIDX"\n",u[0],u[1]);
            if (u[0] != -1 && u[1] != -1) 
            {
                g[0] = vwgt[u[0]] - rinfo[u[0]].edegrees[1];
                g[1] = vwgt[u[1]] - rinfo[u[1]].edegrees[0];

                to = (g[0] > g[1] ? 0 : (g[0] < g[1] ? 1 : pass % 2)); 
                
                if (pwgts[to] + vwgt[u[to]] > badmaxpwgt) 
                    to = (to + 1) % 2;
            }
            else if (u[0] == -1 && u[1] == -1) 
                break;
            else if (u[0] != -1 && pwgts[0] + vwgt[u[0]] <= badmaxpwgt)
                to = 0;
            else if (u[1] != -1 && pwgts[1] + vwgt[u[1]] <= badmaxpwgt)
                to = 1;
            else
                break;

            other = (to+1)%2;

            higain = priority_queue_GetTop(queues[to]);
            if (moved[higain] == -1) /* Delete if it was in the separator originally */
                priority_queue_Delete(queues[other], higain);

            /* The following check is to ensure we break out if there is a posibility
                of over-running the mind array.  */
            if (nmind + xadj[higain + 1] - xadj[higain] >= 2 * nvtxs - 1) 
                break;

            pwgts[2] -= (vwgt[higain] - rinfo[higain].edegrees[other]);

            newdiff = lyj_abs(pwgts[to] + vwgt[higain] - (pwgts[other] - rinfo[higain].edegrees[other]));
            if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) 
            {
                mincut = pwgts[2];
                mincutorder = nswaps;
                mindiff = newdiff;
                // printf("nswaps=%"PRIDX" mincutorder=%"PRIDX" pwgts[2]=%"PRIDX" mincut=%"PRIDX" mindiff=%"PRIDX"\n",nswaps,mincutorder,pwgts[2],mincut,mindiff);
                
            }
            else 
            {
                if (nswaps - mincutorder > 2 * limit || 
                    (nswaps - mincutorder > limit && pwgts[2] > 1.10 * mincut)) 
                {
                    pwgts[2] += (vwgt[higain] - rinfo[higain].edegrees[other]);
                    // printf("nswaps-mincutorder > 2 * limit=%d || nswaps - mincutorder > limit && pwgts[2] > 1.10 * mincut=%d\n",\
                    //     nswaps-mincutorder > 2 * limit,nswaps - mincutorder > limit && pwgts[2] > 1.10 * mincut);
                    // printf("nswaps=%"PRIDX" mincutorder=%"PRIDX" limit=%"PRIDX" pwgts[2]=%"PRIDX" mincut=%"PRIDX"\n",nswaps,mincutorder,limit,pwgts[2],mincut);
                
                    break; /* No further improvement, break out */
                }
            }

            nbnd = delete_queue(nbnd,bndptr,bndind,higain);
            pwgts[to] += vwgt[higain];
            where[higain] = to;
            moved[higain] = nswaps;
            swaps[nswaps] = higain;  

            /**********************************************************
             * Update the degrees of the affected nodes
             ***********************************************************/
            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];
                if (where[k] == 2) 
                { 
                    /* For the in-separator vertices modify their edegree[to] */
                    oldgain = vwgt[k] - rinfo[k].edegrees[to];
                    rinfo[k].edegrees[to] += vwgt[higain];
                    if (moved[k] == -1 || moved[k] == -(2 + other))
                        priority_queue_Update(queues[other], k, oldgain - vwgt[higain]);
                }
                else if (where[k] == other) 
                { 
                    /* This vertex is pulled Hunyuan_int_to the separator */
                    nbnd = insert_queue(nbnd,bndptr,bndind,k);

                    mind[nmind++] = k;  /* Keep track for rollback */
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
                            if (moved[kk] == -1 || moved[kk] == -(2 + to))
                                priority_queue_Update(queues[to], kk, oldgain + vwgt[k]);
                        }
                    }

                    /* Insert the new vertex Hunyuan_int_to the priority queue. Only one side! */
                    if (moved[k] == -1) 
                    {
                        priority_queue_Insert(queues[to], k, vwgt[k] - edegrees[other]);
                        moved[k] = -(2+to);
                    }
                }
            }
            mptr[nswaps + 1] = nmind;

            // printf("Moved %6"PRIDX" to %3"PRIDX", Gain: %5"PRIDX" [%5"PRIDX"] [%4"PRIDX" %4"PRIDX"] \t[%5"PRIDX" %5"PRIDX" %5"PRIDX"]\n", higain, to, g[to], g[other], vwgt[u[to]], vwgt[u[other]], pwgts[0], pwgts[1], pwgts[2]);
        }

        // exam_where(graph);
        // exam_pwgts(graph);
        /****************************************************************
        * Roll back computation 
        *****************************************************************/
        for (nswaps--; nswaps > mincutorder; nswaps--) 
        {
            higain = swaps[nswaps];

            to = where[higain];
            other = (to + 1) % 2;
            pwgts[2] += vwgt[higain];
            pwgts[to] -= vwgt[higain];
            where[higain] = 2;
            nbnd = insert_queue(nbnd,bndptr,bndind,higain);

            edegrees = rinfo[higain].edegrees;
            edegrees[0] = edegrees[1] = 0;
            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];
                if (where[k] == 2) 
                    rinfo[k].edegrees[to] -= vwgt[higain];
                else
                    edegrees[where[k]] += vwgt[k];
            }

            /* Push nodes out of the separator */
            for (j = mptr[nswaps]; j < mptr[nswaps + 1]; j++) 
            {
                k = mind[j];
                where[k] = other;
                pwgts[other] += vwgt[k];
                pwgts[2] -= vwgt[k];
                nbnd = delete_queue(nbnd,bndptr,bndind,k);
                for (jj = xadj[k]; jj < xadj[k + 1]; jj++) 
                {
                    kk = adjncy[jj];
                    if (where[kk] == 2) 
                        rinfo[kk].edegrees[other] += vwgt[k];
                }
            }
        }

        // exam_where(graph);
        // exam_pwgts(graph);
        // printf("\tMinimum sep: %6"PRIDX" at %5"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", mincut, mincutorder, pwgts[0], pwgts[1], nbnd);

        graph->mincut = mincut;
        graph->nbnd = nbnd;

        if (mincutorder == -1 || mincut >= initcut)
            break;
    }

    check_free(mind, sizeof(Hunyuan_int_t) * nvtxs * 2, "FM_2WayNodeRefine2Sided: mind");
    check_free(mptr, sizeof(Hunyuan_int_t) * (nvtxs + 1), "FM_2WayNodeRefine2Sided: mptr");
    check_free(swaps, sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeRefine2Sided: swaps");
    check_free(moved, sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeRefine2Sided: moved");
    priority_queue_Destroy(queues[1]);
    priority_queue_Destroy(queues[0]);
}

/*************************************************************************/
/*! This function performs a node-based FM refinement. 
    Each refinement iteration is split Hunyuan_int_to two sub-iterations. 
    In each sub-iteration only moves to one of the left/right partitions 
    is allowed; hence, it is one-sided. 
*/
/**************************************************************************/
void FM_2WayNodeRefine1Sided(graph_t *graph, Hunyuan_int_t niter)
{
    Hunyuan_int_t i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, nmind, iend;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
    Hunyuan_int_t *mptr, *mind, *swaps;
    priority_queue_t *queue; 
    nrinfo_t *rinfo;
    Hunyuan_int_t higain, mincut, initcut, mincutorder;	
    Hunyuan_int_t pass, to, other, limit;
    Hunyuan_int_t badmaxpwgt, mindiff, newdiff;
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

    queue = priority_queue_Create(nvtxs);

    swaps = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeRefine1Sided: swaps");
    mptr   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * (nvtxs + 1), "FM_2WayNodeRefine1Sided: mptr");
    mind   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs * 2, "FM_2WayNodeRefine1Sided: mind");
    
    mult = 0.5 * 1.2000499;
    badmaxpwgt = (Hunyuan_int_t)(mult * (pwgts[0] + pwgts[1] + pwgts[2]));

    // printf("Partitions-N1: [%6"PRIDX" %6"PRIDX"] Nv-Nb[%6"PRIDX" %6"PRIDX"]. ISep: %6"PRIDX"\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut);

    to = (pwgts[0] < pwgts[1] ? 1 : 0);
    for (pass = 0; pass< 2 * niter; pass++) 
    {  
        /* the 2*niter is for the two sides */
        other = to; 
        to    = (to + 1) % 2;

        priority_queue_Reset(queue);

        mincutorder = -1;
        initcut = mincut = graph->mincut;
        nbnd = graph->nbnd;

        /* use the swaps array in place of the traditional perm array to save memory */
        irandArrayPermute(nbnd, swaps, nbnd, 1);
        for (ii=0; ii<nbnd; ii++) 
        {
            i = bndind[swaps[ii]];
            priority_queue_Insert(queue, i, vwgt[i]-rinfo[i].edegrees[other]);
        }

        limit = (0 ? lyj_min(5 * nbnd, 500) : lyj_min(3 * nbnd, 300));

        /******************************************************
        * Get Hunyuan_int_to the FM loop
        *******************************************************/
        mptr[0] = nmind = 0;
        mindiff = lyj_abs(pwgts[0] - pwgts[1]);
        for (nswaps = 0; nswaps < nvtxs; nswaps++) 
        {
            if ((higain = priority_queue_GetTop(queue)) == -1)
                break;

            /* The following check is to ensure we break out if there is a posibility
                of over-running the mind array.  */
            if (nmind + xadj[higain + 1] - xadj[higain] >= 2 * nvtxs - 1) 
                break;

            if (pwgts[to] + vwgt[higain] > badmaxpwgt) 
                break;  /* No poHunyuan_int_t going any further. Balance will be bad */

            pwgts[2] -= (vwgt[higain] - rinfo[higain].edegrees[other]);

            newdiff = lyj_abs(pwgts[to] + vwgt[higain] - (pwgts[other] - rinfo[higain].edegrees[other]));
            if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) 
            {
                mincut      = pwgts[2];
                mincutorder = nswaps;
                mindiff     = newdiff;
            }
            else 
            {
                if (nswaps - mincutorder > 3 * limit || 
                    (nswaps - mincutorder > limit && pwgts[2] > 1.10 * mincut)) 
                {
                    pwgts[2] += (vwgt[higain]-rinfo[higain].edegrees[other]);
                    break; /* No further improvement, break out */
                }
            }

            nbnd = delete_queue(nbnd,bndptr,bndind,higain);
            pwgts[to]     += vwgt[higain];
            where[higain]  = to;
            swaps[nswaps]  = higain;  


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
                { 
                    /* This vertex is pulled Hunyuan_int_to the separator */
                    nbnd = insert_queue(nbnd,bndptr,bndind,k);
        
                    mind[nmind++] = k;  /* Keep track for rollback */
                    where[k] = 2;
                    pwgts[other] -= vwgt[k];

                    edegrees = rinfo[k].edegrees;
                    edegrees[0] = edegrees[1] = 0;
                    for (jj = xadj[k], iend = xadj[k + 1]; jj < iend; jj++) 
                    {
                        kk = adjncy[jj];
                        if (where[kk] != 2) 
                            edegrees[where[kk]] += vwgt[kk];
                        else 
                        {
                            rinfo[kk].edegrees[other] -= vwgt[k];

                            /* Since the moves are one-sided this vertex has not been moved yet */
                            priority_queue_Update(queue, kk, vwgt[kk] - rinfo[kk].edegrees[other]); 
                        }
                    }

                    /* Insert the new vertex Hunyuan_int_to the priority queue. Safe due to one-sided moves */
                    priority_queue_Insert(queue, k, vwgt[k]-edegrees[other]);
                }
            }
            mptr[nswaps+1] = nmind;

            // printf("Moved %6"PRIDX" to %3"PRIDX", Gain: %5"PRIDX" [%5"PRIDX"] \t[%5"PRIDX" %5"PRIDX" %5"PRIDX"] [%3"PRIDX" %2"PRIDX"]\n", \
                higain, to, (vwgt[higain]-rinfo[higain].edegrees[other]), vwgt[higain], \
                pwgts[0], pwgts[1], pwgts[2], nswaps, limit);
        }


        /****************************************************************
        * Roll back computation 
        *****************************************************************/
        for (nswaps--; nswaps > mincutorder; nswaps--) 
        {
            higain = swaps[nswaps];

            pwgts[2] += vwgt[higain];
            pwgts[to] -= vwgt[higain];
            where[higain] = 2;
            nbnd = insert_queue(nbnd,bndptr,bndind,higain);
        
            edegrees = rinfo[higain].edegrees;
            edegrees[0] = edegrees[1] = 0;
            for (j = xadj[higain]; j < xadj[higain + 1]; j++) 
            {
                k = adjncy[j];
                if (where[k] == 2) 
                    rinfo[k].edegrees[to] -= vwgt[higain];
                else
                    edegrees[where[k]] += vwgt[k];
            }

            /* Push nodes out of the separator */
            for (j = mptr[nswaps]; j < mptr[nswaps + 1]; j++) 
            {
                k = mind[j];
                where[k] = other;
                pwgts[other] += vwgt[k];
                pwgts[2] -= vwgt[k];
                nbnd = delete_queue(nbnd,bndptr,bndind,k);
                for (jj = xadj[k], iend = xadj[k+1]; jj < iend; jj++) 
                {
                    kk = adjncy[jj];
                    if (where[kk] == 2) 
                        rinfo[kk].edegrees[other] += vwgt[k];
                }
            }
        }

        // printf("\tMinimum sep: %6"PRIDX" at %5"PRIDX", PWGTS: [%6"PRIDX" %6"PRIDX"], NBND: %6"PRIDX"\n", mincut, mincutorder, pwgts[0], pwgts[1], nbnd);

        graph->mincut = mincut;
        graph->nbnd   = nbnd;

        if (pass % 2 == 1 && (mincutorder == -1 || mincut >= initcut))
            break;
    }

    check_free(mind, sizeof(Hunyuan_int_t) * nvtxs * 2, "FM_2WayNodeRefine1Sided: mind");
    check_free(mptr, sizeof(Hunyuan_int_t) * (nvtxs + 1), "FM_2WayNodeRefine1Sided: mptr");
    check_free(swaps, sizeof(Hunyuan_int_t) * nvtxs, "FM_2WayNodeRefine1Sided: swaps");

    priority_queue_Destroy(queue);
}

void Refine2WayNode(graph_t *graph, graph_t *origraph)
{
    if (graph == origraph) 
        Compute_Reorder_Informetion_2way(graph);
    else 
    {
        do 
        {
            graph = graph->finer;
            // printf("Refine2WayNode 0\n");
            project_Reorder(graph);
            // printf("Refine2WayNode 1\n");
            // exam_where(graph);
            CONTROL_COMMAND(control, FMNODEBALANCE_Time, gettimebegin(&start_fmnodebalance, &end_fmnodebalance, &time_fmnodebalance));
            FM_2WayNodeBalance(graph);
            CONTROL_COMMAND(control, FMNODEBALANCE_Time, gettimeend(&start_fmnodebalance, &end_fmnodebalance, &time_fmnodebalance));

            // printf("Refine2WayNode 2\n");
            // exam_where(graph);
            CONTROL_COMMAND(control, FM1SIDENODEREFINE_Time, gettimebegin(&start_fm1sidenoderefine, &end_fm1sidenoderefine, &time_fm1sidenoderefine));
            FM_2WayNodeRefine1Sided(graph, 10);
            CONTROL_COMMAND(control, FM1SIDENODEREFINE_Time, gettimeend(&start_fm1sidenoderefine, &end_fm1sidenoderefine, &time_fm1sidenoderefine));
 
            // printf("Refine2WayNode 3\n");
            // exam_where(graph);

        } while (graph != origraph);
    }
}

/*************************************************************************/
/*! This function is the entry point of refinement */
/*************************************************************************/
void Refine2WayPartition(graph_t *orggraph, graph_t *graph, Hunyuan_real_t *ntpwgts, Hunyuan_real_t *balance_factor)
{
    /* Compute the parameters of the coarsest graph */
    CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimebegin(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
    Compute_Partition_Informetion_2way(graph);
    CONTROL_COMMAND(control, PARTITIOBINF2WAY, gettimeend(&start_partitioninf2way, &end_partitioninf2way, &time_partitioninf2way));
        

    for (;;) 
    {
        // Balance2Way(ctrl, graph, tpwgts);
        CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, gettimebegin(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
        Balance2Way_partition(graph, ntpwgts, balance_factor);
        CONTROL_COMMAND(control, FM2WAYCUTBALANCE_Time, gettimeend(&start_fm2waycutbalance, &end_fm2waycutbalance, &time_fm2waycutbalance));
        
        // FM_2WayRefine(ctrl, graph, tpwgts, ctrl->niter); 
        CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimebegin(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));
        FM_2WayCutRefine(graph, ntpwgts, 10);
        CONTROL_COMMAND(control, FM2WAYCUTREFINE_Time, gettimeend(&start_fm2waycutrefine, &end_fm2waycutrefine, &time_fm2waycutrefine));
        // if(exam_correct_gp(graph->where, graph->nvtxs, 2))
        // {
        //     printf("The Answer is Right\n");
        //     printf("edgecut=%"PRIDX" \n", compute_edgecut(graph->where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt));
        // }
        // else 
        //     printf("The Answer is Error\n");
        if (graph == orggraph)
            break;

        graph = graph->finer;
        
        Project_2WayPartition(graph);
    }
}

#endif