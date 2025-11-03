#ifndef KWAYREFINE_H
#define KWAYREFINE_H

#include "struct.h"
#include "graph.h"
#include "memory.h"
#include "queue.h"

#include <math.h>

void Compute_Partition_Informetion_Kway(graph_t *graph, Hunyuan_int_t nparts)
{
    Hunyuan_int_t nvtxs, nbnd, mincut, k;
    Hunyuan_int_t *xadj, *vwgt, *adjncy, *adjwgt, *where, *bndptr, *bndind, *ed, *id, *pwgts;

    nvtxs = graph->nvtxs;

    xadj   = graph->xadj;
    vwgt   = graph->vwgt;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;

    where  = graph->where;
    bndptr = graph->bndptr;
    bndind = graph->bndind;
    pwgts  = graph->pwgts;

    mincut = 0;

    // printf("Compute_Partition_Informetion_Kway 0\n");

    //  init nbnd, bndptr and bndind
    bndind = graph->bndind = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Compute_Partition_Informetion_Kway: bndind");
    bndptr = graph->bndptr = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Compute_Partition_Informetion_Kway: bndptr");
    nbnd = init_queue(nbnd, bndptr, nvtxs);

    //  init pwgts
    pwgts  = graph->pwgts = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "Compute_Partition_Informetion_Kway: pwgts");
    set_value_int(nparts, 0, pwgts);

    if(graph->ckrinfo == NULL)
        graph->ckrinfo = (ckrinfo_t *)check_malloc(sizeof(ckrinfo_t) * nvtxs, "Compute_Partition_Informetion_Kway: graph->ckrinfo");

    memset(graph->ckrinfo, 0, sizeof(ckrinfo_t) * nvtxs);

    cnbrReset(graph);
    // for(Hunyuan_int_t i = 0;i < nvtxs;i++)
    //     printf("%"PRIDX" %"PRIDX" \n",i,where[i]);
    // printf("Compute_Partition_Informetion_Kway 1\n");

    // printf("%"PRIDX" \n",where[1]);

    //  compute array nbnd, bndptr, bndind, ckrinfo
    ckrinfo_t *myrinfo;
    cnbr_t *mynbrs;
    for(Hunyuan_int_t i = 0;i < nvtxs;i++)
    {
        Hunyuan_int_t partition = where[i];
        pwgts[partition] += vwgt[i];

        myrinfo = graph->ckrinfo + i;

        Hunyuan_int_t begin = xadj[i];
        Hunyuan_int_t end   = xadj[i + 1];
        Hunyuan_int_t ted   = 0;
        Hunyuan_int_t tid   = 0;
        for(Hunyuan_int_t j = begin;j < end;j++)
        {
            Hunyuan_int_t k = adjncy[j];
            if(partition != where[k])
                ted += adjwgt[j];
            else 
                tid += adjwgt[j];
        }
        
        myrinfo->id = tid;
        myrinfo->ed = ted;

        // printf("i=%d flag_boundary=%d\n",i,flag_boundary);
        
        if(ted > 0)
        {
            mincut += ted;

            myrinfo->inbr = cnbrGetNext(graph, end - begin + 1);
            mynbrs = graph->cnbr + myrinfo->inbr;

            for (Hunyuan_int_t j = begin; j < end; j++) 
            {
                Hunyuan_int_t other = where[adjncy[j]];
                if (partition != other) 
                {
                    for (k = 0; k < myrinfo->nnbrs; k++) 
                    {
                        if (mynbrs[k].pid == other) 
                        {
                            mynbrs[k].ed += adjwgt[j];
                            break;
                        }
                    }
                    if (k == myrinfo->nnbrs) 
                    {
                        mynbrs[k].pid = other;
                        mynbrs[k].ed  = adjwgt[j];
                        myrinfo->nnbrs++;
                    }
                }
            }

            /* Only ed-id>=0 nodes are considered to be in the boundary */
            if (ted - tid >= 0)
            {
                // printf("%"PRIDX" \n", i);
                nbnd = insert_queue(nbnd, bndptr, bndind, i);
            }
        }
        else 
        {
            myrinfo->inbr = -1;
        }
    }

    // for(Hunyuan_int_t i = 0;i < nvtxs;i++)
    //     printf("i=%"PRIDX" id=%"PRIDX" ed=%"PRIDX" inbr=%"PRIDX" nnbrs=%"PRIDX"\n", i, graph->ckrinfo[i].ed, graph->ckrinfo[i].id, graph->ckrinfo[i].inbr, graph->ckrinfo[i].nnbrs);
    // printf("\n");
    // for(Hunyuan_int_t i = 0;i < nbnd;i++)
    //     printf("i=%"PRIDX" bndind[i]=%"PRIDX"\n", i, bndind[i]);
    // printf("\n");
    
    // printf("Compute_Partition_Informetion_Kway 2\n");

    // printf("nbnd=%"PRIDX"\n",nbnd);

    graph->nbnd   = nbnd;
    graph->mincut = mincut / 2;
}

/*************************************************************************
* These macros deal with id/ed updating during k-way refinement
**************************************************************************/
Hunyuan_int_t UpdateMovedVertexInfoAndBND(Hunyuan_int_t i, Hunyuan_int_t from, Hunyuan_int_t k, Hunyuan_int_t to, ckrinfo_t *myrinfo, cnbr_t *mynbrs, \
    Hunyuan_int_t *where, Hunyuan_int_t nbnd, Hunyuan_int_t *bndptr, Hunyuan_int_t *bndind, Hunyuan_int_t bndtype) 
{
    Hunyuan_int_t j;
    do 
    { 
        where[i] = to; 
        myrinfo->ed += myrinfo->id - mynbrs[k].ed; 
        lyj_swap(myrinfo->id, mynbrs[k].ed, j); 
        if (mynbrs[k].ed == 0) 
            mynbrs[k] = mynbrs[--myrinfo->nnbrs]; 
        else 
            mynbrs[k].pid = from;
        
        /* Update the boundary information. Both deletion and addition is \
            allowed as this routine can be used for moving arbitrary nodes. */
        if (bndtype == 1) 
        {
            if (bndptr[i] != -1 && myrinfo->ed - myrinfo->id < 0)
                nbnd = delete_queue(nbnd, bndptr,  bndind, i);
            if (bndptr[i] == -1 && myrinfo->ed - myrinfo->id >= 0)
                nbnd = insert_queue(nbnd, bndptr,  bndind, i);
        }
        else 
        {
            if (bndptr[i] != -1 && myrinfo->ed <= 0)
                nbnd = delete_queue(nbnd, bndptr,  bndind, i);
            if (bndptr[i] == -1 && myrinfo->ed > 0)
                nbnd = insert_queue(nbnd, bndptr,  bndind, i);
        }
    } while(0);

    return nbnd;
}

Hunyuan_int_t UpdateAdjacentVertexInfoAndBND(graph_t *graph, Hunyuan_int_t vid, Hunyuan_int_t adjlen, Hunyuan_int_t me, Hunyuan_int_t from, Hunyuan_int_t to, \
    ckrinfo_t *myrinfo, Hunyuan_int_t ewgt, Hunyuan_int_t nbnd, Hunyuan_int_t *bndptr, Hunyuan_int_t *bndind, Hunyuan_int_t bndtype)
{
   do 
   {
        Hunyuan_int_t k; 
        cnbr_t *mynbrs; 
        
        if (myrinfo->inbr == -1) 
        {
            myrinfo->inbr  = cnbrGetNext(graph, adjlen + 1); 
            myrinfo->nnbrs = 0; 
        }
     
        mynbrs = graph->cnbr + myrinfo->inbr; 
    
        /* Update global ID/ED and boundary */ 
        if (me == from) 
        {
            myrinfo->ed += ewgt;
            myrinfo->id -= ewgt;
            if (bndtype == 1) 
            {
                if (myrinfo->ed - myrinfo->id >= 0 && bndptr[(vid)] == -1)
                    nbnd = insert_queue(nbnd, bndptr,  bndind, (vid));
            }
            else 
            {
                if (myrinfo->ed > 0 && bndptr[(vid)] == -1)
                    nbnd = insert_queue(nbnd, bndptr,  bndind, (vid));
            }
        }
        else if (me == to) 
        {
            myrinfo->id += ewgt;
            myrinfo->ed -= ewgt;
            if (bndtype == 1) 
            {
                if (myrinfo->ed - myrinfo->id < 0 && bndptr[(vid)] != -1)
                    nbnd = delete_queue(nbnd, bndptr,  bndind, (vid));
            }
            else
            {
                if (myrinfo->ed <= 0 && bndptr[(vid)] != -1)
                    nbnd = delete_queue(nbnd, bndptr,  bndind, (vid));
            }
        }
    
        /* Remove contribution from the .ed of 'from' */
        if (me != from) 
        {
            for (k = 0; k < myrinfo->nnbrs; k++) 
            {
                if (mynbrs[k].pid == from) 
                {
                    if (mynbrs[k].ed == (ewgt))
                        mynbrs[k] = mynbrs[--myrinfo->nnbrs];
                    else
                        mynbrs[k].ed -= (ewgt);
                    break;
                }
            }
        }
    
        /* Add contribution to the .ed of 'to' */
        if (me != to) 
        {
            for (k = 0; k < myrinfo->nnbrs; k++) 
            {
                if (mynbrs[k].pid == to) 
                {
                    mynbrs[k].ed += (ewgt);
                    break;
                }
            }
            if (k == myrinfo->nnbrs) 
            {
                mynbrs[k].pid  = to;
                mynbrs[k].ed   = (ewgt);
                myrinfo->nnbrs++;
            }
        }
    } while(0);

    return nbnd;
}

Hunyuan_int_t UpdateQueueInfo(priority_queue_real_t *queue, Hunyuan_int_t *vstatus, Hunyuan_int_t vid, Hunyuan_int_t me, Hunyuan_int_t from, Hunyuan_int_t to, \
    ckrinfo_t *myrinfo, Hunyuan_int_t oldnnbrs, Hunyuan_int_t nupd, Hunyuan_int_t *updptr, Hunyuan_int_t *updind, Hunyuan_int_t bndtype)
{
    do 
    {
        Hunyuan_real_t rgain; 

        if (me == to || me == from || oldnnbrs != myrinfo->nnbrs) 
        {
            rgain = (myrinfo->nnbrs > 0 ? 1.0 * myrinfo->ed / sqrt(myrinfo->nnbrs) : 0.0) - myrinfo->id;
                
            if (bndtype == 1) 
            {
                if (vstatus[(vid)] == 1) 
                {
                    if (myrinfo->ed-myrinfo->id >= 0)
                        priority_queue_real_Update(queue, (vid), rgain);
                    else 
                    {
                        priority_queue_real_Delete(queue, (vid));
                        vstatus[(vid)] = 3;
                        nupd = delete_queue(nupd, updptr,  updind, (vid));
                    }
                }
                else if (vstatus[(vid)] == 3 && myrinfo->ed - myrinfo->id >= 0) 
                {
                    priority_queue_real_Insert(queue, (vid), rgain);
                    vstatus[(vid)] = 1;
                    nupd = insert_queue(nupd, updptr,  updind, (vid));
                }
            }
            else 
            {
                if (vstatus[(vid)] == 1) 
                {
                    if (myrinfo->ed > 0)
                        priority_queue_real_Update(queue, (vid), rgain);
                    else 
                    {
                        priority_queue_real_Delete(queue, (vid));
                        vstatus[(vid)] = 3;
                        nupd = delete_queue(nupd, updptr,  updind, (vid));
                    }
                }
                else if (vstatus[(vid)] == 3 && myrinfo->ed > 0) 
                {
                    priority_queue_real_Insert(queue, (vid), rgain);
                    vstatus[(vid)] = 1;
                    nupd = insert_queue(nupd, updptr,  updind, (vid));
                }
            }
        }
    } while(0);

    return nupd;
}

/*************************************************************************/
/*! K-way partitioning optimization in which the vertices are visited in 
    decreasing ed/sqrt(nnbrs)-id order. Note this is just an 
    approximation, as the ed is often split across different subdomains 
    and the sqrt(nnbrs) is just a crude approximation.

  \param graph is the graph that is being refined.
  \param niter is the number of refinement iterations.
  \param ffactor is the \em fudge-factor for allowing positive gain moves 
         to violate the max-pwgt constraint.
  \param omode is the type of optimization that will performed among
         OMODE_REFINE and OMODE_BALANCE 
         

*/
/**************************************************************************/
void RefineKWayCut(graph_t *graph, Hunyuan_int_t niter, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor, Hunyuan_real_t ffactor, Hunyuan_int_t omode)
{
    /* Common variables to all types of kway-refinement/balancing routines */
    Hunyuan_int_t i, ii, iii, j, k, l, pass, nvtxs, gain; 
    Hunyuan_int_t from, me, to, oldcut, vwgt;
    Hunyuan_int_t *xadj, *adjncy, *adjwgt;
    Hunyuan_int_t *where, *pwgts, *perm, *bndptr, *bndind, *minwgt, *maxwgt, *itpwgts;
    Hunyuan_int_t nmoved, nupd, *vstatus, *updptr, *updind;
    Hunyuan_int_t maxndoms, *safetos=NULL, *nads=NULL, *doms=NULL, **adids=NULL, **adwgts=NULL;
    Hunyuan_int_t *bfslvl=NULL, *bfsind=NULL, *bfsmrk=NULL;
    Hunyuan_int_t bndtype = (omode == 1 ? 1 : 2);

    /* Edgecut-specific/different variables */
    Hunyuan_int_t nbnd, oldnnbrs;
    priority_queue_real_t *queue;
    Hunyuan_real_t rgain;
    ckrinfo_t *myrinfo;
    cnbr_t *mynbrs;

    /* Link the graph fields */
    nvtxs  = graph->nvtxs;
    xadj   = graph->xadj;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;

    bndind = graph->bndind;
    bndptr = graph->bndptr;

    where = graph->where;
    pwgts = graph->pwgts;

    /* Setup the weight intervals of the various subdomains */
    minwgt  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "RefineKWayCut: minwgt");
    maxwgt  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "RefineKWayCut: maxwgt");
    itpwgts = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "RefineKWayCut: itpwgts");
    perm    = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "RefineKWayCut: perm");

    for (i = 0; i < nparts; i++) 
    {
        itpwgts[i] = tpwgts[i] * graph->tvwgt[0];
        maxwgt[i]  = tpwgts[i] * graph->tvwgt[0] * balance_factor[0];
        minwgt[i]  = tpwgts[i] * graph->tvwgt[0] * (1.0 / balance_factor[0]);
        // printf("itpwgts[i]=%"PRIDX" maxwgt[i]=%"PRIDX" minwgt[i]=%"PRIDX"\n",itpwgts[i], maxwgt[i], minwgt[i]);
    }

    /* This stores the valid target subdomains. It is used when ctrl->minconn to
        control the subdomains to which moves are allowed to be made. 
        When ctrl->minconn is false, the default values of 2 allow all moves to
        go through and it does not interfere with the zero-gain move selection. */
    safetos = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "RefineKWayCut: safetos");
    set_value_int(nparts, 2, safetos);

    /* Setup updptr, updind like boundary info to keep track of the vertices whose
        vstatus's need to be reset at the end of the inner iteration */
    vstatus = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "RefineKWayCut: vstatus");
    updptr  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "RefineKWayCut: updptr");
    updind  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "RefineKWayCut: updind");
    set_value_int(nvtxs, 3, vstatus);
    set_value_int(nvtxs, -1, updptr);

    queue = priority_queue_real_Create(nvtxs);

    // printf("RefineKWayCut 0\n");

    // if(exam_correct_gp(graph->where, graph->nvtxs, nparts))
    // {
	// 	printf("The Answer is Right\n");
    //     printf("edgecut=%"PRIDX" \n", compute_edgecut(graph->where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt));
    // }
	// else 
	// 	printf("The Answer is Error\n");

    /*=====================================================================
    * The top-level refinement loop 
    *======================================================================*/
    for (pass = 0; pass < niter; pass++) 
    {
        // printf("pass=%"PRIDX"\n",pass);
        if (omode == 2) 
        {   /* Check to see if things are out of balance, given the tolerance */
            for (i = 0; i < nparts; i++) 
            {
                if (pwgts[i] > maxwgt[i])
                    break;
            }
            if (i == nparts) /* Things are balanced. Return right away */
                break;
        }

        oldcut = graph->mincut;
        nbnd   = graph->nbnd;
        nupd   = 0;

        // printf("nbnd=%"PRIDX"\n",nbnd);
        // printf("RefineKWayCut 1\n");

        /* Insert the boundary vertices in the priority queue */
        // printf("pass=%"PRIDX" random_count=%"PRIDX"\n",pass,rand_count());
        irandArrayPermute(nbnd, perm, nbnd / 4, 1);

        // for(Hunyuan_int_t i = 0;i < nbnd;i++)
        //     printf("%"PRIDX" ", bndind[i]);
        // printf("\n");

        // printf("RefineKWayCut 1.1\n");
        for (ii = 0; ii < nbnd; ii++) 
        {
            i = bndind[perm[ii]];
            rgain = (graph->ckrinfo[i].nnbrs > 0 ? 
                    1.0 * graph->ckrinfo[i].ed / sqrt(graph->ckrinfo[i].nnbrs) : 0.0) 
                    - graph->ckrinfo[i].id;
            priority_queue_real_Insert(queue, i, rgain);
            vstatus[i] = 1;
            nupd = insert_queue(nupd, updptr, updind, i);
        }
        // printf("nbnd=%"PRIDX" nupd=%"PRIDX"\n",nbnd,nupd);

        // printf("RefineKWayCut 2\n");

        /* Start extracting vertices from the queue and try to move them */
        for (nmoved = 0, iii = 0;;iii++) 
        {
            if ((i = priority_queue_real_GetTop(queue)) == -1) 
                break;
            vstatus[i] = 2;

            myrinfo = graph->ckrinfo + i;
            mynbrs  = graph->cnbr + myrinfo->inbr;

            from = where[i];
            vwgt = graph->vwgt[i];

            /* Prevent moves that make 'from' domain underbalanced */
            if (omode == 1) 
            {
                if (myrinfo->id > 0 && pwgts[from] - vwgt < minwgt[from]) 
                {
                    // printf("myrinfo->id=%"PRIDX" > 0 && pwgts[from]=%"PRIDX" - vwgt=%"PRIDX" < minwgt[from]=%"PRIDX" from=%"PRIDX"\n",myrinfo->id,pwgts[from],vwgt, minwgt[from], from);
                    continue;   
                }
            }
            else 
            { /* OMODE_BALANCE */
                if (pwgts[from] - vwgt < minwgt[from]) 
                    continue;   
            }

            /* Find the most promising subdomain to move to */
            if (omode == 1) 
            {
                for (k = myrinfo->nnbrs - 1; k >= 0; k--) 
                {
                    if (!safetos[to = mynbrs[k].pid])
                        continue;
                    gain = mynbrs[k].ed - myrinfo->id; 
                    if (gain >= 0 && pwgts[to] + vwgt <= maxwgt[to] + ffactor * gain)  
                        break;
                }
                if (k < 0)
                    continue;  /* break out if you did not find a candidate */

                for (j = k - 1; j >= 0; j--) 
                {
                    if (!safetos[to = mynbrs[j].pid])
                        continue;
                    gain = mynbrs[j].ed - myrinfo->id; 
                    if ((mynbrs[j].ed > mynbrs[k].ed && pwgts[to] + vwgt <= maxwgt[to] + ffactor * gain) ||
                        (mynbrs[j].ed == mynbrs[k].ed && itpwgts[mynbrs[k].pid] * pwgts[to] < itpwgts[to] * pwgts[mynbrs[k].pid]))
                        k = j;
                }

                to = mynbrs[k].pid;
                gain = mynbrs[k].ed - myrinfo->id;
                if (!
                        (gain > 0 || 
                            (gain == 0 && 
                                (pwgts[from] >= maxwgt[from] || 
                                    itpwgts[to] * pwgts[from] > itpwgts[from] * (pwgts[to] + vwgt) || 
                                    (iii % 2 == 0 && 
                                        safetos[to] == 2
                                    )
                                )
                            )
                        )
                    )
                    continue;
            }
            else 
            {  /* OMODE_BALANCE */
                for (k = myrinfo->nnbrs - 1; k >= 0; k--) 
                {
                    if (!safetos[to = mynbrs[k].pid])
                        continue;
                    if (pwgts[to] + vwgt <= maxwgt[to] || 
                        itpwgts[from] * (pwgts[to] + vwgt) <= itpwgts[to] * pwgts[from]) 
                        break;
                }
                if (k < 0)
                    continue;  /* break out if you did not find a candidate */

                for (j = k - 1; j >= 0; j--) 
                {
                    if (!safetos[to = mynbrs[j].pid])
                        continue;
                    if (itpwgts[mynbrs[k].pid] * pwgts[to] < itpwgts[to] * pwgts[mynbrs[k].pid]) 
                        k = j;
                }

                to = mynbrs[k].pid;

                if (pwgts[from] < maxwgt[from] && pwgts[to] > minwgt[to] && 
                    mynbrs[k].ed - myrinfo->id < 0) 
                    continue;
            }

            /*=====================================================================
            * If we got here, we can now move the vertex from 'from' to 'to' 
            *======================================================================*/
            graph->mincut -= mynbrs[k].ed - myrinfo->id;
            nmoved++;

            // printf("\t\tnmoved %6"PRIDX" iii %6"PRIDX" nbnd %6"PRIDX" Moving %6"PRIDX" to %3"PRIDX". Gain: %4"PRIDX". Cut: %6"PRIDX"\n", 
            //   nmoved, iii, nbnd, i, to, mynbrs[k].ed-myrinfo->id, graph->mincut);
            // exam_priority_real_queue(queue);

            /* Update ID/ED and BND related information for the moved vertex */
            pwgts[to] += vwgt;
            pwgts[from] -= vwgt;
            // if(nvtxs == 255119)
            //     printf("iii=%"PRIDX" befor nbnd=%"PRIDX"\n",iii,nbnd);
            nbnd = UpdateMovedVertexInfoAndBND(i, from, k, to, myrinfo, mynbrs, where, nbnd, bndptr, bndind, bndtype);
            // if(nvtxs == 255119)
            //     printf("iii=%"PRIDX" end   nbnd=%"PRIDX"\n",iii,nbnd);
      
            /* Update the degrees of adjacent vertices */
            for (j = xadj[i]; j < xadj[i + 1]; j++) 
            {
                ii = adjncy[j];
                me = where[ii];
                myrinfo = graph->ckrinfo + ii;

                oldnnbrs = myrinfo->nnbrs;

                // if(nvtxs == 255119 || iii == 47)
                // {
                //     printf("ii=%"PRIDX" befor nbnd=%"PRIDX"\n",ii,nbnd);
                // }
                nbnd = UpdateAdjacentVertexInfoAndBND(graph, ii, xadj[ii + 1] - xadj[ii], me, from, to, myrinfo, adjwgt[j], \
                    nbnd, bndptr, bndind, bndtype);
                // if(nvtxs == 255119 || iii == 47)
                // {
                //     printf("ii=%"PRIDX" end   nbnd=%"PRIDX"\n",ii, nbnd);
                //     printf("ii=%"PRIDX" befor nupd=%"PRIDX"\n",ii,nupd);
                //     exam_priority_queue(queue);
                // }
                nupd = UpdateQueueInfo(queue, vstatus, ii, me, from, to, myrinfo, oldnnbrs, nupd, updptr, updind, bndtype);
                // if(nvtxs == 255119 || iii == 47)
                // {
                //     exam_priority_queue(queue);
                //     printf("ii=%"PRIDX" end   nupd=%"PRIDX" queue_topval=%"PRIDX"\n",ii, nupd, priority_queue_SeeTopVal(queue));
                // }
            }
        }

        // printf("RefineKWayCut 3 iii=%"PRIDX"\n",iii);

        graph->nbnd = nbnd;

        /* Reset the vstatus and associated data structures */
        for (i = 0; i < nupd; i++) 
        {
            vstatus[updind[i]] = 3;
            updptr[updind[i]]  = -1;
        }

        if (nmoved == 0 || (omode == 1 && graph->mincut == oldcut))
            break;
        
        // printf("RefineKWayCut 4\n");
    }

    // if(exam_correct_gp(graph->where, graph->nvtxs, nparts))
	// {
	// 	printf("The Answer is Right\n");
    //     printf("edgecut=%"PRIDX" \n", compute_edgecut(graph->where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt));
    // }
	// else 
	// 	printf("The Answer is Error\n");
    check_free(minwgt, sizeof(Hunyuan_int_t) * nparts, "RefineKWayCut: minwgt");
    check_free(maxwgt, sizeof(Hunyuan_int_t) * nparts, "RefineKWayCut: maxwgt");
    check_free(itpwgts, sizeof(Hunyuan_int_t) * nparts, "RefineKWayCut: itpwgts");
    check_free(perm, sizeof(Hunyuan_int_t) * nvtxs, "RefineKWayCut: perm");

    check_free(safetos, sizeof(Hunyuan_int_t) * nparts, "RefineKWayCut: safetos");
    check_free(vstatus, sizeof(Hunyuan_int_t) * nvtxs, "RefineKWayCut: vstatus");
    check_free(updptr, sizeof(Hunyuan_int_t) * nvtxs, "RefineKWayCut: updptr");
    check_free(updind, sizeof(Hunyuan_int_t) * nvtxs, "RefineKWayCut: updind");

    priority_queue_real_Destroy(queue);
}

/*************************************************************************/
/*! This function computes the boundary definition for balancing. */
/*************************************************************************/
void ComputeKWayBoundary(graph_t *graph, Hunyuan_int_t bndtype)
{
    Hunyuan_int_t i, nvtxs, nbnd;
    Hunyuan_int_t *bndind, *bndptr;

    nvtxs  = graph->nvtxs;
    bndind = graph->bndind;
    bndptr = graph->bndptr;
    
    nbnd = init_queue(nbnd, bndptr, nvtxs);

    /* Compute the boundary */
    if (bndtype == 1) 
    { /* BNDTYPE_REFINE */
        for (i = 0; i < nvtxs; i++) 
        {
            if (graph->ckrinfo[i].ed - graph->ckrinfo[i].id >= 0) 
                nbnd = insert_queue(nbnd, bndptr, bndind, i);
        }
    }
    else 
    { /* BNDTYPE_BALANCE */
        for (i = 0; i < nvtxs; i++) 
        {
            if (graph->ckrinfo[i].ed > 0) 
                nbnd = insert_queue(nbnd, bndptr, bndind, i);
        }
    }

    graph->nbnd = nbnd;
}

void Project_KWayPartition(graph_t *graph, Hunyuan_int_t nparts)
{
    Hunyuan_int_t i, j, k, nvtxs, me, other, istart, iend, tid, ted, nbnd;
    Hunyuan_int_t *xadj, *adjncy, *adjwgt;
    Hunyuan_int_t *cmap, *where, *bndptr, *bndind, *cwhere, *htable;
    graph_t *cgraph;

    cgraph = graph->coarser;
    cwhere = cgraph->where;

    nvtxs   = graph->nvtxs;
    cmap    = graph->cmap;
    xadj    = graph->xadj;
    adjncy  = graph->adjncy;
    adjwgt  = graph->adjwgt;

    // AllocateKWayPartitionMemory(ctrl, graph);
    graph->cnbr_size    = 2 * graph->nedges;
    graph->cnbr_length  = 0;
    graph->cnb_reallocs = 0;
    graph->cnbr = (cnbr_t *)check_malloc(graph->cnbr_size * sizeof(cnbr_t), "Project_KWayPartition: graph->cnbr");

    // graph->pwgts  = imalloc(ctrl->nparts*graph->ncon, "AllocateKWayPartitionMemory: pwgts");
    // graph->where  = imalloc(graph->nvtxs,  "AllocateKWayPartitionMemory: where");
    // graph->bndptr = imalloc(graph->nvtxs,  "AllocateKWayPartitionMemory: bndptr");
    // graph->bndind = imalloc(graph->nvtxs,  "AllocateKWayPartitionMemory: bndind");
    graph->pwgts   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "Project_KWayPartition: graph->pwgts");
    graph->where   = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Project_KWayPartition: graph->where");
    graph->bndptr  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Project_KWayPartition: graph->bndptr");
    graph->bndind  = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nvtxs, "Project_KWayPartition: graph->bndind");
    graph->ckrinfo = (ckrinfo_t *)check_malloc(sizeof(ckrinfo_t) * nvtxs, "Project_KWayPartition: graph->ckrinfo");

    where  = graph->where;
    bndind = graph->bndind;
    bndptr = graph->bndptr;
    nbnd = init_queue(nbnd, bndptr, nvtxs);

    htable = (Hunyuan_int_t *)check_malloc(sizeof(Hunyuan_int_t) * nparts, "Project_KWayPartition: htable");
    set_value_int(nparts, -1, htable);

    /* Compute the required info for refinement */

    ckrinfo_t *myrinfo;
    cnbr_t *mynbrs;

    /* go through and project partition and compute id/ed for the nodes */
    for (i = 0; i < nvtxs; i++) 
    {
        k        = cmap[i];
        where[i] = cwhere[k];
        cmap[i]  = cgraph->ckrinfo[k].ed;  /* For optimization */
    }

    memset(graph->ckrinfo, 0, sizeof(ckrinfo_t) * nvtxs);
    cnbrReset(graph);

    for (i = 0; i < nvtxs; i++) 
    {
        istart = xadj[i];
        iend   = xadj[i + 1];
        myrinfo = graph->ckrinfo + i;

        if (cmap[i] == 0) 
        { /* Interior node. Note that cmap[i] = crinfo[cmap[i]].ed */
            for (tid = 0, j = istart; j < iend; j++) 
                tid += adjwgt[j];

            myrinfo->id   = tid;
            myrinfo->inbr = -1;
        }
        else 
        { /* Potentially an interface node */
            myrinfo->inbr = cnbrGetNext(graph, iend - istart + 1);
            mynbrs        = graph->cnbr + myrinfo->inbr;

            me = where[i];
            for (tid = 0, ted = 0, j = istart; j < iend; j++) 
            {
                other = where[adjncy[j]];
                if (me == other) 
                {
                    tid += adjwgt[j];
                }
                else 
                {
                    ted += adjwgt[j];
                    if ((k = htable[other]) == -1) 
                    {
                        htable[other]               = myrinfo->nnbrs;
                        mynbrs[myrinfo->nnbrs].pid  = other;
                        mynbrs[myrinfo->nnbrs++].ed = adjwgt[j];
                    }
                    else 
                    {
                        mynbrs[k].ed += adjwgt[j];
                    }
                }
            }
            myrinfo->id = tid;
            myrinfo->ed = ted;
      
            /* Remove space for edegrees if it was interior */
            if (ted == 0) 
            { 
                graph->cnbr_length -= iend - istart + 1;
                myrinfo->inbr       = -1;
            }
            else 
            {
                if (ted - tid >= 0) 
                    nbnd = insert_queue(nbnd, bndptr, bndind, i);
        
                for (j = 0; j < myrinfo->nnbrs; j++)
                    htable[mynbrs[j].pid] = -1;
            }
        }
    }
      
    graph->nbnd = nbnd;

    graph->mincut = cgraph->mincut;
    copy_int(nparts, cgraph->pwgts, graph->pwgts);

    check_free(htable, sizeof(Hunyuan_int_t) * nparts, "Project_KWayPartition: htable");

    FreeGraph(&graph->coarser, nparts);
    graph->coarser = NULL;
}

/*************************************************************************/
/*! This function is the entry point of cut-based refinement */
/*************************************************************************/
void RefineKWayPartition(graph_t *orggraph, graph_t *graph, Hunyuan_int_t nparts, Hunyuan_real_t *tpwgts, Hunyuan_real_t *balance_factor)
{
    Hunyuan_int_t i, nlevels, contig = 0;
    graph_t *ptr;

    /* Determine how many levels are there */
    for (ptr = graph, nlevels = 0; ptr != orggraph; ptr = ptr->finer, nlevels++); 

    // printf("RefineKWayPartition 0\n");

    /* Compute the parameters of the coarsest graph */
    Compute_Partition_Informetion_Kway(graph, nparts);
    // exam_pwgts(graph, nparts);

    // printf("RefineKWayPartition 1\n");

    /* Refine each successively finer graph */
    for (i = 0; ;i++) 
    {
        // printf("RefineKWayPartition %"PRIDX" %"PRIDX"\n", 2 * i, nlevels);
        if (2 * i >= nlevels && !IsBalancedKway(graph, nparts, tpwgts, balance_factor, .02)) 
        {
            // printf("RefineKWayPartition 2\n");
            // printf("nvtxs=%"PRIDX"\n",graph->nvtxs);
            // exam_where(graph);
            ComputeKWayBoundary(graph, 2);
            // printf("RefineKWayPartition 3\n");
            RefineKWayCut(graph, 1, nparts, tpwgts, balance_factor, 0, 2); 
            // printf("RefineKWayPartition 4\n");
            ComputeKWayBoundary(graph, 1);
            // printf("RefineKWayPartition 5\n");
        }

        // printf("nvtxs=%"PRIDX"\n",graph->nvtxs);
        // exam_xadj(graph);
        // exam_vwgt(graph);
        // exam_where(graph);
        // exam_adjncy_adjwgt(graph);
        RefineKWayCut(graph, 10, nparts, tpwgts, balance_factor, 5.0, 1);
        // if(exam_correct_gp(graph->where, graph->nvtxs, nparts))
        // {
        //     printf("The Answer is Right\n");
        //     printf("edgecut=%"PRIDX" \n", compute_edgecut(graph->where, graph->nvtxs, graph->xadj, graph->adjncy, graph->adjwgt));
        // }
        // else 
        //     printf("The Answer is Error\n");
        // printf("nvtxs=%"PRIDX"\n",graph->nvtxs);
        // exam_where(graph);
        // printf("RefineKWayPartition 6\n");

        if (graph == orggraph)
            break;

        graph = graph->finer;

        Project_KWayPartition(graph, nparts);
        // exam_pwgts(graph, nparts);
        // printf("RefineKWayPartition 7\n");
    }

    /* Deal with contiguity requirement at the end */
    if (!IsBalancedKway(graph, nparts, tpwgts, balance_factor, 0.0)) 
    {
        // printf("RefineKWayPartition 8\n");
        ComputeKWayBoundary(graph, 2);
        // printf("RefineKWayPartition 9\n");
        RefineKWayCut(graph, 10, nparts, tpwgts, balance_factor, 0, 2); 
        // printf("RefineKWayPartition 10\n");

        ComputeKWayBoundary(graph, 1);
        // printf("RefineKWayPartition 11\n");
        RefineKWayCut(graph, 10, nparts, tpwgts, balance_factor, 0, 1); 
        // printf("RefineKWayPartition 12\n");
    }
}

#endif