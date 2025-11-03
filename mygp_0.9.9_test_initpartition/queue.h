#ifndef CTRL_H
#define CTRL_H

#include "struct.h"
#include "memory.h"


//	priority queue
/*************************************************************************/
/*! This function initializes the data structures of the priority queue */
/**************************************************************************/
void priority_queue_Init(priority_queue_t *queue, size_t maxnodes)
{
	queue->nownodes = 0;
	queue->maxnodes = maxnodes;
	queue->heap     = (node_t *)check_malloc(sizeof(node_t) * maxnodes, "priority_queue_Init: heap");
	queue->locator  = (size_t *)check_malloc(sizeof(size_t) * maxnodes, "priority_queue_Init: locator");
	for(Hunyuan_int_t i = 0;i < maxnodes;i++)
		queue->locator[i] = -1;
}

/*************************************************************************/
/*! This function creates and initializes a priority queue */
/**************************************************************************/
priority_queue_t *priority_queue_Create(size_t maxnodes)
{
	priority_queue_t *queue; 

	queue = (priority_queue_t *)check_malloc(sizeof(priority_queue_t), "priority_queue_Create: queue");
	priority_queue_Init(queue, maxnodes);

	return queue;
}

/*************************************************************************/
/*! This function resets the priority queue */
/**************************************************************************/
void priority_queue_Reset(priority_queue_t *queue)
{
	Hunyuan_int_t i;
	size_t *locator = queue->locator;
	node_t *heap    = queue->heap;

	for (i = queue->nownodes - 1; i >= 0; i--)
		locator[heap[i].val] = -1;
	queue->nownodes = 0;
}

/*************************************************************************/
/*! This function frees the Hunyuan_int_ternal datastructures of the priority queue */
/**************************************************************************/
void priority_queue_Free(priority_queue_t *queue)
{
	if (queue == NULL) return;
	check_free(queue->locator, sizeof(size_t) * queue->maxnodes, "priority_queue_Free: queue->locator");
	check_free(queue->heap, sizeof(node_t) * queue->maxnodes, "priority_queue_Free: queue->heap");
	queue->maxnodes = 0;
}

/*************************************************************************/
/*! This function frees the Hunyuan_int_ternal datastructures of the priority queue 
    and the queue itself */
/**************************************************************************/
void priority_queue_Destroy(priority_queue_t *queue)
{
	if (queue == NULL) return;
	priority_queue_Free(queue);
	check_free(queue, sizeof(priority_queue_t), "priority_queue_Destroy: queue");
}

/*************************************************************************/
/*! This function returns the length of the queue */
/**************************************************************************/
size_t priority_queue_Length(priority_queue_t *queue)
{
	return queue->nownodes;
}

/*************************************************************************/
/*! This function adds an item in the priority queue */
/**************************************************************************/
Hunyuan_int_t priority_queue_Insert(priority_queue_t *queue, Hunyuan_int_t node, Hunyuan_int_t key)
{
	Hunyuan_int_t i, j;
	size_t *locator=queue->locator;
	node_t *heap = queue->heap;

	i = queue->nownodes++;
	while (i > 0) 
	{
		j = (i - 1) >> 1;
		if (m_gt_n(key, heap[j].key)) 
		{
			heap[i] = heap[j];
			locator[heap[i].val] = i;
			i = j;
		}
		else
			break;
	}
  
	heap[i].key   = key;
	heap[i].val   = node;
	locator[node] = i;

	return 0;
}

/*************************************************************************/
/*! This function deletes an item from the priority queue */
/**************************************************************************/
Hunyuan_int_t priority_queue_Delete(priority_queue_t *queue, Hunyuan_int_t node)
{
	Hunyuan_int_t i, j, nownodes;
	Hunyuan_int_t newkey, oldkey;
	size_t *locator = queue->locator;
	node_t *heap = queue->heap;

	i = locator[node];
	locator[node] = -1;

	if (--queue->nownodes > 0 && heap[queue->nownodes].val != node) 
	{
		node   = heap[queue->nownodes].val;
		newkey = heap[queue->nownodes].key;
		oldkey = heap[i].key;

		if (m_gt_n(newkey, oldkey)) 
		{ /* Filter-up */
			while (i > 0) 
			{
				j = (i - 1) >> 1;
				if (m_gt_n(newkey, heap[j].key)) 
				{
					heap[i] = heap[j];
					locator[heap[i].val] = i;
					i = j;
				}
				else
					break;
			}
		}
		else 
		{ /* Filter down */
			nownodes = queue->nownodes;
			while ((j = (i << 1) + 1) < nownodes) 
			{
				if (m_gt_n(heap[j].key, newkey)) 
				{
					if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, heap[j].key))
						j++;
					heap[i] = heap[j];
					locator[heap[i].val] = i;
					i = j;
				}
				else if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, newkey)) 
				{
					j++;
					heap[i] = heap[j];
					locator[heap[i].val] = i;
					i = j;
				}
				else
					break;
			}
		}

		heap[i].key   = newkey;
		heap[i].val   = node;
		locator[node] = i;
	}

	return 0;
}

/*************************************************************************/
/*! This function updates the key values associated for a particular item */ 
/**************************************************************************/
void priority_queue_Update(priority_queue_t *queue, Hunyuan_int_t node, Hunyuan_int_t newkey)
{
	Hunyuan_int_t i, j, nownodes;
	Hunyuan_int_t oldkey;
	size_t *locator = queue->locator;
	node_t *heap = queue->heap;

	oldkey = heap[locator[node]].key;

	i = locator[node];

	if (m_gt_n(newkey, oldkey)) 
	{ /* Filter-up */
		while (i > 0) 
		{
			j = (i - 1) >> 1;
			if (m_gt_n(newkey, heap[j].key)) 
			{
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else
				break;
		}
	}
	else 
	{ /* Filter down */
		nownodes = queue->nownodes;
		while ((j = (i << 1) + 1) < nownodes) 
		{
			if (m_gt_n(heap[j].key, newkey)) 
			{
				if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, heap[j].key))
					j++;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, newkey)) 
			{
				j++;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else
				break;
		}
	}

	heap[i].key   = newkey;
	heap[i].val   = node;
	locator[node] = i;

	return;
}

/*************************************************************************/
/*! This function returns the item at the top of the queue and removes
    it from the priority queue */
/**************************************************************************/
Hunyuan_int_t priority_queue_GetTop(priority_queue_t *queue)
{
	Hunyuan_int_t i, j;
	size_t *locator;
	node_t *heap;
	Hunyuan_int_t vtx, node;
	Hunyuan_int_t key;

	if (queue->nownodes == 0)
		return -1;

	queue->nownodes--;

	heap    = queue->heap;
	locator = queue->locator;

	vtx = heap[0].val;
	locator[vtx] = -1;

	if ((i = queue->nownodes) > 0) 
	{
		key  = heap[i].key;
		node = heap[i].val;
		i = 0;
		while ((j = 2 * i + 1) < queue->nownodes) 
		{
			if (m_gt_n(heap[j].key, key)) 
			{
				if (j + 1 < queue->nownodes && m_gt_n(heap[j + 1].key, heap[j].key))
					j = j+1;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else if (j + 1 < queue->nownodes && m_gt_n(heap[j + 1].key, key)) 
			{
				j = j + 1;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else
				break;
		}

		heap[i].key   = key;
		heap[i].val   = node;
		locator[node] = i;
	}

	return vtx;
}

/*************************************************************************/
/*! This function returns the item at the top of the queue. The item is not
    deleted from the queue. */
/**************************************************************************/
Hunyuan_int_t priority_queue_SeeTopVal(priority_queue_t *queue)
{
  return (queue->nownodes == 0 ? -1 : queue->heap[0].val);
}

/*************************************************************************/
/*! This function returns the item at the top of the queue. The item is not
    deleted from the queue. */
/**************************************************************************/
Hunyuan_int_t priority_queue_SeeTopKey(priority_queue_t *queue)
{
  return (queue->nownodes == 0 ? -1 : queue->heap[0].key);
}

void exam_priority_queue(priority_queue_t *queue)
{
	printf("nownodes=%"PRIDX" maxnodes=%"PRIDX"\n",queue->nownodes,queue->maxnodes);
	printf("key:");
	for(Hunyuan_int_t i = 0;i < queue->nownodes;i++)
		printf("%"PRIDX" ",queue->heap[i].key);
	printf("\n");
	printf("val:");
	for(Hunyuan_int_t i = 0;i < queue->nownodes;i++)
		printf("%"PRIDX" ",queue->heap[i].val);
	printf("\n");
}



// priority_queue_real_t
/*************************************************************************/
/*! This function initializes the data structures of the priority queue */
/**************************************************************************/
void priority_queue_real_Init(priority_queue_real_t *queue, size_t maxnodes)
{
	queue->nownodes = 0;
	queue->maxnodes = maxnodes;
	queue->heap     = (node_real_t *)check_malloc(sizeof(node_real_t) * maxnodes, "priority_queue_real_Init: heap");
	queue->locator  = (size_t *)check_malloc(sizeof(size_t) * maxnodes, "priority_queue_real_Init: locator");
	for(Hunyuan_int_t i = 0;i < maxnodes;i++)
		queue->locator[i] = -1;
}

/*************************************************************************/
/*! This function creates and initializes a priority queue */
/**************************************************************************/
priority_queue_real_t *priority_queue_real_Create(size_t maxnodes)
{
	priority_queue_real_t *queue; 

	queue = (priority_queue_real_t *)check_malloc(sizeof(priority_queue_real_t), "priority_queue_real_Create: queue");
	priority_queue_real_Init(queue, maxnodes);

	return queue;
}

/*************************************************************************/
/*! This function resets the priority queue */
/**************************************************************************/
void priority_queue_real_Reset(priority_queue_real_t *queue)
{
	Hunyuan_int_t i;
	size_t *locator = queue->locator;
	node_real_t *heap    = queue->heap;

	for (i = queue->nownodes - 1; i >= 0; i--)
		locator[heap[i].val] = -1;
	queue->nownodes = 0;
}

/*************************************************************************/
/*! This function frees the Hunyuan_int_ternal datastructures of the priority queue */
/**************************************************************************/
void priority_queue_real_Free(priority_queue_real_t *queue)
{
	if (queue == NULL) return;
	check_free(queue->locator, sizeof(size_t) * queue->maxnodes, "priority_queue_real_Free: queue->locator");
	check_free(queue->heap, sizeof(node_real_t) * queue->maxnodes, "priority_queue_real_Free: queue->heap");
	queue->maxnodes = 0;
}

/*************************************************************************/
/*! This function frees the Hunyuan_int_ternal datastructures of the priority queue 
    and the queue itself */
/**************************************************************************/
void priority_queue_real_Destroy(priority_queue_real_t *queue)
{
	if (queue == NULL) return;
	priority_queue_real_Free(queue);
	check_free(queue, sizeof(priority_queue_real_t), "priority_queue_real_Destroy: queue");
}

/*************************************************************************/
/*! This function returns the length of the queue */
/**************************************************************************/
size_t priority_queue_real_Length(priority_queue_real_t *queue)
{
	return queue->nownodes;
}

/*************************************************************************/
/*! This function adds an item in the priority queue */
/**************************************************************************/
Hunyuan_int_t priority_queue_real_Insert(priority_queue_real_t *queue, Hunyuan_int_t node, Hunyuan_real_t key)
{
	Hunyuan_int_t i, j;
	size_t *locator=queue->locator;
	node_real_t *heap = queue->heap;

	i = queue->nownodes++;
	while (i > 0) 
	{
		j = (i - 1) >> 1;
		if (m_gt_n(key, heap[j].key)) 
		{
			heap[i] = heap[j];
			locator[heap[i].val] = i;
			i = j;
		}
		else
			break;
	}
  
	heap[i].key   = key;
	heap[i].val   = node;
	locator[node] = i;

	return 0;
}

/*************************************************************************/
/*! This function deletes an item from the priority queue */
/**************************************************************************/
Hunyuan_int_t priority_queue_real_Delete(priority_queue_real_t *queue, Hunyuan_int_t node)
{
	Hunyuan_int_t i, j, nownodes;
	Hunyuan_real_t newkey, oldkey;
	size_t *locator = queue->locator;
	node_real_t *heap = queue->heap;

	i = locator[node];
	locator[node] = -1;

	if (--queue->nownodes > 0 && heap[queue->nownodes].val != node) 
	{
		node   = heap[queue->nownodes].val;
		newkey = heap[queue->nownodes].key;
		oldkey = heap[i].key;

		if (m_gt_n(newkey, oldkey)) 
		{ /* Filter-up */
			while (i > 0) 
			{
				j = (i - 1) >> 1;
				if (m_gt_n(newkey, heap[j].key)) 
				{
					heap[i] = heap[j];
					locator[heap[i].val] = i;
					i = j;
				}
				else
					break;
			}
		}
		else 
		{ /* Filter down */
			nownodes = queue->nownodes;
			while ((j = (i << 1) + 1) < nownodes) 
			{
				if (m_gt_n(heap[j].key, newkey)) 
				{
					if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, heap[j].key))
						j++;
					heap[i] = heap[j];
					locator[heap[i].val] = i;
					i = j;
				}
				else if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, newkey)) 
				{
					j++;
					heap[i] = heap[j];
					locator[heap[i].val] = i;
					i = j;
				}
				else
					break;
			}
		}

		heap[i].key   = newkey;
		heap[i].val   = node;
		locator[node] = i;
	}

	return 0;
}

/*************************************************************************/
/*! This function updates the key values associated for a particular item */ 
/**************************************************************************/
void priority_queue_real_Update(priority_queue_real_t *queue, Hunyuan_int_t node, Hunyuan_real_t newkey)
{
	Hunyuan_int_t i, j, nownodes;
	Hunyuan_real_t oldkey;
	size_t *locator = queue->locator;
	node_real_t *heap = queue->heap;

	oldkey = heap[locator[node]].key;

	i = locator[node];

	if (m_gt_n(newkey, oldkey)) 
	{ /* Filter-up */
		while (i > 0) 
		{
			j = (i - 1) >> 1;
			if (m_gt_n(newkey, heap[j].key)) 
			{
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else
				break;
		}
	}
	else 
	{ /* Filter down */
		nownodes = queue->nownodes;
		while ((j = (i << 1) + 1) < nownodes) 
		{
			if (m_gt_n(heap[j].key, newkey)) 
			{
				if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, heap[j].key))
					j++;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else if (j + 1 < nownodes && m_gt_n(heap[j + 1].key, newkey)) 
			{
				j++;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else
				break;
		}
	}

	heap[i].key   = newkey;
	heap[i].val   = node;
	locator[node] = i;

	return;
}

/*************************************************************************/
/*! This function returns the item at the top of the queue and removes
    it from the priority queue */
/**************************************************************************/
Hunyuan_int_t priority_queue_real_GetTop(priority_queue_real_t *queue)
{
	Hunyuan_int_t i, j;
	size_t *locator;
	node_real_t *heap;
	Hunyuan_int_t vtx, node;
	Hunyuan_real_t key;

	if (queue->nownodes == 0)
		return -1;

	queue->nownodes--;

	heap    = queue->heap;
	locator = queue->locator;

	vtx = heap[0].val;
	locator[vtx] = -1;

	if ((i = queue->nownodes) > 0) 
	{
		key  = heap[i].key;
		node = heap[i].val;
		i = 0;
		while ((j = 2 * i + 1) < queue->nownodes) 
		{
			if (m_gt_n(heap[j].key, key)) 
			{
				if (j + 1 < queue->nownodes && m_gt_n(heap[j + 1].key, heap[j].key))
					j = j+1;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else if (j + 1 < queue->nownodes && m_gt_n(heap[j + 1].key, key)) 
			{
				j = j + 1;
				heap[i] = heap[j];
				locator[heap[i].val] = i;
				i = j;
			}
			else
				break;
		}

		heap[i].key   = key;
		heap[i].val   = node;
		locator[node] = i;
	}

	return vtx;
}

/*************************************************************************/
/*! This function returns the item at the top of the queue. The item is not
    deleted from the queue. */
/**************************************************************************/
Hunyuan_int_t priority_queue_real_SeeTopVal(priority_queue_real_t *queue)
{
  return (queue->nownodes == 0 ? -1 : queue->heap[0].val);
}

/*************************************************************************/
/*! This function returns the item at the top of the queue. The item is not
    deleted from the queue. */
/**************************************************************************/
Hunyuan_real_t priority_queue_real_SeeTopKey(priority_queue_real_t *queue)
{
  return (queue->nownodes == 0 ? -1.0 : queue->heap[0].key);
}

void exam_priority_real_queue(priority_queue_real_t *queue)
{
	printf("nownodes=%"PRIDX" maxnodes=%"PRIDX"\n",queue->nownodes,queue->maxnodes);
	printf("key:");
	for(Hunyuan_int_t i = 0;i < queue->nownodes;i++)
		printf("%"PRREAL" ",queue->heap[i].key);
	printf("\n");
	printf("val:");
	for(Hunyuan_int_t i = 0;i < queue->nownodes;i++)
		printf("%"PRIDX" ",queue->heap[i].val);
	printf("\n");
}


//	common queue
Hunyuan_int_t init_queue(Hunyuan_int_t ptr, Hunyuan_int_t *bndptr, Hunyuan_int_t nvtxs)
{
	set_value_int(nvtxs, -1, bndptr);
	return 0;
}

/*************************************************************************/
/*! Execution process:	(n -> empty)
		nbnd:	 5
		bndind:	 6  4  5  9  2  n  n  n  n  n
		bndptr:	-1 -1  4 -1  1  2  0 -1 -1  3
	aftering insert 8
		nbnd:	 6
		bndind:	 6  4  5  9  2  8  n  n  n  n
		bndptr:	-1 -1  4 -1  1  2  0 -1  5  3
 */
/**************************************************************************/
																//  /\		//
Hunyuan_int_t insert_queue(Hunyuan_int_t nbnd, Hunyuan_int_t *bndptr, Hunyuan_int_t *bndind, Hunyuan_int_t vertex)// /  \		//
{													//				||
	bndind[nbnd]   = vertex;						//	bndind[5] = 8
	bndptr[vertex] = nbnd;							//	bndptr[8] = 5
	nbnd ++;										//	nbnd      = 6

	return nbnd;
}

/*************************************************************************/
/*! Execution process:	(n -> empty)
		nbnd:	 6
		bndind:	 6  4  5  9  2  8  n  n  n  n
		bndptr:	-1 -1  4 -1  1  2  0 -1  5  3
	aftering delete 4
		nbnd:	 5
		bndind:	 6  8  5  9  2  n  n  n  n  n
		bndptr:	-1 -1  4 -1 -1  2  0 -1  1  3
 */
/**************************************************************************/
																//  /\		//
Hunyuan_int_t delete_queue(Hunyuan_int_t nbnd, Hunyuan_int_t *bndptr, Hunyuan_int_t *bndind, Hunyuan_int_t vertex)// /  \		//
{													//				||
	bndind[bndptr[vertex]]   = bndind[nbnd - 1];	//	bndind[1] = 8
	bndptr[bndind[nbnd - 1]] = bndptr[vertex];		//	bndptr[8] = 1
	bndptr[vertex] = -1;							//	bndptr[4] = -1
	nbnd --;										//	nbnd      = 5

	return nbnd;
}

#endif