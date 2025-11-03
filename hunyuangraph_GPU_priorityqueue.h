#ifndef _H_GPU_PRIORITYQUEUE
#define _H_GPU_PRIORITYQUEUE

#include "hunyuangraph_struct.h"


__device__ void priority_queue_Init(priority_queue_t *queue, int maxnodes)
{
	queue->nownodes = 0;
	queue->maxnodes = maxnodes;
	for(int i = 0;i < maxnodes;i++)
		queue->locator[i] = -1;
}

__device__ void priority_queue_Reset(priority_queue_t *queue, int maxnodes)
{
	for (int i = queue->nownodes - 1; i >= 0; i--)
		queue->locator[queue->val[i]] = -1;
	queue->nownodes = 0;
    queue->maxnodes = maxnodes;
}

__device__ int priority_queue_Length(priority_queue_t *queue)
{
	return queue->nownodes;
}

__device__ int priority_queue_Insert(priority_queue_t *queue, int node, int key)
{
	int i, j;

	i = queue->nownodes++;
	while (i > 0) 
	{
		j = (i - 1) >> 1;
		if (key > queue->key[j]) 
		{
            queue->val[i] = queue->val[j];
            queue->key[i] = queue->key[j];
			queue->locator[queue->val[i]] = i;
			i = j;
		}
		else
			break;
	}
  
	queue->key[i]   = key;
	queue->val[i]   = node;
	queue->locator[node] = i;

	return 0;
}

__device__ int priority_queue_Delete(priority_queue_t *queue, int node)
{
	int i, j, nownodes;
	int newkey, oldkey;

	i = queue->locator[node];
	queue->locator[node] = -1;

	if (--queue->nownodes > 0 && queue->val[queue->nownodes] != node) 
	{
		node   = queue->val[queue->nownodes];
		newkey = queue->key[queue->nownodes];
		oldkey = queue->key[i];

		if (newkey > oldkey) 
		{ /* Filter-up */
			while (i > 0) 
			{
				j = (i - 1) >> 1;
				if (newkey > queue->key[j]) 
				{
                    queue->val[i] = queue->val[j];
                    queue->key[i] = queue->key[j];
					queue->locator[queue->val[i]] = i;
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
				if (queue->key[j] > newkey) 
				{
					if (j + 1 < nownodes && queue->key[j + 1] > queue->key[j])
						j++;
                    queue->val[i] = queue->val[j];
                    queue->key[i] = queue->key[j];
					queue->locator[queue->val[i]] = i;
					i = j;
				}
				else if (j + 1 < nownodes && queue->key[j + 1] > newkey)
				{
					j++;
                    queue->val[i] = queue->val[j];
                    queue->key[i] = queue->key[j];
					queue->locator[queue->val[i]] = i;
					i = j;
				}
				else
					break;
			}
		}

		queue->key[i] = newkey;
		queue->val[i] = node;
		queue->locator[node] = i;
	}

	return 0;
}

__device__ void priority_queue_Update(priority_queue_t *queue, int node, int newkey)
{
	int i, j, nownodes;
	int oldkey;

	oldkey = queue->key[queue->locator[node]];

	i = queue->locator[node];

	if (newkey > oldkey) 
	{ /* Filter-up */
		while (i > 0) 
		{
			j = (i - 1) >> 1;
			if (newkey > queue->key[j]) 
			{
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
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
			if (queue->key[j] > newkey)
			{
				if (j + 1 < nownodes && queue->key[j + 1] > queue->key[j])
					j++;
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
				i = j;
			}
			else if (j + 1 < nownodes && queue->key[j + 1] > newkey) 
			{
				j++;
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
				i = j;
			}
			else
				break;
		}
	}

	queue->key[i] = newkey;
	queue->val[i] = node;
	queue->locator[node] = i;

	return;
}

__device__ int priority_queue_GetTop(priority_queue_t *queue)
{
	int i, j;
	int vtx, node;
	int key;

	if (queue->nownodes == 0)
		return -1;

	queue->nownodes--;

	vtx = queue->val[0];
	queue->locator[vtx] = -1;

	if ((i = queue->nownodes) > 0) 
	{
		key  = queue->key[i];
		node = queue->val[i];
		i = 0;
		while ((j = 2 * i + 1) < queue->nownodes) 
		{
			if (queue->key[j] > key) 
			{
				if (j + 1 < queue->nownodes && queue->key[j + 1] > queue->key[j])
					j = j + 1;
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
				i = j;
			}
			else if (j + 1 < queue->nownodes && queue->key[j + 1] > key) 
			{
				j = j + 1;
                queue->val[i] = queue->val[j];
                queue->key[i] = queue->key[j];
				queue->locator[queue->val[i]] = i;
				i = j;
			}
			else
				break;
		}

		queue->key[i] = key;
		queue->val[i] = node;
		queue->locator[node] = i;
	}

	return vtx;
}

#endif