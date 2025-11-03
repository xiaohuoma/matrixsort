#ifndef SEARCHTREE_H
#define SEARCHTREE_H

#include "struct.h"
#include "memory.h"
#include "define.h"

void binary_search_tree_Init(binary_search_tree_t *tree)
{
    tree->nownodes = 0;
	tree->treenode = NULL;
}

binary_search_tree_t *binary_search_tree_Create()
{
    binary_search_tree_t *tree;

    tree = (binary_search_tree_t *)check_malloc(sizeof(binary_search_tree_t), "binary_search_tree_Create: tree");
    binary_search_tree_Init(tree);

    return tree;
}

void Free_Treenode(treenode_t *node)
{
	if(node != NULL)
	{
		Free_Treenode(node->left);
		Free_Treenode(node->right);
		check_free(node, sizeof(treenode_t), "Free_Treenode: node");
	}
}

void binary_search_tree_Free(binary_search_tree_t *tree)
{
	if (tree == NULL) return;
	Free_Treenode(tree->treenode);
	// check_free(tree->locator);
	tree->nownodes = 0;
}

void binary_search_tree_Destroy(binary_search_tree_t *tree)
{
	if (tree == NULL) return;
	binary_search_tree_Free(tree);
	check_free(tree, sizeof(binary_search_tree_t), "binary_search_tree_Destroy: tree");
}

Hunyuan_int_t binary_search_tree_Length(binary_search_tree_t *tree)
{
	return tree->nownodes;
}

treenode_t *Create_TreeNode(Hunyuan_int_t val, Hunyuan_int_t key)
{
	treenode_t *newnode = (treenode_t *)check_malloc(sizeof(treenode_t), "Create_TreeNode: newnode");
    
	newnode->val = val;
	newnode->key = key;
    newnode->left = newnode->right = NULL;
    
	return newnode;
}

treenode_t *Insert_TreeNode(treenode_t *node, Hunyuan_int_t val, Hunyuan_int_t key, Hunyuan_int_t *nownodes)
{
	// if empty
    if (node == NULL) 
	{
		node = Create_TreeNode(val, key);
		(*nownodes)++;
		return node;
	}

    // if less than
    if (val < node->val)
        node->left = Insert_TreeNode(node->left, val, key, nownodes);
    // if greater than
    else if (val > node->val)
        node->right = Insert_TreeNode(node->right, val, key, nownodes);
	
	//	if equal
	else
		node->key += key;

    return node;
}

void binary_search_tree_Insert(binary_search_tree_t *tree, Hunyuan_int_t val, Hunyuan_int_t key)
{
	treenode_t *root = tree->treenode;
	
	root = Insert_TreeNode(root, val, key, &tree->nownodes);
	tree->treenode = root;

	return ;
}

Hunyuan_int_t InorderTraversal_TreeNode(treenode_t *root, Hunyuan_int_t *dst1, Hunyuan_int_t *dst2, Hunyuan_int_t *ptr) 
{
    if (root != NULL) 
	{
        *ptr = InorderTraversal_TreeNode(root->left,dst1,dst2,ptr);
		//	do operation
        // printf("root->val=%"PRIDX" root->key=%"PRIDX" ", root->val,root->key);
		// if(dst != NULL) 
		// {
			dst1[*ptr] = root->val;
			dst2[*ptr] = root->key;
			// printf("root->val=%"PRIDX" dst[ptr]=%"PRIDX" ptr=%"PRIDX"\n",root->val, dst[*ptr], *ptr);
			(*ptr) ++;
		// }

        *ptr = InorderTraversal_TreeNode(root->right,dst1,dst2,ptr);
    }

	return *ptr;
}

void binary_search_tree_Traversal(binary_search_tree_t *tree, Hunyuan_int_t *dst1, Hunyuan_int_t *dst2)
{
	treenode_t *root = tree->treenode;
	Hunyuan_int_t ptr = 0;

	InorderTraversal_TreeNode(root, dst1, dst2, &ptr);
}



//  Binary Search Tree Version 2.0
void binary_search_tree_Init2(binary_search_tree2_t *tree, Hunyuan_int_t size)
{
    tree->nownodes = 0;
	tree->maxnodes = size;
	tree->treenode = (treenode2_t *)check_malloc(sizeof(treenode2_t) * size, "binary_search_tree2_Init: tree->treenode");

	for(Hunyuan_int_t i = 0;i < size;i++)
	{
		tree->treenode[i].val = -1;
		tree->treenode[i].key = 0;
	}
}

binary_search_tree2_t *binary_search_tree_Create2(Hunyuan_int_t size)
{
    binary_search_tree2_t *tree;

    tree = (binary_search_tree2_t *)check_malloc(sizeof(binary_search_tree2_t), "binary_search_tree2_Create: tree");
    
	binary_search_tree_Init2(tree, size);

    return tree;
}

void exam_binary_search_tree2(binary_search_tree2_t *tree)
{
	printf("val:");
	for(Hunyuan_int_t i = 0;i < tree->maxnodes;i++)
	{
		printf("%"PRIDX" ",tree->treenode[i].val);
	}
	printf("\n");
	printf("key:");
	for(Hunyuan_int_t i = 0;i < tree->maxnodes;i++)
	{
		printf("%"PRIDX" ",tree->treenode[i].key);
	}
	printf("\n");
}

void exam_binary_search_tree2_flag(binary_search_tree2_t *tree)
{
	Hunyuan_int_t flag = 0;
	printf("val:");
	for(Hunyuan_int_t i = 0;i < tree->maxnodes;i++)
	{
		printf("%"PRIDX" ",tree->treenode[i].val);
		if(tree->treenode[i].val != -1)
			flag = 1;
	}
	printf("\n");
	if(flag == 1)
		printf("flag=1\n");
	printf("key:");
	for(Hunyuan_int_t i = 0;i < tree->maxnodes;i++)
	{
		printf("%"PRIDX" ",tree->treenode[i].key);
		if(tree->treenode[i].key != 0)
			flag = 2;
	}
	printf("\n");
	if(flag == 2)
		printf("flag=2\n");
}

void binary_search_tree_Free2(binary_search_tree2_t *tree)
{
	if (tree == NULL) return;
	check_free(tree->treenode, sizeof(treenode2_t) * tree->maxnodes, "binary_search_tree_Free2: tree->treenode");
	tree->nownodes = 0;
	tree->maxnodes = 0;
}

void binary_search_tree_Destroy2(binary_search_tree2_t *tree)
{
	if (tree == NULL) return;
	binary_search_tree_Free2(tree);
	check_free(tree, sizeof(binary_search_tree2_t), "binary_search_tree_Destroy2: tree");
}

Hunyuan_int_t binary_search_tree_Length2(binary_search_tree2_t *tree)
{
	return tree->nownodes;
}

void Insert_TreeNode2(binary_search_tree2_t *tree, Hunyuan_int_t val, Hunyuan_int_t key)
{
	Hunyuan_int_t ptr = 0;
	treenode2_t *treenode = tree->treenode;

	while (treenode[ptr].val != -1) 
	{
		if(ptr >= tree->maxnodes)
		{
			printf("check_realloc\n");
			treenode = tree->treenode = (treenode2_t *)check_realloc(treenode, sizeof(treenode2_t) * tree->maxnodes * 2, sizeof(Hunyuan_int_t) * tree->maxnodes, "Insert_TreeNode2: treenode");
			for(Hunyuan_int_t i = tree->maxnodes;i < tree->maxnodes * 2;i++)
			{
				treenode[i].val = -1;
				treenode[i].key = 0;
			}
			tree->maxnodes *= 2;
		}

        if (treenode[ptr].val < val) 
            ptr = 2 * ptr + 2;
        else if (treenode[ptr].val > val) 
            ptr = 2 * ptr + 1; 
		else if(treenode[ptr].val == val) 
		{
			treenode[ptr].key += key;
			printf("Update: val=%"PRIDX" key=%"PRIDX" ptr=%"PRIDX"\n",val,treenode[ptr].key,ptr);
            return ;
        }
    }

	printf("Insert: val=%"PRIDX" key=%"PRIDX" ptr=%"PRIDX"\n",val,key,ptr);
    treenode[ptr].val = val;
    treenode[ptr].key = key;
    tree->nownodes++;

    return ;
}

void binary_search_tree_Insert2(binary_search_tree2_t *tree, Hunyuan_int_t val, Hunyuan_int_t key)
{
	Insert_TreeNode2(tree, val, key);
	printf("\n");
	return ;
}

void InorderTraversal_TreeNode2(binary_search_tree2_t *tree, treenode2_t *treenode, Hunyuan_int_t maxnodes, Hunyuan_int_t *dst1, Hunyuan_int_t *dst2, Hunyuan_int_t located, Hunyuan_int_t *ptr) 
{
	printf("InorderTraversal_TreeNode2 1 located=%"PRIDX"\n",located);
	exam_binary_search_tree2(tree);
	if(treenode[located].val == -1)
		return;

	if (2 * located + 1 < maxnodes)
		InorderTraversal_TreeNode2(tree, treenode, maxnodes, dst1, dst2, 2 * located + 1, ptr);
	printf("located=%"PRIDX" ptr=%"PRIDX" val=%"PRIDX" key=%"PRIDX"\n",located,*ptr,treenode[located].val,treenode[located].key);
	exam_binary_search_tree2(tree);
	dst1[*ptr] = treenode[located].val;
	printf("1\n");
	exam_binary_search_tree2(tree);
	dst2[*ptr] = treenode[located].key;
	printf("2\n");
	exam_binary_search_tree2(tree);
	(*ptr)++;
	printf("3\n");
	exam_binary_search_tree2(tree);
	if(2 * located + 2 < maxnodes)
		InorderTraversal_TreeNode2(tree, treenode, maxnodes, dst1, dst2, 2 * located + 2, ptr);
}

void binary_search_tree_Traversal2(binary_search_tree2_t *tree, Hunyuan_int_t *dst1, Hunyuan_int_t *dst2)
{
	treenode2_t *treenode = tree->treenode;

	Hunyuan_int_t ptr = 0;

	InorderTraversal_TreeNode2(tree, treenode, tree->maxnodes, dst1, dst2, 0, &ptr);
}

void Reset_TreeNode2(treenode2_t *treenode, Hunyuan_int_t maxnodes, Hunyuan_int_t located) 
{
	if(treenode[located].val == -1)
		return;
	else
	{
		// if (2 * located + 1 < maxnodes)
		// 	Reset_TreeNode2(treenode, maxnodes, located * 2 + 1);
		// treenode[located].val = -1;
		// treenode[located].key = 0;
		// if (2 * located + 2 < maxnodes)
		// 	Reset_TreeNode2(treenode, maxnodes, located * 2 + 2);
		
        if (2 * located + 1 < maxnodes)
            Reset_TreeNode2(treenode, maxnodes, 2 * located + 1);
        treenode[located].val = -1;
        treenode[located].key = 0;
        if (2 * located + 2 < maxnodes)
            Reset_TreeNode2(treenode, maxnodes, 2 * located + 2);
	}
}

void binary_search_tree_Reset2(binary_search_tree2_t *tree)
{
	treenode2_t *treenode = tree->treenode;

	Reset_TreeNode2(treenode, tree->maxnodes, 0);
}

#endif