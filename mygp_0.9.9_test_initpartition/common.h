#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "typedef.h"
#include "define.h"

/*************************************************************************/
/*! This function prHunyuan_int_ts an error message and exits  
 */
/*************************************************************************/
void error_exit(const char *error_message) 
{
    fprintf(stderr, "Error: %s\n", error_message);

    fflush(stderr);

    exit(1);
}

Hunyuan_int_t ceilUp(Hunyuan_real_t number) 
{
    Hunyuan_int_t integerPart;
    Hunyuan_real_t decimalPart;

    // 分离整数和小数部分
    integerPart = (Hunyuan_int_t)number; // 取整数部分
    decimalPart = number - integerPart; // 取小数部分

    // 如果小数部分大于0，则整数部分加1
    if (decimalPart > 0)
        return integerPart + 1;

    // 如果小数部分为0或者number为负数，则直接返回整数部分
    return integerPart;
}

Hunyuan_int_t floorDown(Hunyuan_real_t number) 
{
    Hunyuan_int_t integerPart = (Hunyuan_int_t)number; // 直接取整数部分
    Hunyuan_real_t decimalPart = number - integerPart; // 计算小数部分
	
	// printf("floorDown %"PRREAL" %"PRIDX" %"PRREAL"\n", number, integerPart, decimalPart);

    // 如果小数部分大于0，说明原数是正数，直接返回整数部分即可
    if (decimalPart > 0)
        return integerPart;
	// 如果小数部分小于0，说明原数是负数，需要减1以实现向下取整
	else
        return (integerPart - 1);
}

/*************************************************************************
* This function returns the log2(x)
**************************************************************************/
Hunyuan_int_t lyj_log2(Hunyuan_int_t a) 
{
    if (a <= 0) 
	{
        fprintf(stderr, "lyj_log2: Input must be greater than 0.\n");
        return -1;
    }

    Hunyuan_int_t i = 0;
    while (a > 1) 
	{
        a >>= 1; // 将 a 右移一位，相当于 a = a / 2
        i++;
    }

    return i;
}

void set_value_int(size_t n, Hunyuan_int_t val, Hunyuan_int_t *src)
{
	size_t i;
	for (i = 0; i < n; i++)
    	src[i] = val;
}

void set_value_double(size_t n, double val, double *src)
{
	size_t i;
	for (i = 0; i < n; i++)
    	src[i] = val;
}

void copy_double(size_t n, double *src, double *dst)
{
	for (size_t i = 0; i < n; i++)
    	dst[i] = src[i];
}

void copy_int(Hunyuan_int_t n, Hunyuan_int_t *src, Hunyuan_int_t *dst)
{
	for (Hunyuan_int_t i = 0; i < n; i++)
    	dst[i] = src[i];
}

Hunyuan_int_t sum_int(size_t n, Hunyuan_int_t *src, Hunyuan_int_t ncon)
{
	Hunyuan_int_t sum = 0;
	for(size_t i = 0;i < n;i++)
		sum += src[i];
	return sum;
}

Hunyuan_real_t sum_real(size_t n, Hunyuan_real_t *src, Hunyuan_int_t ncon)
{
	Hunyuan_real_t sum = 0;
	for(size_t i = 0;i < n;i++)
		sum += src[i];
	return sum;
}

Hunyuan_real_t *rscale_real(size_t n, Hunyuan_real_t wsum, Hunyuan_real_t *dst)
{
	size_t i;
	for(i = 0;i < n;i++, dst += 1)
		(*dst) *= wsum;
	return dst;
}

void select_sort(Hunyuan_int_t *num, Hunyuan_int_t length)
{
	for(Hunyuan_int_t i = 0;i < length;i++)
	{
		Hunyuan_int_t t = i;
		for(Hunyuan_int_t j = i + 1;j < length;j++)
			if(num[j] < num[t]) t = j;
		Hunyuan_int_t z;
		lyj_swap(num[t], num[i],z);
		// printf("i=%d t=%d num: ",i,t);
		// for(Hunyuan_int_t j = 0;j < length;j++)
		// 	printf("%d ",num[j]);
		// printf("\n");
	}
}

void select_sort_val(Hunyuan_int_t *num, Hunyuan_int_t length)
{
	for(Hunyuan_int_t i = 0;i < length;i++)
	{
		Hunyuan_int_t t = i;
		for(Hunyuan_int_t j = i + 1;j < length;j++)
			if(num[j] < num[t]) t = j;
		Hunyuan_int_t z;
		lyj_swap(num[t], num[i], z);
		lyj_swap(num[t + length], num[i + length],z);
	}
}
//	USE_GKRAND ???
void gk_randinit(uint64_t seed)
{
#ifdef USE_GKRAND
  mt[0] = seed;
  for (mti=1; mti<NN; mti++) 
    mt[mti] = (6364136223846793005ULL * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
#else
  srand((unsigned int) seed);
#endif
}

/*************************************************************************/
/*! Initializes the generator */ 
/**************************************************************************/
void isrand(Hunyuan_int_t seed)
{
  gk_randinit((uint64_t) seed);
}

/*************************************************************************/
/*! This function initializes the random number generator 
  */
/*************************************************************************/
void InitRandom(Hunyuan_int_t seed)
{
	isrand((seed == -1 ? 4321 : seed));
}

/* generates a random number on [0, 2^64-1]-Hunyuan_int_terval */
uint64_t gk_randint64(void)
{
#ifdef USE_GKRAND
#else
  return (uint64_t)(((uint64_t) rand()) << 32 | ((uint64_t) rand()));
#endif
}

/* generates a random number on [0, 2^32-1]-Hunyuan_int_terval */
uint32_t gk_randint32(void)
{
#ifdef USE_GKRAND
#else
  return (uint32_t)rand();
#endif
}

/*************************************************************************/
/*! Returns a random number */ 
/**************************************************************************/
Hunyuan_int_t irand()
{
  if (sizeof(Hunyuan_int_t) <= sizeof(int32_t)) 
    return (Hunyuan_int_t)gk_randint32();
  else 
    return (Hunyuan_int_t)gk_randint64(); 
}

Hunyuan_int_t rand_count()
{
	static int ccnt = 0;  
	ccnt++;   
	return ccnt;
}

/*************************************************************************/
/*! Returns a random number between [0, max) */ 
/**************************************************************************/
Hunyuan_int_t irandInRange(Hunyuan_int_t max) 
{
	int t = rand_count(); 
	// if(t % 10000 == 0)  printf("ccnt=%d\n",t);
	return (Hunyuan_int_t)((irand())%max); 
}

/*************************************************************************/
/*! Randomly permutes the elements of an array p[]. 
    flag == 1, p[i] = i prior to permutation, 
    flag == 0, p[] is not initialized. */
/**************************************************************************/
void irandArrayPermute(Hunyuan_int_t n, Hunyuan_int_t *p, Hunyuan_int_t nshuffles, Hunyuan_int_t flag)
{
	Hunyuan_int_t i, u, v;
	Hunyuan_int_t tmp;

	if (flag == 1) 
	{
		for (i = 0; i < n; i++)
			p[i] = (Hunyuan_int_t)i;
	}

	if (n < 10) 
	{
		for (i = 0; i < n; i++) 
		{
			v = irandInRange(n);
			u = irandInRange(n);
			lyj_swap(p[v], p[u], tmp);
		}
	}
	else 
	{
		for (i = 0; i < nshuffles; i++) 
		{
			v = irandInRange(n - 3);
			u = irandInRange(n - 3);
			lyj_swap(p[v + 0], p[u + 2], tmp);
			lyj_swap(p[v + 1], p[u + 3], tmp);
			lyj_swap(p[v + 2], p[u + 0], tmp);
			lyj_swap(p[v + 3], p[u + 1], tmp);
		}
	}
}

#endif