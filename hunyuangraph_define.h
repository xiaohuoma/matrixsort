#ifndef _H_DEFINE
#define _H_DEFINE

/*Define functions*/
// #define ALIGNMENT 4
#define hunyuangraph_GPU_cacheline 128
#define SM_NUM 170
#define IMB 1.04
#define OverLoaded 1
#define IDX_MAX   INT32_MAX
#define IDX_MIN   INT32_MIN
#define hunyuangraph_max(m,n) ((m)>=(n)?(m):(n))
#define hunyuangraph_min(m,n) ((m)>=(n)?(n):(m))
#define hunyuangraph_swap(m,n,temp) do{(temp)=(m);(m)=(n);(n)=(temp);} while(0) 
#define hunyuangraph_tocsr(i,n,c) do{for(i=1;i<n;i++)c[i]+= c[i-1];for(i=n;i>0;i--)c[i]=c[i-1];c[0]=0;} while(0)
#define SHIFTCSR(i, n, a) do {for (i=n; i>0; i--) a[i] = a[i-1]; a[0] = 0; } while(0) 
#define hunyuangraph_add_sub(m,n,temp) do{(m)+=(temp);(n)-=(temp);} while(0)
#define hunyuangraph_listinsert(n,list,lptr,i) do{list[n]=i;lptr[i]=(n)++;} while(0) 
#define hunyuangraph_listdelete(n,list,lptr,i) do{list[lptr[i]]=list[--(n)];lptr[list[n]]=lptr[i];lptr[i]=-1;} while(0) 
#define M_GT_N(m,n) ((m)>(n))

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

#define _GKQSORT_SWAP(a, b, t) ((void)((t = *a), (*a = *b), (*b = t)))
#define _GKQSORT_STACK_SIZE	    (8 * sizeof(size_t))
#define _GKQSORT_PUSH(top, low, high) (((top->_lo = (low)), (top->_hi = (high)), ++top))
#define	_GKQSORT_POP(low, high, top)  ((--top, (low = top->_lo), (high = top->_hi)))
#define	_GKQSORT_STACK_NOT_EMPTY	    (_stack < _top)

#define GK_MKQSORT(GKQSORT_TYPE,GKQSORT_BASE,GKQSORT_NELT,GKQSORT_LT)   \
{									\
  GKQSORT_TYPE *const _base = (GKQSORT_BASE);				\
  const size_t _elems = (GKQSORT_NELT);					\
  GKQSORT_TYPE _hold;							\
									\
  if (_elems == 0)                                                      \
    return;                                                             \
  if (_elems > 4) {					\
    GKQSORT_TYPE *_lo = _base;						\
    GKQSORT_TYPE *_hi = _lo + _elems - 1;				\
    struct {								\
      GKQSORT_TYPE *_hi; GKQSORT_TYPE *_lo;				\
    } _stack[_GKQSORT_STACK_SIZE], *_top = _stack + 1;			\
    while (_GKQSORT_STACK_NOT_EMPTY) {					\
      GKQSORT_TYPE *_left_ptr; GKQSORT_TYPE *_right_ptr;		\
      GKQSORT_TYPE *_mid = _lo + ((_hi - _lo) >> 1);			\
      if (GKQSORT_LT (_mid, _lo))					\
        _GKQSORT_SWAP (_mid, _lo, _hold);				\
      if (GKQSORT_LT (_hi, _mid))					\
        _GKQSORT_SWAP (_mid, _hi, _hold);				\
      else								\
        goto _jump_over;						\
      if (GKQSORT_LT (_mid, _lo))					\
        _GKQSORT_SWAP (_mid, _lo, _hold);				\
  _jump_over:;								\
      _left_ptr  = _lo + 1;						\
      _right_ptr = _hi - 1;						\
      do {								\
        while (GKQSORT_LT (_left_ptr, _mid))				\
         ++_left_ptr;							\
        while (GKQSORT_LT (_mid, _right_ptr))				\
          --_right_ptr;							\
        if (_left_ptr < _right_ptr) {					\
          _GKQSORT_SWAP (_left_ptr, _right_ptr, _hold);			\
          if (_mid == _left_ptr)					\
            _mid = _right_ptr;						\
          else if (_mid == _right_ptr)					\
            _mid = _left_ptr;						\
          ++_left_ptr;							\
          --_right_ptr;							\
        }								\
        else if (_left_ptr == _right_ptr) {				\
          ++_left_ptr;							\
          --_right_ptr;							\
          break;							\
        }								\
      } while (_left_ptr <= _right_ptr);				\
      if (_right_ptr - _lo <= 4) {			\
        if (_hi - _left_ptr <= 4)			\
          _GKQSORT_POP (_lo, _hi, _top);				\
        else								\
          _lo = _left_ptr;						\
      }									\
      else if (_hi - _left_ptr <= 4)			\
        _hi = _right_ptr;						\
      else if (_right_ptr - _lo > _hi - _left_ptr) {			\
        _GKQSORT_PUSH (_top, _lo, _right_ptr);				\
        _lo = _left_ptr;						\
      }									\
      else {								\
        _GKQSORT_PUSH (_top, _left_ptr, _hi);				\
        _hi = _right_ptr;						\
      }									\
    }									\
  }									\
  {									\
    GKQSORT_TYPE *const _end_ptr = _base + _elems - 1;			\
    GKQSORT_TYPE *_tmp_ptr = _base;					\
    register GKQSORT_TYPE *_run_ptr;					\
    GKQSORT_TYPE *_thresh;						\
    _thresh = _base + 4;				\
    if (_thresh > _end_ptr)						\
      _thresh = _end_ptr;						\
    for (_run_ptr = _tmp_ptr + 1; _run_ptr <= _thresh; ++_run_ptr)	\
      if (GKQSORT_LT (_run_ptr, _tmp_ptr))				\
        _tmp_ptr = _run_ptr;						\
    if (_tmp_ptr != _base)						\
      _GKQSORT_SWAP (_tmp_ptr, _base, _hold);				\
    _run_ptr = _base + 1;						\
    while (++_run_ptr <= _end_ptr) {					\
      _tmp_ptr = _run_ptr - 1;						\
      while (GKQSORT_LT (_run_ptr, _tmp_ptr))				\
        --_tmp_ptr;							\
      ++_tmp_ptr;							\
      if (_tmp_ptr != _run_ptr) {					\
        GKQSORT_TYPE *_trav = _run_ptr + 1;				\
        while (--_trav >= _run_ptr) {					\
          GKQSORT_TYPE *_hi; GKQSORT_TYPE *_lo;				\
          _hold = *_trav;						\
          for (_hi = _lo = _trav; --_lo >= _tmp_ptr; _hi = _lo)		\
            *_hi = *_lo;						\
          *_hi = _hold;							\
        }								\
      }									\
    }									\
  }									\
}

#endif