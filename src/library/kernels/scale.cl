//TEST KERNEL

#ifndef INDEX_TYPE
#error INDEX_TYPE undefined!
#endif

#ifndef VALUE_TYPE
#error VALUE_TYPE undefined!
#endif

#ifndef SIZE_TYPE
#error SIZE_TYPE undefined!
#endif

// v = v*alpha
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void scale ( __global VALUE_TYPE* v,
             __global VALUE_TYPE* alpha,
             const SIZE_TYPE size)
{
    const int i = get_global_id(0);

    if (i >= size) return;

    v[i] = v[i]* alpha[0];
}
