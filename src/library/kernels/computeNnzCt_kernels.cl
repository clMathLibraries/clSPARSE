R"(

#define GROUPSIZE_256 256

__kernel
void compute_nnzCt_kernel(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global int *d_csrRowPtrCt,
        const int m)
{
    __local int s_csrRowPtrA[GROUPSIZE_256+1];
    int global_id = get_global_id(0);
    int start, stop, index, strideB, row_size_Ct = 0;

    if (global_id < m)
    {
        start = d_csrRowPtrA[global_id];
        stop = d_csrRowPtrA[global_id + 1];

        for (int i = start; i < stop; i++)
        {
            index = d_csrColIndA[i];
            strideB = d_csrRowPtrB[index + 1] - d_csrRowPtrB[index];
            row_size_Ct += strideB;
        }

        d_csrRowPtrCt[global_id] = row_size_Ct;
    }

    if (global_id == 0)
        d_csrRowPtrCt[m] = 0;
}

)"
