R"(

#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#ifndef INDEX_TYPE
#error INDEX_TYPE undefined!
#endif

#ifndef VALUE_TYPE
#error VALUE_TYPE undefined!
#endif

__kernel
void process_scaninput ( int total,
			__global  VALUE_TYPE *A,
			__global  int *scan_input )
{

    int tid   = get_global_id(0);         // global workitem id
    if (tid >= total)
        return;
	
    if (A[tid] != 0)
       scan_input[tid] = 1;
    else
       scan_input[tid] = 0;
    
}

__kernel
void spread_value( int m, int n, int total,
                  __global VALUE_TYPE *A,
                  __global int *scan_input,
                  __global int *scan_output,
                  __global int *row,
                  __global int *col,
                  __global VALUE_TYPE*val){
					  
	int tid   = get_global_id(0);
	if (scan_input[tid] == 1 && tid < total){
	    int x_idx = tid / n;
            int y_idx = tid % m;
            row[scan_output[tid]] = x_idx;		
            col[scan_output[tid]] = y_idx;
            val[scan_output[tid]] = A[tid];
	}				
}

)"
