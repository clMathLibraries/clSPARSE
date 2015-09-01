R"(
//////////////////////////////////////////////////////////////////////////
// < A CUDA/OpenCL General Sparse Matrix-Matrix Multiplication Program >
//
// < See paper:
// Weifeng Liu and Brian Vinter, "An Efficient GPU General Sparse
// Matrix-Matrix Multiplication for Irregular Data," Parallel and
// Distributed Processing Symposium, 2014 IEEE 28th International
// (IPDPS '14), pp.370-381, 19-23 May 2014
// for details. >
//////////////////////////////////////////////////////////////////////////

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define TUPLE_QUEUE 6
//typedef double   vT;
#define vT float

inline
void binarysearch(__local int   *s_key,
                  __local vT *s_val,
                  int            key_input,
                  vT          val_input,
                  int            merged_size,
                  bool          *is_new_col)
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = s_key[median];
        
        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            s_val[median] += val_input;
            *is_new_col = 0;
            break;
        }
    }
    //return start;
}

inline
void binarysearch_sub(__local int   *s_key,
                  __local vT *s_val,
                  int            key_input,
                  vT          val_input,
                  int            merged_size)
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = s_key[median];

        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            s_val[median] -= val_input;
            break;
        }
    }
    //return start;
}

inline
void binarysearch_global(__global int   *d_key,
                  __global vT *d_val,
                  int            key_input,
                  vT          val_input,
                  int            merged_size,
                  bool          *is_new_col)
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = d_key[median];

        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            d_val[median] += val_input;
            *is_new_col = 0;
            break;
        }
    }
    //return start;
}

inline
void binarysearch_global_sub(__global int   *d_key,
                  __global vT *d_val,
                  int            key_input,
                  vT          val_input,
                  int            merged_size)
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = d_key[median];

        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            d_val[median] -= val_input;
            break;
        }
    }
    //return start;
}

inline
void scan_32(__local volatile short *s_scan,
              const int      local_id)
{
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    if (local_id < 16)  { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[31] += s_scan[15]; s_scan[32] = s_scan[31]; s_scan[31] = 0; temp = s_scan[15]; s_scan[15] = 0; s_scan[31] += temp; }
    if (local_id < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16)  { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

inline
void scan_64(__local volatile short *s_scan,
              const int      local_id)
{
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    if (local_id < 32) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[63] += s_scan[31]; s_scan[64] = s_scan[63]; s_scan[63] = 0; temp = s_scan[31]; s_scan[31] = 0; s_scan[63] += temp; }
    if (local_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

inline
void scan_128(__local volatile short *s_scan,
              const int      local_id)
{
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    if (local_id < 64) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[127] += s_scan[63]; s_scan[128] = s_scan[127]; s_scan[127] = 0; temp = s_scan[63]; s_scan[63] = 0; s_scan[127] += temp; }
    if (local_id < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

inline
void scan_256(__local volatile short *s_scan,
              const int      local_id)
{
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    if (local_id < 128) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }
    if (local_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 128) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}


inline
void scan(__local volatile short *s_scan,
          const int               local_id,
          const int               local_size)
{
    switch (local_size)
    {
    case 32:
        scan_32(s_scan, local_id);
        break;
    case 64:
        scan_64(s_scan, local_id);
        break;
    case 128:
        scan_128(s_scan, local_id);
        break;
    case 256:
        scan_256(s_scan, local_id);
        break;
    }
}

inline
int y_pos(const int x_pos,
          const int b_length,
          const int offset)
{
    int pos = b_length - (x_pos + b_length - offset); //offset - x_pos
    return pos > b_length ? b_length : pos;
}

inline
int mergepath_partition_liu(__local int   *s_a_key,
                            const int      a_length,
                            __local int   *s_b_key,
                            const int      b_length,
                            const int      offset)
{
    int x_start = offset > b_length ? offset - b_length : 0;
    int x_stop  = offset > a_length ? a_length : offset;
    
    int x_median;
    
    while (x_stop >= x_start)
    {
        x_median = (x_stop + x_start) / 2;
        
        if (s_a_key[x_median] > s_b_key[y_pos(x_median, b_length, offset) - 1])
        {
            if (s_a_key[x_median - 1] < s_b_key[y_pos(x_median, b_length, offset)])
            {
                break;
            }
            else
            {
                x_stop = x_median - 1;
            }
        }
        else
        {
            x_start = x_median + 1;
        }
    }
    
    return x_median;
}

inline
void mergepath_serialmerge_liu(__local int          *s_a_key,
                               __local vT           *s_a_val,
                               const int             a_length,
                               __local int          *s_b_key,
                               __local vT           *s_b_val,
                               const int             b_length,
                               int                  *reg_key,
                               vT                   *reg_val)
{
    int a_pointer = 0;
    int b_pointer = 0;
    
    for (int c_pointer = 0; c_pointer < a_length + b_length; c_pointer++)
    {
        if (a_pointer < a_length && (b_pointer >= b_length || s_a_key[a_pointer] <= s_b_key[b_pointer]))
        {
            reg_key[c_pointer] = s_a_key[a_pointer];
            reg_val[c_pointer] = s_a_val[a_pointer];
            a_pointer += 1;
        }
        else
        {
            reg_key[c_pointer] = s_b_key[b_pointer];
            reg_val[c_pointer] = s_b_val[b_pointer];
            b_pointer += 1;
        }
    }
}

inline
void mergepath_liu(__local int          *s_a_key,
                   __local vT               *s_a_val,
                   const int                 a_length,
                   __local int              *s_b_key,
                   __local vT               *s_b_val,
                   const int                 b_length,
                   __local  int             *s_a_border,
                   __local  int             *s_b_border,
                   int                      *reg_key,
                   vT                       *reg_val)//,
                   //__global int          *d_a_border,
                   //__global int          *d_b_border)
{
    if (b_length == 0)
        return;

    if (s_a_key[a_length-1] < s_b_key[0])
        return;

    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    int delta = ceil((float)(a_length + b_length) / (float)local_size);
    int active_threads = ceil((float)(a_length + b_length) / (float)delta);

    int offset = delta * local_id;

    int a_start, a_stop, b_start, b_stop;

    if (!local_id)
    {
        s_a_border[active_threads] = a_length;
        s_b_border[active_threads] = b_length;
        //d_a_border[active_threads] = s_a_border[active_threads];
        //d_b_border[active_threads] = s_b_border[active_threads];
    }

    if (local_id < active_threads)
    {
        s_a_border[local_id] = a_start = mergepath_partition_liu(s_a_key, a_length, s_b_key, b_length, offset);
        s_b_border[local_id] = b_start = y_pos(s_a_border[local_id], b_length, offset);
        //d_a_border[local_id] = s_a_border[local_id];
        //d_b_border[local_id] = s_b_border[local_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < active_threads)
    {
        a_stop = s_a_border[local_id+1];
        b_stop = s_b_border[local_id+1];
    }

    if (local_id < active_threads)
    {
        mergepath_serialmerge_liu(&s_a_key[a_start],
                                  &s_a_val[a_start],
                                  a_stop - a_start,
                                  &s_b_key[b_start],
                                  &s_b_val[b_start],
                                  b_stop - b_start,
                                  reg_key, reg_val);

        //d_a_border[local_id] = (s_a_border[local_id+1] - s_a_border[local_id]);
        //d_b_border[local_id] = (s_b_border[local_id+1] - s_b_border[local_id]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (local_id == active_threads - 1)
    //{
    //    for (int i = 0; i < (s_b_border[local_id+1] - s_b_border[local_id]); i++)
    //    {
    //        d_b_border[i] = reg_key[i];
    //    }
    //}

    if (local_id < active_threads)
    {
        for (int is = 0; is < (a_stop - a_start) + (b_stop - b_start); is++)
        {
            s_a_key[offset + is] = reg_key[is];
            s_a_val[offset + is] = reg_val[is];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
)"

R"(

inline
int mergepath_partition_global_liu(__global int   *s_a_key,
                            const int      a_length,
                            __global int   *s_b_key,
                            const int      b_length,
                            const int      offset)
{
    int x_start = offset > b_length ? offset - b_length : 0;
    int x_stop  = offset > a_length ? a_length : offset;
    
    int x_median;
    
    while (x_stop >= x_start)
    {
        x_median = (x_stop + x_start) / 2;
        
        if (s_a_key[x_median] > s_b_key[y_pos(x_median, b_length, offset) - 1])
        {
            if (s_a_key[x_median - 1] < s_b_key[y_pos(x_median, b_length, offset)])
            {
                break;
            }
            else
            {
                x_stop = x_median - 1;
            }
        }
        else
        {
            x_start = x_median + 1;
        }
    }
    
    return x_median;
}

inline
void mergepath_serialmerge_global_liu(__global int          *s_a_key,
                               __global vT           *s_a_val,
                               const int             a_length,
                               __global int          *s_b_key,
                               __global vT           *s_b_val,
                               const int             b_length,
                               int                  *reg_key,
                               vT                   *reg_val)
{
    int a_pointer = 0;
    int b_pointer = 0;
    
    for (int c_pointer = 0; c_pointer < a_length + b_length; c_pointer++)
    {
        if (a_pointer < a_length && (b_pointer >= b_length || s_a_key[a_pointer] <= s_b_key[b_pointer]))
        {
            reg_key[c_pointer] = s_a_key[a_pointer];
            reg_val[c_pointer] = s_a_val[a_pointer];
            a_pointer += 1;
        }
        else
        {
            reg_key[c_pointer] = s_b_key[b_pointer];
            reg_val[c_pointer] = s_b_val[b_pointer];
            b_pointer += 1;
        }
    }
}

inline
void mergepath_global_liu(__global int          *s_a_key,
                   __global vT               *s_a_val,
                   const int                 a_length,
                   __global int              *s_b_key,
                   __global vT               *s_b_val,
                   const int                 b_length,
                   __local  int             *s_a_border,
                   __local  int             *s_b_border,
                   int                      *reg_key,
                   vT                       *reg_val)
{
    if (b_length == 0)
        return;

    if (s_a_key[a_length-1] < s_b_key[0])
        return;

    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    int delta = ceil((float)(a_length + b_length) / (float)local_size);
    int active_threads = ceil((float)(a_length + b_length) / (float)delta);

    int offset = delta * local_id;

    int a_start, a_stop, b_start, b_stop;

    if (!local_id)
    {
        s_a_border[active_threads] = a_length;
        s_b_border[active_threads] = b_length;
    }

    if (local_id < active_threads)
    {
        s_a_border[local_id] = a_start = mergepath_partition_global_liu(s_a_key, a_length, s_b_key, b_length, offset);
        s_b_border[local_id] = b_start = y_pos(s_a_border[local_id], b_length, offset);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < active_threads)
    {
        a_stop = s_a_border[local_id+1];
        b_stop = s_b_border[local_id+1];
    }

    if (local_id < active_threads)
    {
        mergepath_serialmerge_global_liu(&s_a_key[a_start],
                                  &s_a_val[a_start],
                                  a_stop - a_start,
                                  &s_b_key[b_start],
                                  &s_b_val[b_start],
                                  b_stop - b_start,
                                  reg_key, reg_val);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < active_threads)
    {
        for (int is = 0; is < (a_stop - a_start) + (b_stop - b_start); is++)
        {
            s_a_key[offset + is] = reg_key[is];
            s_a_val[offset + is] = reg_val[is];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}


inline
void mergepath_global_2level_liu(__global int          *s_a_key,
                   __global vT               *s_a_val,
                   const int                 a_length,
                   __global int              *s_b_key,
                   __global vT               *s_b_val,
                   const int                 b_length,
                   __local  int             *s_a_border,
                   __local  int             *s_b_border,
                   int                      *reg_key,
                   vT                       *reg_val,
                   __local  int             *s_key,
                   __local  vT              *s_val,
                   __global int              *d_temp_key,
                   __global vT               *d_temp_val)
{
    if (b_length == 0)
        return;

    if (s_a_key[a_length-1] < s_b_key[0])
        return;

    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    int delta_2level = local_size * 9;
    int loop_2level = ceil((float)(a_length + b_length) / (float)delta_2level);

    int a_border_2level_l, b_border_2level_l, a_border_2level_r, b_border_2level_r;

    for (int i = 0; i < loop_2level; i++)
    {
        // compute `big' borders
        int offset_2level = delta_2level * i;
        a_border_2level_l = i == 0 ? 0 : a_border_2level_r; //mergepath_partition_global_liu(s_a_key, a_length, s_b_key, b_length, offset_2level);
        b_border_2level_l = i == 0 ? 0 : b_border_2level_r; //y_pos(a_border_2level_l, b_length, offset_2level);

        int offset_2level_next = delta_2level * (i + 1);

        if (i == (loop_2level - 1)){
            a_border_2level_r = a_length;
            b_border_2level_r = b_length;
        }
        else
        {
           s_a_border[local_id] = a_border_2level_r = local_id < 64 ? mergepath_partition_global_liu(s_a_key, a_length, s_b_key, b_length, offset_2level_next) : 0;
           barrier(CLK_LOCAL_MEM_FENCE);
           a_border_2level_r = local_id < 64 ? a_border_2level_r : s_a_border[local_id % 64];
           b_border_2level_r = y_pos(a_border_2level_r, b_length, offset_2level_next);
        }

        //barrier(CLK_GLOBAL_MEM_FENCE);

        // load entries in the borders
        int a_size = a_border_2level_r - a_border_2level_l;
        int b_size = b_border_2level_r - b_border_2level_l;

        for (int j = local_id; j < a_size; j += local_size)
        {
            s_key[j] = s_a_key[a_border_2level_l + j];
            s_val[j] = s_a_val[a_border_2level_l + j];
        }

        for (int j = local_id; j < b_size; j += local_size)
        {
            s_key[a_size + j] = s_b_key[b_border_2level_l + j];
            s_val[a_size + j] = s_b_val[b_border_2level_l + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // merge path in local mem
        mergepath_liu(s_key, s_val, a_size,
                      &s_key[a_size], &s_val[a_size], b_size,
                      s_a_border, s_b_border, reg_key, reg_val);
        barrier(CLK_LOCAL_MEM_FENCE);

        // dump the merged part to device mem (temp)
        for (int j = local_id; j < a_size + b_size; j += local_size)
        {
            d_temp_key[offset_2level + j] = s_key[j];
            d_temp_val[offset_2level + j] = s_val[j];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // dump the temp data to the target place, both in device mem
    for (int j = local_id; j < a_length + b_length; j += local_size)
    {
        s_a_key[j] = d_temp_key[j];
        s_a_val[j] = d_temp_val[j];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

inline
void readwrite_mergedlist(__global int   *d_csrColIndCt,
                          __global vT *d_csrValCt,
                          __local  int   *s_key_merged,
                          __local  vT *s_val_merged,
                          const int       merged_size,
                          const int       row_offset,
                          const bool      is_write)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    int stride, offset_local_id, global_offset;
    int loop = ceil((float)merged_size / (float)local_size);

    for (int i = 0; i < loop; i++)
    {
        stride = i != loop - 1 ? local_size : merged_size - i * local_size;
        offset_local_id = i * local_size + local_id;
        global_offset = row_offset + offset_local_id;

        if (local_id < stride)
        {
            if (is_write)
            {
                d_csrColIndCt[global_offset] = s_key_merged[offset_local_id];
                d_csrValCt[global_offset]    = s_val_merged[offset_local_id];
            }
            else
            {
                s_key_merged[offset_local_id] = d_csrColIndCt[global_offset];
                s_val_merged[offset_local_id] = d_csrValCt[global_offset];
            }
        }
    }
}

inline
void readwrite_mergedlist_global(__global int   *d_csrColIndCt,
                          __global vT *d_csrValCt,
                          __global  int   *d_key_merged,
                          __global  vT *d_val_merged,
                          const int       merged_size,
                          const int       row_offset,
                          const bool      is_write)
{
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    int stride, offset_local_id, global_offset;
    int loop = ceil((float)merged_size / (float)local_size);

    for (int i = 0; i < loop; i++)
    {
        stride = i != loop - 1 ? local_size : merged_size - i * local_size;
        offset_local_id = i * local_size + local_id;
        global_offset = row_offset + offset_local_id;

        if (local_id < stride)
        {
            if (is_write)
            {
                d_csrColIndCt[global_offset] = d_key_merged[offset_local_id];
                d_csrValCt[global_offset]    = d_val_merged[offset_local_id];
            }
            else
            {
                d_key_merged[offset_local_id] = d_csrColIndCt[global_offset];
                d_val_merged[offset_local_id] = d_csrValCt[global_offset];
            }
        }
    }
}
)"

R"(

__kernel
void EM_mergepath(__global int            *d_queue,
                  __global const int      *d_csrRowPtrA,
                  __global const int      *d_csrColIndA,
                  __global const vT    *d_csrValA,
                  __global const int      *d_csrRowPtrB,
                  __global const int      *d_csrColIndB,
                  __global const vT    *d_csrValB,
                  __global int            *d_csrRowPtrC,
                  __global const int      *d_csrRowPtrCt,
                  __global int            *d_csrColIndCt,
                  __global vT          *d_csrValCt,
                  __local  int            *s_key_merged,        // BUFFSIZE_MERGED
                  __local  vT          *s_val_merged,        // BUFFSIZE_MERGED
                  __local  volatile short *s_scan,              // LOCAL_SIZE + 1
                  const int                queue_offset,
                  const int                mergebuffer_size,
                  __local  int              *s_a_border,
                  __local  int              *s_b_border)
{
    int queue_id = TUPLE_QUEUE * (queue_offset + get_group_id(0));

    // if merged size equals -1, kernel return since this row is done
    int merged_size = d_queue[queue_id + 2];

    int local_id = get_local_id(0); //threadIdx.x;
    int row_id = d_queue[queue_id];

    int   local_size = get_local_size(0);
    float local_size_float = local_size;

    int stride, loop;
    int reg_reuse1;

    int   col_Ct;      // index_type
    vT val_Ct;      // value_type
    vT val_A;       // value_type

    int start_col_index_A, stop_col_index_A;  // index_type
    int start_col_index_B, stop_col_index_B;  // index_type

    int k, is;

    bool  is_new_col;
    bool  is_last;
    int   VT, diag, mp;
    int   reg_key[9];
    vT reg_val[9];

    start_col_index_A = d_csrRowPtrA[row_id];
    stop_col_index_A  = d_csrRowPtrA[row_id + 1];

    if (merged_size == 0)
    {
        is_last = false;

        // read the first set of current nnzCt row to merged list
        reg_reuse1 = d_csrColIndA[start_col_index_A];      // reg_reuse1 = row_id_B
        val_A    = d_csrValA[start_col_index_A];

        start_col_index_B = d_csrRowPtrB[reg_reuse1];      // reg_reuse1 = row_id_B
        stop_col_index_B  = d_csrRowPtrB[reg_reuse1 + 1];  // reg_reuse1 = row_id_B

        stride = stop_col_index_B - start_col_index_B;
        loop   = ceil(stride / local_size_float); //ceil((float)stride / (float)local_size);

        start_col_index_B += local_id;

        for (k = 0; k < loop; k++)
        {
            reg_reuse1 = k != loop - 1 ? local_size : stride - k * local_size; // reg_reuse1 = input_size

            // if merged_size + reg_reuse1 > mergebuffer_size, write it to global mem and return
            if (merged_size + reg_reuse1 > mergebuffer_size)
            {
                // write a signal to some place, not equals -1 means next round is needed
                if (local_id == 0)
                {
                    d_queue[queue_id + 2] = merged_size;
                    d_queue[queue_id + 3] = start_col_index_A;
                    d_queue[queue_id + 4] = start_col_index_B;
                }

                // dump current data to global mem
                reg_reuse1 = d_queue[queue_id + 1];
                readwrite_mergedlist(d_csrColIndCt, d_csrValCt, s_key_merged, s_val_merged, merged_size, reg_reuse1, 1);

                return;
            }

            if (start_col_index_B < stop_col_index_B)
            {
                col_Ct = d_csrColIndB[start_col_index_B];
                val_Ct = d_csrValB[start_col_index_B] * val_A;

                s_key_merged[merged_size + local_id] = col_Ct;
                s_val_merged[merged_size + local_id] = val_Ct;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            merged_size += reg_reuse1;   // reg_reuse1 = input_size
            start_col_index_B += local_size;
        }

        start_col_index_A++;
    }
    else
    {
        is_last = true;
        start_col_index_A = d_queue[queue_id + 3];

        // load existing merged list
        reg_reuse1 = d_queue[queue_id + 5];
        readwrite_mergedlist(d_csrColIndCt, d_csrValCt, s_key_merged, s_val_merged, merged_size, reg_reuse1, 0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // merge the rest of sets of current nnzCt row to the merged list
    while (start_col_index_A < stop_col_index_A)
    {
        reg_reuse1 = d_csrColIndA[start_col_index_A];                      // reg_reuse1 = row_id_B
        val_A    = d_csrValA[start_col_index_A];

        start_col_index_B = is_last ? d_queue[queue_id + 4] : d_csrRowPtrB[reg_reuse1];      // reg_reuse1 = row_id_B
        is_last = false;
        stop_col_index_B  = d_csrRowPtrB[reg_reuse1 + 1];  // reg_reuse1 = row_id_B

        stride = stop_col_index_B - start_col_index_B;
        loop  = ceil(stride / local_size_float); //ceil((float)stride / (float)local_size);

        start_col_index_B += local_id;

        for (k = 0; k < loop; k++)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            is_new_col = 0;

            if (start_col_index_B < stop_col_index_B)
            {
                col_Ct = d_csrColIndB[start_col_index_B];
                val_Ct = d_csrValB[start_col_index_B] * val_A;

                // binary search on existing sorted list
                // if the column is existed, add the value to the position
                // else, set scan value to 1, and wait for scan
                is_new_col = 1;
                binarysearch(s_key_merged, s_val_merged, col_Ct, val_Ct, merged_size, &is_new_col);
            }

            s_scan[local_id] = is_new_col;
            barrier(CLK_LOCAL_MEM_FENCE);

            // scan with half-local_size work-items
            // s_scan[local_size] is the size of input non-duplicate array
            scan(s_scan, local_id, local_size);
            barrier(CLK_LOCAL_MEM_FENCE);

            // if all elements are absorbed into merged list,
            // the following work in this inner-loop is not needed any more
            if (s_scan[local_size] == 0)
            {
                start_col_index_B += local_size;
                continue;
            }

            // check if the total size is larger than the capicity of merged list
            if (merged_size + s_scan[local_size] > mergebuffer_size)
            {
                // roll back 'binary serach plus' in this round
                if (start_col_index_B < stop_col_index_B)
                {
                    binarysearch_sub(s_key_merged, s_val_merged, col_Ct, val_Ct, merged_size);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                // write a signal to some place, not equals -1 means next round is needed
                if (local_id == 0)
                {
                    d_queue[queue_id + 2] = merged_size;
                    d_queue[queue_id + 3] = start_col_index_A;
                    d_queue[queue_id + 4] = start_col_index_B;
                }

                // dump current data to global mem
                reg_reuse1 = d_queue[queue_id + 1]; //d_csrRowPtrCt[row_id];
                readwrite_mergedlist(d_csrColIndCt, d_csrValCt, s_key_merged, s_val_merged, merged_size, reg_reuse1, 1);

                return;
            }

            // write compact input to free place in merged list
            if(is_new_col)
            {
                reg_reuse1 = merged_size + s_scan[local_id];
                s_key_merged[reg_reuse1] = col_Ct;
                s_val_merged[reg_reuse1] = val_Ct;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // merge path partition
            
            reg_reuse1 = s_scan[local_size]; // reg_reuse1 = size_b;
            
            mergepath_liu(s_key_merged, s_val_merged, merged_size, 
                          &s_key_merged[merged_size], &s_val_merged[merged_size], reg_reuse1,
                          s_a_border, s_b_border, reg_key, reg_val); //, d_a_border, d_b_border);
             

            merged_size += reg_reuse1; // reg_reuse1 = size_b = s_scan[local_size];
            start_col_index_B += local_size;
        }

        start_col_index_A++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0)
    {
        d_csrRowPtrC[row_id] = merged_size;
        d_queue[queue_id + 2] = -1;
    }

    // write merged list to global mem
    reg_reuse1 = d_queue[queue_id + 1]; //d_csrRowPtrCt[row_id];
    readwrite_mergedlist(d_csrColIndCt, d_csrValCt, s_key_merged, s_val_merged, merged_size, reg_reuse1, 1);
}
)"

R"(

__kernel
void EM_mergepath_global(__global int            *d_queue,
                  __global const int      *d_csrRowPtrA,
                  __global const int      *d_csrColIndA,
                  __global const vT    *d_csrValA,
                  __global const int      *d_csrRowPtrB,
                  __global const int      *d_csrColIndB,
                  __global const vT    *d_csrValB,
                  __global int            *d_csrRowPtrC,
                  __global const int      *d_csrRowPtrCt,
                  __global int            *d_csrColIndCt,
                  __global vT          *d_csrValCt,
                  __local  int            *s_key_merged_l1,        // mergebuffer_size
                  __local  vT          *s_val_merged_l1,        // mergebuffer_size
                  __local  volatile short *s_scan,              // LOCAL_SIZE + 1
                  const int                queue_offset,
                         const int                mergebuffer_size,
                         __local  int              *s_a_border,
                         __local  int              *s_b_border)
{
    int queue_id = TUPLE_QUEUE * (queue_offset + get_group_id(0));

    // if merged size equals -1, kernel return since this row is done
    int merged_size_l2 = d_queue[queue_id + 2];
    int merged_size_l1 = 0;

    int local_id = get_local_id(0); //threadIdx.x;
    int row_id = d_queue[queue_id];

    int   local_size = get_local_size(0);
    float local_size_float = local_size;

    int stride, loop;
    int reg_reuse1;

    int   col_Ct;      // index_type
    vT val_Ct;      // value_type
    vT val_A;       // value_type

    int start_col_index_A, stop_col_index_A;  // index_type
    int start_col_index_B, stop_col_index_B;  // index_type

    int k, is;

    bool  is_new_col;
    bool  is_last;
    int   VT, diag, mp;
    int   reg_key[9];
    vT reg_val[9];

    start_col_index_A = d_csrRowPtrA[row_id];
    stop_col_index_A  = d_csrRowPtrA[row_id + 1];

    is_last = true;
    start_col_index_A = d_queue[queue_id + 3];

    // load existing merged list
    reg_reuse1 = d_queue[queue_id + 1];
    __global int   *d_key_merged = &d_csrColIndCt[reg_reuse1];
    __global vT *d_val_merged = &d_csrValCt[reg_reuse1];

    reg_reuse1 = d_queue[queue_id + 5];
    readwrite_mergedlist_global(d_csrColIndCt, d_csrValCt, d_key_merged, d_val_merged, merged_size_l2, reg_reuse1, 0);
    barrier(CLK_GLOBAL_MEM_FENCE);

    // merge the rest of sets of current nnzCt row to the merged list
    while (start_col_index_A < stop_col_index_A)
    {
        reg_reuse1 = d_csrColIndA[start_col_index_A];                      // reg_reuse1 = row_id_B
        val_A    = d_csrValA[start_col_index_A];

        start_col_index_B = is_last ? d_queue[queue_id + 4] : d_csrRowPtrB[reg_reuse1];      // reg_reuse1 = row_id_B
        is_last = false;
        stop_col_index_B  = d_csrRowPtrB[reg_reuse1 + 1];  // reg_reuse1 = row_id_B

        stride = stop_col_index_B - start_col_index_B;
        loop  = ceil(stride / local_size_float); //ceil((float)stride / (float)local_size);

        start_col_index_B += local_id;

        for (k = 0; k < loop; k++)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            is_new_col = 0;

            if (start_col_index_B < stop_col_index_B)
            {
                col_Ct = d_csrColIndB[start_col_index_B];
                val_Ct = d_csrValB[start_col_index_B] * val_A;

                // binary search on existing sorted list
                // if the column is existed, add the value to the position
                // else, set scan value to 1, and wait for scan
                is_new_col = 1;

                // search on l2
                binarysearch_global(d_key_merged, d_val_merged, col_Ct, val_Ct, merged_size_l2, &is_new_col);

                // search on l1
                if (is_new_col == 1)
                    binarysearch(s_key_merged_l1, s_val_merged_l1, col_Ct, val_Ct, merged_size_l1, &is_new_col);
            }

            s_scan[local_id] = is_new_col;
            barrier(CLK_LOCAL_MEM_FENCE);

            // scan with half-local_size work-items
            // s_scan[local_size] is the size of input non-duplicate array
            scan(s_scan, local_id, local_size);
            barrier(CLK_LOCAL_MEM_FENCE);

            // if all elements are absorbed into merged list,
            // the following work in this inner-loop is not needed any more
            if (s_scan[local_size] == 0)
            {
                start_col_index_B += local_size;
                continue;
            }

            // check if the total size is larger than the capicity of merged list
            if (merged_size_l1 + s_scan[local_size] > mergebuffer_size)
            {
                if (start_col_index_B < stop_col_index_B)
                {
                    // rollback on l2
                    binarysearch_global_sub(d_key_merged, d_val_merged, col_Ct, val_Ct, merged_size_l2);

                    // rollback on l1
                    binarysearch_sub(s_key_merged_l1, s_val_merged_l1, col_Ct, val_Ct, merged_size_l1);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                // write a signal to some place, not equals -1 means next round is needed
                if (local_id == 0)
                {
                    d_queue[queue_id + 2] = merged_size_l2 + merged_size_l1;
                    d_queue[queue_id + 3] = start_col_index_A;
                    d_queue[queue_id + 4] = start_col_index_B;
                }

                // dump l1 to global
                readwrite_mergedlist(d_key_merged, d_val_merged, s_key_merged_l1, s_val_merged_l1,
                                     merged_size_l1, merged_size_l2, 1);
                barrier(CLK_GLOBAL_MEM_FENCE);
                
                mergepath_global_2level_liu(d_key_merged, d_val_merged, merged_size_l2,
                                     &d_key_merged[merged_size_l2], &d_val_merged[merged_size_l2], merged_size_l1,
                                     s_a_border, s_b_border,
                                     reg_key, reg_val,
                                     s_key_merged_l1, s_val_merged_l1,
                                     &d_key_merged[merged_size_l2 + merged_size_l1],
                                     &d_val_merged[merged_size_l2 + merged_size_l1]);

                return;
            }

            // write compact input to free place in merged list
            if(is_new_col)
            {
                reg_reuse1 = merged_size_l1 + s_scan[local_id];
                s_key_merged_l1[reg_reuse1] = col_Ct;
                s_val_merged_l1[reg_reuse1] = val_Ct;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // merge path partition on l1
            reg_reuse1 = s_scan[local_size]; // reg_reuse1 = size_b;

            mergepath_liu(s_key_merged_l1, s_val_merged_l1, merged_size_l1, 
                          &s_key_merged_l1[merged_size_l1], &s_val_merged_l1[merged_size_l1], reg_reuse1,
                          s_a_border, s_b_border, reg_key, reg_val);

            merged_size_l1 += reg_reuse1; // reg_reuse1 = size_b = s_scan[local_size];
            start_col_index_B += local_size;
        }

        start_col_index_A++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0)
    {
        d_csrRowPtrC[row_id] = merged_size_l2 + merged_size_l1;
        d_queue[queue_id + 2] = -1;
    }

    // dump l1 to global
    readwrite_mergedlist(d_key_merged, d_val_merged, s_key_merged_l1, s_val_merged_l1,
                         merged_size_l1, merged_size_l2, 1);
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    mergepath_global_2level_liu(d_key_merged, d_val_merged, merged_size_l2,
                  &d_key_merged[merged_size_l2], &d_val_merged[merged_size_l2], merged_size_l1,
                  s_a_border, s_b_border,
                  reg_key, reg_val,
                  s_key_merged_l1, s_val_merged_l1,
                  &d_key_merged[merged_size_l2 + merged_size_l1],
                  &d_val_merged[merged_size_l2 + merged_size_l1]);
}

)"
