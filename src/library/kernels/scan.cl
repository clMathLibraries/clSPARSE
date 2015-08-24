R"(
/* ************************************************************************
* Copyright 2015 Vratis, Ltd.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ************************************************************************ */

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

#ifndef VALUE_TYPE
#error VALUE_TYPE undefined!
#endif

#ifndef SIZE_TYPE
#error SIZE_TYPE undefined!
#endif

#ifndef WG_SIZE
#error WG_SIZE undefined!
#endif
)"

R"(
VALUE_TYPE operation(VALUE_TYPE A, VALUE_TYPE B)
{
#ifdef OP_EW_PLUS
return A + B;
#elif OP_EW_MIN
return min(A, B);
#elif OP_EW_MAX
return max(A, B);
#else
return -99999;
#endif
}
)"

R"(
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void per_block_inclusive_scan( __global VALUE_TYPE* input_ptr,
                                        VALUE_TYPE  identity,
                                  const SIZE_TYPE   vecSize,
                                __local VALUE_TYPE* lds,
                               __global VALUE_TYPE* scanBuffer,
                               __global VALUE_TYPE* scanBuffer1,
                                    int exclusive) // do exclusive scan ?
{
    // 2 thread per element
    size_t gloId = get_global_id(0);
    size_t groId = get_group_id(0);
    size_t locId = get_local_id(0);
    size_t wgSize = get_local_size(0);

    wgSize *=2;

    //input_iter.init( input_ptr );
    size_t offset = 1;

    // load input into shared memory
    if(groId * wgSize + locId < vecSize)
        lds[locId] = input_ptr[groId * wgSize + locId];
    else
        lds[locId] = 0;//input_ptr[vecSize - 1];

    if(groId * wgSize + locId + (wgSize / 2) < vecSize)
        lds[locId + (wgSize / 2)] = input_ptr[groId * wgSize + locId + (wgSize / 2)];
    else
        lds[locId + (wgSize / 2)] = 0;//input_ptr[vecSize - 1];

        // Exclusive case
    if(exclusive && gloId == 0)
    {
        VALUE_TYPE start_val = input_ptr[0];
        lds[locId] = operation(identity, start_val);//(*binaryOp)(identity, start_val);
    }

    for (size_t start = wgSize>>1; start > 0; start >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locId < start)
        {
            size_t temp1 = offset*(2*locId+1)-1;
            size_t temp2 = offset*(2*locId+2)-1;
            VALUE_TYPE y = lds[temp2];
            VALUE_TYPE y1 =lds[temp1];
            lds[temp2] = operation(y, y1);

        }
        offset *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locId == 0)
    {
        scanBuffer[ groId ] = lds[wgSize -1];
        scanBuffer1[ groId ] = lds[wgSize/2 -1];
    }
}
)"


R"(
kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void intra_block_inclusive_scan(
                __global VALUE_TYPE* postSumArray,
                __global VALUE_TYPE* preSumArray,
                VALUE_TYPE identity,
                const SIZE_TYPE vecSize,
                __local VALUE_TYPE* lds,
                const SIZE_TYPE workPerThread)
{
    size_t gloId = get_global_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );
    uint mapId  = gloId * workPerThread;

    // do offset of zero manually
    uint offset;
    VALUE_TYPE workSum;

    if (mapId < vecSize)
    {
        // accumulate zeroth value manually
        offset = 0;
        workSum = preSumArray[mapId+offset];

        //  Serial accumulation
        for ( offset = offset + 1; offset < workPerThread; offset += 1 )
        {
            if (mapId + offset < vecSize)
            {
                VALUE_TYPE y = preSumArray[mapId+offset];
                workSum = operation(workSum,y); //(*binaryOp)( workSum, y );
            }
        }
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    VALUE_TYPE scanSum = workSum;

    lds[ locId ] = workSum;

    offset = 1;
  // scan in lds
    for ( offset = offset*1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        if (mapId < vecSize)
        {
            if (locId >= offset)
            {
                VALUE_TYPE y = lds[ locId - offset ];
                scanSum = operation(scanSum, y);//(*binaryOp)( scanSum, y );
            }
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        lds[ locId ] = scanSum;

    } // for offset
    barrier( CLK_LOCAL_MEM_FENCE );

    // write final scan from pre-scan and lds scan
     workSum = preSumArray[mapId];
     if (locId > 0)
     {
        VALUE_TYPE y = lds[locId-1];
        workSum = operation(workSum, y);//(*binaryOp)(workSum, y);
        postSumArray[ mapId] = workSum;
     }
     else
     {
       postSumArray[ mapId] = workSum;
     }

     for ( offset = 1; offset < workPerThread; offset += 1 )
     {
        barrier( CLK_GLOBAL_MEM_FENCE );

        if (mapId < vecSize && locId > 0)
        {
            VALUE_TYPE y  = preSumArray[ mapId + offset ] ;
            VALUE_TYPE y1 = operation(y, workSum);//(*binaryOp)(y, workSum);
            postSumArray[ mapId + offset ] = y1;
            workSum = y1;

        } // thread in bounds

        else
        {
            VALUE_TYPE y  = preSumArray[ mapId + offset ] ;
            postSumArray[ mapId + offset ] = operation(y, workSum); //(*binaryOp)(y, workSum);
            workSum = postSumArray[ mapId + offset ];
        }

    } // for
})"


R"(
kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void per_block_addition(
                __global VALUE_TYPE* output_ptr,
                __global VALUE_TYPE* input_ptr,
                __global VALUE_TYPE* postSumArray,
                __global VALUE_TYPE* preSumArray,
                __local  VALUE_TYPE* lds,
                const  SIZE_TYPE vecSize,
                int exclusive,
                VALUE_TYPE identity )
{

// 1 thread per element
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );
    //output_iter.init( output_ptr );
    //input_iter.init( input_ptr );

  // if exclusive, load gloId=0 w/ identity, and all others shifted-1
    VALUE_TYPE val;

    if (gloId < vecSize)
    {
        if (exclusive)
        {
            if (gloId > 0)
            { // thread>0
                val = input_ptr[gloId-1];
                lds[ locId ] = val;
            }
            else
            { // thread=0
                val = identity;
                lds[ locId ] = val;
            }
        }
        else
        {
          val = input_ptr[gloId];
          lds[ locId ] = val;
        }
    }

    VALUE_TYPE scanResult = lds[locId];
    VALUE_TYPE postBlockSum, newResult;
    VALUE_TYPE y, y1, sum;

    if (locId == 0 && gloId < vecSize)
    {
        if (groId > 0)
        {
            if (groId % 2 == 0)
                postBlockSum = postSumArray[ groId/2 -1 ];
            else if (groId == 1)
                postBlockSum = preSumArray[0];
            else
            {
                y = postSumArray[ groId/2 -1 ];
                y1 = preSumArray[groId/2];
                postBlockSum = operation(y, y1);//(*binaryOp)(y, y1);
            }

         if (exclusive)
            newResult = postBlockSum;
        else
            newResult = operation(scanResult, postBlockSum); //(*binaryOp)( scanResult, postBlockSum );

        }
        else
        {
            newResult = scanResult;
        }
        lds[ locId ] = newResult;
    }

    //  Computes a scan within a workgroup
    sum = lds[ locId ];
    for ( size_t offset = 1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );

        if (locId >= offset)
        {
            VALUE_TYPE y = lds[ locId - offset ];
            sum = operation(sum, y);//(*binaryOp)( sum, y );
        }

        barrier( CLK_LOCAL_MEM_FENCE );

        lds[ locId ] = sum;
    }

    barrier( CLK_LOCAL_MEM_FENCE );

    //  Abort threads that are passed the end of the input vector
    if (gloId >= vecSize) return;

    output_ptr[ gloId ] = sum;
}
)"



