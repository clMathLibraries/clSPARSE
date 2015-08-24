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

#ifndef KEY_TYPE
#error KEY_TYPE undefined!
#endif

#ifndef WG_SIZE
#error WG_SIZE undefined!
#endif
)"


R"(
//Kernels puts 1 in the places where the kay is changing;
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void offset_calculation(global KEY_TYPE* keys,
                        global KEY_TYPE* output,
                        const  SIZE_TYPE size)
{
    size_t gloId = get_global_id( 0 );

    if (gloId >= size) return;

    KEY_TYPE key, prev_key;

    if(gloId > 0)
    {
        key = keys[ gloId ];
        prev_key = keys[ gloId - 1];

        if(key == prev_key)
            output[ gloId ] = 0;
        else
            output[ gloId ] = 1;
    }
    else
    {
         output[ gloId ] = 0;
    }
}
)"

R"(
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void per_block_scan_by_key( global KEY_TYPE *keys,          //offset array
                            global VALUE_TYPE *vals,        //input values
                            global VALUE_TYPE *output,      //offsetValues
                            const SIZE_TYPE size,           //size of the system
                            local KEY_TYPE* ldsKeys,        //local mem for I keys
                            local VALUE_TYPE* ldsVals,      //temp Values
                            global KEY_TYPE *keyBuffer,     //partial sum array for keys
                            global VALUE_TYPE *valBuffer)   //partial sum array for values
{
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    KEY_TYPE key;
    VALUE_TYPE val;

    if(gloId < size)
    {
        key = keys[ gloId ];
        val = vals[ gloId ];
        ldsKeys[ locId ] = key;
        ldsVals[ locId ] = val;
    }
    else
    {
        ldsKeys[ locId ] = keys[size-1];
        ldsVals[ locId ] = 0;
    }

    // Computes a scan within a workgroup
    // updates vals in lds but not keys
    VALUE_TYPE sum = val;
    for( size_t offset = 1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        KEY_TYPE key2 = ldsKeys[locId - offset];
        if (locId >= offset && key == key2)
        {
            VALUE_TYPE y = ldsVals[ locId - offset ];
            sum = sum + y;
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        ldsVals[ locId ] = sum;
    }
    barrier( CLK_LOCAL_MEM_FENCE ); // needed for large data types

    //  Abort threads that are passed the end of the input vector
    if (gloId >= size) return;

    // Each work item writes out its calculated scan result, relative to the beginning
    // of each work group
    KEY_TYPE key2 = -1;
    if (gloId < size -1 )
        key2 = keys[gloId + 1];
    if(key != key2)
       output[ gloId ] = sum;

    if (locId == 0)
    {
        keyBuffer[ groId ] = ldsKeys[ wgSize-1 ];
        valBuffer[ groId ] = ldsVals[ wgSize-1 ];
    }
}

)"

R"(
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void intra_block_inclusive_scan_by_key(
      global KEY_TYPE*   keySumArray,
      global VALUE_TYPE* preSumArray,
      global VALUE_TYPE* postSumArray,
      const  SIZE_TYPE   size,
      local  KEY_TYPE*   ldsKeys,
      local  VALUE_TYPE* ldsVals,
      const  SIZE_TYPE   workPerThread)
{
    size_t groId = get_group_id( 0 );
    size_t gloId = get_global_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );
    uint mapId  = gloId * workPerThread;

    // do offset of zero manually
    SIZE_TYPE  offset;
    KEY_TYPE   key;
    VALUE_TYPE workSum;

    if (mapId < size)
    {
        KEY_TYPE prevKey;

        // accumulate zeroth value manually
        offset = 0;
        key = keySumArray[ mapId+offset ];
        workSum = preSumArray[ mapId+offset ];
        postSumArray[ mapId+offset ] = workSum;

        //  Serial accumulation
        for( offset = offset+1; offset < workPerThread; offset += 1 )
        {
            prevKey = key;
            key = keySumArray[ mapId+offset ];
            if (mapId+offset < size)
            {
                VALUE_TYPE y = preSumArray[ mapId+offset ];
                if (key == prevKey)
                {
                    workSum = workSum + y;
                }
                else
                {
                    workSum = y;
                }
                postSumArray[ mapId+offset ] = workSum;
            }
        }
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    VALUE_TYPE scanSum = workSum;
    offset = 1;
    // load LDS with register sums
    ldsVals[ locId ] = workSum;
    ldsKeys[ locId ] = key;
    // scan in lds
    for( offset = offset*1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        if (mapId < size)
        {
            if (locId >= offset  )
            {
                VALUE_TYPE y    = ldsVals[ locId - offset ];
                KEY_TYPE   key1 = ldsKeys[ locId ];
                KEY_TYPE   key2 = ldsKeys[ locId-offset ];

                if ( key1 == key2 )
                {
                   scanSum = scanSum + y;
                }
                else
                   scanSum = ldsVals[ locId ];
             }
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        ldsVals[ locId ] = scanSum;
    } // for offset
    barrier( CLK_LOCAL_MEM_FENCE );

    // write final scan from pre-scan and lds scan
    for( offset = 0; offset < workPerThread; offset += 1 )
    {
        barrier( CLK_GLOBAL_MEM_FENCE );

        if (mapId < size && locId > 0)
        {
            VALUE_TYPE y    = postSumArray[ mapId+offset ];
            KEY_TYPE   key1 = keySumArray[ mapId+offset ]; // change me
            KEY_TYPE   key2 = ldsKeys[ locId-1 ];
            if ( key1 == key2 )
            {
                VALUE_TYPE y2 = ldsVals[locId-1];
                y = y + y2;
            }
            postSumArray[ mapId+offset ] = y;
        } // thread in bounds
    } // for
} // end kernel
)"

R"(
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void per_block_addition_by_key(
            global KEY_TYPE*   keySumArray,  //InputBuffer
            global VALUE_TYPE* postSumArray, //InputBuffer
            global KEY_TYPE*   keys,
            global VALUE_TYPE* output,       //offsetValArray
            const SIZE_TYPE size)
{

    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );

    //  Abort threads that are passed the end of the input vector
    if( gloId >= size) return;

    // accumulate prefix
    KEY_TYPE key1 = keySumArray[ groId-1 ];
    KEY_TYPE key2 = keys[ gloId ];
    KEY_TYPE key3 = -1;

    if(gloId < size -1 )
        key3 =  keys[ gloId + 1];
    if (groId > 0 && key1 == key2 && key2 != key3)
    {
          VALUE_TYPE scanResult = output[ gloId ];
          VALUE_TYPE postBlockSum = postSumArray[ groId-1 ];
          VALUE_TYPE newResult = scanResult + postBlockSum;
          output[ gloId ] = newResult;

    }
}
)"

R"(
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void key_value_mapping(
    global KEY_TYPE     *keys,         // global input keys
    global KEY_TYPE     *keys_output,  // global output keys
    global VALUE_TYPE   *vals_output,  // global output values
    global KEY_TYPE     *offsetArray,   // temp offset array
    global VALUE_TYPE   *offsetValArray,// temp val array
    const SIZE_TYPE size)               // global size
{



    size_t gloId = get_global_id( 0 );
    size_t locId = get_local_id( 0 );

    //  Abort threads that are passed the end of the input vector
    if( gloId >= size) return;

    SIZE_TYPE numSections = *(offsetArray+size-1) + 1;
    if(gloId < (size-1) && offsetArray[ gloId ] != offsetArray[ gloId +1])
    {
          keys_output[ offsetArray [ gloId ]] = keys[ gloId];
          vals_output[ offsetArray [ gloId ]] = offsetValArray [ gloId];
    }

    if( gloId == (size-1) )
    {
        keys_output[ numSections - 1 ] = keys[ gloId ]; //Copying the last key directly. Works either ways
        vals_output[ numSections - 1 ] = offsetValArray [ gloId ];
        offsetArray [ gloId ] = numSections;
    }
}
)"
