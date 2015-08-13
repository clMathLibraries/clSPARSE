R"(
/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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

#define NUM_ITER 16
#define MIN(X,Y) X<Y?X:Y;
#define MAX(X,Y) X>Y?X:Y;
/******************************************************************************
 *  Kernel 2
 *****************************************************************************/
__kernel void perBlockAddition(
				__global int *output_ptr,
				__global int *input_ptr,
				__global int *preSumArray,
				  const uint vecSize,
				  const uint load_per_wg,
				         int identity
)
{

	// 1 thread per element
	size_t gloId = get_global_id( 0 );
	size_t groId = get_group_id( 0 );
	size_t locId = get_local_id( 0 );
	size_t wgSize = get_local_size( 0 );
        __local int lds[256]; //make a parameter

	int scanResult;
	const uint input_offset = groId * load_per_wg + locId;

	int sum, temp;
	for (uint i = 0; i < load_per_wg; i += wgSize)
	{
#if EXCLUSIVE
		if(gloId == 0 && i == 0 )
		{
			  scanResult = identity;
		}
		else
		{
		   if((input_offset + i) < vecSize)
		   {
			  scanResult = input_ptr[input_offset + i -1];
		   }
		   if(groId > 0 && i == 0 && locId == 0)
		   {
			  scanResult = preSumArray[groId-1];
		   }
		   if(locId == 0 && i > 0)
		   {
			 temp = lds[wgSize-1];
			 scanResult = scanResult + temp;
		   }
		}
#else
	   if((input_offset + i) < vecSize)
	   {
		  scanResult = input_ptr[input_offset + i];
	   }
	   if(groId > 0 && i == 0 && locId == 0)
	   {
		  temp = preSumArray[groId-1];
		  scanResult = scanResult + temp;
	   }
	   if(locId == 0 && i > 0)
	   {
		   temp = lds[wgSize-1];
		   scanResult = scanResult +  temp;
	   }
#endif
	   barrier( CLK_LOCAL_MEM_FENCE );
	   lds[locId] = scanResult;
	   sum = lds[locId];
	   for( size_t offset = 1; offset < (wgSize); offset *= 2 )
	   {
			barrier( CLK_LOCAL_MEM_FENCE );
			if (locId >= offset)
			{
				int y = lds[ locId - offset ];
				//sum = (*binaryOp)( sum, y );
				sum = sum + y;
			}
			barrier( CLK_LOCAL_MEM_FENCE );
			lds[ locId ] = sum;
		}
		if((input_offset + i) < vecSize)
			 output_ptr[input_offset + i] = sum;

		barrier( CLK_LOCAL_MEM_FENCE );
	}

}

/******************************************************************************
 *  Kernel 1
 *****************************************************************************/
__kernel void intraBlockInclusiveScan(
				__global int* preSumArray,
				         int  identity,
				const   uint vecSize,
				const   uint workPerThread
				)
{
	size_t gloId = get_global_id( 0 );
	size_t locId = get_local_id( 0 );
	size_t wgSize = get_local_size( 0 );

        __local int lds[256];//make a parameter
	uint mapId  = gloId * workPerThread;
	// do offset of zero manually
	uint offset;
	int workSum;
	if (mapId < vecSize)
	{
		// accumulate zeroth value manually
		offset = 0;
		workSum = preSumArray[mapId+offset];

		//  Serial accumulation
		for( offset = offset+1; offset < workPerThread; offset += 1 )
		{
			if (mapId+offset<vecSize)
			{
				int y = preSumArray[mapId+offset];
				//workSum = (*binaryOp)( workSum, y );
				workSum = workSum + y ;
			}
		}
	}
	barrier( CLK_LOCAL_MEM_FENCE );
	int scanSum = workSum;
	lds[ locId ] = workSum;
	offset = 1;
	// scan in lds
	for( offset = offset*1; offset < wgSize; offset *= 2 )
	{
		barrier( CLK_LOCAL_MEM_FENCE );
		if (mapId < vecSize)
		{
			if (locId >= offset)
			{
				int y = lds[ locId - offset ];
				//scanSum = (*binaryOp)( scanSum, y );
				scanSum = scanSum + y;
			}
		}
		barrier( CLK_LOCAL_MEM_FENCE );
		lds[ locId ] = scanSum;

	} // for offset
	barrier( CLK_LOCAL_MEM_FENCE );
	// write final scan from pre-scan and lds scan
	 workSum = preSumArray[mapId];
	 if(locId > 0){
		int y = lds[locId-1];
		//workSum = (*binaryOp)(workSum, y);
		workSum = workSum + y;
		preSumArray[ mapId] = workSum;
	 }
	 else{
	   preSumArray[ mapId] = workSum;
	}
	for( offset = 1; offset < workPerThread; offset += 1 )
	{
		barrier( CLK_GLOBAL_MEM_FENCE );

		if ((mapId + offset) < vecSize && locId > 0)
		{
			int y  = preSumArray[ mapId + offset ] ;
			//iPtrType y1 = (*binaryOp)(y, workSum);
			int y1 = y + workSum;
			preSumArray[ mapId + offset ] = y1;
			workSum = y1;

		} // thread in bounds
		else if((mapId + offset) < vecSize){
		   //iPtrType y  = preSumArray[ mapId + offset ] ;
		   int y  = preSumArray[ mapId + offset ] ;
		   //preSumArray[ mapId + offset ] = (*binaryOp)(y, workSum);
		   preSumArray[ mapId + offset ] = y + workSum;
		   workSum = preSumArray[ mapId + offset ];
		}

	} // for


} // end kernel


/******************************************************************************
 *  Kernel 0
 *****************************************************************************/
__kernel void perBlockInclusiveScan(
				__global int* input_ptr,
				         int  identity,
				const    uint vecSize,
				__global int* preSumArray,
				const    uint load_per_wg)
{
	// 2 thread per element
	size_t gloId = get_global_id( 0 );
	size_t groId = get_group_id( 0 );
	size_t locId = get_local_id( 0 );
	size_t wgSize = get_local_size( 0 )  ;

        __local int lds[256];//make a parameter

	const uint input_offset = groId * load_per_wg + locId;

	int local_sum;
	if((input_offset) < vecSize)
	    local_sum = input_ptr[input_offset];

#if EXCLUSIVE
	if(gloId == 0)
	{
		local_sum = local_sum + identity;
	}
#endif
	for (uint i = wgSize; i < load_per_wg; i += wgSize)
	{
	   if((input_offset + i) < vecSize)
	   {
			//typename iIterType::value_type temp = input_iter[input_offset + i];
			int temp = input_ptr[input_offset + i];
			//local_sum = (*binaryOp)(local_sum, temp);
			local_sum = local_sum + temp;
	   }
	}
	lds[locId] = local_sum;

	int sum = lds[locId];
	for( size_t offset = 1; offset < (wgSize); offset *= 2 )
	{
		barrier( CLK_LOCAL_MEM_FENCE );
		if (locId >= offset)
		{
			//typename iIterType::value_type y = lds[ locId - offset ];
		    int y = lds[ locId - offset ];
			//sum = (*binaryOp)( sum, y );
			sum = sum + y;
		}
		barrier( CLK_LOCAL_MEM_FENCE );
		lds[ locId ] = sum;
	}
	if (locId == 0)
	{
		preSumArray[groId] = lds[wgSize-1];
	}
}

)"
