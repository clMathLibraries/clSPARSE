R"(
#pragma OPENCL EXTENSION cl_amd_printf : enable


/******************************************************************************
 *  Kernel 0
 *****************************************************************************/
__kernel void OffsetCalculation(
    __global int *ikeys, //input keys
    __global int *tempArray, //offsetKeys
    const uint vecSize
    )
{

    //keys.init( ikeys );

    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    if (gloId >= vecSize) return;

    int key, prev_key;

    if(gloId > 0){
      key = ikeys[ gloId ];
	  prev_key = ikeys[ gloId - 1];
	  //if((*binaryPred)(key, prev_key))
	  if(key == prev_key)
	    tempArray[ gloId ] = 0;
	  else
		tempArray[ gloId ] = 1;
	}
	else{
		 tempArray[ gloId ] = 0;
	}
}

/******************************************************************************
 *  Kernel 1
 *****************************************************************************/
__kernel void perBlockScanByKey(
    __global int *keys,
    __global int *ivals, //input values
    const uint vecSize,
    __global int *keyBuffer,
    __global int *valBuffer)
{

    __local int ldsKeys[256];
    __local int ldsVals[256];

    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    // if exclusive, load gloId=0 w/ init, and all others shifted-1
    int key;
    int val;

    if(gloId < vecSize){
      key = keys[ gloId ];
      val = ivals[ gloId ];
      ldsKeys[ locId ] = key;
      ldsVals[ locId ] = val;
    }
    // Computes a scan within a workgroup
    // updates vals in lds but not keys
    int sum = val;
    for( size_t offset = 1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        int key2 = ldsKeys[locId - offset];
        if (locId >= offset && key == key2)
        {
            int y = ldsVals[ locId - offset ];
            //sum = (*binaryFunct)( sum, y );
			sum = sum + y ;
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        ldsVals[ locId ] = sum;
    }
    barrier( CLK_LOCAL_MEM_FENCE ); // needed for large data types
    //  Abort threads that are passed the end of the input vector


    if (gloId >= vecSize) return;

    if (locId == 0)
    {
        keyBuffer[ groId ] = ldsKeys[ wgSize-1 ];
        valBuffer[ groId ] = ldsVals[ wgSize-1 ];
    }
}


/******************************************************************************
 *  Kernel 2
 *****************************************************************************/
__kernel void intraBlockInclusiveScanByKey(
    __global int *keySumArray,
    __global int *preSumArray,
    __global int *postSumArray,
    const uint vecSize,
    const uint workPerThread
)
{
    size_t groId = get_group_id( 0 );
    size_t gloId = get_global_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );
    uint mapId  = gloId * workPerThread;

    __local int ldsKeys[256];
    __local int ldsVals[256];
	
    // do offset of zero manually
    uint offset;
    int key;
    int workSum;

    if (mapId < vecSize)
    {
        int prevKey;

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
            if (mapId+offset<vecSize )
            {
                int  y = preSumArray[ mapId+offset ];
                if ( key == prevKey )
                {
                    //workSum = (*binaryFunct)( workSum, y );
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

    int scanSum = workSum;
    offset = 1;
    // load LDS with register sums
    ldsVals[ locId ] = workSum;
    ldsKeys[ locId ] = key;
    // scan in lds
    for( offset = offset*1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        if (mapId < vecSize)
        {
            if (locId >= offset  )
            {
                int y = ldsVals[ locId - offset ];
                int key1 = ldsKeys[ locId ];
                int key2 = ldsKeys[ locId-offset ];
                if ( key1 == key2 )
                {
                   //scanSum = (*binaryFunct)( scanSum, y );
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

        if (mapId < vecSize && locId > 0)
        {
            int  y = postSumArray[ mapId+offset ];
            int key1 = keySumArray[ mapId+offset ]; // change me
            int key2 = ldsKeys[ locId-1 ];
            if ( key1 == key2 )
            {
                int  y2 = ldsVals[locId-1];
                //y = (*binaryFunct)( y, y2 );
                  y = y + y2;
            }
            postSumArray[ mapId+offset ] = y;
        } // thread in bounds
    } // for

} // end kernel


/******************************************************************************
 *  Kernel 3
 *****************************************************************************/
__kernel void keyValueMapping(
    __global int *ikeys,
    __global int *ikeys_output,
    __global int  *ivals, //input values
    __global int *ivals_output,
    __global int *newkeys,
    __global int *keySumArray, //InputBuffer
    __global int *postSumArray, //InputBuffer
    const uint vecSize
)
{
    size_t gloId = get_global_id( 0 );
    size_t groId = get_group_id( 0 );
    size_t locId = get_local_id( 0 );
    size_t wgSize = get_local_size( 0 );

    __local int ldsKeys[256];
    __local int ldsVals[256];
	
    // if exclusive, load gloId=0 w/ init, and all others shifted-1
    int key;
    int  val;

    if(gloId < vecSize){
      key = newkeys[ gloId ];
      val = ivals[ gloId ];
      ldsKeys[ locId ] = key;
      ldsVals[ locId ] = val;
    }
    // Computes a scan within a workgroup
    // updates vals in lds but not keys
    int  sum = val;
    for( size_t offset = 1; offset < wgSize; offset *= 2 )
    {
        barrier( CLK_LOCAL_MEM_FENCE );
        int key2 = ldsKeys[locId - offset];
        if (locId >= offset && key == key2)
        {
            int  y = ldsVals[ locId - offset ];
            //sum = (*binaryFunct)( sum, y );
			sum = sum + y;
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        ldsVals[ locId ] = sum;
    }
    barrier( CLK_LOCAL_MEM_FENCE ); // needed for large data types

    //  Abort threads that are passed the end of the input vector
    if (gloId >= vecSize) return;

    // accumulate prefix
    int  key1 =  keySumArray[ groId-1 ];
    int  key2 =  newkeys[ gloId ];
    int  key3 = -1;
    if(gloId < vecSize -1 )
      key3 =  newkeys[ gloId + 1];
    if (groId > 0 && key1 == key2 && key2 != key3)
    {
        int  scanResult = sum;
        int  postBlockSum = postSumArray[ groId-1 ];
        //int  newResult = (*binaryFunct)( scanResult, postBlockSum );
        int  newResult = scanResult + postBlockSum ;
        sum = newResult;

    }

    unsigned int count_number_of_sections = 0;		
    count_number_of_sections = newkeys[vecSize-1] + 1;
    if(gloId < (vecSize-1) && newkeys[ gloId ] != newkeys[ gloId +1])
    {
        ikeys_output [newkeys [ gloId ]] = ikeys[ gloId];
        ivals_output[ newkeys [ gloId ]] = sum;
    }
    //printf("keys_output %d vals_output %d newkeys [ gloId ]: %d\n", keys_output [newkeys [ gloId ]], vals_output[ newkeys [ gloId ]], newkeys [ gloId ]);
    if( gloId == (vecSize-1) )
    {
        ikeys_output[ count_number_of_sections - 1 ] = ikeys[ gloId ]; //Copying the last key directly. Works either ways
        ivals_output[ count_number_of_sections - 1 ] = sum;
        newkeys [ gloId ] = count_number_of_sections;
    }

}
)"
