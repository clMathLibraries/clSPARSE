#ifndef CUBLAS_BENCHMARK_COMMON_HXX__
#define CUBLAS_BENCHMARK_COMMON_HXX__

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusparse_v2.h>


template<typename T>
static T
makeScalar(double val)
{
    return static_cast<T>(val);
}

std::string
prettyPrintClStatus( const int& status )
{
    switch( status )
    {
    case cudaErrorMissingConfiguration:
        return "cudaErrorMissingConfiguration";
    case cudaErrorMemoryAllocation:
        return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:
        return "cudaErrorInitializationError";
    case cudaErrorLaunchFailure:
        return "cudaErrorLaunchFailure";
    case cudaErrorPriorLaunchFailure:
        return "cudaErrorPriorLaunchFailure";
    case cudaErrorLaunchTimeout:
        return "cudaErrorLaunchTimeout";
    case cudaErrorLaunchOutOfResources:
        return "cudaErrorLaunchOutOfResources";
    case cudaErrorInvalidDeviceFunction:
        return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidConfiguration:
        return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice:
        return "cudaErrorInvalidDevice";
    case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";
    case cudaErrorInvalidPitchValue:
        return "cudaErrorInvalidPitchValue";
    case cudaErrorInvalidSymbol:
        return "cudaErrorInvalidSymbol";
    case cudaErrorMapBufferObjectFailed:
        return "cudaErrorMapBufferObjectFailed";
    case cudaErrorUnmapBufferObjectFailed:
        return "cudaErrorUnmapBufferObjectFailed";
    case cudaErrorInvalidHostPointer:
        return "cudaErrorInvalidHostPointer";
    case cudaErrorInvalidDevicePointer:
        return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidTexture:
        return "cudaErrorInvalidTexture";
    case cudaErrorInvalidTextureBinding:
        return "cudaErrorInvalidTextureBinding";
    case cudaErrorInvalidChannelDescriptor:
        return "cudaErrorInvalidChannelDescriptor";
    case cudaErrorInvalidMemcpyDirection:
        return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorAddressOfConstant:
        return "cudaErrorAddressOfConstant";
    case cudaErrorTextureFetchFailed:
        return "cudaErrorTextureFetchFailed";
    case cudaErrorTextureNotBound:
        return "cudaErrorTextureNotBound";
    case cudaErrorSynchronizationError:
        return "cudaErrorSynchronizationError";
    case cudaErrorInvalidFilterSetting:
        return "cudaErrorInvalidFilterSetting";
    case cudaErrorInvalidNormSetting:
        return "cudaErrorInvalidNormSetting";
    case cudaErrorMixedDeviceExecution:
        return "cudaErrorMixedDeviceExecution";
    case cudaErrorCudartUnloading:
        return "cudaErrorCudartUnloading";
    case cudaErrorUnknown:
        return "cudaErrorUnknown";
    case cudaErrorNotYetImplemented:
        return "cudaErrorNotYetImplemented";
    case cudaErrorMemoryValueTooLarge:
        return "cudaErrorMemoryValueTooLarge";
    case cudaErrorInvalidResourceHandle:
        return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady:
        return "cudaErrorNotReady";
    case cudaErrorInsufficientDriver:
        return "cudaErrorInsufficientDriver";
    case cudaErrorSetOnActiveProcess:
        return "cudaErrorSetOnActiveProcess";
    case cudaErrorInvalidSurface:
        return "cudaErrorInvalidSurface";
    case cudaErrorNoDevice:
        return "cudaErrorNoDevice";
    case cudaErrorECCUncorrectable:
        return "cudaErrorECCUncorrectable";
    case cudaErrorSharedObjectSymbolNotFound:
        return "cudaErrorSharedObjectSymbolNotFound";
    case cudaErrorSharedObjectInitFailed:
        return "cudaErrorSharedObjectInitFailed";
    case cudaErrorUnsupportedLimit:
        return "cudaErrorUnsupportedLimit";
    case cudaErrorDuplicateVariableName:
        return "cudaErrorDuplicateVariableName";
    case cudaErrorDuplicateTextureName:
        return "cudaErrorDuplicateTextureName";
    case cudaErrorDuplicateSurfaceName:
        return "cudaErrorDuplicateSurfaceName";
    case cudaErrorDevicesUnavailable:
        return "cudaErrorDevicesUnavailable";
    case cudaErrorInvalidKernelImage:
        return "cudaErrorInvalidKernelImage";
    case cudaErrorNoKernelImageForDevice:
        return "cudaErrorNoKernelImageForDevice";
    case cudaErrorIncompatibleDriverContext:
        return "cudaErrorIncompatibleDriverContext";
    case cudaErrorPeerAccessAlreadyEnabled:
        return "cudaErrorPeerAccessAlreadyEnabled";
    case cudaErrorPeerAccessNotEnabled:
        return "cudaErrorPeerAccessNotEnabled";
    case cudaErrorDeviceAlreadyInUse:
        return "cudaErrorDeviceAlreadyInUse";
    case cudaErrorProfilerDisabled:
        return "cudaErrorProfilerDisabled";
    case cudaErrorProfilerNotInitialized:
        return "cudaErrorProfilerNotInitialized";
    case cudaErrorProfilerAlreadyStarted:
        return "cudaErrorProfilerAlreadyStarted";
    case cudaErrorProfilerAlreadyStopped:
        return "cudaErrorProfilerAlreadyStopped";
    case cudaErrorAssert:
        return "cudaErrorAssert";
    case cudaErrorTooManyPeers:
        return "cudaErrorTooManyPeers";
    case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorHostMemoryNotRegistered:
        return "cudaErrorHostMemoryNotRegistered";
    case cudaErrorOperatingSystem:
        return "cudaErrorOperatingSystem";
    case cudaErrorPeerAccessUnsupported:
        return "cudaErrorPeerAccessUnsupported";
    case cudaErrorLaunchMaxDepthExceeded:
        return "cudaErrorLaunchMaxDepthExceeded";
    case cudaErrorLaunchFileScopedTex:
        return "cudaErrorLaunchFileScopedTex";
    case cudaErrorLaunchFileScopedSurf:
        return "cudaErrorLaunchFileScopedSurf";
    case cudaErrorSyncDepthExceeded:
        return "cudaErrorSyncDepthExceeded";
    case cudaErrorLaunchPendingCountExceeded:
        return "cudaErrorLaunchPendingCountExceeded";
    case cudaErrorNotPermitted:
        return "cudaErrorNotPermitted";
    case cudaErrorNotSupported:
        return "cudaErrorNotSupported";
    case cudaErrorHardwareStackError:
        return "cudaErrorHardwareStackError";
    case cudaErrorIllegalInstruction:
        return "cudaErrorIllegalInstruction";
    case cudaErrorMisalignedAddress:
        return "cudaErrorMisalignedAddress";
    case cudaErrorInvalidAddressSpace:
        return "cudaErrorInvalidAddressSpace";
    case cudaErrorInvalidPc:
        return "cudaErrorInvalidPc";
    case cudaErrorIllegalAddress:
        return "cudaErrorIllegalAddress";
    case cudaErrorInvalidPtx:
        return "cudaErrorInvalidPtx";
    case cudaErrorInvalidGraphicsContext:
        return "cudaErrorInvalidGraphicsContext";
    case cudaErrorStartupFailure:
        return "cudaErrorStartupFailure";
    case cudaErrorApiFailureBase:
        return "cudaErrorApiFailureBase";
    default:
        return "Error code not defined";
        break;
    }
}

inline int
CUDA_V_Throw( int res, const std::string& msg, size_t lineno )
{
    switch( res )
    {
    case cudaSuccess: /**< No error */
        break;
    default:
        {
            std::stringstream tmp;

            tmp << "CUDA_V_THROWERROR< ";
            tmp << prettyPrintClStatus(res) ;
            tmp << " > (";
            tmp << lineno;
            tmp << "): ";
            tmp << msg;
            std::string errorm(tmp.str());
            std::cout << errorm<< std::endl;
            throw std::runtime_error( errorm );
        }
    }

    return res;
}

#define CUDA_V_THROW(_status,_message) CUDA_V_Throw(_status, _message, \
                                                        __LINE__)

class cusparseFunc
{
public:
    cusparseFunc(StatisticalTimer& _timer)
          : timer(_timer)
    {
        timer_id = timer.getUniqueID( "cufunc", 0 );

        /* Setup cublas. */
        cuSparseStatus = cusparseCreate( &handle );

        CUDA_V_THROW( cuSparseStatus, "cusparseCreate() failed with %d\n" );
    }

    virtual ~cusparseFunc()
    {
        cusparseDestroy( handle );
    }

    double time_in_ns()
    {
        return timer.getAverageTime( timer_id ) * 1e9;
    }

    virtual void call_func() = 0;
    virtual double gflops( ) = 0;
    virtual std::string gflops_formula( ) = 0;
    virtual double bandwidth( ) = 0;
    virtual std::string bandwidth_formula( ) = 0;
    virtual void setup_buffer( double alpha, double beta, const std::string& path ) = 0;
    virtual void initialize_cpu_buffer() = 0;
    virtual void initialize_gpu_buffer() = 0;
    virtual void reset_gpu_write_buffer() = 0;
	virtual void read_gpu_buffer() = 0;
	virtual void releaseGPUBuffer_deleteCPUBuffer()=0;

    StatisticalTimer& timer;
    StatisticalTimer::sTimerID timer_id;

protected:
    virtual void initialize_scalars(double alpha, double beta) = 0;

protected:
    cudaError_t err;
    cusparseHandle_t handle;
    cusparseStatus_t cuSparseStatus;
};

#endif // ifndef CUBLAS_BENCHMARK_COMMON_HXX__
