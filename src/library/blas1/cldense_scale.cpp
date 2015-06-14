#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "internal/clsparse_internal.hpp"
#include "internal/data_types/clvector.hpp"


//TODO:: add offset to the scale kernel
template <typename T>
clsparseStatus
scale(clsparse::array_base<T>& pVector,
      const clsparse::array_base<T>& pAlpha,
      clsparseControl control)
{
    const int group_size = 256;
    //const int group_size = control->max_wg_size;

    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE="+ OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(group_size);

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "blas1", "scale",
                                         params);
    KernelWrap kWrapper(kernel);

    cl_ulong size = pVector.size();
    cl_ulong offset = 0;

    kWrapper << size
             << pVector.data()
             << offset
             << pAlpha.data()
             << offset;

    int blocksNum = (size + group_size - 1) / group_size;
    int globalSize = blocksNum * group_size;

    cl::NDRange local(group_size);
    cl::NDRange global (globalSize);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}


clsparseStatus
cldenseSscale (clsparseVector* y,
                const clsparseScalar* alpha,
                const clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparse::vector<cl_float> pY(control, y->values, y->n );
    clsparse::vector<cl_float> pAlpha(control, alpha->value, 1);

    cl_float pattern = 0.0f;

    if (pAlpha[0] == 0.f)
    {

        cl_int status = pY.fill(control, pattern);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        return clsparseSuccess;
    }

    return scale(pY, pAlpha, control);
}

clsparseStatus
cldenseDscale (clsparseVector* y,
                const clsparseScalar* alpha,
                const clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparse::vector<cl_double> pY(control, y->values, y->n );
    clsparse::vector<cl_double> pAlpha(control, alpha->value, 1);

    cl_double pattern = 0.0;

    if (pAlpha[0] == 0.0)
    {

        cl_int status = pY.fill(control, pattern);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        return clsparseSuccess;
    }

    return scale(pY, pAlpha, control);
}
