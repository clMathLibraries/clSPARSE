#if defined ( _WIN32 )
#define NOMINMAX
#endif

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"
#include "resources/matrix_utils.h"

//boost ublas
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>


clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

namespace po = boost::program_options;
namespace uBLAS = boost::numeric::ublas;

//number of columns in dense B matrix;
cl_int B_num_cols;
cl_double B_values;



template<typename T>
clsparseStatus generateResult( cldenseMatrix& matB, clsparseScalar& alpha,
                               cldenseMatrix& matC, clsparseScalar& beta )
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

    if(typeid(T) == typeid(float))
    {

        return clsparseScsrmm( &alpha, &CSRE::csrSMatrix, &matB,
                               &beta, &matC, CLSE::control );


    }

    if(typeid(T) == typeid(double))
    {
        return clsparseDcsrmm( &alpha, &CSRE::csrDMatrix, &matB,
                               &beta, &matC, CLSE::control );

    }
    return clsparseSuccess;
}

template <typename T>
class TestCSRMM : public ::testing::Test
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

public:

    void SetUp()
    {
        // alpha and beta scalars are not yet supported in generating reference result;
        alpha = T(CSRE::alpha);
        beta = T(CSRE::beta);

        B = uBLASDenseM(CSRE::n_cols, B_num_cols, T(B_values));
        C = uBLASDenseM(CSRE::n_rows, B_num_cols, T(0));


        cl_int status;
        cldenseInitMatrix( &deviceMatB );
        deviceMatB.values = clCreateBuffer( CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   B.data().size( ) * sizeof( T ), B.data().begin(), &status );

        deviceMatB.num_rows = B.size1();
        deviceMatB.num_cols = B.size2();
        deviceMatB.lead_dim = std::min(B.size1(), B.size2());


        ASSERT_EQ(CL_SUCCESS, status);

        cldenseInitMatrix( &deviceMatC );
        deviceMatC.values = clCreateBuffer( CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   C.data().size( ) * sizeof( T ), C.data().begin(), &status );


        deviceMatC.num_rows = C.size1();
        deviceMatC.num_cols = C.size2();
        deviceMatC.lead_dim = std::min(C.size1(), C.size2());
        ASSERT_EQ(CL_SUCCESS, status);

        clsparseInitScalar( &gAlpha );
        gAlpha.value = clCreateBuffer(CLSE::context,
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(T), &alpha, &status);
        ASSERT_EQ(CL_SUCCESS, status);

        clsparseInitScalar( &gBeta );
        gBeta.value = clCreateBuffer(CLSE::context,
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(T), &beta, &status);
        ASSERT_EQ(CL_SUCCESS, status);

    }

    void TearDown()
    {
        ::clReleaseMemObject(gAlpha.value);
        ::clReleaseMemObject(gBeta.value);

        clsparseInitScalar(&gAlpha);
        clsparseInitScalar(&gBeta);

        ::clReleaseMemObject(deviceMatB.values);
        ::clReleaseMemObject(deviceMatC.values);

        cldenseInitMatrix( &deviceMatB );
        cldenseInitMatrix( &deviceMatC );

    }


    typedef typename uBLAS::matrix<T, uBLAS::row_major, uBLAS::unbounded_array<T> > uBLASDenseM;
    uBLASDenseM B;
    uBLASDenseM C;


    cldenseMatrix deviceMatB;
    cldenseMatrix deviceMatC;

    T alpha;
    T beta;

    clsparseScalar gAlpha;
    clsparseScalar gBeta;
};

typedef ::testing::Types<float,double> TYPES;
//typedef ::testing::Types<float> TYPES;
TYPED_TEST_CASE( TestCSRMM, TYPES );



// This test may give you false failure result due to multiplication order.
TYPED_TEST(TestCSRMM, multiply)
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

    cl::Event event;
    clsparseEnableAsync(CLSE::control, true);

    //control object is global and it is updated here;
    clsparseStatus status =
            generateResult<TypeParam>(this->deviceMatB, this->gAlpha,
            this->deviceMatC, this->gBeta );

    EXPECT_EQ(clsparseSuccess, status);

    status = clsparseGetEvent(CLSE::control, &event());
    EXPECT_EQ(clsparseSuccess, status);
    event.wait();

    std::vector<TypeParam> result(this->C.data().size());

    cl_int cl_status = clEnqueueReadBuffer(CLSE::queue,
                                            this->deviceMatC.values, CL_TRUE, 0,
                                            result.size()*sizeof(TypeParam),
                                            result.data(), 0, NULL, NULL);
    EXPECT_EQ(CL_SUCCESS, cl_status);

    // Generate referencee result;
    if (typeid(TypeParam) == typeid(float))
    {
         this->C = uBLAS::sparse_prod(CSRE::ublasSCsr, this->B, this->C, false);
    }

    if (typeid(TypeParam) == typeid(double))
    {
         this->C = uBLAS::sparse_prod(CSRE::ublasDCsr, this->B, this->C, false);
    }


    if(typeid(TypeParam) == typeid(float))
        for (int l = 0; l < std::min(this->C.size1(), this->C.size2()); l++)
            for( int i = 0; i < this->C.data().size(); i++ )
            {
                ASSERT_NEAR(this->C.data()[i], result[i], 5e-3);
            }

    if(typeid(TypeParam) == typeid(double))
        for (int l = 0; l < std::min(this->C.size1(), this->C.size2()); l++)
            for( int i = 0; i < this->C.data().size(); i++ )
            {
                ASSERT_NEAR(this->C.data()[i], result[i], 5e-10);
            };
}


int main (int argc, char* argv[])
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;
    //pass path to matrix as an argument, We can switch to boost po later

    std::string path;
    double alpha;
    double beta;
    std::string platform;
    cl_platform_type pID;
    cl_uint dID;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("path,p", po::value(&path)->required(), "Path to matrix in mtx format.")
            ("platform,l", po::value(&platform)->default_value("AMD"),
             "OpenCL platform: AMD or NVIDIA.")
            ("device,d", po::value(&dID)->default_value(0),
             "Device id within platform.")
            ("alpha,a", po::value(&alpha)->default_value(1.0),
             "Alpha parameter for eq: \n\ty = alpha * M * x + beta * y")
            ("beta,b", po::value(&beta)->default_value(0.0),
             "Beta parameter for eq: \n\ty = alpha * M * x + beta * y")
            ("cols,c", po::value(&B_num_cols)->default_value(8),
             "Number of columns in B matrix while calculating sp_A * d_B = d_C")
            ("vals,v", po::value(&B_values)->default_value(1.0),
             "Initial value of B columns");


    //	Parse the command line options, ignore unrecognized options and collect them into a vector of strings
    //  Googletest itself accepts command line flags that we wish to pass further on
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser( argc, argv ).options( desc ).allow_unregistered( ).run( );

    try {
        po::store( parsed, vm );
        po::notify( vm );
    }
    catch( po::error& error )
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what( ) << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    std::vector< std::string > to_pass_further = po::collect_unrecognized( parsed.options, po::include_positional );

    //check platform
    if(vm.count("platform"))
    {
        if ("AMD" == platform)
        {
            pID = AMD;
        }
        else if ("NVIDIA" == platform)
        {
            pID = NVIDIA;
        }
        else
        {

            std::cout << "The platform option is missing or is ill defined!\n";
            std::cout << "Given [" << platform << "]" << std::endl;
            platform = "AMD";
            pID = AMD;
            std::cout << "Setting [" << platform << "] as default" << std::endl;
        }

    }

    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE(pID, dID));
    ::testing::AddGlobalTestEnvironment( new CSRE(path, alpha, beta,
                                                  CLSE::queue, CLSE::context));
    return RUN_ALL_TESTS();
}
