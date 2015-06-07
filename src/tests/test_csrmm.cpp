#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"
#include "resources/matrix_utils.h"

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

namespace po = boost::program_options;


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
    //if(typeid(T) == typeid(double))
    //{
    //    return clsparseDcsrmm( &alpha, &CSRE::csrDMatrix, &matB,
    //                           &beta, &matC, CLSE::control );

    //}
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
        alpha = T(CSRE::alpha);
        beta = T(CSRE::beta);

        // This needs to be more flexible, to be able to specify matB column values from command line
        matB = matrix<T>( CSRE::n_cols, CSRE::n_cols, CSRE::n_cols );
        matC = matrix<T>( CSRE::n_rows, CSRE::n_cols, CSRE::n_cols );

        std::fill( matB.data.begin( ), matB.data.end( ), T( 1 ) );
        std::fill( matC.data.begin( ), matC.data.end( ), T( 2 ) );

        cl_int status;
        cldenseInitMatrix( &deviceMatB );
        deviceMatB.values = clCreateBuffer( CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   matB.data.size( ) * sizeof( T ), matB.data.data( ), &status );
        deviceMatB.num_rows = matB.num_rows;
        deviceMatB.num_cols = matB.num_cols;
        deviceMatB.lead_dim = matB.leading_dim;

        ASSERT_EQ(CL_SUCCESS, status);

        cldenseInitMatrix( &deviceMatC );
        deviceMatC.values = clCreateBuffer( CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   matC.data.size( ) * sizeof( T ), matC.data.data( ), &status );

        deviceMatC.num_rows = matC.num_rows;
        deviceMatC.num_cols = matC.num_cols;
        deviceMatC.lead_dim = matC.leading_dim;
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


        generateReference( matB, alpha, matC, beta );

    }


    void generateReference (const matrix<float>& x,
                            const float alpha,
                            matrix<float>& y,
                            const float beta)
    {
        csrmm(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
              CSRE::row_offsets, CSRE::col_indices, CSRE::f_values,
              x, alpha, y, beta);
    }

    void generateReference( const matrix<double>& x,
                            const double alpha,
                            matrix<double>& y,
                            const double beta)
    {
        csrmm(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
              CSRE::row_offsets, CSRE::col_indices, CSRE::d_values,
              x, alpha, y, beta);
    }

    matrix<T> matB;
    matrix<T> matC;

    cldenseMatrix deviceMatB;
    cldenseMatrix deviceMatC;

    T alpha;
    T beta;

    clsparseScalar gAlpha;
    clsparseScalar gBeta;
};

//typedef ::testing::Types<float,double> TYPES;
typedef ::testing::Types<float> TYPES;
TYPED_TEST_CASE( TestCSRMM, TYPES );

TYPED_TEST(TestCSRMM, multiply)
{
    cl::Event event;
    clsparseEnableAsync(ClSparseEnvironment::control, true);

    //control object is global and it is updated here;
    clsparseStatus status =
            generateResult<TypeParam>(this->deviceMatB, this->gAlpha,
            this->deviceMatC, this->gBeta );

    EXPECT_EQ(clsparseSuccess, status);

    status = clsparseGetEvent(ClSparseEnvironment::control, &event());
    EXPECT_EQ(clsparseSuccess, status);
    event.wait();

    std::vector<TypeParam> result(this->matC.data.size());

    clEnqueueReadBuffer(ClSparseEnvironment::queue,
                         this->deviceMatC.values, CL_TRUE, 0,
                        result.size()*sizeof(TypeParam),
                        result.data(), 0, NULL, NULL);

    if(typeid(TypeParam) == typeid(float))
        for( int i = 0; i < this->matC.data.size( ); i++ )
            ASSERT_NEAR( this->matC.data[ i ], result[ i ], 5e-4 );

    if(typeid(TypeParam) == typeid(double))
        for( int i = 0; i < this->matC.data.size( ); i++ )
            ASSERT_NEAR( this->matC.data[ i ], result[ i ], 5e-14 );
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
             "Beta parameter for eq: \n\ty = alpha * M * x + beta * y");

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
