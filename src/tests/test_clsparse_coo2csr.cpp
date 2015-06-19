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
std::string path;

namespace po = boost::program_options;

template <typename T>
class TestCOO2CSR : public ::testing::Test
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

public:
#if 0
    void SetUp()
    {
        cl_int status;

        clsparseInitCooMatrix( &cooMatx );
		
	clsparseCooHeaderfromFile( &cooMatx, path.c_str( ) );
	
        cooMatx.values     = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                               cooMatx.nnz * sizeof(T), NULL, &status );

        cooMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           cooMatx.nnz * sizeof( cl_int ), NULL, &status );
        cooMatx.rowIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           cooMatx.nnz * sizeof( cl_int ), NULL, &status );
        clsparseCooMatrixfromFile( &cooMatx, path.c_str( ), CLSE::control );
        
        row = (int *)malloc((cooMatx.m + 1) * sizeof(int));
        col = (int *)malloc((cooMatx.nnz) * sizeof(int));
        val = (T *)malloc(cooMatx.nnz * sizeof(T)); 
    }
#endif 

    void SetUp()
    {
        cl_int status;

        int nnz1, row1, col1;
        clsparseStatus fileError = clsparseHeaderfromFile( &nnz1, &row1, &col1, path.c_str( ) );
        if( fileError != clsparseSuccess )
           throw std::runtime_error( "Could not read matrix market header from disk" );

        clsparseInitCooMatrix( &cooMatx );
        cooMatx.nnz = nnz1;
        cooMatx.m = row1;
        cooMatx.n = col1;

        //clsparseCooHeaderfromFile( &cooMatx, path.c_str( ) );

        cooMatx.values     = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                               cooMatx.nnz * sizeof(T), NULL, &status );

        cooMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           cooMatx.nnz * sizeof( cl_int ), NULL, &status );
        cooMatx.rowIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           cooMatx.nnz * sizeof( cl_int ), NULL, &status );
        clsparseCooMatrixfromFile( &cooMatx, path.c_str( ), CLSE::control );

        row = (int *)malloc((cooMatx.m + 1) * sizeof(int));
        col = (int *)malloc((cooMatx.nnz) * sizeof(int));
        val = (T *)malloc(cooMatx.nnz * sizeof(T));
    }

    clsparseCooMatrix cooMatx;
    
    int *row;
    int *col;
    T * val;

    
};

typedef ::testing::Types<float> TYPES;
TYPED_TEST_CASE(TestCOO2CSR, TYPES);

TYPED_TEST(TestCOO2CSR, transform)
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

    cl_int status;
    cl_event event = NULL;

    clsparseCsrMatrix csrMatx;
    clsparseInitCsrMatrix( &csrMatx );

    if(typeid(TypeParam) == typeid(float))
      csrMatx.values = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                         this->cooMatx.nnz * sizeof( cl_float ), NULL, &status );

    if(typeid(TypeParam) == typeid(double))
      csrMatx.values = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                         this->cooMatx.nnz * sizeof( cl_double ), NULL, &status );

    csrMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                          this->cooMatx.nnz * sizeof( cl_int ), NULL, &status );
    csrMatx.rowOffsets = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                       ( this->cooMatx.m + 1 ) * sizeof( cl_int ), NULL, &status );

    if(typeid(TypeParam) == typeid(float))										   
           clsparseScoo2csr(&(this->cooMatx),
                            &csrMatx,
                            CLSE::control);

    if(typeid(TypeParam) == typeid(double))
            clsparseDcoo2csr(&(this->cooMatx),
                            &csrMatx,
                            CLSE::control);	
				  
    clEnqueueReadBuffer(CLSE::queue,
                        csrMatx.rowOffsets,
                        1,
                        0,
                       (csrMatx.m+1) * sizeof(int),
                        this->row,
                        0,
                        0,
                        0);

    clEnqueueReadBuffer(CLSE::queue,
                        csrMatx.colIndices,
                        1,
                        0,
                       (csrMatx.nnz) * sizeof(int),
                        this->col,
                        0,
                        0,
                        0);

    if(typeid(TypeParam) == typeid(float)){
       clEnqueueReadBuffer(CLSE::queue,
                           csrMatx.values,
                           1,
                           0,
                          (csrMatx.nnz) * sizeof(float),
                           this->val,
                           0,
                           0,
                           0);
    }

    if(typeid(TypeParam) == typeid(double)){
       clEnqueueReadBuffer(CLSE::queue,
                           csrMatx.values,
                           1,
                           0,
                          (csrMatx.nnz) * sizeof(double),
                           this->val,
                           0,
                           0,
     	                   0);
    }
#if 0
    double *temp = (double *)malloc(sizeof(double) *  csrMatx.nnz);
    clEnqueueReadBuffer(CLSE::queue,
                            csrMatx.values,
                            1,
                            0,
                           (csrMatx.nnz) * sizeof(double),
                            temp,
                            0,
                            0,
                            0);


     for(int i = 0; i < csrMatx.nnz; i++){
         std::cout << std::setprecision (16) << temp[i] << " ";
     }
#endif		
     for(int i = 0; i < csrMatx.m + 1; i++){
         ASSERT_EQ (this->row[i], CSRE::row_offsets[i]);
     }
		
     for(int i = 0; i < csrMatx.nnz; i++){
        ASSERT_EQ(this->col[i], CSRE::col_indices[i]);
     }

     if(typeid(TypeParam) == typeid(float)){    
        for(int i = 0; i < csrMatx.nnz; i++){
            ASSERT_EQ(this->val[i], CSRE::f_values[i]);
        }
     }

     if(typeid(TypeParam) == typeid(double)){
        for(int i = 0; i < csrMatx.nnz; i++){
            //std::cout << std::setprecision (16) << this->val[i] << " " <<  CSRE::d_values[i] << std::endl;
            ASSERT_EQ(this->val[i], CSRE::d_values[i]);
        }
     }

}


int main (int argc, char* argv[])
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;
    //pass path to matrix as an argument, We can switch to boost po later

//    std::string path;
    double alpha = 0;
    double beta  = 0;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("path,p", po::value(&path)->required(), "Path to matrix in mtx format.");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (po::error& error)
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what() << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE());
    ::testing::AddGlobalTestEnvironment( new CSRE(path, alpha, beta,
                                                  CLSE::queue, CLSE::context));
    return RUN_ALL_TESTS();
}

