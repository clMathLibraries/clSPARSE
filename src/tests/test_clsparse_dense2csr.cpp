#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"
#include "resources/matrix_utils.h"

#define HERE std::cout<<"here"<<std::endl;

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;
std::string path;

namespace po = boost::program_options;

template <typename T>
class TestDENSE2CSR : public ::testing::Test
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

public:

    void SetUp()
    {
        cl_int err;

	CSRE::csrSMatrix.rowOffsets = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE, (CSRE::n_rows + 1) * sizeof(float), NULL, &err );
	err = clEnqueueWriteBuffer(CLSE::queue, CSRE::csrSMatrix.rowOffsets, 1, 0,
                                  (CSRE::n_rows + 1) * sizeof(int),
                                   CSRE::row_offsets.data(),
                                   0, NULL, NULL);
								   
	CSRE::csrSMatrix.colIndices = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE, (CSRE::csrSMatrix).nnz * sizeof(float), NULL, &err );
	err = clEnqueueWriteBuffer(CLSE::queue, CSRE::csrSMatrix.colIndices, 1, 0,
                                  (CSRE::csrSMatrix.nnz) * sizeof(int),
                                   CSRE::col_indices.data(),
                                   0, NULL, NULL);

	CSRE::csrSMatrix.values = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE, (CSRE::csrSMatrix).nnz * sizeof(float), NULL, &err );
	err = clEnqueueWriteBuffer(CLSE::queue, CSRE::csrSMatrix.values, 1, 0,
				  (CSRE::csrSMatrix).nnz * sizeof(float),
                                  (CSRE::f_values).data(),
                                   0, NULL, NULL);

	if(err!=CL_SUCCESS) fprintf(stderr, "malloc failed rowOffsets\n");
		
    }

};

typedef ::testing::Types<float> TYPES;
TYPED_TEST_CASE(TestDENSE2CSR, TYPES);

TYPED_TEST(TestDENSE2CSR, transform)
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

    clsparseDenseMatrix A;
    clsparseInitDenseMatrix(&A);
	
    cl_int status;

    A.values = clCreateBuffer(CLSE::context,
                              CL_MEM_READ_WRITE,
                              CSRE::n_cols * CSRE::n_rows * sizeof(cl_float), NULL, &status);

    A.m = CSRE::n_rows;
    A.n = CSRE::n_cols;
	
    clsparseScsr2dense(&CSRE::csrSMatrix,
                       &A,
                       CLSE::control);
					   
    clsparseCsrMatrix csrMatx;
    clsparseInitCsrMatrix( &csrMatx );

    csrMatx.values = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           (CSRE::csrSMatrix).nnz * sizeof( cl_float ), NULL, &status );
    csrMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           (CSRE::csrSMatrix).nnz * sizeof( cl_int ), NULL, &status );
    csrMatx.rowOffsets = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           (CSRE::n_rows + 1) * sizeof( cl_int ), NULL, &status );

    //call dense2csr	
    clsparseSdense2csr(&csrMatx,
                       &A,
                       CLSE::control);				   

    int *row = (int *)malloc(sizeof(int) * (csrMatx.m + 1));
    int *col = (int *)malloc(sizeof(int) * csrMatx.nnz);
    float *val = (float *)malloc(sizeof(float) * csrMatx.nnz);

    //std::cout<< "csrMatx.m: " << csrMatx.m <<std::endl;

    clEnqueueReadBuffer(CLSE::queue,
                        csrMatx.rowOffsets,
                        1,
                        0,
                        (csrMatx.m + 1) * sizeof(int),
                        row,
                        0,
                        0,
                        0);

    clEnqueueReadBuffer(CLSE::queue,
                        csrMatx.colIndices,
                        1,
                        0,
                       (csrMatx.nnz) * sizeof(int),
                        col,
                        0,
                        0,
                        0);
 
    clEnqueueReadBuffer(CLSE::queue,
                        csrMatx.values,
                        1,
                        0,
                       (csrMatx.nnz) * sizeof(float),
                        val,
                        0,
                        0,
                        0);
 
    for(int i = 0; i < csrMatx.m + 1; i++){
         ASSERT_EQ (row[i], CSRE::row_offsets[i]);
    }

    for(int i = 0; i < csrMatx.nnz; i++){
        ASSERT_EQ(col[i], CSRE::col_indices[i]);
    }
    if(typeid(TypeParam) == typeid(float)){
       for(int i = 0; i < csrMatx.nnz; i++){
          //std::cout << i << ":" << val[i] << " " << CSRE::f_values[i] << std::endl; 
          ASSERT_EQ(val[i], CSRE::f_values[i]);
        }
    }

    free(row);
    free(col);
    free(val);
}


int main (int argc, char* argv[])
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;
    //pass path to matrix as an argument, We can switch to boost po later

    //std::string path;
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

