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
clsparseStatus generateResult(cl_mem g_dense)
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

    if(typeid(T) == typeid(float))
    {
        return clsparseScsr2dense(CSRE::n_rows,
                                  CSRE::n_cols,
                                  CSRE::n_vals,
                                  CSRE::cl_row_offsets,
                                  CSRE::cl_col_indices,
                                  CSRE::cl_f_values,
                                  g_dense,
                                  CLSE::control);

    }

    if(typeid(T) == typeid(double))
    {
       return clsparseDcsr2dense(CSRE::n_rows,
                                 CSRE::n_cols,
                                 CSRE::n_vals,
                                 CSRE::cl_row_offsets,
                                 CSRE::cl_col_indices,
                                 CSRE::cl_d_values,
                                 g_dense,
                                 CLSE::control);

    }

}

template <typename T>
class TestCSR2DENSE : public ::testing::Test
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

public:

    void SetUp()
    {
        //TODO:: take the values from cmdline;

        dense = std::vector<T>(CSRE::n_cols * CSRE::n_rows, 0);

        cl_int status;
        g_dense = clCreateBuffer(CLSE::context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 dense.size() * sizeof(T), dense.data(), &status);

        ASSERT_EQ(CL_SUCCESS, status); //is it wise to use this here?


        generateReference(dense);

    }


    void generateReference (std::vector<float>& dense)
    {
            csr2dense(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
                      CSRE::row_offsets, CSRE::col_indices, CSRE::f_values,
                      dense);
            
            //std::cout << "print out matrix" << std::endl; 
            //for(int i = 0; i < CSRE::n_rows; i++){
            //   for(int j = 0; j < CSRE::n_cols; j++){
            //       std::cout << dense[i * CSRE::n_cols + j] << " ";
            //   }
            //   std::cout <<std:: endl;
            //}
    }

    void generateReference (std::vector<double>& dense)
    {
            csr2dense(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
                      CSRE::row_offsets, CSRE::col_indices, CSRE::d_values,
                      dense);
    }

    cl_mem g_dense;
    std::vector<T> dense;

};

typedef ::testing::Types<float, double> TYPES;
TYPED_TEST_CASE(TestCSR2DENSE, TYPES);

TYPED_TEST(TestCSR2DENSE, transform)
{

    cl_event event = NULL;
    //clsparseSetupEvent(ClSparseEnvironment::control, &event );

    clsparseStatus status =
            generateResult<TypeParam>(this->g_dense);

    EXPECT_EQ(clsparseSuccess, status);
    //clsparseSynchronize(ClSparseEnvironment::control);

    std::vector<TypeParam> result(this->dense.size());

    clEnqueueReadBuffer(ClSparseEnvironment::queue,
                        this->g_dense, 1, 0,
                        result.size()*sizeof(TypeParam),
                        result.data(), 0, NULL, NULL);

    if(typeid(TypeParam) == typeid(float))
        for(int i = 0; i < this->dense.size(); i++)
            ASSERT_EQ(this->dense[i], result[i]);

    if(typeid(TypeParam) == typeid(double))
        for(int i = 0; i < this->dense.size(); i++)
            ASSERT_EQ(this->dense[i], result[i]);

}


int main (int argc, char* argv[])
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;
    //pass path to matrix as an argument, We can switch to boost po later

    std::string path;
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
