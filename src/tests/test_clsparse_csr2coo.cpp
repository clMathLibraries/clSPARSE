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
clsparseStatus generateResult(cl_mem g_src, cl_mem g_dst, cl_mem g_value)
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

    if(typeid(T) == typeid(float))
    {
        return clsparseScsr2coo(CSRE::n_rows,
                                CSRE::n_cols,
                                CSRE::n_vals,
                                CSRE::cl_row_offsets,
                                CSRE::cl_col_indices,
                                CSRE::cl_f_values,
                                g_src,
                                g_dst,
                                g_value,
                                CLSE::control);

    }

    if(typeid(T) == typeid(double))
    {
       return clsparseDcsr2coo(CSRE::n_rows,
                               CSRE::n_cols,
                               CSRE::n_vals,
                               CSRE::cl_row_offsets,
                               CSRE::cl_col_indices,
                               CSRE::cl_d_values,
                               g_src,
                               g_dst,
                               g_value,
                               CLSE::control);

    }

}

template <typename T>
class TestCSR2COO : public ::testing::Test
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

public:

    void SetUp()
    {
        //TODO:: take the values from cmdline;

        src   = std::vector<int>(CSRE::n_vals, 0);
        dst   = std::vector<int>(CSRE::n_vals, 0);
        value = std::vector<T>(CSRE::n_vals, 0);

        cl_int status;
        g_src = clCreateBuffer(CLSE::context,
                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               src.size() * sizeof(int), src.data(), &status);

        ASSERT_EQ(CL_SUCCESS, status); //is it wise to use this here?

        g_dst = clCreateBuffer(CLSE::context,
                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               dst.size() * sizeof(int), dst.data(), &status);

        ASSERT_EQ(CL_SUCCESS, status); //is it wise to use this here?

        g_value = clCreateBuffer(CLSE::context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 value.size() * sizeof(T), value.data(), &status);

        ASSERT_EQ(CL_SUCCESS, status); //is it wise to use this here?


        generateReference(src, dst, value);

    }


    void generateReference (std::vector<int>& src, std::vector<int>& dst, std::vector<float>& value)
    {
            csr2coo(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
                    CSRE::row_offsets, CSRE::col_indices, CSRE::f_values,
                    src, dst, value);
    }

    void generateReference (std::vector<int>& src, std::vector<int>& dst, std::vector<double>& value)
    {
            csr2coo(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
                    CSRE::row_offsets, CSRE::col_indices, CSRE::d_values,
                    src, dst, value);
    }

    cl_mem g_src;
    cl_mem g_dst;
    cl_mem g_value;

    std::vector<int> src;
    std::vector<int> dst;
    std::vector<T> value;

};

typedef ::testing::Types<float, double> TYPES;
TYPED_TEST_CASE(TestCSR2COO, TYPES);

TYPED_TEST(TestCSR2COO, transform)
{

    cl_event event = NULL;
    //clsparseSetupEvent(ClSparseEnvironment::control, &event );

    clsparseStatus status;

    status = generateResult<TypeParam>(this->g_src, this->g_dst, this->g_value);

    EXPECT_EQ(clsparseSuccess, status);

    //clsparseSynchronize(ClSparseEnvironment::control);

    std::vector<int> resultS(this->src.size());
    std::vector<int> resultD(this->dst.size());
    std::vector<TypeParam> resultV(this->value.size());


    clEnqueueReadBuffer(ClSparseEnvironment::queue,
                        this->g_src, 1, 0,
                        resultS.size()*sizeof(int),
                        resultS.data(), 0, NULL, NULL);

    clEnqueueReadBuffer(ClSparseEnvironment::queue,
                        this->g_dst, 1, 0,
                        resultD.size()*sizeof(int),
                        resultD.data(), 0, NULL, NULL);

    clEnqueueReadBuffer(ClSparseEnvironment::queue,
                        this->g_value, 1, 0,
                        resultV.size()*sizeof(TypeParam),
                        resultV.data(), 0, NULL, NULL);

    if(typeid(TypeParam) == typeid(float)){
        for(int i = 0; i < this->src.size(); i++){
            ASSERT_EQ(this->src[i], resultS[i]);
            //std::cout << this->src[i] << " " << resultS[i] << std::endl;
        }
        for(int i = 0; i < this->dst.size(); i++){
            ASSERT_EQ(this->dst[i], resultD[i]);
        }
        for(int i = 0; i < this->value.size(); i++){
            ASSERT_EQ(this->value[i], resultV[i]);
        }
    }

    if(typeid(TypeParam) == typeid(double)){
        for(int i = 0; i < this->src.size(); i++){
            ASSERT_EQ(this->src[i], resultS[i]);
            //std::cout << this->src[i] << " " << resultS[i] << std::endl;
        }
        for(int i = 0; i < this->dst.size(); i++){
            ASSERT_EQ(this->dst[i], resultD[i]);
            //std::cout << this->dst[i] << " " << resultD[i] << std::endl;
        }
        for(int i = 0; i < this->value.size(); i++){
            ASSERT_EQ(this->value[i], resultV[i]);
            //std::cout << this->value[i] << " " << resultV[i] << std::endl;
        }
    }


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

