#include <gtest/gtest.h>
#include <vector>
#include <string>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"
#include "resources/matrix_utils.h"

cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;


template <typename T>
class TestCSRMV : public ::testing::Test
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

public:

    void SetUp()
    {
        //TODO:: take the values from cmdline;
        alpha = T(1);
        beta = T(0);

        x = std::vector<T>(CSRE::n_cols);
        y = std::vector<T>(CSRE::n_rows);

        std::fill(x.begin(), x.end(), T(1));
        std::fill(x.begin(), x.end(), T(0));

        cl_int status;
        gx = clCreateBuffer(CLSE::context,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            x.size() * sizeof(T), x.data(), &status);

        EXPECT_EQ(CL_SUCCESS, status); //is it wise to use this here?
        if (status != CL_SUCCESS)
        {
            std::cerr << "Problem with allocation of gx vector for tests "
                      << "(" << status << ")" << std::endl;
            exit(-1);
        }

        gy = clCreateBuffer(CLSE::context,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            y.size() * sizeof(T), y.data(), &status);
        if (status != CL_SUCCESS)
        {
            std::cerr << "Problem with allocation of gy vector for tests "
                      << "(" << status << ")" << std::endl;
            exit(-1);
        }

        generateReference(x, alpha, y, beta);


    }


private:
    void generateReference (const std::vector<float>& x,
                            const float alpha,
                            std::vector<float>& y,
                            const float beta)
    {
            csrmv(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
                CSRE::row_offsets, CSRE::col_indices, CSRE::f_values,
                  x, alpha, y, beta);
    }

    void generateReference (const std::vector<double>& x,
                            const double alpha,
                            std::vector<double>& y,
                            const double beta)
    {
            csrmv(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
                CSRE::row_offsets, CSRE::col_indices, CSRE::d_values,
                  x, alpha, y, beta);
    }

    cl_mem gx;
    cl_mem gy;
    std::vector<T> x;
    std::vector<T> y;

    T alpha;
    T beta;


};

typedef ::testing::Types<float, double> TYPES;
TYPED_TEST_CASE(TestCSRMV, TYPES);

TYPED_TEST(TestCSRMV, multiply)
{

}


int main (int argc, char* argv[])
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;
    //pass path to matrix as an argument, We can switch to boost po later
    std::string command_line_arg(argc == 2 ? argv[1] : "");

    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE());
    ::testing::AddGlobalTestEnvironment( new CSRE(command_line_arg,
                                                  CLSE::queue, CLSE::context));
    return RUN_ALL_TESTS();
}
