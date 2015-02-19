#include <gtest/gtest.h>
#include <vector>
#include <string>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"
#include "resources/matrix_utils.h"

cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;


TEST (csrmv, simple_host_test)
{
    using CSRE = CSREnvironment;

    std::vector<float> x(CSRE::n_cols);
    std::vector<float> y(CSRE::n_rows);

    std::fill(x.begin(), x.end(), 1.0);
    std::fill(y.begin(), y.end(), 0.0);

    float alpha = 1.0;
    float beta = 0.0;

    csrmv(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
          CSRE::row_offsets, CSRE::col_indices, CSRE::f_values,
          x, alpha, y, beta);

    ASSERT_TRUE(true);
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
