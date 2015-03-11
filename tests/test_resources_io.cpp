#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "resources/matrix_utils.h"
#include "resources/matrix_market.h"

/** Just a simple test checking if the io functions for matrices are ok */

namespace po = boost::program_options;
std::string path;

const std::string out_path = "Aout.mtx";

TEST (matrix_io, load_save)
{
    std::cout << "Loading: " << path << std::endl;

    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;

    int n_rows;
    int n_cols;
    int n_vals;

    bool status = readMatrixMarketCSR(row_offsets, col_indices, values,
                        n_rows, n_cols, n_vals, path);

    ASSERT_EQ(true, status);

    status = writeMatrixMarketCSR(row_offsets, col_indices, values,
                                  n_rows, n_cols, n_vals, out_path);

    ASSERT_EQ(true, status);


    std::vector<int> row_offsets1;
    std::vector<int> col_indices1;
    std::vector<float> values1;

    int n_rows1;
    int n_cols1;
    int n_vals1;

    status = readMatrixMarketCSR(row_offsets1, col_indices1, values1,
                        n_rows1, n_cols1, n_vals1, out_path);

    ASSERT_EQ(true, status);

    ASSERT_EQ(n_cols, n_cols1);
    ASSERT_EQ(n_rows, n_rows1);
    ASSERT_EQ(n_vals, n_vals1);

    for(int i = 0; i < row_offsets.size(); i++)
        EXPECT_EQ(row_offsets[i], row_offsets1[i]);

    for(int i = 0; i < col_indices.size(); i++)
        EXPECT_EQ(col_indices[i], col_indices1[i]);

    for(int i = 0; i < values.size(); i++)
        EXPECT_NEAR(values[i], values1[i], 1e-5);

}


int main(int argc, char* argv[])
{

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("path,p", po::value(&path)->required(),
             "Path to matrix in mtx format.");


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

    return RUN_ALL_TESTS();
}
