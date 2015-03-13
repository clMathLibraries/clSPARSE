#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "resources/matrix_utils.h"
#include "resources/matrix_market.h"


namespace po = boost::program_options;
std::string path;

const std::string out_path = "AT_out.mtx";

TEST (matrix_transpose, transpose)
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


    std::vector<int> row_offsets_t;
    std::vector<int> col_indices_t;
    std::vector<float> values_t;

    csr_transpose(n_rows, n_cols, n_vals,
                  row_offsets, col_indices, values,
                  row_offsets_t, col_indices_t, values_t);

    status = writeMatrixMarketCSR(row_offsets_t, col_indices_t, values_t,
                                 n_rows, n_cols, n_vals, out_path);
    ASSERT_EQ(true, status);
}

int main (int argc, char* argv[])
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
