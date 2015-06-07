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

TEST (CG, float)
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;

    std::vector<cl_float> x(CSRE::n_cols, 10.0);
    std::vector<cl_float> b(CSRE::n_rows, 1.0);

    clsparseVector gx;
    clsparseVector gb;
    clsparseInitVector(&gx);
    clsparseInitVector(&gb);

    cl_int status;
    gx.values = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               x.size() * sizeof(cl_float), x.data(), &status);

    gx.n = x.size();
    ASSERT_EQ(CL_SUCCESS, status);

    gb.values = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               b.size() * sizeof(cl_float), b.data(), &status);
    gb.n = b.size();

    ASSERT_EQ(CL_SUCCESS, status);

    clSParseSolverControl solver_control =
            clsparseCreateSolverControl(NOPRECOND, 5600, 1e-8, 0);

    ASSERT_NE(nullptr, solver_control);
    clsparseSolverPrintMode(solver_control, NORMAL);

    status = clsparseScsrcg(&gx, &CSRE::csrSMatrix, &gb, solver_control, CLSE::control);

    ASSERT_EQ(CL_SUCCESS, status);

    status = clsparseReleaseSolverControl(solver_control);
    ASSERT_EQ(CL_SUCCESS, status);

    ::clReleaseMemObject(gx.values);
    ::clReleaseMemObject(gb.values);
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
