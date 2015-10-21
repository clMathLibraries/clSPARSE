/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>



//boost ublas
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"

//#include "resources/uBLAS-linalg/ublas_pcg.hpp"

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;
//cl_uint ClSparseEnvironment::N = 1024;

static cl_bool explicit_zeroes = true;

namespace po = boost::program_options;
namespace uBLAS = boost::numeric::ublas;

// Solver parameters;
double absoluteTolerance;
double relativeTolerance;
int maxIterations;
PRECONDITIONER precond;
PRINT_MODE printMode;


// Initial values of rhs and x vectors
double initialRhsValue;
double initialUnknownsValue;


template <typename T>
class Solver : public ::testing::Test
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;

public:


    void SetUp()
    {
        // Setup solver control
        clsparseStatus status;
        solverControl = clsparseCreateSolverControl(precond,
                                                    maxIterations,
                                                    relativeTolerance,
                                                    absoluteTolerance);
        ASSERT_NE(nullptr, solverControl);

        status = clsparseSolverPrintMode(solverControl, printMode);
        ASSERT_EQ(clsparseSuccess, status);

        // Setup rhs and vector of unknowns

        hX = uBLAS::vector<T>(CSRE::n_cols, (T) initialUnknownsValue);
        hB = uBLAS::vector<T>(CSRE::n_rows, (T) initialRhsValue);

        clsparseInitVector(&gX);
        clsparseInitVector(&gB);

        cl_int cl_status;

        gX.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   hX.size() * sizeof(T), hX.data().begin(),
                                   &cl_status);
        gX.num_values = hX.size();
        ASSERT_EQ(CL_SUCCESS, cl_status);

        gB.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   hB.size() * sizeof(T), hB.data().begin(),
                                   &cl_status);
        gB.num_values = hB.size();
        ASSERT_EQ(CL_SUCCESS, cl_status);

    }

    void TearDown()
    {
        clsparseReleaseSolverControl(solverControl);

        ::clReleaseMemObject(gX.values);
        ::clReleaseMemObject(gB.values);

        clsparseInitVector(&gX);
        clsparseInitVector(&gB);
    }

    void test_cg()
    {

        if ( typeid(T) == typeid(cl_float) )
        {
            clsparseStatus status =
                    clsparseScsrcg(&gX, &CSRE::csrSMatrix, &gB, solverControl, CLSE::control);
            ASSERT_EQ(clsparseSuccess, status);

//            std::cout << "Running CPU Solver" << std::endl;
//            std::shared_ptr<Preconditioner<CSRE::sMatrixType, uBLAS::vector<T>> > ublas_precond;

//            if (precond == DIAGONAL )
//            {
//                ublas_precond = std::shared_ptr<Preconditioner<CSRE::sMatrixType, uBLAS::vector<T>> >
//                        (new DiagonalPreconditioner<CSRE::sMatrixType, uBLAS::vector<T>> (CSRE::ublasSCsr));
//            }
//            else
//            {
//                ublas_precond = std::shared_ptr<Preconditioner<CSRE::sMatrixType, uBLAS::vector<T>> >
//                        (new IdentityPreconditioner<CSRE::sMatrixType, uBLAS::vector<T>> (CSRE::ublasSCsr));
//            }

//            size_t iters = pcg_solve(CSRE::ublasSCsr, hX, hB, *ublas_precond,
//                                   maxIterations, relativeTolerance, absoluteTolerance);

//            double ublas_final_residual =
//                    uBLAS::norm_2(prod( CSRE::ublasSCsr, hX) - hB) / uBLAS::norm_2(hB);

//            std::cout << "uBLAS Solver finished calculations" << std::endl;
//            std::cout << "\tfinal residual = " << ublas_final_residual
//                      << "\titerations = " << iters << std::endl;

        }

        if ( typeid(T) == typeid(cl_double))
        {
            clsparseStatus status =
                    clsparseDcsrcg(&gX, &CSRE::csrDMatrix, &gB, solverControl, CLSE::control);
            ASSERT_EQ(clsparseSuccess, status);

//            std::cout << "Running CPU Solver" << std::endl;
//            std::shared_ptr<Preconditioner<CSRE::dMatrixType, uBLAS::vector<T>> > ublas_precond;

//            if (precond == DIAGONAL )
//            {
//                ublas_precond = std::shared_ptr<Preconditioner<CSRE::dMatrixType, uBLAS::vector<T>> >
//                        (new DiagonalPreconditioner<CSRE::dMatrixType, uBLAS::vector<T>> (CSRE::ublasDCsr));
//            }
//            else
//            {
//                ublas_precond = std::shared_ptr<Preconditioner<CSRE::dMatrixType, uBLAS::vector<T>> >
//                        (new IdentityPreconditioner<CSRE::dMatrixType, uBLAS::vector<T>> (CSRE::ublasDCsr));
//            }

//            size_t iters = pcg_solve(CSRE::ublasDCsr, hX, hB, *ublas_precond,
//                                   maxIterations, relativeTolerance, absoluteTolerance);

//            double ublas_final_residual =
//                    uBLAS::norm_2(prod( CSRE::ublasDCsr, hX) - hB) / uBLAS::norm_2(hB);

//            std::cout << "uBLAS Solver finished calculations" << std::endl;
//            std::cout << "\tfinal residual = " << ublas_final_residual
//                      << "\titerations = " << iters << std::endl;

        }

    }

    void test_bicg()
    {

        if ( typeid(T) == typeid(cl_float) )
        {
            clsparseStatus status =
                    clsparseScsrbicgStab(&gX, &CSRE::csrSMatrix, &gB, solverControl, CLSE::control);
            ASSERT_EQ(clsparseSuccess, status);
        }

        if ( typeid(T) == typeid(cl_double))
        {
            clsparseStatus status =
                    clsparseDcsrbicgStab(&gX, &CSRE::csrDMatrix, &gB, solverControl, CLSE::control);
            ASSERT_EQ(clsparseSuccess, status);
        }
    }



    uBLAS::vector<T> hX;
    uBLAS::vector<T> hB;

    cldenseVector gX;
    cldenseVector gB;

    clSParseSolverControl solverControl;



};


typedef ::testing::Types<cl_float, cl_double> TYPES;
TYPED_TEST_CASE(Solver, TYPES);

TYPED_TEST(Solver, cg)
{
    this->test_cg();
}


TYPED_TEST(Solver, bicgstab)
{
    this->test_bicg();
}


int main (int argc, char* argv[])
{

    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;

    std::string path;


    std::string strPrecond;
    std::string strPrintMode;



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
            ("relative,r", po::value(&relativeTolerance)->default_value(1e-2),
             "Final residuum value relative to initial (r = b - A*x)")
            ("absolute,a", po::value(&absoluteTolerance)->default_value(1e-5),
             "Absolute level of residuum value")
            ("iterations,i", po::value(&maxIterations)->default_value(100),
             "Maximum number of solver iterations")
            ("precond,c", po::value(&strPrecond)->default_value("Diagonal"),
             "Type of preconditioner, Diagonal or Void")
            ("mode,m", po::value(&strPrintMode)->default_value("Normal"),
             "Solver Print mode, Verbose, Normal, Quiet")
            ("initx,x", po::value(&initialUnknownsValue)->default_value(0),
             "Initial value for vector of unknowns")
            ("initb,b", po::value(&initialRhsValue)->default_value(1),
             "Initial value for rhs vector")
            ("no_zeroes,z", po::bool_switch()->default_value(false),
             "Disable reading explicit zeroes from the input matrix market file.");


    po::variables_map vm;
    po::parsed_options parsed =
            po::command_line_parser( argc, argv ).options( desc ).allow_unregistered( ).run( );

    try {
        po::store(parsed, vm);
        po::notify(vm);
    }
    catch (po::error& error)
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what() << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    std::vector< std::string > to_pass_further =
            po::collect_unrecognized( parsed.options, po::include_positional );

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

    if (vm["no_zeroes"].as<bool>())
        explicit_zeroes = false;

    //pickup preconditioner
    if ( boost::iequals(strPrecond, "Diagonal"))
    {
        precond = DIAGONAL;
    }
    else if (boost::iequals(strPrecond, "Void"))
    {
        precond = NOPRECOND;
    }

    else
    {
        std::cerr << "Invalid preconditioner [" << strPrecond << "] "
                  << "using Diagonal instead" << std::endl;
        precond = DIAGONAL;
    }

    //pickup solver print mode;
    if ( boost::iequals(strPrintMode, "Normal"))
    {
        printMode = NORMAL;
    }
    else if (boost::iequals(strPrintMode, "Quiet"))
    {
        printMode = QUIET;
    }
    else if (boost::iequals(strPrintMode, "Verbose"))
    {
        printMode = VERBOSE;
    }
    else
    {
        std::cerr << "Invalid print mode given [" << strPrintMode << "]"
                  << " using Quiet instead" << std::endl;
        printMode = QUIET;
    }


    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE(pID, dID));
    ::testing::AddGlobalTestEnvironment( new CSRE(path, 0, 0,
                                                  CLSE::queue, CLSE::context, explicit_zeroes));

    return RUN_ALL_TESTS();

}
