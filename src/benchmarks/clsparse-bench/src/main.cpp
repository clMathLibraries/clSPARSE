/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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

#include <iostream>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <clSPARSE.h>

#include "clsparseTimer-extern.hpp"
#include "loadDynamicLibrary.hpp"
#include "functions/clfunc_xSpMdV.hpp"
#include "functions/clfunc-xSpMdM.hpp"
#include "functions/clfunc_xCG.hpp"
#include "functions/clfunc_xBiCGStab.hpp"
#include "functions/clfunc_xDense2Csr.hpp"
#include "functions/clfunc_xCsr2Dense.hpp"
#include "functions/clfunc_xCsr2Coo.hpp"
#include "functions/clfunc_xCoo2Csr.hpp"
#include "functions/clfunc_xSpMSpM.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;


struct recursive_directory_range
{
    typedef fs::recursive_directory_iterator dir_iterator;

    recursive_directory_range( fs::path p ): p_( p ) {}

    dir_iterator begin( ) { return fs::recursive_directory_iterator( p_ ); }
    dir_iterator end( ) { return fs::recursive_directory_iterator( ); }

    fs::path p_;
};

/**
* @brief findMatrices
* @param root path to the directory where to search for files with extension
* @param extension matrix files extension without "." just mtx
* @param matrix_files vector of found files with given extension
* @return true if any files were found
*/
bool findMatrices( const std::string& root,
                   const std::string& extension,
                   std::vector<fs::path>& matrix_files )
{
    fs::path dir( root );

    recursive_directory_range recursive_directory_it( dir );

    const boost::regex filter( ".*\\.\\" + extension );
    bool found = false;

    if( fs::exists( dir ) && fs::is_directory( dir ) )
    {
        for( auto it : recursive_directory_range( dir ) )
        {
            //std::cout << "Checking:" << it << std::endl;
            if( fs::is_regular_file( it.status( ) ) )
            {
                std::string fname = it.path( ).filename( ).string( );

                std::string fname_suffix = fname.substr( fname.size( ) - 6 );

                if( boost::regex_match( fname, filter ) )
                {
                    std::cout << "\tAdding:" << it.path( ) << std::endl;
                    matrix_files.push_back( it.path( ) );
                    found = true;
                }
            }
        }
    }
    else
    {
        std::cerr << dir << " does not name a directory or directory does not exists!" << std::endl;
        return false;
    }

    return found;
}

std::vector< fs::path > enumMatrices( const std::string& root_dir )
{
    std::vector< fs::path > matxVec;

    if( !findMatrices( root_dir, "mtx", matxVec ) )
        throw std::runtime_error( "Could not read matrix files from disk" );

    // Potential RVO optimization should make returning vector copy speedy
    return matxVec;
}

int main( int argc, char *argv[ ] )
{
    cl_double alpha, beta;
    clsparseIdx_t rows, columns;
    size_t profileCount;
    std::string function;
    std::string precision;
    std::string root_dir;

    po::options_description desc( "clSPARSE bench command line options" );
    desc.add_options( )
        ( "help,h", "produces this help message" )
        ( "dirpath,d", po::value( &root_dir ), "Matrix directory" )
        ( "alpha,a", po::value<cl_double>( &alpha )->default_value( 1.0f ), "specifies the scalar alpha" )
        ( "beta,b", po::value<cl_double>( &beta )->default_value( 0.0f ), "specifies the scalar beta" )
        ( "rows", po::value<clsparseIdx_t>( &rows )->default_value( 16 ), "specifies the number of rows for matrix data" )
        ( "columns", po::value<clsparseIdx_t>( &columns )->default_value( 16 ), "specifies the number of columns for matrix data" )
        ( "function,f", po::value<std::string>( &function )->default_value( "SpMdV" ), "Sparse functions to test. Options: "
                    "SpMdV, SpMdM, SpMSpM, CG, BiCGStab, Csr2Dense, Dense2Csr, Csr2Coo, Coo2Csr" )
        ( "precision,r", po::value<std::string>( &precision )->default_value( "s" ), "Options: s,d,c,z" )
        ( "profile,p", po::value<size_t>( &profileCount )->default_value( 20 ), "Number of times to run the desired test function" )
        ( "extended,e", po::bool_switch()->default_value(false), "Use compensated summation to improve accuracy by emulating extended precision" )
        ( "no_zeroes,z", po::bool_switch()->default_value(false), "Disable reading explicit zeroes from the input matrix market file.")
        ;

    po::variables_map vm;
    po::store( po::parse_command_line( argc, argv, desc ), vm );
    po::notify( vm );

    if( vm.count( "help" ) )
    {
        std::cout << desc << std::endl;
        return 0;
    }

    if( precision != "s" && precision != "d" ) // && precision != "c" && precision != "z" )
    {
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }

    if( vm.count( "dirpath" ) == 0 )
    {
        std::cerr << "The [" << "root" << "] parameter is missing!" << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    //	Discover and load the timer module if present
    void* timerLibHandle = LoadSharedLibrary( "lib", "clsparseTimer", false );
    if( timerLibHandle == NULL )
    {
        std::cerr << "Could not find the external timing library; timings disabled" << std::endl;
    }

    cl_bool extended_precision = false;
    if (vm["extended"].as<bool>())
        extended_precision = true;
    cl_bool explicit_zeroes = true;
    if (vm["no_zeroes"].as<bool>())
        explicit_zeroes = false;

    //	Timer module discovered and loaded successfully
    //	Initialize function pointers to call into the shared module
    void* funcPtr = LoadFunctionAddr( timerLibHandle, "clsparseGetTimer" );
    PFCLSPARSETIMER sparseGetTimer = *static_cast<PFCLSPARSETIMER*>( static_cast<void*>( &funcPtr ) );

    std::unique_ptr< clsparseFunc > my_function;
    if( boost::iequals( function, "SpMdV" ) )
    {
        if( precision == "s" )
            my_function = std::unique_ptr< clsparseFunc >( new xSpMdV< float >( sparseGetTimer, profileCount, extended_precision, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
        else if( precision == "d" )
            my_function = std::unique_ptr< clsparseFunc >( new xSpMdV< double >( sparseGetTimer, profileCount, extended_precision, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
        else
        {
            std::cerr << "Unknown spmdv precision" << std::endl;
            return -1;
        }
    }
    else if( boost::iequals( function, "CG" ) )
    {
        if( precision == "s" )
            my_function = std::unique_ptr< clsparseFunc >( new xCG< float >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
        else
            my_function = std::unique_ptr< clsparseFunc >( new xCG< double >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
    }

    else if( boost::iequals( function, "BiCGStab" ) )
    {
        if( precision == "s" )
            my_function = std::unique_ptr< clsparseFunc >( new xBiCGStab< float >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
        else
            my_function = std::unique_ptr< clsparseFunc >( new xBiCGStab< double >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
    }
    else if( boost::iequals( function, "SpMdM" ) )
    {
        if( precision == "s" )
            my_function = std::unique_ptr< clsparseFunc >( new xSpMdM< cl_float >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, columns, explicit_zeroes ) );
        else
            my_function = std::unique_ptr< clsparseFunc >( new xSpMdM< cl_double >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, columns, explicit_zeroes ) );
    }
    else if (boost::iequals(function, "SpMSpM"))
    {
        if (precision == "s")
            my_function = std::unique_ptr< clsparseFunc>(new xSpMSpM< cl_float >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
        else
            my_function = std::unique_ptr< clsparseFunc >(new xSpMSpM< cl_double >(sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
    }
    else if( boost::iequals( function, "Coo2Csr" ) )
    {
        if( precision == "s" )
            my_function = std::unique_ptr< clsparseFunc >( new xCoo2Csr< float >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
        else
            my_function = std::unique_ptr< clsparseFunc >( new xCoo2Csr< double >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
    }
    else if( boost::iequals( function, "Dense2Csr" ) )
    {
        if( precision == "s" )
            my_function = std::unique_ptr< clsparseFunc >( new xDense2Csr< float >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
        else
            my_function = std::unique_ptr< clsparseFunc >( new xDense2Csr< double >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
    }
    else if( boost::iequals( function, "Csr2Dense" ) )
    {
        if( precision == "s" )
            my_function = std::unique_ptr< clsparseFunc >( new xCsr2Dense< cl_float >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
        else
            my_function = std::unique_ptr< clsparseFunc >( new xCsr2Dense< cl_double >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
    }
    else if( boost::iequals( function, "Csr2Coo" ) )
    {
        if( precision == "s" )
            my_function = std::unique_ptr< clsparseFunc >( new xCsr2Coo< cl_float >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
        else
            my_function = std::unique_ptr< clsparseFunc >( new xCsr2Coo< cl_double >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU, explicit_zeroes ) );
    }
    else
    {
        std::cerr << "Benchmarking unknown function" << std::endl;
        return -1;
    }

    try
    {
        std::vector< fs::path > matrix_files = enumMatrices( root_dir );

        for( auto& file : matrix_files )
        {
            std::string path = file.string( );
            try {
                my_function->setup_buffer( alpha, beta, path );
            }
            // I expect to catch trow from clsparseHeaderfromFile
            // If io_exception then we don't need to cleanup.
            // If runtime_exception is catched we are doomed!
            catch( clsparse::io_exception& io_exc )
            {
                std::cout << io_exc.what( ) << std::endl;
                continue;
            }
            my_function->initialize_cpu_buffer( );
            my_function->initialize_gpu_buffer( );

            for( int i = 0; i < profileCount; ++i )
            {
                my_function->call_func( );
                my_function->reset_gpu_write_buffer( );
            }
            my_function->cleanup( );

            //std::cout << "clSPARSE kernel execution time < ns >: " << my_function->time_in_ns( ) << std::endl;
            //std::cout << "clSPARSE kernel execution Gflops < " <<
            //    my_function->bandwidth_formula( ) << " >: " << my_function->bandwidth( ) << std::endl << std::endl;
        }
    }
    catch( std::exception& exc )
    {
        std::cerr << exc.what( ) << std::endl;
        return 1;
    }

    FreeSharedLibrary( timerLibHandle );

    return 0;
}
