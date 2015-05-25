/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 * ************************************************************************/


#include <iostream>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <clSPARSE.h>

#include "clsparseTimer.extern.hpp"
#include "loadDynamicLibrary.hpp"
#include "functions/clfunc_xSpMdV.hpp"
#include "functions/clfunc_xCGM.hpp"


namespace po = boost::program_options;
namespace fs = boost::filesystem;

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
    fs::directory_iterator end_iter;
    const boost::regex filter( ".*\\.\\" + extension );
    bool found = false;

    if( fs::exists( dir ) && fs::is_directory( dir ) )
    {
        for( fs::directory_iterator dir_iter( dir ); dir_iter != end_iter; ++dir_iter )
        {
            if( fs::is_regular_file( dir_iter->status( ) ) )
            {
                std::string fname = dir_iter->path( ).filename( ).string( );

                if( boost::regex_match( fname, filter ) )
                {
                    // std::cout << "Adding: " << dir_iter->path( ) << std::endl;
                    matrix_files.push_back( dir_iter->path( ) );
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
    double alpha;
    double beta;
    size_t profileCount;
    //int transA_option;
    std::string function;
    std::string precision;
    std::string root_dir;

    po::options_description desc( "clSPARSE bench command line options" );
    desc.add_options( )
        ( "help,h", "produces this help message" )
        ( "dirpath,d", po::value( &root_dir ), "Matrix directory" )
        ( "alpha", po::value<double>( &alpha )->default_value( 1.0f ), "specifies the scalar alpha" )
        ( "beta", po::value<double>( &beta )->default_value( 0.0f ), "specifies the scalar beta" )
        //( "transposeA", po::value<int>( &transA_option )->default_value( 0 ), "0 = no transpose, 1 = transpose, 2 = conjugate transpose" )
        ( "function,f", po::value<std::string>( &function )->default_value( "SpMdV" ), "Sparse functions to test. Options: SpMdV, CGM, CGK" )
        ( "precision,r", po::value<std::string>( &precision )->default_value( "s" ), "Options: s,d,c,z" )
        ( "profile,p", po::value<size_t>( &profileCount )->default_value( 20 ), "Time and report the kernel speed (default: profiling off)" )
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


    //	Timer module discovered and loaded successfully
    //	Initialize function pointers to call into the shared module
    void* funcPtr = LoadFunctionAddr( timerLibHandle, "clsparseGetTimer" );
    PFCLSPARSETIMER sparseGetTimer = *static_cast<PFCLSPARSETIMER*>( static_cast<void*>( &funcPtr ) );

    std::unique_ptr< clsparseFunc > my_function;
    if( boost::iequals( function, "SpMdV" ) )
    {
        if( precision == "s" )
            my_function = std::unique_ptr< clsparseFunc >( new xSpMdV< float >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU ) );
        else if( precision == "d" )
            my_function = std::unique_ptr< clsparseFunc >( new xSpMdV< double >( sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU) );
        else
        {
            std::cerr << "Unknown spmdv precision" << std::endl;
            return -1;
        }
    }

    else if (boost::iequals(function, "CGM" ))
    {
        if (precision == "s")
            my_function = std::unique_ptr< clsparseFunc > ( new xCGM< float >(sparseGetTimer, profileCount, CL_DEVICE_TYPE_GPU ) );
        else
        {
            std::cerr << "Unknown CG precision (double not yet implemented)" << std::endl;
            return -1;
        }
    }
    //else if( boost::iequals( function, "Csr2dense" ) )
    //{
    //    if( precision == "s" )
    //        my_function = std::make_unique< xCsr2dense< float > >( timer );
    //    else if( precision == "d" )
    //        my_function = std::make_unique< xCsr2dense< double > >( timer );
    //    else
    //    {
    //        std::cerr << "Unknown xCsr2dense precision" << std::endl;
    //        return -1;
    //    }
    //}
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
            my_function->setup_buffer( alpha, beta, path );
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
