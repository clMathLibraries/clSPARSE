/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * ************************************************************************/


// Standard system include files
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

// Cuda runtime and libraries files
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusparse_v2.h>

// This project files
#include "include/statisticalTimer.h"
#include "functions/cufunc_xSpMdV.hpp"
#include "functions/cufunc_xCsr2dense.hpp"

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
    std::cout << "Searching for files like: " << filter.str( ) << std::endl;
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
                    std::cout << "Adding: " << dir_iter->path( ) << std::endl;
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

int main(int argc, char *argv[])
{
  double alpha;
  double beta;
  int profileCount;
  //int transA_option;
  std::string function;
  std::string precision;
  std::string root_dir;

  po::options_description desc( "cuSPARSE bench command line options" );
  desc.add_options()
    ( "help,h", "produces this help message" )
    ( "dirpath,d", po::value( &root_dir ), "Matrix directory" )
    ( "alpha", po::value<double>( &alpha )->default_value( 1.0f ), "specifies the scalar alpha" )
    ( "beta", po::value<double>( &beta )->default_value( 0.0f ), "specifies the scalar beta" )
    //( "transposeA", po::value<int>( &transA_option )->default_value( 0 ), "0 = no transpose, 1 = transpose, 2 = conjugate transpose" )
    ( "function,f", po::value<std::string>( &function )->default_value( "SpMdV" ), "Sparse functions to test. Options: SpMdV" )
    ( "precision,r", po::value<std::string>( &precision )->default_value( "s" ), "Options: s,d,c,z" )
    ( "profile,p", po::value<int>( &profileCount )->default_value( 20 ), "Time and report the kernel speed (default: profiling off)" )
    ;

  po::variables_map vm;
  po::store( po::parse_command_line( argc, argv, desc ), vm );
  po::notify( vm );

  if( vm.count( "help" ) )
  {
    std::cout << desc << std::endl;
    return 0;
  }

  if( precision != "s" && precision != "d" && precision != "c" && precision != "z" )
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

  StatisticalTimer& timer = StatisticalTimer::getInstance( );
  timer.Reserve( 3, profileCount );
  timer.setNormalize( true );

  std::unique_ptr< cusparseFunc > my_function;
  if( boost::iequals( function, "SpMdV" ) )
  {
    if( precision == "s" )
        my_function = std::unique_ptr< cusparseFunc >( new xSpMdV< float >( timer ) );
    //else if( precision == "d" )
    //    my_function = std::make_unique< xSpMdV< double > >( timer );
    else
    {
      std::cerr << "Unknown spmdv precision" << std::endl;
      return -1;
    }
  }
  else if( boost::iequals( function, "Csr2dense" ) )
  {
      if( precision == "s" )
          my_function = std::unique_ptr< cusparseFunc >( new xCsr2dense< float >( timer ) );
      //else if( precision == "d" )
      //    my_function = std::make_unique< xCsr2dense< double > >( timer );
      else
      {
          std::cerr << "Unknown xCsr2dense precision" << std::endl;
          return -1;
      }
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
          timer.Reset( );

          std::string path = file.string( );
          my_function->setup_buffer( alpha, beta, path );
          my_function->initialize_cpu_buffer( );
          my_function->initialize_gpu_buffer( );

          for( int i = 0; i < profileCount; ++i )
          {
              my_function->call_func( );
              my_function->reset_gpu_write_buffer( );
          }
          my_function->releaseGPUBuffer_deleteCPUBuffer( );

        timer.pruneOutliers( 3.0 );
        std::cout << "cuSPARSE matrix: " << path << std::endl;
        std::cout << "cuSPARSE kernel execution time < ns >: " << my_function->time_in_ns( ) << std::endl;
        std::cout << "cuSPARSE kernel execution Gflops < " <<
            my_function->bandwidth_formula( ) << " >: " << my_function->bandwidth( ) << std::endl << std::endl;
      }

  }
  catch( std::exception& exc )
  {
      std::cerr << exc.what( ) << std::endl;
      return 1;
  }

  return 0;
}

