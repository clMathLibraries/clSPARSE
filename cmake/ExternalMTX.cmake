# ########################################################################
# Copyright 2015 Advanced Micro Devices, Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################

message( STATUS "Downloading MTX sparse Matrix Market files..." )

option( DOWNLOAD_MTX_BELL_GARLAND "Sparse matrix files used in Bell-Garlands paper" OFF )
option( DOWNLOAD_MTX_LARGE "MTX files where majority of time is in kernel" OFF )
option( DOWNLOAD_MTX_SMALL "MTX files less than 4GB in dense format" OFF )
option( DOWNLOAD_MTX_SPGEMM "MTX files used in Weifeng Paper for benchmarking SpGEMM" OFF )

# Emulate a mapping data structure in CMake
macro( MAP_ENTRY K V )
  SET( "MD5_${K}" "${V}" )
endmacro( )

function( MTX_process_files MTX_file_list MTX_file_path )
  message( STATUS "Optional file() flags = ${ARGV2} ")
  foreach( MTX ${${MTX_file_list}} )
    message( STATUS "Downloading MTX=${MTX} ")
    get_filename_component( MTX_DIR ${MTX} DIRECTORY )
    get_filename_component( MTX_NAME ${MTX} NAME )

    file( DOWNLOAD http://www.cise.ufl.edu/research/sparse/MM/${MTX}
      ${${MTX_file_path}}/${MTX_NAME}
      INACTIVITY_TIMEOUT 60
      EXPECTED_HASH MD5=${MD5_${MTX}}
      ${ARGV2} # Can be used to pass additional flags, such as SHOW_PROGRESS
      # TIMEOUT 300
      # [STATUS status]
      # [LOG log]
      # [TLS_VERIFY on|off]
      # [TLS_CAINFO file]
      )

    message( STATUS "Unzipping MTX=${MTX} ")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf ${${MTX_file_path}}/${MTX_NAME}
      WORKING_DIRECTORY ${${MTX_file_path}}
    )
  endforeach( )
endfunction( )

# Not found from the Bell-Garland paper
# dense2
# qcd5_4

##### Bell-Garland sparse data #####
set( MTX_Bell_Garland_files
  Williams/pdb1HYS.tar.gz
  Williams/consph.tar.gz
  Williams/cant.tar.gz
  Boeing/pwtk.tar.gz
  Bova/rma10.tar.gz
  DNVS/shipsec1.tar.gz
  Williams/mac_econ_fwd500.tar.gz
  Williams/mc2depi.tar.gz
  Williams/cop20k_A.tar.gz
  Hamm/scircuit.tar.gz
  Williams/webbase-1M.tar.gz
  Mittelmann/rail4284.tar.gz
)

MAP_ENTRY( "Williams/pdb1HYS.tar.gz" "fdbfaa0edb11e799f67870ebf16adfb0" )
MAP_ENTRY( "Williams/consph.tar.gz" "797e984d97d0057c9b88a3e4dd9af139" )
MAP_ENTRY( "Williams/cant.tar.gz" "a5360391e462640583e59a7b36fa216c" )
MAP_ENTRY( "Boeing/pwtk.tar.gz" "51617ea77ac8212ca3bf5c1eb041061b" )
MAP_ENTRY( "Bova/rma10.tar.gz" "a899a0c48b9a58d081c52ffd88a84955" )
MAP_ENTRY( "DNVS/shipsec1.tar.gz" "73372e7d6a0848f8b19d64a924fab73e" )
MAP_ENTRY( "Williams/mac_econ_fwd500.tar.gz" "f1b0e56fbb75d1d6862874e3d7d33060" )
MAP_ENTRY( "Williams/mc2depi.tar.gz" "8c8633eada6455c1784269b213c85ea6" )
MAP_ENTRY( "Williams/cop20k_A.tar.gz" "beb2302025bbfae6fd1f5604957ffe52" )
MAP_ENTRY( "Hamm/scircuit.tar.gz" "3e62f7ea83914f7e20019aefb2a5176f" )
MAP_ENTRY( "Williams/webbase-1M.tar.gz" "2d4c239daad6f12d66a1e6a2af44cbdb" )
MAP_ENTRY( "Mittelmann/rail4284.tar.gz" "6279700b7d44b44fd630c079b31eee46" )

set( Bell_Garland_MTX_path ${PROJECT_BINARY_DIR}/Externals/MTX/Bell_Garland )

if( DOWNLOAD_MTX_BELL_GARLAND )
  MTX_process_files( MTX_Bell_Garland_files Bell_Garland_MTX_path )
endif( )

##### Large Sparse Matrices #####

set( MTX_Large_files
  DIMACS10/delaunay_n24.tar.gz
  DIMACS10/kron_g500-logn21.tar.gz
  DIMACS10/europe_osm.tar.gz
  DIMACS10/rgg_n_2_24_s0.tar.gz
  LAW/hollywood-2009.tar.gz
  LAW/indochina-2004.tar.gz
  LAW/arabic-2005.tar.gz
#  LAW/webbase-2001.tar.gz
  Schenk/nlpkkt240.tar.gz
  Janna/Flan_1565.tar.gz
#  Fluorem/HV15R.tar.gz
)

MAP_ENTRY( "DIMACS10/delaunay_n24.tar.gz" "3d73b37d1d1a14247d143223ed3a4c7c" )
MAP_ENTRY( "DIMACS10/kron_g500-logn21.tar.gz" "06a52f2ffad6678c74407a6a21ea3ed0" )
MAP_ENTRY( "DIMACS10/europe_osm.tar.gz" "e3ae56c2799970fa7eb0053332a2ee3c" )
MAP_ENTRY( "DIMACS10/rgg_n_2_24_s0.tar.gz" "f91634d6c606385c8b17b903c3cd505d" )
MAP_ENTRY( "LAW/hollywood-2009.tar.gz" "df8816b08443d823de1c9e017210c36e" )
MAP_ENTRY( "LAW/indochina-2004.tar.gz" "7ab1e181adc460c2d70becf4c36bf577" )
MAP_ENTRY( "LAW/arabic-2005.tar.gz" "1933f0601bf87d51cec75fd687a079fd" )
# MAP_ENTRY( "LAW/webbase-2001.tar.gz" "a3e35e010e868255ef16a357dd219d3d" )
MAP_ENTRY( "Schenk/nlpkkt240.tar.gz" "84c13480dd4c02dccac47d6c9e8ed2f3" )
MAP_ENTRY( "Janna/Flan_1565.tar.gz" "8777f46c3b033e8b4fc24fa31dc79b4a" )
# MAP_ENTRY( "Fluorem/HV15R.tar.gz" "df9e3ea1b295c96655332b6726f31612" )

set( MTX_Large_path ${PROJECT_BINARY_DIR}/Externals/MTX/Large )

if( DOWNLOAD_MTX_LARGE )
  MTX_process_files( MTX_Large_files MTX_Large_path SHOW_PROGRESS )
endif( )

##### Small Sparse Matrices #####

set( MTX_Small_files
  AG-Monien/3elt.tar.gz
  AG-Monien/crack.tar.gz
  Alemdar/Alemdar.tar.gz
  Andrianov/fxm3_6.tar.gz
  Gset/G51.tar.gz
  Hamm/add20.tar.gz
  MathWorks/tomography.tar.gz
  MathWorks/QRpivot.tar.gz
  Meszaros/gas11.tar.gz
  Newman/celegansneural.tar.gz
  NYPA/Maragal_6.tar.gz
  Pajek/Cities.tar.gz
  Pajek/Reuters911.tar.gz
  PARSEC/Na5.tar.gz
  PARSEC/Si10H16.tar.gz
  Qaplib/lp_nug07.tar.gz
  QCD/conf5_0-4x4-18.tar.gz
  Rajat/rajat04.tar.gz
  Sandia/oscil_dcop_11.tar.gz
  UTEP/Dubcova1.tar.gz
  YCheng/psse1.tar.gz
  Zitney/hydr1c.tar.gz
)

MAP_ENTRY( "AG-Monien/3elt.tar.gz" "4699f0d2633d813c5ddfee7bfb551a54" )
MAP_ENTRY( "AG-Monien/crack.tar.gz" "82612407cbd68f3172adc9a6dc5562e6" )
MAP_ENTRY( "Alemdar/Alemdar.tar.gz" "ef8f5d3574ea68e23d1f8a7a06f649e5" )
MAP_ENTRY( "Andrianov/fxm3_6.tar.gz" "0b521cfe9a6be30bbbb70dba5adfd861" )
MAP_ENTRY( "Gset/G51.tar.gz" "b04be35eff02c18b92aaf3b9776eb81d" )
MAP_ENTRY( "Hamm/add20.tar.gz" "d01b99b95eb8fef566f1b49433d68c93" )
MAP_ENTRY( "MathWorks/tomography.tar.gz" "0079a0ad79a3861b8aab031300a0474d" )
MAP_ENTRY( "MathWorks/QRpivot.tar.gz" "cf5e3464656a77020d405ce2bcbb8eb6" )
MAP_ENTRY( "Meszaros/gas11.tar.gz" "265df346e4d7387c6ffa0f222049970f" )
MAP_ENTRY( "Newman/celegansneural.tar.gz" "7495ee4699a7c07ed904ee37b493e13c" )
MAP_ENTRY( "NYPA/Maragal_6.tar.gz" "276670cea54d8f57a3b88256f41b3bb3" )
MAP_ENTRY( "Pajek/Cities.tar.gz" "fbd4a17e1a887bc40bc866382edae037" )
MAP_ENTRY( "Pajek/Reuters911.tar.gz" "28df7f7830d134b5cee1313eb30f7041" )
MAP_ENTRY( "PARSEC/Na5.tar.gz" "29c5595e95c1f2476ada6fb397f0e2c9" )
MAP_ENTRY( "PARSEC/Si10H16.tar.gz" "81a89740c4cd77052af9df5edaae2d1c" )
MAP_ENTRY( "Qaplib/lp_nug07.tar.gz" "6cc995b3c4ec92cddaf9706ef2ac7457" )
MAP_ENTRY( "QCD/conf5_0-4x4-18.tar.gz" "082f48c8d5af68c67ba229685691d09e" )
MAP_ENTRY( "Rajat/rajat04.tar.gz" "bac8efa0265bf9bd19d287677c2f455e" )
MAP_ENTRY( "Sandia/oscil_dcop_11.tar.gz" "bacaaa14de4398c981ba749665459324" )
MAP_ENTRY( "UTEP/Dubcova1.tar.gz" "1eaf7aaf38385287505fe1d504f8a716" )
MAP_ENTRY( "YCheng/psse1.tar.gz" "8e3f802005c2bdf3cdf644d51665374f" )
MAP_ENTRY( "Zitney/hydr1c.tar.gz" "356935778c2fc0011a3a21b645ede961" )

set( MTX_Small_path ${PROJECT_BINARY_DIR}/Externals/MTX/Small )

if( DOWNLOAD_MTX_SMALL )
  MTX_process_files( MTX_Small_files MTX_Small_path )
endif( )

##### SPGEMM Square MTX sparse data #####
set( MTX_SPGEMM_BENCH_files
  Williams/cant.tar.gz  
  Williams/mac_econ_fwd500.tar.gz
  Williams/mc2depi.tar.gz
  Williams/cop20k_A.tar.gz
  Williams/webbase-1M.tar.gz
  Boeing/pwtk.tar.gz
  DNVS/shipsec1.tar.gz
  Hamm/scircuit.tar.gz  
  Oberwolfach/filter3D.tar.gz
  Um/2cubes_sphere.tar.gz
  vanHeukelum/cage12.tar.gz
  GHS_psdef/hood.tar.gz
  JGD_Homology/m133-b3.tar.gz
  QLi/majorbasis.tar.gz
  GHS_indef/mario002.tar.gz
  FreeFieldTechnologies/mono_500Hz.tar.gz
  Um/offshore.tar.gz
  Pajek/patents_main.tar.gz
  FEMLAB/poisson3Da.tar.gz  
)

MAP_ENTRY( "Oberwolfach/filter3D.tar.gz" "fdbfaa0edb11e799f67870ebf16adfb0" )
MAP_ENTRY( "Um/2cubes_sphere.tar.gz" "797e984d97d0057c9b88a3e4dd9af139" )
MAP_ENTRY( "Williams/cant.tar.gz" "a5360391e462640583e59a7b36fa216c" )
MAP_ENTRY( "Boeing/pwtk.tar.gz" "51617ea77ac8212ca3bf5c1eb041061b" )
MAP_ENTRY( "vanHeukelum/cage12.tar.gz" "a899a0c48b9a58d081c52ffd88a84955" )
MAP_ENTRY( "DNVS/shipsec1.tar.gz" "73372e7d6a0848f8b19d64a924fab73e" )
MAP_ENTRY( "Williams/mac_econ_fwd500.tar.gz" "f1b0e56fbb75d1d6862874e3d7d33060" )
MAP_ENTRY( "Williams/mc2depi.tar.gz" "8c8633eada6455c1784269b213c85ea6" )
MAP_ENTRY( "Williams/cop20k_A.tar.gz" "beb2302025bbfae6fd1f5604957ffe52" )
MAP_ENTRY( "Hamm/scircuit.tar.gz" "3e62f7ea83914f7e20019aefb2a5176f" )
MAP_ENTRY( "Williams/webbase-1M.tar.gz" "2d4c239daad6f12d66a1e6a2af44cbdb" )
MAP_ENTRY( "GHS_psdef/hood.tar.gz" "6279700b7d44b44fd630c079b31eee46" )
MAP_ENTRY( "JGD_Homology/m133-b3.tar.gz" "29c5595e95c1f2476ada6fb397f0e2c9" )
MAP_ENTRY( "QLi/majorbasis.tar.gz" "81a89740c4cd77052af9df5edaae2d1c" )
MAP_ENTRY( "GHS_indef/mario002.tar.gz" "6cc995b3c4ec92cddaf9706ef2ac7457" )
MAP_ENTRY( "FreeFieldTechnologies/mono_500Hz.tar.gz" "082f48c8d5af68c67ba229685691d09e" )
MAP_ENTRY( "Um/offshore.tar.gz" "bac8efa0265bf9bd19d287677c2f455e" )
MAP_ENTRY( "Pajek/patents_main.tar.gz" "bacaaa14de4398c981ba749665459324" )
MAP_ENTRY( "FEMLAB/poisson3Da.tar.gz" "1eaf7aaf38385287505fe1d504f8a716" )

set( SPGEMM_BENCH_MTX_path ${PROJECT_BINARY_DIR}/Externals/MTX/SpGemmData )

if( DOWNLOAD_MTX_SPGEMM )
  MTX_process_files( MTX_SPGEMM_BENCH_files SPGEMM_BENCH_MTX_path )
endif( )
