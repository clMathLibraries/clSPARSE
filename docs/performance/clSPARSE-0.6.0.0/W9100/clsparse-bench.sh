#!/bin/bash

#Executable
clsparse_bench=~/code/github/clMathLibraries/bin/clSPARSE/release/clSPARSE-build/staging/clsparse-bench

#Data directories
mtx_cant=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/cant
mtx_consph=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/consph
mtx_cop20k_A=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/cop20k_A
mtx_mac_econ_fwd500=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/mac_econ_fwd500
mtx_mc2depi=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/mc2depi
mtx_pdb1HYS=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/pdb1HYS
mtx_pwtk=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/pwtk
mtx_rail4284=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/rail4284
mtx_rma10=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/rma10
mtx_scircuit=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/scircuit
mtx_shipsec1=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/shipsec1
mtx_webbase_1M=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland/webbase-1M

clsparse_timing_dir=timings/spm-dv/14.301
mkdir -p ${clsparse_timing_dir}
${clsparse_bench} -f spmdv -d ${mtx_cant} > ${clsparse_timing_dir}/cant.txt
${clsparse_bench} -f spmdv -d ${mtx_consph} > ${clsparse_timing_dir}/consph.txt
${clsparse_bench} -f spmdv -d ${mtx_cop20k_A} > ${clsparse_timing_dir}/cop20k_A.txt
${clsparse_bench} -f spmdv -d ${mtx_mac_econ_fwd500} > ${clsparse_timing_dir}/mac_econ_fwd500.txt
${clsparse_bench} -f spmdv -d ${mtx_mc2depi} > ${clsparse_timing_dir}/mc2depi.txt
${clsparse_bench} -f spmdv -d ${mtx_pdb1HYS} > ${clsparse_timing_dir}/pdb1HYS.txt
${clsparse_bench} -f spmdv -d ${mtx_pwtk} > ${clsparse_timing_dir}/pwtk.txt
${clsparse_bench} -f spmdv -d ${mtx_rail4284} > ${clsparse_timing_dir}/rail4284.txt
${clsparse_bench} -f spmdv -d ${mtx_rma10} > ${clsparse_timing_dir}/rma10.txt
${clsparse_bench} -f spmdv -d ${mtx_scircuit} > ${clsparse_timing_dir}/scircuit.txt
${clsparse_bench} -f spmdv -d ${mtx_shipsec1} > ${clsparse_timing_dir}/shipsec1.txt
${clsparse_bench} -f spmdv -d ${mtx_webbase_1M} > ${clsparse_timing_dir}/webbase_1M.txt
