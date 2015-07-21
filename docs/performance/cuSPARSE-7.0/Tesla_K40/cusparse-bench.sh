#!/bin/bash

#Executable
cusparse_bench=~/code/github/clMathLibraries/bin/clSPARSE/release/clSPARSE-build/staging/cusparse-bench

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

mkdir -p timings-cusparse-spmdv
${cusparse_bench} -f spmdv -d ${mtx_cant} > timings-cusparse-spmdv/cant.txt
${cusparse_bench} -f spmdv -d ${mtx_consph} > timings-cusparse-spmdv/consph.txt
${cusparse_bench} -f spmdv -d ${mtx_cop20k_A} > timings-cusparse-spmdv/cop20k_A.txt
${cusparse_bench} -f spmdv -d ${mtx_mac_econ_fwd500} > timings-cusparse-spmdv/mac_econ_fwd500.txt
${cusparse_bench} -f spmdv -d ${mtx_mc2depi} > timings-cusparse-spmdv/mc2depi.txt
${cusparse_bench} -f spmdv -d ${mtx_pdb1HYS} > timings-cusparse-spmdv/pdb1HYS.txt
${cusparse_bench} -f spmdv -d ${mtx_pwtk} > timings-cusparse-spmdv/pwtk.txt
${cusparse_bench} -f spmdv -d ${mtx_rail4284} > timings-cusparse-spmdv/rail4284.txt
${cusparse_bench} -f spmdv -d ${mtx_rma10} > timings-cusparse-spmdv/rma10.txt
${cusparse_bench} -f spmdv -d ${mtx_scircuit} > timings-cusparse-spmdv/scircuit.txt
${cusparse_bench} -f spmdv -d ${mtx_shipsec1} > timings-cusparse-spmdv/shipsec1.txt
${cusparse_bench} -f spmdv -d ${mtx_webbase_1M} > timings-cusparse-spmdv/webbase_1M.txt
