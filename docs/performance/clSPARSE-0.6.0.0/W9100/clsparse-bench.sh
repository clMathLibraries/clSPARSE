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

#!/bin/bash

#Executable
bench_root_dir=~/code/github/kknox/bin/clSPARSE/release/clSPARSE-build/staging
clsparse_bench=${bench_root_dir}/clsparse-bench
function=spmdv

#Data directories
data_root_dir=~/code/github/clMathLibraries/bin/deps/release/Externals/MTX/Bell_Garland
mtx_cant=${data_root_dir}/cant
mtx_consph=${data_root_dir}/consph
mtx_cop20k_A=${data_root_dir}/cop20k_A
mtx_mac_econ_fwd500=${data_root_dir}/mac_econ_fwd500
mtx_mc2depi=${data_root_dir}/mc2depi
mtx_pdb1HYS=${data_root_dir}/pdb1HYS
mtx_pwtk=${data_root_dir}/pwtk
mtx_rail4284=${data_root_dir}/rail4284
mtx_rma10=${data_root_dir}/rma10
mtx_scircuit=${data_root_dir}/scircuit
mtx_shipsec1=${data_root_dir}/shipsec1
mtx_webbase_1M=${data_root_dir}/webbase-1M

clsparse_timing_dir=timings/${function}/14.301
mkdir -p ${clsparse_timing_dir}

${clsparse_bench} -f ${function} -d ${mtx_cant} > ${clsparse_timing_dir}/cant.txt
${clsparse_bench} -f ${function} -d ${mtx_consph} > ${clsparse_timing_dir}/consph.txt
${clsparse_bench} -f ${function} -d ${mtx_cop20k_A} > ${clsparse_timing_dir}/cop20k_A.txt
${clsparse_bench} -f ${function} -d ${mtx_mac_econ_fwd500} > ${clsparse_timing_dir}/mac_econ_fwd500.txt
${clsparse_bench} -f ${function} -d ${mtx_mc2depi} > ${clsparse_timing_dir}/mc2depi.txt
${clsparse_bench} -f ${function} -d ${mtx_pdb1HYS} > ${clsparse_timing_dir}/pdb1HYS.txt
${clsparse_bench} -f ${function} -d ${mtx_pwtk} > ${clsparse_timing_dir}/pwtk.txt
${clsparse_bench} -f ${function} -d ${mtx_rail4284} > ${clsparse_timing_dir}/rail4284.txt
${clsparse_bench} -f ${function} -d ${mtx_rma10} > ${clsparse_timing_dir}/rma10.txt
${clsparse_bench} -f ${function} -d ${mtx_scircuit} > ${clsparse_timing_dir}/scircuit.txt
${clsparse_bench} -f ${function} -d ${mtx_shipsec1} > ${clsparse_timing_dir}/shipsec1.txt
${clsparse_bench} -f ${function} -d ${mtx_webbase_1M} > ${clsparse_timing_dir}/webbase_1M.txt
