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
bench_root_dir=~/code/github/kknox/bin/clsparse/release/clSPARSE-build/staging
cusparse_bench=${bench_root_dir}/cusparse-bench
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

cusparse_timing_dir=timings/${function}/346.47
mkdir -p ${cusparse_timing_dir}

${cusparse_bench} -f ${function} -d ${mtx_cant} > ${cusparse_timing_dir}/cant.txt
${cusparse_bench} -f ${function} -d ${mtx_consph} > ${cusparse_timing_dir}/consph.txt
${cusparse_bench} -f ${function} -d ${mtx_cop20k_A} > ${cusparse_timing_dir}/cop20k_A.txt
${cusparse_bench} -f ${function} -d ${mtx_mac_econ_fwd500} > ${cusparse_timing_dir}/mac_econ_fwd500.txt
${cusparse_bench} -f ${function} -d ${mtx_mc2depi} > ${cusparse_timing_dir}/mc2depi.txt
${cusparse_bench} -f ${function} -d ${mtx_pdb1HYS} > ${cusparse_timing_dir}/pdb1HYS.txt
${cusparse_bench} -f ${function} -d ${mtx_pwtk} > ${cusparse_timing_dir}/pwtk.txt
${cusparse_bench} -f ${function} -d ${mtx_rail4284} > ${cusparse_timing_dir}/rail4284.txt
${cusparse_bench} -f ${function} -d ${mtx_rma10} > ${cusparse_timing_dir}/rma10.txt
${cusparse_bench} -f ${function} -d ${mtx_scircuit} > ${cusparse_timing_dir}/scircuit.txt
${cusparse_bench} -f ${function} -d ${mtx_shipsec1} > ${cusparse_timing_dir}/shipsec1.txt
${cusparse_bench} -f ${function} -d ${mtx_webbase_1M} > ${cusparse_timing_dir}/webbase_1M.txt
