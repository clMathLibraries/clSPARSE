#ifndef _MATRIX_STATISTICS_H_
#define _MATRIX_STATISTICS_H_

#include <string>
#include <iostream>
#include <iomanip>


struct MatrixStatistics
{
    int n_rows;
    int n_cols;
    int nnz;
    int nnz_per_row;
    double avg_spmv_time;
    double instruction_bandwidth;
    double memory_bandwidth;
    std::string name;

};


void printMatrixStatistics(const std::vector<MatrixStatistics>& statistics)
{
    std::cout <<
        "<i> \t<matrix> \t<nrows> \t<ncols> \t<nnz> \t<npr> \t<avgTime[s]> \t<instr BW[GB/s]> \t<mem BW[GB/s]>"
              << std::endl;

    int index = 0;
    for(auto& e : statistics)
    {
        std::cout << ++index  << " \t"
                  << e.name   << " \t"
                  << e.n_rows << " \t"
                  << e.n_cols << " \t"
                  << e.nnz    << " \t"
                  << e.nnz_per_row << " \t"
                  << std::fixed << std::setprecision(8) << e.avg_spmv_time << " \t"
                  << std::fixed << std::setprecision(8) << e.instruction_bandwidth << " \t"
                  << std::fixed << std::setprecision(8) << e.memory_bandwidth << " \t"
                  << std::endl;
    }

}

#endif //_MATRIX_STATISTICS_H_
