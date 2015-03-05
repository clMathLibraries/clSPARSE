#ifndef _MATRIX_UTILS_H_
#define _MATRIX_UTILS_H_


#include <vector>
#include <tuple>
#include <algorithm>
#include <cassert>

// convert row indices vector to csr row_offsets vector
// lenght of indices == matrix nnz
// lenght of offsets == n_rows+1;
template<typename INDEX_TYPE>
void indicesToOffsets(const std::vector<INDEX_TYPE>& row_indices,
                      const INDEX_TYPE n_rows,
                      std::vector<INDEX_TYPE>& row_offsets)
{
    INDEX_TYPE index = -1;
    INDEX_TYPE nnz = row_indices.size();

    if (row_offsets.size() != (n_rows + 1))
        row_offsets.resize(n_rows + 1);

    for (int i = 0; i < nnz; i++)
    {
        if (row_indices[i] != index)
        {
            index++;
            row_offsets[index] = i;
        }
    }

    row_offsets[n_rows] = nnz;
}

//convert csr row offsets to row indices
// lenght of indices == matrix nnz!
// lenght of offsets == n_rows+1;
template<typename INDEX_TYPE>
void offsetsToIndices(const std::vector<INDEX_TYPE>& row_offsets,
                      const size_t n_rows,
                      std::vector<INDEX_TYPE>& row_indices)
{
    INDEX_TYPE nnz = row_offsets[n_rows];

    if(row_indices.size() != nnz)
        row_indices.resize(nnz);

    for( size_t i = 0; i < n_rows; i++ )
        for (int j = row_offsets[i]; j < row_offsets[i+1]; j++)
            row_indices[j] = i;

}


// sort coo matrix by (row, col) tuple
// Think about something faster, avoid rewritings?!
template <typename VALUE_TYPE, typename INDEX_TYPE>
void sortByRowCol(std::vector<INDEX_TYPE>& rows,
                  std::vector<INDEX_TYPE>& cols,
                  std::vector<VALUE_TYPE>&   vals)
{
    typedef std::tuple<INDEX_TYPE, INDEX_TYPE, VALUE_TYPE> Element;

    size_t size = vals.size( );

    std::vector<Element> matrix;//(size);

    for (int i = 0; i < size; i++)
    {
        matrix.push_back(std::make_tuple(rows[i], cols[i], vals[i]));
    }

    std::sort(matrix.begin(), matrix.end(),
              [](const Element& left, const Element& right)
                {
                    if (std::get<0>(left) == std::get<0>(right))
                    {
                        return (std::get<1>(left) < std::get<1>(right));
                    }
                    else
                    {
                        return (std::get<0>(left) < std::get<0>(right));
                    }
                }

              );

    for(int i = 0; i < size; i++)
        std::tie(rows[i], cols[i], vals[i]) = matrix[i];
}


//simple spmv for csr matrix to obtain reference results;
template<typename VALUE_TYPE, typename INDEX_TYPE>
void csrmv(int n_rows, int n_cols, int nnz,
      const std::vector<INDEX_TYPE>& row_offsets,
      const std::vector<INDEX_TYPE>& col_indices,
      const std::vector<VALUE_TYPE>& values,
      const std::vector<VALUE_TYPE>& x,
      const VALUE_TYPE alpha,
      std::vector<VALUE_TYPE>& y,
      const VALUE_TYPE beta)
{
    assert(x.size() == n_cols);
    assert(y.size() == n_rows);

    assert(row_offsets.size() == n_rows + 1);
    assert(row_offsets[n_rows] == nnz);
    assert(col_indices.size() == nnz);
    assert(values.size() == nnz);


    for (int i = 0; i < n_rows; i++)
    {
        VALUE_TYPE sum = (VALUE_TYPE)0;
        for(int j = row_offsets[i]; j < row_offsets[i+1]; j++)
        {
            sum += alpha * values[j] * x[col_indices[j]];
        }
        y[i] = sum + beta * y[i];

    }
}

//simple spmv for csr matrix to obtain reference results;
template<typename VALUE_TYPE, typename INDEX_TYPE>
void coomv(int n_rows, int n_cols, int nnz,
      const std::vector<INDEX_TYPE>& row_indices,
      const std::vector<INDEX_TYPE>& col_indices,
      const std::vector<VALUE_TYPE>& values,
      const std::vector<VALUE_TYPE>& x,
      const VALUE_TYPE alpha,
      std::vector<VALUE_TYPE>& y,
      const VALUE_TYPE beta)
{
    assert(x.size() == n_cols);
    assert(y.size() == n_rows);

    assert(row_indices.size() == nnz);
    assert(col_indices.size() == nnz);
    assert(values.size() == nnz);

    for (int i = 0; i < nnz; i++)
    {
        y[row_indices[i]] += (alpha * values[i] * x[col_indices[i]])
                                + beta * y[row_indices[i]];
    }
}


#endif

