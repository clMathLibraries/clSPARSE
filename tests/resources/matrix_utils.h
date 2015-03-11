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

/*
 * Generate A^T where A is n_rows x n_cols CSR matrix
 *
 */

template<typename VALUE_TYPE, typename INDEX_TYPE>
void csr_transpose(int n_rows, int n_cols, int nnz,
                   const std::vector<INDEX_TYPE>& row_offsets,
                   const std::vector<INDEX_TYPE>& col_indices,
                   const std::vector<VALUE_TYPE>& values,
                   std::vector<INDEX_TYPE>& row_offsets_t,
                   std::vector<INDEX_TYPE>& col_indices_t,
                   std::vector<VALUE_TYPE>& values_t)
{

    row_offsets_t.resize(n_cols+1);
    col_indices_t.resize(nnz);
    values_t.resize(nnz);

    std::vector<INDEX_TYPE> col_nnz(n_cols);
    // need to be zeroed because we will be counting
    std::fill(col_nnz.begin(), col_nnz.end(), 0);

    //the col_indices have repeating data depends on the nnz per row;
    //data are sorted
    /* example of col_indices vector [index] = col_indices[index]
        0 = 0
        1 = 1
        2 = 2
        3 = 3
        4 = 4
        5 = 11
        6 = 12
        7 = 40
        8 = 41
        9 = 49
        10 = 50
        ...
        36 = 1
        37 = 2
        38 = 11
        39 = 12
        40 = 40
        41 = 41
        42 = 49
        43 = 50
        ...
        54 = 0
        55 = 1
        56 = 2
        57 = 3
        58 = 4
        59 = 10
        60 = 11
        61 = 12
        62 = 13
        63 = 14
        64 = 15
        65 = 16
        66 = 17
        67 = 18
        68 = 40
        69 = 41
        70 = 49
        71 = 50
        72 = 51
        73 = 52
        74 = 53
        75 = 54
    */
    //we have to count how many zeros, ones, twos, fours are in this vector;

    //This looks like gather / reduce operation. maybe with
    // help of row_offsets it can be done in parallel mode nicely!
    //or reduce. but requires atomic due to indirect mem access.
    for (int i = 0; i < nnz; i++)
        col_nnz[col_indices[i]] += 1;

    //calculate col offsets; its easy since we know how many nnz in each col
    //we have from previous loop
    row_offsets_t[0] = 0;
    for (int i = 1; i <= n_cols; i++)
    {
        row_offsets_t[i] = row_offsets_t[i-1] + col_nnz[i - 1];
        col_nnz[i - 1] = 0;
    }

    //calculate row_indices;
    //this might look similar to the csr multiply algorithm
    //or offsets to indices on gpu
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = row_offsets[i]; j < row_offsets[i+1]; j++)
        {
            VALUE_TYPE v = values[j];
            int k = col_indices[j];
            int l = row_offsets_t[k] + col_nnz[k];

            col_indices_t[l] = i;
            values_t[l] = v;

            col_nnz[k] += 1;
        }
    }
}

#endif

