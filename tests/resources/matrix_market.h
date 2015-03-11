#ifndef _MATRIX_MARKET_H
#define _MATRIX_MARKET_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include "matrix_utils.h"
/**
  * Functions to read matrix in matrix market (*.mtx) format
  */
inline
void tokenize(std::vector<std::string>& tokens,
              const std::string& str,
              const std::string& delimiters = "\n\r\t ")
{
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

struct matrix_market_banner
{
    std::string storage;    // "array" or "coordinate"
    std::string symmetry;   // "general", "symmetric", "hermitian", or "skew-symmetric"
    std::string type;       // "complex", "real", "integer", or "pattern"
};

template <typename Stream>
bool read_matrix_market_banner(matrix_market_banner& banner, Stream& input)
{
    std::string line;
    std::vector<std::string> tokens;

    // read first line
    std::getline(input, line);
    tokenize(tokens, line);

    if (tokens.size() != 5 || tokens[0] != "%%MatrixMarket" || tokens[1] != "matrix")
    {
        std::cerr << "invalid MatrixMarket banner" << std::endl;
        return false;
    }

    banner.storage  = tokens[2];
    banner.type     = tokens[3];
    banner.symmetry = tokens[4];

    if (banner.storage != "array" && banner.storage != "coordinate")
    {
        std::cerr << "invalid MatrixMarket storage format [" + banner.storage + "]";
        return false;
    }

    if (banner.type != "complex" && banner.type != "real"
            && banner.type != "integer" && banner.type != "pattern")
    {
        std::cerr << "invalid MatrixMarket data type [" + banner.type + "]";
        return false;
    }

    if (banner.symmetry != "general" && banner.symmetry != "symmetric"
            && banner.symmetry != "hermitian" && banner.symmetry != "skew-symmetric")
    {
        std::cerr << "invalid MatrixMarket symmetry [" + banner.symmetry + "]";
        return false;
    }
    return true;
}



/**
 * Read matrix in coo format
 */
template <typename T>
bool readMatrixMarket(std::vector<int>& rows,
                      std::vector<int>& cols,
                      std::vector<T>&    vals,
                      int& n_rows,
                      int& n_cols,
                      int& n_vals,
                      const std::string& filename)
{

    typedef int IndexType;
    typedef T    ValueType;

    std::ifstream file(filename.c_str());
    if (!file)
    {
        std::cerr << "Problem with opening file ["<< filename <<"]"<< std::endl;
        return false;
    }

    matrix_market_banner banner;
    if(!read_matrix_market_banner(banner, file))
    {
        return false;
    }

    if (banner.storage == "coordinate")
    {
        //read coordinate
        std::string line;

        //skip over banner and comments;
        do
        {
            std::getline(file, line);
        } while (line[0] == '%');

        //this line should contain size of the matrix;
        std::vector<std::string> tokens;
        tokenize(tokens, line);

        if (tokens.size() != 3)
        {
            std::cerr << "Invalid MatrixMarket coordinate format" <<std::endl;
            return false;
        }

        // store the matrix size params
        std::istringstream(tokens[0]) >> n_rows;
        std::istringstream(tokens[1]) >> n_cols;
        std::istringstream(tokens[2]) >> n_vals;

        // allocate temoral data containers
        std::vector<IndexType> row_indices(n_vals);
        std::vector<IndexType> col_indices(n_vals);
        std::vector<ValueType> values(n_vals);


        IndexType num_entries_read = 0;

        if(banner.type == "pattern")
        {
            while (num_entries_read < n_vals && !file.eof())
            {
                file >> row_indices[num_entries_read];
                file >> col_indices[num_entries_read];
                num_entries_read++;
            }

            std::fill(values.begin(), values.end(), ValueType(1));

        }

        else if (banner.type == "real" || banner.type == "integer")
        {
            while (num_entries_read < n_vals && !file.eof())
            {
                double real;

                file >> row_indices[num_entries_read];
                file >> col_indices[num_entries_read];
                file >> real;
                values[num_entries_read] = (ValueType)real; //easy conversion

                num_entries_read++;
            }
        }
        else if (banner.type == "complex")
        {
            std::cerr
                    << "Complex Matrix Market format is not supported"
                    << std::endl;
            return false;
        }
        else
        {
            std::cerr << "Invalid Matrix Market data type" << std::endl;
            return false;
        }

        if (num_entries_read != n_vals)
        {
            std::cerr << "Unexpected EOF while reading MatrixMarket entries "
                << "from file [" << filename << "]" << std::endl;
             return false;
        }

        //check validity
        if (n_vals > 0)
        {
            //JPA: redundand?! IndexType is unsigned int
            int min_row_index = *std::min_element(&row_indices[0],
                    &row_indices[n_vals-1]);
            int min_col_index = *std::min_element(&col_indices[0],
                    &col_indices[n_vals-1]);
            int max_row_index = *std::max_element(&row_indices[0],
                    &row_indices[n_vals-1]);
            int max_col_index = *std::max_element(&col_indices[0],
                    &col_indices[n_vals-1]);

            if (min_row_index < 1)
            {
                std::cerr << "found invalid row index < 1 " << std::endl;
                return false;
            }
            if (min_col_index < 1)
            {
                std::cerr << "found invalid col index < 1"<< std::endl;
                return false;
            }
            if (max_row_index > n_rows)
            {
                std::cerr << "found invalid row_index > n_rows" << std::endl;
                 return false;
            }
            if (max_col_index > n_cols)
            {
                std::cerr << "found invalid col_index > num_rows" << std::endl;
                return false;
            }

            //convert to 0 base indices;
            for (IndexType i = 0; i < n_vals; i++)
            {
                row_indices[i] -= 1;
                col_indices[i] -= 1;
            }

            //expand symmetric formats to general format
            if (banner.symmetry != "general")
            {
                IndexType off_diagonals = 0;

                for (IndexType i = 0; i < n_vals; i++)
                    if (row_indices[i] != col_indices[i])
                        off_diagonals++;

                //if matrix is not symmetrix the off_diagonals will be 0;
                IndexType general_num_entries = n_vals + off_diagonals;


                std::vector<IndexType> general_rows(general_num_entries);
                std::vector<IndexType> general_cols(general_num_entries);
                std::vector<ValueType> general_vals(general_num_entries);

                if (banner.symmetry == "symmetric")
                {
                    IndexType nnz = 0;

                    for (IndexType i = 0; i < n_vals; i++)
                    {
                        general_rows[nnz] = row_indices[i];
                        general_cols[nnz] = col_indices[i];
                        general_vals[nnz] = values[i];
                        nnz++;

                        if (row_indices[i] != col_indices[i])
                        {
                            general_rows[nnz] = col_indices[i];
                            general_cols[nnz] = row_indices[i];
                            general_vals[nnz] = values[i];
                            nnz++;
                        }
                    }
                }
                else if (banner.symmetry == "hermitian")
                {
                    std::cerr << "Hermitian matrices are not supported" << std::endl;
                    return false;
                }
                else if (banner.symmetry == "skew-symmetric")
                {
                    std::cerr << "Skew-symmetric matrices are not supported" << std::endl;
                    return false;
                }

                // resize output data;
                n_vals = general_num_entries;
                //rows.resize(general_num_entries);
                //cols.resize(general_num_entries);
                //vals.resize(general_num_entries);

                //copy the data;
                rows = std::move(general_rows);
                cols = std::move(general_cols);
                vals = std::move(general_vals);
            }
            else
            {
                rows = std::move(row_indices);
                cols = std::move(col_indices);
                vals = std::move(values);

            }

        }

    }
    else
    {
        std::cerr <<
                     "Matrix storage type [array] is not supported." << std::endl;
        return false;
    }

    sortByRowCol(rows, cols, vals);
    std::cout << "[" << filename << "]"
              << ": " << n_rows  << ", " << n_cols << ", " << n_vals << std::endl;
    return true;
}

template <typename T>
bool readMatrixMarketCOO(std::vector<int>& rows,
                         std::vector<int>& cols,
                         std::vector<T>&    vals,
                         int& n_rows,
                         int& n_cols,
                         int& n_vals,
                         const std::string& filename)
{
    return readMatrixMarket(rows, cols, vals, n_rows, n_cols, n_vals, filename);
}

template <typename T>
bool readMatrixMarketCSR(std::vector<int>& row_offsets,
                         std::vector<int>& col_indices,
                         std::vector<T>&    values,
                         int& n_rows,
                         int& n_cols,
                         int& n_vals,
                         const std::string& filename)
{
    std::vector<int> rows;
    bool status =
            readMatrixMarket(rows, col_indices, values, n_rows, n_cols, n_vals,
                             filename);
    if (!status)
    {
        return status;
    }
    indicesToOffsets(rows, n_rows, row_offsets);
    return status;
}

/**
  * Write matrix market
  */
template <typename T>
bool writeMatrixMarket(std::vector<int>& rows,
                       std::vector<int>& cols,
                       std::vector<T>& vals,
                       int n_rows,
                       int n_cols,
                       int n_vals,
                       const std::string& filename)
{

    std::ofstream file(filename.c_str());

    if(!file)
    {
        return false;
    }

    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "\t" << n_rows << "\t" << n_cols << "\t" << n_vals << "\n";

    for(size_t i = 0; i < n_vals; i++)
    {
        file << rows[i] + 1 << " ";
        file << cols[i] + 1 << " ";
        file <<  vals[i];
        file << "\n";
    }

    file.close();
    return true;
}

template <typename T>
bool writeMatrixMarketCOO(std::vector<int>& row_indices,
                          std::vector<int>& col_indices,
                          std::vector<T>& values,
                          int n_rows,
                          int n_cols,
                          int n_vals,
                          const std::string& filename)
{
    return writeMatrixMarket(row_indices, col_indices, values,
                             n_rows, n_cols, n_vals, filename);

}


template <typename T>
bool writeMatrixMarketCSR(std::vector<int>& row_offsets,
                          std::vector<int>& col_indices,
                          std::vector<T>&    values,
                          int n_rows,
                          int n_cols,
                          int n_vals,
                          const std::string& filename)
{
    std::vector<int> row_indices;
    offsetsToIndices(row_offsets, n_rows, row_indices);

    return writeMatrixMarket(row_indices, col_indices, values,
                             n_rows, n_cols, n_vals, filename);



}

#endif //_MATRIX_MARKET_H_

