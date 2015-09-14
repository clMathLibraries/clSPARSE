/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ************************************************************************ */

/*! \file
* \brief sparse_matrix_fill.hpp provides functionality to fill CSR matrix values with 
*  random powers of 2 or integer values.
* 
*/

#pragma once
#ifndef _SPARSE_MATRIX_FILL_HXX
#define _SPARSE_MATRIX_FILL_HXX

#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>


template <typename T>
class clsparse_matrix_fill {
    typedef boost::minstd_rand base_generator_type;
public:
    clsparse_matrix_fill(int seed, int lR, int hR)
    {
        generator = new base_generator_type(seed);
        minR = lR;
        maxR = hR;
    }// 
    ~clsparse_matrix_fill()
    {
        delete generator;
    }

    void fillMtxTwoPowers(T* A, size_t sz)
    {
        for (size_t i = 0; i < sz; i++)
        {
            int tmp = roll_die();
            A[i] = static_cast<T>(1 << tmp);
        }
    }

    void fillMtxIntegers(T* A, size_t sz)
    {
        for (size_t i = 0; i < sz; i++)
        {
            A[i] = static_cast<T>(roll_die());
        }
    }

    void fillMtxOnes(T* A, size_t sz)
    {
        for (size_t i = 0; i < sz; i++)
        {
            A[i] = static_cast<T>(1);
        }
    }

private:
    base_generator_type *generator;
    int minR;  // Range [minR, maxR]
    int maxR;
    int roll_die()
    {
        boost::random::uniform_int_distribution<> dist(minR, maxR);

        //generator.seed(static_cast<unsigned int>(std::time(0)));
        return dist(*generator);
    }
};

#endif // _SPARSE_MATRIX_FILL_HXX