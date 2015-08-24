/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
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

#pragma once
#ifndef _BLAS1_ENVIRONMENT_H_
#define _BLAS1_ENVIRONMENT_H_

#include <gtest/gtest.h>

class Blas1Environment : public ::testing::Environment
{
public:

    Blas1Environment(double alpha, double beta)
    {
        this->alpha = alpha;
        this->beta = beta;
    }

    void SetUp( )  {  }

    void TearDown( )  {  }

    static double alpha;
    static double beta;
};


#endif //_BLAS1_ENVIRONMENT_H_

