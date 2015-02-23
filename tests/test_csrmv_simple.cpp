#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"
#include "resources/matrix_utils.h"

cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

namespace po = boost::program_options;

template<typename T>
clsparseStatus generateResult(cl_mem x, cl_mem alpha,
                              cl_mem y, cl_mem beta,
                              cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event* event)
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

    if(typeid(T) == typeid(float))
    {
        return clsparseScsrmv(CSRE::n_rows,
                       CSRE::n_cols,
                       CSRE::n_vals,
                       alpha,
                       CSRE::cl_row_offsets,
                       CSRE::cl_col_indices,
                       CSRE::cl_f_values,
                       x,
                       beta,
                       y,
                       CLSE::queue,
                       num_events_in_wait_list,
                       event_wait_list,
                       event);

    }
    if(typeid(T) == typeid(double))
    {
       return clsparseDcsrmv(CSRE::n_rows,
                       CSRE::n_cols,
                       CSRE::n_vals,
                       alpha,
                       CSRE::cl_row_offsets,
                       CSRE::cl_col_indices,
                       CSRE::cl_d_values,
                       x,
                       beta,
                       y,
                       CLSE::queue,
                       num_events_in_wait_list,
                       event_wait_list,
                       event);

    }
}

template <typename T>
class TestCSRMV : public ::testing::Test
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

public:

    void SetUp()
    {
        //TODO:: take the values from cmdline;
        alpha = T(CSRE::alpha);
        beta = T(CSRE::beta);

        x = std::vector<T>(CSRE::n_cols);
        y = std::vector<T>(CSRE::n_rows);

        std::fill(x.begin(), x.end(), T(1));
        std::fill(y.begin(), y.end(), T(2));

        cl_int status;
        gx = clCreateBuffer(CLSE::context,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            x.size() * sizeof(T), x.data(), &status);

        ASSERT_EQ(CL_SUCCESS, status); //is it wise to use this here?

        gy = clCreateBuffer(CLSE::context,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            y.size() * sizeof(T), y.data(), &status);

        ASSERT_EQ(CL_SUCCESS, status);

        galpha = clCreateBuffer(CLSE::context,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(T), &alpha, &status);

        ASSERT_EQ(CL_SUCCESS, status);

        void* hgalpha = clEnqueueMapBuffer(CLSE::queue, galpha, true, CL_MAP_READ,
                                                          0, sizeof(T), 0, NULL, NULL, NULL);
        if(typeid(T) == typeid(float))
            printf("hgalpha = %f\n", *(T*)hgalpha);
        if(typeid(T) == typeid(double))
            printf("hgalpha = %g\n", *(T*)hgalpha);
        clEnqueueUnmapMemObject(CLSE::queue, galpha, hgalpha, 0, NULL, NULL);


        gbeta = clCreateBuffer(CLSE::context,
                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof(T), &beta, &status);

        ASSERT_EQ(CL_SUCCESS, status);
        void* hgbeta = clEnqueueMapBuffer(CLSE::queue, gbeta, true, CL_MAP_READ,
                                                          0, sizeof(T), 0, NULL, NULL, NULL);
        if(typeid(T) == typeid(float))
            printf("hgbeta = %f\n", *(T*)hgbeta);
        if(typeid(T) == typeid(double))
            printf("hgbeta = %g\n", *(T*)hgbeta);
        clEnqueueUnmapMemObject(CLSE::queue, gbeta, hgbeta, 0, NULL, NULL);

        generateReference(x, alpha, y, beta);


    }


    void generateReference (const std::vector<float>& x,
                            const float alpha,
                            std::vector<float>& y,
                            const float beta)
    {
            csrmv(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
                CSRE::row_offsets, CSRE::col_indices, CSRE::f_values,
                  x, alpha, y, beta);
    }

    void generateReference (const std::vector<double>& x,
                            const double alpha,
                            std::vector<double>& y,
                            const double beta)
    {
            csrmv(CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
                CSRE::row_offsets, CSRE::col_indices, CSRE::d_values,
                  x, alpha, y, beta);
    }

    cl_mem gx;
    cl_mem gy;
    std::vector<T> x;
    std::vector<T> y;

    T alpha;
    T beta;

    cl_mem galpha;
    cl_mem gbeta;


};

typedef ::testing::Types<float, double> TYPES;
TYPED_TEST_CASE(TestCSRMV, TYPES);

TYPED_TEST(TestCSRMV, multiply)
{
    clsparseStatus status =
            generateResult<TypeParam>(this->gx,
                                      this->galpha,
                                      this->gy,
                                      this->gbeta,
                                      0, NULL, NULL);
    EXPECT_EQ(clsparseSuccess, status);

    std::vector<TypeParam> result(this->y.size());

    clEnqueueReadBuffer(ClSparseEnvironment::queue,
                        this->gy, 1, 0,
                        result.size()*sizeof(TypeParam),
                        result.data(), 0, NULL, NULL);

    if(typeid(TypeParam) == typeid(float))
        for(int i = 0; i < this->y.size(); i++)
            EXPECT_NEAR(this->y[i], result[i], 5e-5);

    if(typeid(TypeParam) == typeid(double))
        for(int i = 0; i < this->y.size(); i++)
            EXPECT_NEAR(this->y[i], result[i], 5e-14);


}


int main (int argc, char* argv[])
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;
    //pass path to matrix as an argument, We can switch to boost po later

    std::string path;
    double alpha;
    double beta;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("path,p", po::value(&path)->required(), "Path to matrix in mtx format.")
            ("alpha,a", po::value(&alpha)->default_value(1.0),
             "Alpha parameter for eq: \n\ty = alpha * M * x + beta * y")
            ("beta,b", po::value(&beta)->default_value(0.0),
             "Beta parameter for eq: \n\ty = alpha * M * x + beta * y");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (po::error& error)
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what() << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }



    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE());
    ::testing::AddGlobalTestEnvironment( new CSRE(path, alpha, beta,
                                                  CLSE::queue, CLSE::context));
    return RUN_ALL_TESTS();
}
