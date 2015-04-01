#include <iostream>
#include <boost/program_options.hpp>
#include "Params.h"
#include "Executor.h"

//namespace fs = boost::filesystem;


namespace po = boost::program_options;


int main(int argc, char* argv[])
{

    po::options_description desc("Allowed options");

    std::string root_dir;
    std::string platform;
    cl_uint device;
    cl_platform_type platform_type;

    double alpha = 1.0;
    double beta = 0.0;
    int number_of_tries;
    int number_of_warmups;
    bool prec;


    desc.add_options()
            ("help,h", "Produce this message")
            ("root,r", po::value(&root_dir), "Matrix directory")
            ("platform,l", po::value(&platform)->default_value("AMD"),
             "OpenCL platform: AMD or NVIDIA.")
            ("device,d", po::value(&device)->default_value(0),
             "Device id within platform.")
            ("benchmarks,t",
             po::value(&number_of_tries)->default_value(10),
             "Number of benchmark iterations")
            ("warmups,w",
             po::value(&number_of_warmups)->default_value(2),
             "Number of warmups iterations")
            ("dprecision,p", po::value(&prec)->default_value(true),
             "Double precision calculations")
            ("alpha,a", po::value(&alpha)->default_value(1.0),
             "Alpha value for oper y = alpha * A * x + beta * y")
            ("beta,b", po::value(&beta)->default_value(0.0),
             "Alpha value for oper y = alpha * A * x + beta * y");

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

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    if (vm.count("root") == 0)
    {
        std::cerr << "The ["<< "root" << "] parameter is missing!" << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    //check platform
    if(vm.count("platform"))
    {
        if ("AMD" == platform)
        {
            platform_type = AMD;
        }
        else if ("NVIDIA" == platform)
        {
            platform_type = NVIDIA;
        }
        else
        {

            std::cout << "The platform option is missing or is ill defined!\n";
            std::cout << "Given [" << platform << "]" << std::endl;
            platform = "AMD";
            platform_type = AMD;
            std::cout << "Setting [" << platform << "] as default" << std::endl;
        }
    }

    if(prec)
    {
        std::cout << "Executing benchmark in double precision" << std::endl;
        Params<double> params;
        params.alpha = alpha;
        params.beta = beta;
        params.number_of_tries = number_of_tries;
        params.number_of_warmups = number_of_warmups;
        params.root_dir = root_dir;
        params.dID = device;
        params.pID = platform_type;

        Executor<double> executor(params);
        executor.exec();
    }
    else
    {
        std::cout << "Executing benchmark in single precision" << std::endl;
        Params<float> params;
        params.alpha = alpha;
        params.beta = beta;
        params.number_of_tries = number_of_tries;
        params.number_of_warmups = number_of_warmups;
        params.root_dir = root_dir;
        params.dID = device;
        params.pID = platform_type;

        Executor<float> executor(params);
        executor.exec();
    }


//    CPerfCounter timer;
//    timer.Start();

//    cl_int status = CL_SUCCESS;

//    cl_platform_id* platforms = NULL;
//    cl_uint num_platforms = 0;

//    status = getPlatforms(&platforms, &num_platforms);
//    if (status != CL_SUCCESS)
//    {
//        std::cerr << "Problem with setting up the OpneCL platforms" << std::endl;
//        return status;
//    }

//    printPlatforms(platforms, num_platforms);

//    cl_device_id device = NULL;
//    status = getDevice(platforms[0], &device, CL_DEVICE_TYPE_GPU);
//    if (status != CL_SUCCESS)
//    {
//        std::cerr << "Problem with initializing GPU device" << std::endl;
//        return status;
//    }

//    printDeviceInfo(device);

//    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
//    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

//    clsparseSetup();
//    free(platforms);


//    std::vector<fs::path> matrix_files;

//    if (!findMatrices("/media/jpola/Storage/matrices/small_set", "mtx", matrix_files))
//        return -100;

//    auto file = matrix_files.begin();

//    std::string path = (*file).native();

//    //host matrix definition
//    std::vector<int> row_offsets;
//    std::vector<int> col_indices;
//    std::vector<float> values;
//    int n_rows = 0;
//    int n_cols = 0;
//    int n_vals = 0;

//    if (!readMatrixMarketCSR(row_offsets, col_indices, values,
//                        n_rows, n_cols, n_vals, path))
//    {
//        return -101;
//    }

//    //gpu matrix definition
//    cl_mem cl_row_offsets;
//    cl_mem cl_col_indices;
//    cl_mem cl_values;

//    if (!allocateMatrix(row_offsets, col_indices, values,
//                   n_rows, n_cols,
//                   &cl_row_offsets, &cl_col_indices, &cl_values,
//                   queue, context, &status))
//    {
//        std::cerr << "Problem with allocating matrix "
//                  << (*file).filename().native()
//                  <<  " OpenCL status " << status << std::endl;
//        return status;
//    }


//    //TODO:: remember about offsets;
//    cl_mem alpha;
//    cl_mem beta;

//    cl_mem x;
//    cl_mem y;

//    if (!allocateVector<float>(&alpha, 1, 1, queue, context, &status))
//    {
//        std::cerr << "Problem with allocation alpha " << status << std::endl;
//        return status;
//    }

//    if (!allocateVector<float>(&beta, 1 , 1, queue, context, &status))
//    {
//        std::cerr << "Problem with allocation beta " << status << std::endl;
//        return status;
//    }

//    if (!allocateVector<float>(&x, n_cols , 1, queue, context, &status))
//    {
//        std::cerr << "Problem with allocation beta " << status << std::endl;
//        return status;
//    }

//    if (!allocateVector<float>(&y, n_rows , 0, queue, context, &status))
//    {
//        std::cerr << "Problem with allocation beta " << status << std::endl;
//        return status;
//    }

//    executeCSRMultiply<float>(n_rows, n_cols, n_vals,
//                       alpha, 0,
//                       cl_row_offsets,
//                       cl_col_indices,
//                       cl_values,
//                       x,
//                       0,
//                       beta,
//                       0,
//                       y,
//                       0,
//                       queue,
//                       1
//                       );



//    clsparseTeardown();
//    timer.Stop();
//    std::cout << "Execution time "
//              << timer.GetElapsedTime() << " s." << std::endl;



	return 0;
}
