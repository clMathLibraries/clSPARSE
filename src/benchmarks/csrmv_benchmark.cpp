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

	return 0;
}
