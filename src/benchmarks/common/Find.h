#ifndef _FIND_H_
#define _FIND_H_

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <string>
#include <vector>

namespace fs = boost::filesystem;

/**
 * @brief findMatrices
 * @param root path to the directory where to search for files with extension
 * @param extension matrix files extension without "." just mtx
 * @param matrix_files vector of found files with given extension
 * @return true if any files were found
 */
bool findMatrices(const std::string& root,
                  const std::string& extension,
                  std::vector<fs::path>& matrix_files)
{


    fs::path dir(root);
    fs::directory_iterator end_iter;
    const boost::regex filter( ".*\\.\\" + extension );
    std::cout << "Searching for files like: " << filter.str() << std::endl;
    bool found = false;

    if (fs::exists(dir) && fs::is_directory(dir))
    {
        for (fs::directory_iterator dir_iter(dir); dir_iter != end_iter; ++dir_iter)
        {
            if (fs::is_regular_file(dir_iter->status()) )
            {
                std::string fname = dir_iter->path().filename().native();

                if(boost::regex_match(fname, filter))
                {
                    std::cout << "Adding: " << dir_iter->path() << std::endl;
                    matrix_files.push_back(dir_iter->path());
                    found = true;
                }
            }
        }
    }
    else
    {
        std::cerr << dir << " does not name a directory or directory does not exists!" << std::endl;
        return false;
    }

    return found;
}

#endif // _FIND_H_
