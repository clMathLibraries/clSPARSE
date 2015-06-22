## Project badges
**coming soon**

Pre-built binaries **not yet** available on our releases page **coming soon**

# clSPARSE
an OpenCL&copy; library implementing Sparse linear algebra.  This project started
as a collaboration between [AMD Inc.](http://www.amd.com/) and
[Vratis Ltd.](http://www.vratis.com/).  In opening the source to the public, we
invite all interested parties to [contribute](CONTRIBUTING.md) to the source.

## Introduction to clSPARSE
The library source compiles cross-platform on the back of an advanced cmake build
system allowing users to build the library, benchmarks, tests and takes care of
dependencies for them.  True in spirit with
the other clMath libraries, clSPARSE exports a “C” interface to allow
projects to build wrappers around clSPARSE in any language they need.   With
respect to the API, a great deal of thought and effort went into designing the
API’s to make them less ‘cluttered’ compared to the older clMath libraries.  
OpenCL state is not explicitly passed through the API, which enables the library
to be forward compatible when users are ready to switch from OpenCL 1.2 to OpenCL
2.0.  Lastly, the API’s are designed such that users are in control of where
input and output buffers live, and they maintain absolute control of when data
transfers to/from device memory need to happen, so that there are no performance
surprises.

At release, clSPARSE provides these fundamental sparse operations for OpenCL:
-  Sparse Matrix - dense Vector multiply (SpM-dV)
-  Sparse Matrix - dense Matrix multiply (SpM-dM)
-  Iterative conjugate gradient solver (CG)
-  Iterative biconjugate gradient stabilized solver (BiCGStab)
-  Dense to CSR conversions (& converse)
-  COO to CSR conversions (& converse)
-  Functions to read matrix market files in COO or CSR format

### clSPARSE build information
A [Build primer](https://github.com/kknox/clSPARSE/wiki/Build) is available

### clSPARSE library user documentation
**API documentation** not yet available

### Google Groups
Two mailing lists have been created for the clMath projects:

-   clmath@googlegroups.com - group whose focus is to answer
    questions on using the library or reporting issues

-   clmath-developers@googlegroups.com - group whose focus is for
    developers interested in contributing to the library code itself

### Contributing code
Please refer to and read the [Contributing](CONTRIBUTING.md) document for guidelines on
how to contribute code to this open source project. Code in the
/master branch is considered to be stable and new library releases are made
when commits are merged into /master.  Active development and pull-requests should
be made to the /develop branch.

## Samples
clSPARSE contains a directory of simple [OpenCL samples](./samples) that demonstrate the use
of the API in both C and C++.  The [superbuild](http://www.kitware.com/media/html/BuildingExternalProjectsWithCMake2.8.html)
script for clSPARSE also builds the samples as an external project, to demonstrate
how an application would find and link to clSPARSE with cmake.

## Build dependencies
clSPARSE is primarily written with C++ using C++11 core features.  It does export
a 'C' interface for compatibility with other languages.

### Compiling for Windows
-  Windows&reg; 7/8
-  Visual Studio 2013 and above
-  CMake 2.8.12 (download from [Kitware](http://www.cmake.org/download/))
  -  Solution (.sln) or
  -  Nmake makefiles
-   An OpenCL SDK, such as [APP SDK 3.0](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)

### Compiling for Linux
-  GCC 4.8 and above
-  CMake 2.8.12 (install with distro package manager )
   -  Unix makefiles or
   -  KDevelop or
   -  QT Creator
   -  An OpenCL SDK, such as [APP SDK 3.0](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)

### Compiling for Mac OSX
-   CMake 2.8.12 (install via [brew](http://brew.sh/))
   -  Unix makefiles or
   -  XCode
- An OpenCL SDK (installed via `xcode-select --install`)

### Test infrastructure dependencies
-   Googletest v1.7
-   Boost v1.58

### Benchmark infrastructure dependencies
-   Boost v1.58

  [API documentation]: http://clmathlibraries.github.io/clSPARSE/
  [binary_release]: https://github.com/clMathLibraries/clSPARSE/releases
