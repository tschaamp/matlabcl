#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"
#include <cstddef>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include "math.h"
#include "mex.h"
#include <sstream>

using namespace cl;
using namespace std;

//Create vars
Program         program;
vector<Device>  devices;
Context         context;
Context         *contextptr;
CommandQueue    *queue;
Kernel          kernel;

vector<Buffer*> inputBuffers;
cl_double       *result;

void initCL() {
    // Get queue and devices
    try { 
        // Get available platforms
        vector<Platform> platforms;
        Platform::get(&platforms);
 
        // Select the platform and create a context
        cl_context_properties cps[3] = { 
            CL_CONTEXT_PLATFORM, 
            (cl_context_properties)(platforms[0])(), 
            0 
        };
        contextptr = new Context( CL_DEVICE_TYPE_GPU, cps);
        context = *contextptr;
        // Get a list of devices on this platform
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
 
        // Create a command queue and use the first device
        queue = new CommandQueue(context, devices[0]);
    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
}

void createBuffers(int nrhs, const mxArray *prhs[]) {
    Buffer *inputBuffer;
    try {
        for(int i=1; i<nrhs; i++) {
            const mxArray *cur = prhs[i];
            int m = mxGetM(cur);
            int n = mxGetN(cur);
            std::size_t cursize = n*m*mxGetElementSize(cur);
            inputBuffer = new Buffer(context, CL_MEM_READ_ONLY, cursize);
            inputBuffers.push_back(inputBuffer);
            (*queue).enqueueWriteBuffer(*inputBuffer, CL_TRUE, 0, cursize, mxGetData(cur));
            kernel.setArg(i, *inputBuffer);
        } 
    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
}

void dynamicReplace(std::string *sourceCode, int nrhs) {
    std::stringstream vars;
    std::stringstream func;
    func << 0;
    for (uint i=1; i<nrhs; i++) {
        vars << ", __global __read_only double *m" << i;
        func << "+m" << i << "[pos]";
    }
    std::string token1 = "?1";
    std::string token2 = "?2";

    // find token1
    int begin1 = (*sourceCode).find(token1);
    while (begin1 != -1 && begin1 < (*sourceCode).length()) {
        (*sourceCode).replace(begin1, token1.length(), vars.str());
        begin1 = (*sourceCode).find(token1);
    }

    // find token2
    int begin2 = (*sourceCode).find(token2);
    while (begin2 != -1 && begin2 < (*sourceCode).length()) {
        (*sourceCode).replace(begin2, token2.length(), func.str());
        begin2 = (*sourceCode).find(token2);
    }
}

void buildDynamicKernel(std::string kernelname, int nrhs) {
    // Build Kernel
    try { 
        // Read source file
        std::ifstream sourceFile("kernel.cl");
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));

        // replace dynamic tokens
        dynamicReplace(&sourceCode, nrhs);

        Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        // Check if kernelname in sourcefile
        if (sourceCode.find("__kernel void " + kernelname, 0) == 
            std::string::npos) {
            throw "Kernel not found.";
        }

        // Make program of the source code in the context
        program = Program(context, source);
 
        // Build program for these specific devices
        program.build(devices);

        // Make kernel
        kernel = Kernel(program, kernelname.c_str());
    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
       std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
       std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    }
}

double* runKernel(unsigned int width, unsigned int height) {
    // Create the host image
    std::size_t matrixsize = width * height * sizeof(cl_double);
    cl_double *matrix      = (cl_double*) malloc(matrixsize);

    // Run Kernel
    try { 
        // Create memory buffers
        Buffer outputBuffer = Buffer(context, CL_MEM_WRITE_ONLY, matrixsize);
        (*queue).enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, matrixsize, matrix); 
        // Set arguments to kernel
        kernel.setArg(0, outputBuffer);
        // Run the kernel on specific range
        NDRange global(width, height);
        NDRange local(1, 1);
        (*queue).enqueueNDRangeKernel(kernel, NullRange, global, local);
 
        // Read buffer into a local list
        (*queue).enqueueReadBuffer(outputBuffer, CL_TRUE, 0, matrixsize, matrix);

    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
       std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
       std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    }

    return matrix; 
}

// clean up
void cleanup() {
    free(queue);
    free(contextptr);
    // free Buffers
    while (!inputBuffers.empty()) {
        free(inputBuffers.back());
        inputBuffers.pop_back();
    }
}

// Standard method needed by Matlab
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // Free memory done in exitfunction
    mexAtExit(cleanup);

    // Skip if too less Arguments
    if (nrhs < 3) {
        std::cout << "Not enough arguments." << std::endl;
        return;
    }
    
    // Determine rows and columns
    int rows = 1;
    int columns = 1;
    for(int i=1; i<nrhs; i++) {
        int m = mxGetM(prhs[i]);
        int n = mxGetN(prhs[i]);
        if (m>rows) rows = m;
        if (n>columns) columns = n;
    }

    // Init CL environment if not already done
    if(!contextptr) {
        initCL();
    }

    // Call and create named Kernel
    int nChars = mxGetN(prhs[0])+1;  //Add one extra for the \0
    vector<char> buffer;
    mxGetString(prhs[0], &buffer[0], nChars);
    buffer[nChars-1] = 0;
    try {
        buildDynamicKernel(&buffer[0], nrhs);
    } catch(...) {
        std::cout << "Kernel not found." << std::endl;
        return;
    }

    // Create Buffers
    createBuffers(nrhs, prhs);

    // Run the Kernel
    result = runKernel(rows, columns);    


    // always return mxArray
    // mxReal is our data-type
    plhs[0] = mxCreateDoubleMatrix(rows, columns, mxREAL); 

    // Set data to result
    mxSetData(plhs[0], result);

    return;
}
