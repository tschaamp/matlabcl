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

using namespace cl;

//Create vars
Program program;
vector<Device> devices;
Context context;
CommandQueue queue;
Kernel kernel;

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
        Context contextl( CL_DEVICE_TYPE_GPU, cps);
        context = contextl;
        // Get a list of devices on this platform
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
 
        // Create a command queue and use the first device
        queue = CommandQueue(context, devices[0]);
    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
}

void createBuffers(int nrhs, const mxArray *prhs[]) {
    //Buffer *inputBuffers = (Buffer*) malloc(nrhs*sizeof(Buffer));
    try {
        for(int i=0; i<nrhs; i++) {
            const mxArray *cur = prhs[i];
            int m = mxGetM(cur);
            int n = mxGetN(cur);
            std::size_t cursize = n*m*mxGetElementSize(cur);
            Buffer inputBuffer = Buffer(context, CL_MEM_READ_ONLY, cursize);
            //inputBuffers[i] = inputBuffer;
            queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, cursize, mxGetData(cur));
            kernel.setArg(i+1, inputBuffer);
        } 
    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
    //free(inputBuffers);
}

void buildKernel() {
    // Build and Run Kernel
    try { 
        // Read source file
        std::ifstream sourceFile("test.cl");
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
        Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
 
        // Make program of the source code in the context
        program = Program(context, source);
 
        // Build program for these specific devices
        program.build(devices);

        // Make kernel
        Kernel lokernel(program, "test");
        kernel = lokernel;
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
    cl_double *matrix   = (cl_double*) malloc(matrixsize);

    // Run Kernel
    try { 
        // Create memory buffers
        Buffer outputBuffer = Buffer(context, CL_MEM_WRITE_ONLY, matrixsize);
 
        // Set arguments to kernel
        kernel.setArg(0, outputBuffer);
        // Run the kernel on specific range
        NDRange global(width, height);
        NDRange local(1, 1);
        queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
 
        // Read buffer into a local list
        queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, matrixsize, matrix);

    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
       std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
       std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    }

    return matrix; 
}

// Prints given Matrix to stdout
void printMatrixToStdOut(double *matrix, unsigned int width, unsigned int height) {
    for(unsigned int i=0; i<height; i++) {
        for(unsigned int j=0; j<width; j++) {
            printf("x:%i,y:%i=%f\n", i, j, matrix[(i*width)+j]);
        }
    }
}

// Standard method needed by Matlab
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    // Allocate memory for output
    cl_double *result;

    // Determine rows and columns
    //TODO Correct Ranges for Kernel to work properly
    int rows = 1;
    int columns = 1;
    for(int i=0; i<=nrhs-1; i++) {
        int m = mxGetM(prhs[i]);
        int n = mxGetN(prhs[i]);
        if (m>rows) rows = m;
        if (n>columns) columns = n;
        if (m != columns) {
            std::cout << "dim mismatch:"<<m<<"!="<<columns<<std::endl;
            return;
        }
        std::cout << "M:"<<mxGetM(prhs[i])<<",N:"<<mxGetN(prhs[i])<<std::endl;
    }

    // Init CL environment
    initCL();

    // Parse String and create dynamic Kernel
    //TODO
    buildKernel();

    // Create Buffers
    createBuffers(nrhs, prhs);

    // Run the Kernel
    result = runKernel(rows, columns);    


    // If left-hand arguments given - write back, else print to stdout
    // OVERRIDE: always return mxArray
    //if (nlhs == 1) {
        // Allocate memory and assign output pointer
        double *outArray;

        // mxReal is our data-type
        plhs[0] = mxCreateDoubleMatrix(rows, columns, mxREAL); 

        // Set data to result
        mxSetData(plhs[0], result);
    //}
    // Free memory
    // not neccessary since matlab sets pointer at mxSetData
    //free(result);
    return;
}
