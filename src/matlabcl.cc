#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <cstddef>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include "math.h"
#include "mex.h"

using namespace cl;

double* simplecl(unsigned int width, unsigned int height) {
    //int width=10;
    //int height=10;

    //Create vars
    Program program;
    vector<Device> devices;

    // Create the host image
    unsigned int matrixsize = width * height * sizeof(cl_double);
    cl_double *matrix   = (cl_double*) malloc(matrixsize);

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
        Context context( CL_DEVICE_TYPE_GPU, cps);
 
        // Get a list of devices on this platform
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
 
        // Create a command queue and use the first device
        CommandQueue queue = CommandQueue(context, devices[0]);
 
        // Read source file
        std::ifstream sourceFile("test.cl");
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
        Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
 
        // Make program of the source code in the context
        program = Program(context, source);
 
        // Build program for these specific devices
        //int err = CL_SUCCESS;
        program.build(devices);

        // Make kernel
        Kernel kernel(program, "test");
 
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


    return matrix;

    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
       std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
       std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    }
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

    // Determine width and height
    //TODO REALLY Determine size, but how?
    int width = 10;
    int height = 10;

    // Parse String and create dynamic Kernel
    //TODO

    // Run the Kernel
    result = simplecl(width, height);    

    // If left-hand arguments given - write back, else print to stdout
    if (nlhs == 1) {
        // Allocate memory and assign output pointer
        double *outArray;

        // mxReal is our data-type
        plhs[0] = mxCreateDoubleMatrix(width, height, mxREAL); 

        // Get a pointer to the data space in our newly allocated memory
        outArray = mxGetPr(plhs[0]);

/*for(int i=0;i<height;i++)
{
    for(int j=0;j<width;j++)
    {
        outArray[(i*width)+j] = static_cast<double>(matrix[(i*width)+j]);
    }
}*/
        outArray = result;
    } else {
        printMatrixToStdOut(result, width, height);
    }
    return;
}
