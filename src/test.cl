#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void test(__global __write_only double *output, __global __read_only double *v1, __global __read_only double *v2) {
    unsigned int width   = get_global_size(0);
    unsigned int height  = get_global_size(1);
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int pos = x*height+y;
    //output[pos] = 2.0;
    output[pos] = v1[pos];
    output[pos] += v2[pos];
}
