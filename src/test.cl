#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void test(__global __write_only double *output) {
    unsigned int width   = get_global_size(0);
    unsigned int height  = get_global_size(1);
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    output[x*height+y] = 2.0;
}
