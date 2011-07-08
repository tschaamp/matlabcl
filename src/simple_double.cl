#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void add(__global __write_only double *output ?1) {
    uint width  = get_global_size(0);
    uint height = get_global_size(1);
    uint x      = get_global_id(0);
    uint y      = get_global_id(1);
    uint pos    = x*height + y;

    output[pos] = ?2;
}

__kernel void dots(__global __write_only double *output, __global __read_only double *v1, __global __read_only double *v2) {
    uint width  = get_global_size(0);
    uint height = get_global_size(1);
    uint x      = get_global_id(0);
    uint y      = get_global_id(1);
    uint pos    = x*height + y;

    double dotProduct = 0;
    for (int x = 0; x < width; ++x)
        dotProduct += v1[x] * v2[x];

    output[0] = dotProduct;
}

__kernel void smul(__global __write_only double *output, __global __read_only double *m, __global __read_only double *s) {
    uint width  = get_global_size(0);
    uint height = get_global_size(1);
    uint x      = get_global_id(0);
    uint y      = get_global_id(1);
    uint pos    = x*height + y;

    output[pos] = (*s)*m[pos];
}

__kernel void minus(__global __write_only double *output, __global __read_only double *v1, __global __read_only double *v2) {
    uint width  = get_global_size(0);
    uint height = get_global_size(1);
    uint x      = get_global_id(0);
    uint y      = get_global_id(1);
    uint pos    = x*height + y;

    output[pos] = v1[pos] - v2[pos];
}

__kernel void mtimesv(__global __write_only double *output, __global __read_only double *m, __global __read_only double *v) {
    uint width  = get_global_size(0);
    uint height = get_global_size(1);

    uint y = get_global_id(0);
    if (y < height) {
    
        const __global double* row = m + y * width;

        double dotProduct = 0;
        for (int x = 0; x < width; ++x)
            dotProduct += row[x] * m[x];

        output[y] = dotProduct;

    }
}
