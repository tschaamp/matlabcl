clear;
cd '/home/tschaamp/opencl-toolbox';
path(path,'/home/tschaamp/matlabcl/build');

% Set up env
n = 1000;
% Create testvars
m1 = rand(n);
m2 = rand(n);
m3 = rand(n);
s1 = rand;

% CPU test
disp('running cpu');
tic
resultCPU = ((m2 * m1)/s1)*m3;
toc

% GPU test
disp('running cuda');
tic
m1GPU = gpuArray(m1);
m2GPU = gpuArray(m2);
m3GPU = gpuArray(m3);
s1GPU = gpuArray(s1);
resultGPU = gather(((m2GPU * m1GPU)/s1GPU)*m3GPU);
toc

% opencl-toolbox test
disp('running opencl-toolbox');
tic
ocl = opencl();
ocl.initialize();

ocl.addfile('cl/matlab_kernels.cl');
ocl.build();
toc
m1CL1 = clobject(m1);
m2CL1 = clobject(m2);
m3CL1 = clobject(m3);
s1CL1 = clobject(s1);
t1 = clobject(zeros(n));
t2 = clobject(zeros(n));
t3 = clobject(zeros(n));

% ((m2 * m1)/s1)*m3
%  |--t1--
global_work_size = [n,n,0];
local_work_size = [1,1,0];

% addkernel = clkernel('add', global_work_size, local_work_size);
timeskernel = clkernel('times', global_work_size, local_work_size);
divide_scalar = clkernel('divide_scalar', global_work_size, local_work_size);
timeskernel(m2CL1, m1CL1, t1, double(n));
divide_scalar(t1, s1CL1, t2, double(n));
timeskernel(t2, m3CL1, t3, double(n));
resultCL1 = t1.get();
toc

% matlabcl test
disp('running matlabcl');
tic
resultCL2 = matlabcl(zeros(n));
toc
