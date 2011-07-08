A = rand(500);
B = rand(500);
C = rand(500);
D = rand(500);
E = rand(500);
disp('adding 5 Matrices: A,B,C,D,E');
tic;
R = matlabcl('add',A,B,C,D,E);
toc;
check = isequal(R,A+B+C+D+E);
if check == 1
  disp('equal check successful!');
end
