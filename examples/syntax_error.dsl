// negative test for the parser demo: the pipeline's second stage is missing
const N = 8;

input x : fp16 [N, N];
param w : fp16 [N, N];

out = matmul(x, w) -> ;
