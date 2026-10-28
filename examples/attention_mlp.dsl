// Reference workload from the project brief: single-head attention -> 2-layer MLP.
//   SEQ = 64 tokens, D = 256 model/head dim, F = 1024 feed-forward dim.
// Target: GTX 1050 Ti (Pascal sm_61) -- 6 SMs, 48 KB shared/block, 255 regs/thread.

const SEQ = 64;
const D   = 256;
const F   = 1024;

dtype fp16 store fp32 accum;

input Q  : fp16 [SEQ, D];
input K  : fp16 [SEQ, D];
input V  : fp16 [SEQ, D];

param W1 : fp16 [D, F];
param b1 : fp32 [F];
param W2 : fp16 [F, D];
param b2 : fp32 [D];

out = attention(Q, K, V) -> MLP(W1, relu, W2, b1, b2);
