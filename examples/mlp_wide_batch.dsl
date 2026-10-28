// Same graph as attention_mlp.dsl but with 512 tokens.
//
// Why: the GEMM->GEMM (MLP) fusion is rejected at SEQ=64 because the fused kernel's grid
// (M/32 x N/64 = 2 x 4 = 8 blocks) cannot fill the device, and the 1050 Ti floor is 2 blocks/SM.
// At SEQ=512 the same fusion yields 64 blocks, above the floor; the accumulator-register guard
// still rejects it, and --force-fusion measures that rejection to be correct (~2.9x slower).

const SEQ = 512;
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
