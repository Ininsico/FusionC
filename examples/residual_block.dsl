// A pre-norm transformer block fragment: LayerNorm, attention, residual add, GELU MLP.
// Exercises the reduction kernels (softmax, layernorm), the residual elementwise path and GELU.
//
// Four pipelines: a pipeline result is bound to its name, so later pipelines can reference it.
//   norm = LayerNorm(x)
//   att  = attention(Q, K, V)
//   res  = norm + att          (residual)
//   out  = GELU-MLP(res)

const SEQ = 64;
const D   = 256;
const F   = 1024;

dtype fp16 store fp32 accum;

input x    : fp16 [SEQ, D];
input Q    : fp16 [SEQ, D];
input K    : fp16 [SEQ, D];
input V    : fp16 [SEQ, D];

param gain : fp32 [D];
param beta : fp32 [D];
param W1   : fp16 [D, F];
param b1   : fp32 [F];
param W2   : fp16 [F, D];
param b2   : fp32 [D];

norm = layernorm(x, gain, beta);
att  = attention(Q, K, V);
res  = add(norm, att);
out  = res -> MLP(W1, gelu, W2, b1, b2);
