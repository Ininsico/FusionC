// Negative test: the compiler must fail loudly on ops it cannot emit.
// Expected: "error: unsupported op 'conv2d' ..." and a non-zero exit status.

const SEQ = 64;
const D   = 256;

dtype fp16 store fp32 accum;

input img : fp16 [SEQ, D];
param w   : fp16 [D, D];

out = conv2d(img, w);
