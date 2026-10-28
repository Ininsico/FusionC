// device_probe.cu -- report the properties the FusionC heuristics need.
//
// The compiler's thresholds (SMEM_BYTES_PER_SM, REGS_PER_SM, the SM count used for the grid
// occupancy floor) are compile-time constants for sm_61. This probe lets the driver feed the
// *actual* device numbers into the compiler instead of assuming the 1050 Ti from the brief.
//
// build: nvcc -arch=sm_61 device_probe.cu -o device_probe
#include <cstdio>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    int device = 0;
    if (argc > 2 && argv[1][0] == '-' && argv[1][1] == 'd') {
        device = atoi(argv[2]);
    }
    cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, device);
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("DEVICE,%s,%d.%d,%d,%d,%d,%d,%d,%d,%d\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           (int) p.sharedMemPerBlock, (int) p.sharedMemPerMultiprocessor,
           p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor,
           p.maxThreadsPerBlock, p.clockRate);
    return 0;
}
