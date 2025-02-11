# Module 3 Assignment

## Results of the Math Operations

### Results of Add Operation
```bash
ubuntu@ip-172-31-77-247:~/EN605.617/module3$ make assignment && ./assignment.exe 1024 256 add
nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
----- New Run -----
Changed operation to add
Operation: add, Array size: 1024, Array size in bytes: 4096, Number of blocks: 4, Threads per block: 256, Total threads: 1024
Time elapsed GPU = 225178 ns
Results of operation: 
Result[0]: 0, A[0]: 0, B[0], 0
Result[1]: 3, A[1]: 1, B[1], 2
Result[2]: 5, A[2]: 2, B[2], 3
Result[3]: 5, A[3]: 3, B[3], 2
Result[4]: 4, A[4]: 4, B[4], 0
```

### Results of Subtract Operation
```bash
ubuntu@ip-172-31-77-247:~/EN605.617/module3$ make assignment && ./assignment.exe 1024 256 subtract
nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
----- New Run -----
Changed operation to subtract
Operation: subtract, Array size: 1024, Array size in bytes: 4096, Number of blocks: 4, Threads per block: 256, Total threads: 1024
Time elapsed GPU = 227143 ns
Results of operation: 
Result[0]: -3, A[0]: 0, B[0], 3
Result[1]: 1, A[1]: 1, B[1], 0
Result[2]: 2, A[2]: 2, B[2], 0
Result[3]: 0, A[3]: 3, B[3], 3
Result[4]: 4, A[4]: 4, B[4], 0
```

### Results of Multiply Operation
```bash
ubuntu@ip-172-31-77-247:~/EN605.617/module3$ make assignment && ./assignment.exe 1024 256 multiply
nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
----- New Run -----
Changed operation to multiply
Operation: multiply, Array size: 1024, Array size in bytes: 4096, Number of blocks: 4, Threads per block: 256, Total threads: 1024
Time elapsed GPU = 227719 ns
Results of operation: 
Result[0]: 0, A[0]: 0, B[0], 0
Result[1]: 1, A[1]: 1, B[1], 1
Result[2]: 0, A[2]: 2, B[2], 0
Result[3]: 9, A[3]: 3, B[3], 3
Result[4]: 0, A[4]: 4, B[4], 0
```

### Results of Modulus Operation
```bash
ubuntu@ip-172-31-77-247:~/EN605.617/module3$ make assignment && ./assignment.exe 1024 256 mod
nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -o assignment.exe
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
----- New Run -----
Changed operation to mod
Operation: mod, Array size: 1024, Array size in bytes: 4096, Number of blocks: 4, Threads per block: 256, Total threads: 1024
Time elapsed GPU = 230301 ns
Results of operation: 
Result[0]: 0, A[0]: 0, B[0], 3
Result[1]: 1, A[1]: 1, B[1], 3
Result[2]: 2, A[2]: 2, B[2], 3
Result[3]: 0, A[3]: 3, B[3], 3
Result[4]: -1, A[4]: 4, B[4], 0
```

## Effects of Conditional Branching in CUDA
- Create a program that demonstrates the effect of conditional branching in CUDA 
kernels executing similar algorithms. Some charts and result description is 
required as well.
- Use two additional numbers of threads
- Use two additional block sizes
- Include at least one performance comparison chart and a short text file that includes your thoughts on the results.

## Stretch Problem
**The good:**
- Comparison between running host code and GPU using chrono clock
- The main function reads in command line arguments for the number of blocks and threads.

**The bad:**
- Block size of 3 by default is probably a bad default block size to have. It should probably be a multiple of the number of SMs for the GPU, so some large positive number (e.g. 80).
- The arrays are initialized incorrectly, at least in comparison to the current instructions we are given. The first array is initialized with -i, and the second array is initialized with i*i, when it should be i and a random number from 0-3 respectively.
- There are some syntax errors (e.g. missing "<" for the chrono::duration_cast call)
- No GPU synchronization for the profiling. This means that the timing for running the add kernel will just be the time for the CPU to call the kernel. For a correct timing, the code should call the kernel and then synchronize with the GPU so that the CPU can actually know that the GPU has finished executing the kernel.
