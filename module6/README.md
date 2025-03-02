# Module 6 Assignment: 

For this assignment, I've created a program that performs a math operation on two input arrays. One method will only use the default stream, performing a full data copy, performing the kernel operation, and then copying data back. The method utilizing streams will chunk up the data into equal parts and then perform copy-compute-copy on that chunk of data within each stream. This will allow for copy-compute overlap and improve the throughput of the system since data can be worked on and copied to or from device at the same time. Events are utilized in this program to accurately record timing information, since events can be activated and timed after a kernel finishes executing.

## Building and Running
```bash
# Build and run the program
make run

# Build and profile
make profile
```

## Sample Results for running the program

```bash
ubuntu@ip-172-31-77-247:~/EN605.617/module6$ make run
./assignment.exe 1048576 256 1
Array length: 33554432, Array bytes: 134217728, Blocks: 4096, Threads/block: 256, Total threads: 1048576
---- Executing kernel with streams ----
Milliseconds elapsed with streams: 71.492035
Results of operation with streams: 
C[0]: 1, A[0]: 0, B[0], 1
C[1]: 3, A[1]: 1, B[1], 2
C[2]: 3, A[2]: 2, B[2], 1
C[3]: 5, A[3]: 3, B[3], 2
C[4]: 7, A[4]: 4, B[4], 3
C[5]: 8, A[5]: 5, B[5], 3
C[6]: 7, A[6]: 6, B[6], 1
C[7]: 9, A[7]: 7, B[7], 2
C[8]: 10, A[8]: 8, B[8], 2
C[9]: 11, A[9]: 9, B[9], 2
...
---- Executing kernel w/o streams ----
Milliseconds elapsed w/o streams: 86.266144
Results of operation with streams: 
C[0]: 1, A[0]: 0, B[0], 1
C[1]: 3, A[1]: 1, B[1], 2
C[2]: 5, A[2]: 2, B[2], 3
C[3]: 5, A[3]: 3, B[3], 2
C[4]: 6, A[4]: 4, B[4], 2
C[5]: 5, A[5]: 5, B[5], 0
C[6]: 7, A[6]: 6, B[6], 1
C[7]: 8, A[7]: 7, B[7], 1
C[8]: 8, A[8]: 8, B[8], 0
C[9]: 9, A[9]: 9, B[9], 0
...
```