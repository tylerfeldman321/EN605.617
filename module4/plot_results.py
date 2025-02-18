import matplotlib.pyplot as plt
import numpy as np


N_exp = np.arange(8, 31)
N_values = 2 ** N_exp
gpu_time_microseconds = np.array([
    165, 168, 183, 170, 185, 199, 232, 300, 416, 643, 1174, 1748, 2898, 5197,
    9990, 20122, 40668, 82051, 164918, 332241, 664826, 1330236, 2657664
])
plt.figure(figsize=(10, 6))
plt.plot(N_values, gpu_time_microseconds, marker='o', linestyle='-', label="GPU Time")
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel("Number of Data Elements (log scale)")
plt.ylabel("Time Elapsed (microseconds)")
plt.title("GPU Execution Time vs Number of Data Elements")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()



N = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
cpu_add_interleaved = np.array([0.011360, 0.016480, 0.030720, 0.059424, 0.114752, 0.223296, 0.454464, 0.873760, 1.691616])
cpu_add_non_interleaved = np.array([0.009312, 0.017056, 0.032704, 0.061696, 0.156736, 0.260288, 0.569408, 1.126496, 2.251200])
gpu_add_interleaved = np.array([0.257280, 0.249664, 0.241664, 0.242752, 0.239104, 0.230272, 0.218592, 0.181728, 0.134528])
gpu_add_non_interleaved = np.array([0.020704, 0.023552, 0.022528, 0.019904, 0.015936, 0.010368, 0.014688, 0.016352, 0.020128])
plt.figure(figsize=(10, 6))
plt.plot(N, cpu_add_interleaved, marker='o', label="CPU Add Interleaved", linestyle='-')
plt.plot(N, cpu_add_non_interleaved, marker='s', label="CPU Add Non-Interleaved", linestyle='--')
plt.plot(N, gpu_add_interleaved, marker='^', label="GPU Add Interleaved", linestyle='-')
plt.plot(N, gpu_add_non_interleaved, marker='d', label="GPU Add Non-Interleaved", linestyle='--')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel("N (Array Size)")
plt.ylabel("Execution Time (ms)")
plt.title("Performance Comparison of CPU and GPU Add Operations")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()


