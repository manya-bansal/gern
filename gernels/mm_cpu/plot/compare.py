with open("compare.txt", 'r') as file:
    raw_data = file.read()

split_data = [line.split(" ") for line in raw_data.split("\n")]

dimensions = [int(a) for a, b, c in split_data]
blas_time = [float(b) for a, b, c in split_data]
gern_time = [float(c) for a, b, c in split_data]

import matplotlib.pyplot as plt

plt.plot(dimensions, blas_time, label="MKL_BLAS", marker='o')
plt.plot(dimensions, gern_time, label="GERN", marker='x')

plt.xlabel("Dimension n")
plt.ylabel("Time (in seconds)")
plt.title("Comparison of MKL_BLAS and GERN Square Matrix Multiplication Times")
plt.legend()
plt.grid(True)

plt.show()