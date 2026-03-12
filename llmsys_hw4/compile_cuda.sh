mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC
