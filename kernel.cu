#include "cuda_runtime.h"
#include "device_launch_parameters.h"
extern "C" {
	#include "helper_functions.h"
}
#include <stdio.h>
#include <cuda.h>

#ifdef SYNCTHREAD
void __syncthreads();
#endif

__device__	int device_min(int a, int b) {
	if(a < 0 || b < 0)
		return -1;
	return a < b ? a : b;
}

__global__ void minKernel(int *a_in, int *out){
	extern __shared__ int a_s[];
	unsigned int tid_block = threadIdx.x;
	unsigned int tid = blockDim.x * blockIdx.x + tid_block;
	
	a_s[tid_block] = a_in[tid];
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0 ; s >>= 1) {
		if (tid_block < s)
			a_s[tid_block] = device_min(a_s[tid_block], a_s[tid_block + s]);
		__syncthreads();
	}

	if (tid_block == 0)
		out[blockIdx.x] = a_s[0];
}

int find_min_seq(int *a, int size) {
	int min = a[0];
	for (int i = 1; i < size; i++) {
		if (a[i] < min)
			min = a[i];
	}
	return min;
}

int main(){

	int size = 1024 * 1024, block_size = 1024;
	int *a_h, *a_d, *out_d, *device_out_h;
	double elapced;

	initialize_data_random(&a_h, size);
	initialize_data_zero(&device_out_h, block_size);
	
	#ifdef DEBUG
	printf("device_out_h array: \n");
	for(int k=0;k<block_size;k++){printf("%d\t", device_out_h[k]);}
	#endif

	set_clock();

	int min_seq = find_min_seq(a_h, size);

	elapced = get_time();

	printf("TIME Seq: %f\n", elapced);

	set_clock();

	CUDA_CHECK_RETURN(cudaMalloc((void **)&a_d, sizeof(int)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&out_d, sizeof(int)*block_size));

	dim3 grid_dim(1024, 1, 1);
	dim3 block_dim(block_size, 1, 1);

	CUDA_CHECK_RETURN(cudaMemcpy(a_d, a_h, sizeof(int)*size, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	minKernel <<<grid_dim, block_dim, sizeof(int)*block_size, NULL >>> (a_d, out_d);
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	CUDA_CHECK_RETURN(cudaMemcpy(device_out_h, out_d, sizeof(int)*block_size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	int min_parralel = find_min_seq(device_out_h, block_size);

	elapced = get_time();

	printf("TIME parallel: %f\n", elapced);

	#ifdef DEBUG
	printf("device_out_h array: \n");
	for(int k=0; k < block_size ;k++){
		printf("%d\t", device_out_h[k]);
	}
	#endif

	printf("Parallel_min: %d \nSeq_min: %d", min_parralel, min_seq);

    return 0;
}