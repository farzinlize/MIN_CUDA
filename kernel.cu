#include "cuda_runtime.h"
#include "device_launch_parameters.h"
extern "C" {
	#include "helper_functions.h"
	#include "fuzzy_timing.h"
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
	unsigned int tid = (blockDim.x*2) * blockIdx.x + tid_block;
	
	a_s[tid_block] = device_min(a_in[tid], a_in[tid+blockDim.x]);
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

int main(int argc, char * argv[]){

	if(argc != 2){
		printf("Correct way to execute this program is:\n");
		printf("MIN_CUDA stream_count\n");
		printf("For example:\nMIN_CUDA 4\n");
		return 1;
	}

	int stream_count = atoi(argv[1]);

	int size = 1024 * 1024 * 40, block_size = 1024;
	int *a_h, *a_d, *out_d, *device_out_h;
	int min_parralel, min_seq;
	double seq_time, total_time, kernel_time;

	initialize_data_random_cudaMallocHost(&a_h, size);	//initial data on host
	initialize_data_zero_cudaMallocHost(&device_out_h, block_size);
	
	cudaStream_t* streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * stream_count);
	for(int i=0;i<stream_count;i++){
		cudaStreamCreate(&streams[i]);
	}

	set_clock();	//sequentinal run

	min_seq = find_min_seq(a_h, size);

	seq_time = get_elapsed_time();

	printf("[TIME] Sequential: %.4f\n", seq_time);

	int stream_size = size / stream_count;
	int out_size_stream = block_size / stream_count;
	int block_count = (stream_size/block_size)/2;
	dim3 grid_dim(block_count, 1, 1);
	dim3 block_dim(block_size, 1, 1);
	
	set_clock();	//parallel run

	CUDA_CHECK_RETURN(cudaMalloc((void **)&a_d, sizeof(int)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&out_d, sizeof(int)*block_size));

	int offset = 0, out_offset = 0;
	for(int stream_id=0 ; stream_id < stream_count ; stream_id++){
		cudaMemcpyAsync(&a_d[offset], &a_h[offset], stream_size*sizeof(int), cudaMemcpyHostToDevice, streams[stream_id]);
		minKernel<<<grid_dim, block_dim, block_size*sizeof(int), streams[stream_id]>>>(&a_d[offset], &out_d[out_offset]);
		cudaMemcpyAsync(&device_out_h[out_offset], &out_d[out_offset], out_size_stream*sizeof(int), cudaMemcpyDeviceToHost, streams[stream_id]);
		offset+=stream_size;
		out_offset+=out_size_stream;
	}

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	kernel_time = get_elapsed_time();
	set_clock();

	min_parralel = find_min_seq(device_out_h, block_size);

	total_time = get_elapsed_time();
	total_time += kernel_time;

	printf("[TIME] total parallel: %.4f\n", total_time);
	printf("[TIME] kernel_time : %.4f\n", kernel_time);

	#ifdef DEBUG
	printf("device_out_h array: \n");
	for(int k=0; k < block_size ;k++){
		printf("%d\t", device_out_h[k]);
	}
	#endif

	printf("[SPEEDUP] sequentianal / parallel_time  : %.4f\n", seq_time/total_time);
	printf("Parallel_min: %d \tSeq_min: %d", min_parralel, min_seq);

    return 0;
}