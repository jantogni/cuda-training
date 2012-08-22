#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define N 1536
#define Th 512

using namespace std;

__global__ void reduceVector(int * input, int * output, int * sub_sum){
	__shared__ int sh_input[Th];
	int thread = threadIdx.x;
	int thread_desp = threadIdx.x + blockIdx.x * blockDim.x;

	int element = blockDim.x * (blockIdx.x + 1);

	if(thread_desp < element)
		sh_input[thread] = input[thread_desp];
	__syncthreads();

	while(element > 1){
		if(thread < element/2){
			sh_input[thread] = sh_input[thread] + sh_input[element/2 + thread];
			output[thread_desp] = sh_input[thread];
		}
		__syncthreads();

		element = element/2;
	}
	sub_sum[blockIdx.x] = output[0];
}

int main() {
	int i;	
	int blocks = ceil(N/512.0);

	int * input;
	int * sub_sum;
	int * output;
	int * dev_input;
	int * dev_sub_sum;
	int * dev_output;

	input = (int *) malloc (N * sizeof(int));	
	sub_sum = (int *) malloc (blocks * sizeof(int));	
	output = (int *) malloc (N * sizeof(int));	
	cudaMalloc( (void**)&dev_input, N * sizeof(int));
	cudaMalloc( (void**)&dev_output, N * sizeof(int));
	cudaMalloc( (void**)&dev_sub_sum, blocks * sizeof(int));

	for(i=0; i<N; i++){
		input[i] = 1;
	}

	cudaMemcpy(dev_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_output, output, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sub_sum, sub_sum, blocks * sizeof(int), cudaMemcpyHostToDevice);

	reduceVector<<< blocks , 512 >>> (dev_input, dev_output, dev_sub_sum);

	cudaMemcpy(output, dev_output, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(sub_sum, dev_sub_sum, blocks * sizeof(int), cudaMemcpyDeviceToHost);

	//for(i=0; i<N/2; i++)
	//	cout << i << " " << output[i] << " " << endl;

	for(i=0; i<blocks; i++)
		cout << i << " " << sub_sum[i] << " " << endl;

	return 0;
}
