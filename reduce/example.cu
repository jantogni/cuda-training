#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define N 512
#define Th 512

using namespace std;

__global__ void reduceVector(int * input, int * output){
	__shared__ int sh_input[Th];
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int element = blockDim.x;

	if(i < element)
		sh_input[i] = input[i];
	__syncthreads();

	while(element > 1){
		if(i < element/2){
			sh_input[i] = sh_input[i] + sh_input[element/2 + i];
			output[i] = sh_input[i];
		}
		__syncthreads();

		element = element/2;
	}
}

int main() {
	int i;
	
	int * input;
	int * output;
	int * dev_input;
	int * dev_output;

	input = (int *) malloc (N * sizeof(int));	
	output = (int *) malloc (N * sizeof(int));	
	cudaMalloc( (void**)&dev_input, N * sizeof(int));
	cudaMalloc( (void**)&dev_output, N * sizeof(int));

	for(i=0; i<N; i++){
		input[i] = 1;
		output[i] = 0;
	}

	cudaMemcpy(dev_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_output, output, N * sizeof(int), cudaMemcpyHostToDevice);

	reduceVector<<< ceil(N/512.0) , 512 >>> (dev_input, dev_output);

	cudaMemcpy(output, dev_output, N * sizeof(int), cudaMemcpyDeviceToHost);

	for(i=0; i<(N/2); i++)
		cout << i << " " << output[i] << " " << endl;

	return 0;
}
