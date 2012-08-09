#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define N 60

using namespace std;

__global__ void consecutivos() {
	//int t = threadIdx.x;
	//int b = blockIdx.x;
	//int B = blockDim.x;
}


int main() {
	dim3 grid_dim(30,2);
	consecutivos<<< grid_dim, 3 >>>();	
	return 0;
}
