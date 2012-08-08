#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define N 60

using namespace std;

__global__ void add(int *a, int *b, int *c) {
	int i = blockIdx.x;
	if(i<N)
		c[i] = a[i] + b[i];
}

int main() {
	int a[N], b[N], c[N];	
	int *dev_a, *dev_b, *dev_c;

	cudaMalloc( (void**)&dev_a, N * sizeof(int) );
	cudaMalloc( (void**)&dev_b, N * sizeof(int) );
	cudaMalloc( (void**)&dev_c, N * sizeof(int) );

	for( int i=0; i<N; i++){
		a[i] = i;
		b[i] = 2*i;
	}
	
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	
	add<<<N,1>>>(dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	

	for( int i=0; i<N; i++){
		cout << a[i] << " " << b[i] << " " << c[i] << endl;
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
