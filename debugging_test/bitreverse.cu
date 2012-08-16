#include <stdio.h>
#include <stdlib.h>
#define N 256

__global__ void bitreverse(unsigned int *data){
	unsigned int *idata = data;
	unsigned int x = idata[threadIdx.x];
	x = ((0xf0f0f0f0 & x) >> 4) | ((0x0f0f0f0f & x) << 4);
	x = ((0xcccccccc & x) >> 2) | ((0x33333333 & x) << 2);
	x = ((0xaaaaaaaa & x) >> 1) | ((0x55555555 & x) << 1);
	idata[threadIdx.x] = x;
}

int main(void){
	unsigned int *d = NULL; int i;
	unsigned int idata[N], odata[N];
	for (i = 0; i < N; i++)
		idata[i] = (unsigned int)i;
	
	cudaMalloc((void**)&d, sizeof(int)*N);
	cudaMemcpy(d, idata, sizeof(int)*N,cudaMemcpyHostToDevice);

	bitreverse<<<1, N>>>(d);

	cudaMemcpy(odata, d, sizeof(int)*N,cudaMemcpyHostToDevice);

	for (i = 0; i < N; i++)
		printf("%u -> %u\n", idata[i], odata[i]);
	
	cudaFree((void*)d);
	return 0;
}
