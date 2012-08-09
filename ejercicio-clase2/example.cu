#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define N 100000000

using namespace std;

__global__ void elemento_n(float *pi_4){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < N){
		pi_4[i] = pow((float)-1,(float)i);
		pi_4[i] /= (float)(2*i+1);
	}
}

void showDeviceProperties(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	cout << "Max Threads: " << prop.maxThreadsPerBlock << endl;
	cout << "Max Grid Size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << endl;
	cout << "Max Threads Dim: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << endl;
}

int main(){
	//showDeviceProperties();
	int i;
	float val[N];
	float *dev_val;
	float pi_4 = 0;

	cudaMalloc( (void**)&dev_val, N * sizeof(float) );

	elemento_n<<< ceil(N/512.0), 512 >>>(dev_val);
	cudaThreadSynchronize(); 
	cout << cudaGetErrorString(cudaGetLastError()) << endl;

	cudaMemcpy(val, dev_val, N * sizeof(float), cudaMemcpyDeviceToHost);

	//for(i=0; i<N; i++){
	//	cout << "i: " << i << " " << val[i] << endl;
	//}

	for(i=0; i<N; i++){
		pi_4 = pi_4 + val[i];
	}

	cout.precision(100);
	cout << fixed << "El valor de pi/4: " << pi_4 << endl;

	return 0;
}
