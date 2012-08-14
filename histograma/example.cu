#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define N 5120
#define HIST 256

using namespace std;

void histograma_cpu(int * data, int * counter){
	int i;
	for(i=0; i<N; i++)
		counter[data[i]]++;
}

__global__ void histograma_gpu(int * data, int * counter){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N)
		atomicAdd(&(counter[data[tid]]),1);
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
	//#ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
	//	cout << "Not using atomics" << endl;
	//#endif

	/* Variables */
	int * data;
	int * dev_data;

	int * counter;
	int * counter_cpu;
	int * dev_counter;

	int i;

	/*Inicialización de datos y copiado a Device*/
	data = (int *)malloc(N * sizeof(int));
	cudaMalloc((void**)&dev_data, N * sizeof(int));

	counter = (int *)malloc(HIST * sizeof(int));
	counter_cpu = (int *)malloc(HIST * sizeof(int));
	cudaMalloc((void**)&dev_counter, HIST * sizeof(int));

	for(i=0; i<N; i++)
		cin >> data[i];

	for(i=0; i<HIST; i++){
		counter[i] = 0;
		counter_cpu[i] = 0;
	}

	cudaMemcpy(dev_counter, counter, HIST * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

	/* Llamado a CPU*/
	cout << "Resultados histograma" << endl;
	histograma_cpu(data, counter_cpu);

	/*Llamado a GPU*/
	histograma_gpu<<< ceil(N/512.0) , 512 >>> (dev_data, dev_counter);
	//cudaThreadSynchronize();
        //cout << cudaGetErrorString(cudaGetLastError()) << endl;

	/*Copiando al host*/
	cudaMemcpy(counter, dev_counter, HIST * sizeof(int), cudaMemcpyDeviceToHost);

	/*Mostrando por pantalla*/
	cout << "i\tGPU\tCPU" << endl;
	for(i=0; i<HIST; i++)
		cout << i << "\t" << counter[i] << "\t" << counter_cpu[i] << endl;

	return 0;
}
