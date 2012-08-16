#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

#define N 512000
#define HIST 256

using namespace std;

void histograma_cpu(int * data, int * counter){
	int i;
	for(i=0; i<N; i++)
		counter[data[i]]++;
}

__global__ void histograma_gpu_global(int * data, int * counter){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N)
		atomicAdd(&(counter[data[tid]]),1);
}

__global__ void histograma_gpu_shared(int * data, int * counter){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	__shared__ int h_sh[HIST];
	int dev_data;

	if(tid < N){
		if(threadIdx.x < 256)
			h_sh[threadIdx.x] = 0;
	}
	__syncthreads();
	
	if(tid < N){
		dev_data = data[tid];
		atomicAdd(&(h_sh[dev_data]),1);
	}
	__syncthreads();

	if(tid < N){
		if(threadIdx.x < 256)
			atomicAdd(&(counter[threadIdx.x]),h_sh[threadIdx.x]);
	}
	__syncthreads();	
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
	int i;

	int * data;
	int * dev_data;

	int * counter_cpu;

	int * counter_global;
	int * dev_global_counter;

	int * counter_shared;
	int * dev_shared_counter;

	cudaEvent_t start, stop;
	float time_global, time_shared, time_cpu;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*Inicialización de datos y copiado a Device*/
	data = (int *)malloc(N * sizeof(int));
	cudaMalloc((void**)&dev_data, N * sizeof(int));

	counter_global = (int *)malloc(HIST * sizeof(int));
	counter_shared = (int *)malloc(HIST * sizeof(int));
	counter_cpu = (int *)malloc(HIST * sizeof(int));

	cudaMalloc((void**)&dev_global_counter, HIST * sizeof(int));
	cudaMalloc((void**)&dev_shared_counter, HIST * sizeof(int));

	for(i=0; i<N; i++)
		cin >> data[i];

	for(i=0; i<HIST; i++){
		counter_global[i] = 0;
		counter_shared[i] = 0;
		counter_cpu[i] = 0;
	}

	cudaMemcpy(dev_global_counter, counter_global, HIST * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_shared_counter, counter_shared, HIST * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

	/* Llamado a CPU*/
	cout << "Resultados histograma" << endl;
	cudaEventRecord(start, 0);
	histograma_cpu(data, counter_cpu);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_cpu, start, stop);

	/*Llamado a GPU Global*/
	cudaEventRecord(start, 0);
	histograma_gpu_global<<< ceil(N/512.0) , 512 >>> (dev_data, dev_global_counter);
	//cudaThreadSynchronize();
        //cout << cudaGetErrorString(cudaGetLastError()) << endl;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_global, start, stop);

	/*Llamado a GPU Shared*/
	cudaEventRecord(start, 0);
	histograma_gpu_shared<<< ceil(N/512.0) , 512 >>> (dev_data, dev_shared_counter);
	//cudaThreadSynchronize();
        //cout << cudaGetErrorString(cudaGetLastError()) << endl;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_shared, start, stop);

	/*Copiando al host*/
	cudaMemcpy(counter_global, dev_global_counter, HIST * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(counter_shared, dev_shared_counter, HIST * sizeof(int), cudaMemcpyDeviceToHost);



	/*Resultados*/

	cout << "Tiempo para Histograma CPU \t" << time_cpu  << "[ms]" << endl;
	cout << "Tiempo para Histograma Global \t" << time_global  << "[ms]" << endl;
	cout << "Tiempo para Histograma Shared \t" << time_shared  << "[ms]" << endl;

	/*Mostrando por pantalla*/
	//cout << "i\tGLOBAL\tSHARED\tCPU" << endl;
	//for(i=0; i<HIST; i++)
	//	cout << i << "\t" << counter_global[i] << "\t" << counter_shared[i]  << "\t" << counter_cpu[i] << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dev_data);
	cudaFree(dev_global_counter);
	cudaFree(dev_shared_counter);

	free(data);
	free(counter_cpu);
	free(counter_global);
	free(counter_shared);

	return 0;
}
