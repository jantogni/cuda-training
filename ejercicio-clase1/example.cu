#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 60

/* El siguiente es un kernel CUDA que se ejecuta en la GPU.
 * Todas las hebras ejecutan esta funcion en paralelo.  */

__global__ void consecutivos(float a[]) {
    int t = threadIdx.x; /* indice de la hebra (en este ejemplo, entre 0 y 19) */
    int b = blockIdx.x;  /* indice del bloque  (en este ejemplo, entre 0 y 2)  */
    int B = blockDim.x;  /* taman~o del bloque (en este ejemplo siempre es 20) */

    int i = b * B + t;   /* elemento del arreglo que le toca asignar a esta hebra */
    a[i] = i;
}


int main() {

    float *a_gpu;
    float *a_cpu;

    /* Reserva arreglos de taman~o N en la CPU y en la GPU. */
    cudaMalloc((void **) &a_gpu, N * sizeof(float));
    a_cpu = (float *) malloc(N * sizeof(float));

    /* Ejecuta el kernel con 3 bloques de 20 hebras cada uno. */
    consecutivos<<<4, 15>>>(a_gpu);

    /* Copia el arreglo de la GPU a la CPU. */
    cudaMemcpy(a_cpu, a_gpu, N * sizeof(float), cudaMemcpyDeviceToHost);

    int i;
    for (i = 0; i < N; ++i)
        printf("%.1f\n", a_cpu[i]);

    /* Libera los arreglos reservados. */
    free(a_cpu);
    cudaFree(a_gpu);

    return 0;
}
