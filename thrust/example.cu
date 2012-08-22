#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using namespace std;

int main() {
	thrust::host_vector<int> h_vec(24);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	thrust::device_vector<int> d_vec = h_vec;

	thrust::sort(d_vec.begin(), d_vec.end());

	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	return 0;
}
