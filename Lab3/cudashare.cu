#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//max value for element of array
#define MAX 100000
//defined threads per block for cims machines
#define THREADS_PER_BLOCK 1024
//number of warp
#define WARP 32

void generate(int *a, const int size);
__global__ void get_max(int *array, int *maxarray);

//generate array of random numbers
void generate(int *a, const int size){
	int i;
	time_t t;
   
   	srand((unsigned) time(&t));

	for(i = 0; i < size; i++){
		a[i] = random() % MAX;;
	}
}

//get the max value in array
__global__ void get_max(int *array, int *maxarray){
	int temp;
	//declare the shared variable
	__shared__ int max[THREADS_PER_BLOCK];
	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	max[threadIdx.x] = array[index];
	__syncthreads();

	//her the number of threads is the number of block dimensions
	int numThreads = blockDim.x;

	while(numThreads > WARP){
		int half = numThreads / 2;
		if (threadIdx.x < half){
			temp = max[threadIdx.x + half];
			//replace with bigger value
			if (temp > max[threadIdx.x]) {
				max[threadIdx.x] = temp;
			}
		}
		__syncthreads();
		numThreads = numThreads / 2;
	}
	//save the max value to maxarray
	maxarray[blockIdx.x] = max[0];
}

int main(int argc, char *argv[]){
	//declare arrays
	int *host_a;
	int *host_m;
	int *device_a;
	int *device_m;
	//declare max and i
	int max, i;

	//get size of array from command line
	const int size = atoi(argv[1]);

	//malloc host arrays
	host_a = (int *)malloc(size * sizeof(int));
	host_m = (int *)malloc(size * sizeof(int));

	//generate random array
	generate(host_a, size);

	//malloc device arrays
	cudaMalloc((void **)&device_a, sizeof(int) * size);
	cudaMalloc((void **)&device_m, sizeof(int) * size);

	//copy host array into device array
	cudaMemcpy(device_a, host_a, sizeof(int) * size, cudaMemcpyHostToDevice);

	const int NUM_BLOCKS = size / THREADS_PER_BLOCK;
	
	//run get_max function
	get_max<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(device_a, device_m);
	
	//copy device max values to host max array
	cudaMemcpy(host_m, device_m, sizeof(int) * size, cudaMemcpyDeviceToHost);

	max = host_m[0];
	//get the max value among device max value
	for(i = 1; i < size; i++){
		if(host_m[i] > max){
			max = host_m[i];
		}
	}

	printf("Max value in the array: %d\n", max);

	//free arrays
	free(host_a);
	free(host_m);
	cudaFree(device_a);
	cudaFree(device_m);

	return 0;
}
