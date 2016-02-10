#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//max value for element in array
#define MAX 100000
//defined threads per block for cims machines
#define THREADS_PER_BLOCK 1024
//defined warp number
#define WARP 32

void generate(int *a, const int sie);
__global__ void get_max(int *array, const int size);

//generate random array
void generate(int *a, const int size){
	int i;
	time_t t;
   
   	srand((unsigned) time(&t));

	for(i = 0; i < size; i++){
		a[i] = random() % MAX;;
	}
}

//get the maximum value in the array
__global__ void get_max(int *array, const int size){
	int temp;
	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	int numThreads = size;

	//if the number of threads is bigger than size of the warp
	while(numThreads > WARP){
		int half = numThreads / 2;
		if (index < half){
			temp = array[index + half];
			//replace with bigger value
			if (temp > array[index]) {
				array[index] = temp;
			}
		}
		__syncthreads();
		numThreads = numThreads / 2;
	}
}

int main(int argc, char *argv[]){
	
	//initialize arrays
	int *host_a;
	int *device_a;
	//initialize max value
	int max = 0;

	//get size from argument from command line
	const int size = atoi(argv[1]);

	//malloc host array
	host_a = (int *)malloc(size * sizeof(int));

	//generate random array
	generate(host_a, size);

	//malloc for device array
	cudaMalloc( (void **)&device_a, sizeof(int) * size );

	//copy contents of host array into device array
	cudaMemcpy(device_a, host_a, sizeof(int) * size, cudaMemcpyHostToDevice);

	const int NUM_BLOCKS = size / THREADS_PER_BLOCK;
	
	//run get_max function
	get_max<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(device_a, size);
	
	//copy contents of device array back to host array
	cudaMemcpy(host_a, device_a, sizeof(int) * size, cudaMemcpyDeviceToHost);

	//go through host array and get the max value
	int i;
	for(i = 0; i < WARP; i++){
		if(max < host_a[i]){
			max = host_a[i];
		}
	}
	
	printf("Max value in the array: %d\n", max);

	//free the arrays
	free(host_a);
	cudaFree(device_a);

	return 0;
}
