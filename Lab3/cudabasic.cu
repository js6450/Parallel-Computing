#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//define max value of array allement
#define MAX 100000
//defined threads per block for cims machines
#define THREADS_PER_BLOCK 1024

void generate(int *a);
__global__ void get_max(int *array);

//generate random numbers in array
void generate(int *a, int size){
	int i;
	time_t t;
   
   	srand((unsigned) time(&t));

	for(i = 0; i < size; i++){
		a[i] = random() % MAX;;
	}
}

//get the maximum value
__global__ void get_max(int *array, int size){
	int temp;
	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	int numThreads = size;
	int half = numThreads / 2;	

	while(numThreads > 1){
		//if index is in the first half of array
		if (index < half){
			temp = array[index + half];
			if (temp > array[index]) {
				//replace with bigger value
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

	//receive value of size of array
	const int size = atoi(argv[1]);

	//malloc host array
	host_a = (int *)malloc(size * sizeof(int));

	//generate random numbers
	generate(host_a, size);

	//generate for cuda
	cudaMalloc( (void **)&device_a, sizeof(int) * size );

	//copy contents of host array to device array
	cudaMemcpy(device_a, host_a, sizeof(int) * size, cudaMemcpyHostToDevice);

	const int NUM_BLOCKS = size / THREADS_PER_BLOCK;
	
	//run get_max function
	get_max<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(device_a, size);
	
	//copy ocntents of device array bact to host array
	cudaMemcpy(host_a, device_a, sizeof(int) * size, cudaMemcpyDeviceToHost);

	printf("Max value in the array: %d\n", host_a[0]);

	//free memory when done
	free(host_a);
	cudaFree(device_a);

	return 0;
}
