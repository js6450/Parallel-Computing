#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//array
int *a;
//max value for array element
int max_num = 100000;
//size of array
int size;

void generate();
int find_max();


//generate array of random values
void generate(){

	//malloc array
	a = (int *)malloc(size * sizeof(int));
	if(!a){
		printf("Cannot allocate array.\n");
		exit(1);
	}

//	time_t time;
//	srand((unsigned) time(&time));

	int i;
	for(i = 0; i < size; i++){
		a[i] = rand() % max_num;
//		printf("%d element: %d\n", i, a[i]);
	}

}

//sequentially find mex value
int find_max(){
	
	//initially max value is zero
	int max = 0;
	int i;

	for(i = 0; i < size; i++){
		//replace max if array value is bigger
		if(a[i] > max){
			max = a[i];
		}
	}
	
	//print max value
	printf("max: %d\n", max);
	return max;
}

int main(int argc, char *argv[]){

	if(argc != 2){
		printf("Usage: need size of array\n");
	}

	//receive size of array from command line
	size = atoi(argv[1]);

	generate();

	find_max();

}
