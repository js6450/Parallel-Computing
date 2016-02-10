#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/***** Globals ******/
float *a; /* The coefficients */
float *x; /* The unknowns */
float *b; /* The constants */
float *diag; /*diagonal coefficient of each row in a*/
float err; /* The absolute relative error */
int num = 0; /* number of unknowns */


/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input(); /* Read input from file */
int checkErr();

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/


int checkErr( float *new_x, int num){
 int i;
 float absErr;
 for(i = 0; i < num; i++){
 	absErr = (new_x[i] - x[i])/new_x[i];
 	if(absErr > err){
		return 1;
 	} 
 }
	return 0;
}

/*
    Conditions for convergence (diagonal dominance):
    1. diagonal element >= sum of all other elements of the row
    2. At least one diagonal element > sum of all other elements of the row
*/
void check_matrix(){
 int bigger = 0; /* Set to 1 if at least one diag element > sum */
 int i, j; 
 float sum = 0; 
 float aii = 0; 

 for(i = 0; i < num; i++){
	sum = 0;
	aii = fabs(a[i*(num + 1)]);
	for(j = 0; j < num; j++){
		if( j != i)
			 sum += fabs(a[i*num + j]);
	}

 if( aii < sum){
	printf("The matrix will not converge\n");
	exit(1);
 }

 if(aii > sum) 
	bigger++;
 }
//if bigger is not >0, matrix will not converge...
 if( !bigger )
 { 
	printf("The matrix will not converge\n");
	exit(1);
 }
}


/************************************************************/
/* Read input from file */
void get_input(char filename[])
{
 FILE * fp;
 int i, j;

 fp = fopen(filename, "r");
 if(!fp)
 {
	printf("Cannot open file %s\n", filename);
	exit(1);
 }

 fscanf(fp,"%d ",&num);
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float*)malloc(num * num * sizeof(float));
 if(!a)
 {
	printf("Cannot allocate a!\n");
	exit(1);
 }

 x = (float *) malloc(num * sizeof(float));
 if(!x)
 {
	printf("Cannot allocate x!\n");
	exit(1);
 }

 b = (float *) malloc(num * sizeof(float));
 if( !b)
 {
	printf("Cannot allocate b!\n");
	exit(1);
 }

 diag = (float *) malloc(num * sizeof(float));

 /* Now .. Filling the blanks */

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
 {
	fscanf(fp,"%f ", &x[i]);
 }

 for(i = 0; i < num ; i++){
	for(j = 0; j < num; j++){
		fscanf(fp,"%f ",&a[i*num + j]);
		if(i == j){
			diag[i] = a[i*num + j];
		}
	}	

 /* reading the b element */
	fscanf(fp,"%f ",&b[i]);
 }

fclose(fp);

}


/************************************************************/


int main(int argc, char *argv[])
{
 int i, j;
 int nit = 0; /* number of iterations */


 if( argc != 2) {
	printf("Usage: gsref filename\n");
	exit(1);
 }

 /* Read the input file and fill the global data structure above */
 get_input(argv[1]);

 /* Check for convergence condition */
 check_matrix();

 int comm_size;
 int my_rank;
 
 MPI_Init(&argc, &argv);
 MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
 MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

 int localMin = num / comm_size;
 int extra = num % comm_size;
 int counts[comm_size];
 int displs[comm_size];
 int recv[comm_size];
 int disp = 0;
 
 for(i = 0; i < comm_size; i++){
	if(i < extra){
		counts[i] = localMin + 1;
	} 
	else {
		counts[i] = localMin;
	}
	recv[i] = counts[i];
	displs[i] = disp;
	disp = disp + counts[i];
 } 

 int localNum = (int)ceil((double)num / comm_size);
 int numA = localNum * num;

 float *localX = (float *) malloc(localNum * sizeof(float));
 float *localA = (float *) malloc(num * localNum * sizeof(float));
 float *localB = (float *) malloc(localNum * sizeof(float));
 float *curr = (float *) malloc(num * sizeof(float));
 float *localD = (float *) malloc(localNum * sizeof(float));

 MPI_Scatter(a, numA, MPI_FLOAT, localA, numA, MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Scatter(b, localNum, MPI_FLOAT, localB, localNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Scatter(diag, localNum, MPI_FLOAT, localD, localNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Scatterv(x, counts, displs, MPI_FLOAT, localX, recv[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

 for(i = 0; i < num; i++){
	curr[i] = x[i];
 }

 do{
	nit++;
	for(i = 0; i < num; i++){
		x[i] = curr[i];
	}

	for(i = 0; i < counts[my_rank]; i++){
		
		int global_i = i;
		int k;
		for(k = 0; k < my_rank; k++){
			global_i += counts[k];
		}		

		localX[i] = localB[i];
		
		int j;
		for(j = 0; j < global_i; j++){
			localX[i] = localX[i] - localA[i * num + j] * x[j];
		}

		for(j = global_i + 1; j < num; j++){
			localX[i] = localX[i] - localA[i * num + j] * x[j];
		}

		localX[i] = localX[i]/localD[i];
	}
	MPI_Allgatherv(localX, counts[my_rank], MPI_FLOAT, curr, recv, displs, MPI_FLOAT, MPI_COMM_WORLD);
 }while(checkErr(curr, num));

 if( my_rank == 0){
/* Writing to the stdout */
/* Keep that same format */
	for(i = 0; i < num; i++){
		printf("%f\n", x[i]);
	}
	printf("total number of iterations: %d\n", nit);

	free(x);
	free(a);
	free(b);
	free(diag);
 } 

 free(localX);
 free(localA);
 free(localB);
 free(curr);
 free(localD);

 MPI_Barrier(MPI_COMM_WORLD);
 MPI_Finalize();
 
 exit(0);
}
