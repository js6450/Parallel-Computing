#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <assert.h>

#define CITY_PATH_MAX 16

struct Path{
    int num;
    int total;
    int city_path[CITY_PATH_MAX];
    int visited[CITY_PATH_MAX];
};
typedef struct Path Path;

int numCities;
int start = 0; 
int **edges;    
Path *best_path;    

void get_input(char filename[]);    

void init_path(Path *path);
void init_best_path(Path *path);
void add_city(Path *path, int new_city);
void print_path(Path *path);
int check_best_path(Path *path);
void remove_last(Path *p);
Path* copy_paths(Path *original_path);
void compute_path( Path *p);

void get_input(char filename[]){
    FILE * tsm;
    int i, j;
    
    tsm = fopen(filename, "r");

    if(!tsm){
        printf("Cannot open file %s.\n", filename);
        exit(1);
    }
    
    char temp[130];
    while(fgets(temp, sizeof(temp), tsm) != NULL){
	numCities++;
    }
    rewind(tsm);

    edges = (int**)malloc(numCities * sizeof(int*));

    if(!edges){
        printf("Cannot allocate.\n");
        exit(1);
    }

    for(i = 0; i < numCities; i++){
        edges[i] = (int *)malloc(numCities * sizeof(int)); 
        if(!edges[i]){
            printf("Cannot allocate.\n");
            exit(1);
        }
    }

    for(i = 0; i < numCities; i++){
        for(j = 0; j < numCities; j++){
            fscanf(tsm, "%d ", &edges[i][j]);
        }
    }

    best_path = malloc(sizeof(Path));

    fclose(tsm);
}

void init_path(Path *path){
    int i;
    path->num = 1;                   
    for(i = 0; i < CITY_PATH_MAX; i++){
        path->city_path[i] = -1;  
        path->visited[i] = 0;    
    }
    path->city_path[0] = start;   
    path->visited[start] = 1;    
    path->total = 0;                  
}

void init_best_path(Path *path){
    int i;

    init_path(path);

    path->num = numCities;

    for(i = 1; i < numCities; i++){
        path->city_path[i] = i;
        path->total += edges[i-1][i];
        path->visited[i] = 1;
    }
}

void add_city(Path *path, int new_city){
    path->total += edges[path->city_path[path->num - 1]][new_city];
    path->city_path[path->num] = new_city;    
    path->visited[new_city] = 1;           
    path->num++;                              
}

void print_path(Path *path){
    int i;

    printf("%s","Path: " );

    for( i = 0; i < numCities; i++){
        if(path->city_path[i] != -1){
            printf("%d ", path->city_path[i] );
        }
    }
    printf("%s\n"," " );

    printf("Total distance: %d\n", path->total);
}

int check_best_path(Path *p){
    int result = 0;

    if(p->total < best_path->total){
        result = 1;
    }

    return result;
}

void remove_last(Path *p){
    p->total -=  edges[p->city_path[p->num - 2]][p->city_path[p->num - 1]];
    p->visited[p->city_path[p->num - 1]] = 0;   
    p->city_path[p->num - 1] = -1;              
    p->num--;                                
}

Path* copy_paths(Path *original_path){
    int i;
    Path *new_path;
    new_path = malloc(sizeof( Path));
    init_path(new_path);

    new_path->num = original_path->num;
    new_path->total = original_path->total;

    for(i = 0; i < original_path->num; i++){
        new_path->city_path[i] = original_path->city_path[i];
    }

    return new_path;
}

void compute_path( Path *p ){
    int i;

    if(p->num + 1 == numCities){
        for(i = 0; i < numCities; i++){
            if(p->visited[i] == 0){
                add_city(p, i);
            }
        }
        if(check_best_path(p) == 1){
            Path* curr_best_path;

            curr_best_path = copy_paths(p);
            
            #pragma omp critical
            best_path = curr_best_path;
        }

        remove_last(p);
        remove_last(p);

    } else {
        for( i = 0; i < numCities; i++){
            if(p->visited[i] == 0){
                add_city(p, i);
                compute_path(p);
            }
        }
        remove_last(p);
    }
}

int main(int argc, char *argv[]){   
    int i; 

    get_input(argv[1]);

    init_best_path(best_path);

    #pragma omp parallel for
    for(i = 1; i < numCities;i++){
        Path *curr_path;
        curr_path = malloc( sizeof( Path));
        init_path(curr_path);

        add_city(curr_path, i);
        compute_path(curr_path );
    }

    print_path(best_path);



    return 0;
}
