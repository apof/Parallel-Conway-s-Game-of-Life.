#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <timestamp.h>
#include <cuda.h>
#include "header.cuh"
#include "func.cuh"

void initialize_table(long *current_table);
void print_table(long *t);

int main(void)
{
	int i;
	long size = (N+2)*(N+2)*sizeof(long);
	//allocate space for cpu
	long *current_table,*next_table;
	current_table = (long*) malloc(size);
	next_table = (long*) malloc(size);

	initialize_table(current_table);	//random initialization of the table		
				
	//allocate space for gpu
	long *g_current_table, *g_next_table;
	cudaMalloc((void **)&g_current_table, size);	//table for current generation
	cudaMalloc((void **)&g_next_table, size);		//table for next generation

	//copy table to device
	cudaMemcpy(g_current_table, current_table, size, cudaMemcpyHostToDevice);

	dim3 threadNum = dim3(BLSIZE,BLSIZE);
	dim3 blockNum = dim3((N  + (threadNum.x * CELLS_PER_THREAD) - 1) / (threadNum.x * CELLS_PER_THREAD), (N +(threadNum.y * CELLS_PER_THREAD) - 1)/ (threadNum.y * CELLS_PER_THREAD));
	dim3 rowThreadNum(BLSIZE,1);
	dim3 gridNum(N/rowThreadNum.x+1,1);

	//Timer
	timestamp t_start;
	t_start = getTimestamp();

	/*print_table(current_table);
		printf("\n\n");*/
printf("starting loop\n");
	for(i=0;i<GENERATION_NUM;i++)
	{

		//update virtual rows
		updateVirtualRows<<<gridNum,rowThreadNum>>>(g_current_table);
		//update virtual columns
		updateVirtualColumns<<<gridNum,rowThreadNum>>>(g_current_table);
		//update virtual cornes
		updateVirtualCorners<<<1,1>>>(g_current_table);

		//calculate next generation
		calculateNextGen<<<blockNum,threadNum>>>(g_current_table, g_next_table);

		//swap device current and device next table
		g_current_table = g_next_table;

	}

	float msecs = getElapsedtime(t_start);
	printf("Game Of Life with table size = %d\n",N);
	printf("Execution time %.2f msecs\nBandwith %lf GB/sec\n", msecs, (N+2)*(N+2)*sizeof(long)/msecs*1000.0f/(float)(1024*1024*1024));

	cudaMemcpy(current_table, g_current_table, size, cudaMemcpyDeviceToHost);
	
	/*print_table(current_table);
	printf("\n\n");*/

	//free space on cpu and device
	free(current_table);
	free(next_table);
	cudaFree(g_current_table);
	cudaFree(g_next_table);
}

void initialize_table(long *current_table)
{
	long temp_table[N+2][N+2];
	int i,j,count=0;

	srand((unsigned) time(NULL));		//init seed

	//init first and last row and column to 0 else to random
	for(i=0;i<N+2;i++)
		for(j=0;j<N+2;j++)
			if(i == 0 || i == N+1 || j == 0 || j == N+1)	temp_table[i][j] = 0;
			else		temp_table[i][j] = rand() % 2;

	//copy this table to current table
	for(i=0;i<N+2;i++)
		for(j=0;j<N+2;j++)
		{
			current_table[count] = temp_table[i][j];
			count++;
		}
}

void print_table(long *t)
{
	int i;
	for(i=0;i<(N+2)*(N+2);i++)
	{
		if(i%(N+2) == 0) printf("\n");
		printf("%d",t[i] );
	}
}