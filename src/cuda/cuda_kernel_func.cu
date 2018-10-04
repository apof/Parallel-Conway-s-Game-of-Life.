#include "header.cuh"
#include <stdio.h>

__global__ void updateVirtualRows(long* currentTable)
{
	long index_x =  threadIdx.x + blockIdx.x * blockDim.x;

	if(index_x < (N+1))		//only threads required do the job
	{
		//copy bottom line to top
		currentTable[(N+1)*(N+2)+index_x] = currentTable[N+2+index_x];
		//copy top line to bottom
		currentTable[index_x] = currentTable[N*(N+2)+index_x];
	}
}

__global__ void updateVirtualColumns(long* currentTable)
{
	long index_y = threadIdx.x + blockIdx.x * blockDim.x;

	if(index_y < (N+1))		//only threads required do the job
	{
		//copy left line to right
		currentTable[index_y*(N+2)+N+1] = currentTable[index_y*(N+2)+1];
		//copy right line to left
		currentTable[index_y*(N+2)] = currentTable[index_y*(N+2)+N];
	}
}

__global__ void updateVirtualCorners(long* currentTable)
{
	currentTable[0] = currentTable[N*(N+2)+N];
	currentTable[(N+1)*(N+2)+N+1] = currentTable[N+3];
	currentTable[N+1] = currentTable[N*(N+2)+1];
	currentTable[(N+1)*(N+2)] = currentTable[2*N+2];
}


__global__ void calculateNextGen(long* currentTable, long* nextTable)
{
	size_t i,j;

	size_t offset1 = threadIdx.y / (BLSIZE - 1);								//index to traverse the localTable
    size_t offset2 = threadIdx.x / (BLSIZE - 1);								//index to traverse the localTable
    size_t x = (blockIdx.y * blockDim.y + threadIdx.y) * CELLS_PER_THREAD ;		//index to the globalTable
    size_t y = (blockIdx.x * blockDim.x + threadIdx.x) * CELLS_PER_THREAD ;		//index to the globalTable
    size_t Y = threadIdx.y * CELLS_PER_THREAD;									//index to the localTable
    size_t X = threadIdx.x * CELLS_PER_THREAD;									//index to the localTable

    //define the grid of this block
	__shared__ long local_table[BLSIZE*CELLS_PER_THREAD+2][BLSIZE*CELLS_PER_THREAD+2];

    //Copy the suitable part of the globalTable to the localTable
    //The size of the localTable is BLSIZE*CELLS_PER_THREAD+2 x BLSIZE*CELLS_PER_THREAD+2 (+2 for the virtual neighbours)
    for ( i = 0; i < CELLS_PER_THREAD + 1; i++){

      	size_t in2 = (x + offset1 + i ) * (N + 2);//The global grid has a N+2 edge
      	for ( j = 0; j < CELLS_PER_THREAD + 1; j++)
      	{
      	  size_t in1 = y + j + offset2;
      	  local_table[Y + offset1 + i][X + offset2 + j] = currentTable[in2 + in1];
      	}
    }


      //All threads should have reached this point in order to continue
      //Syncthreads() is used to guarantee that whole localTable has been filled uπ 
	__syncthreads();


	//Calculation of the next generation status

	int ii,jj;
	for (i= 1; i < CELLS_PER_THREAD + 1; i++)
    {
      ii = Y + i;
      size_t new_index_y = (x + i)*(N + 2);
      //each thread computes CELLS_PER_THREAD elements of the next_generation_table
      for (j = 1; j < CELLS_PER_THREAD + 1; j++)		
      {
        jj = X + j;
        int livingNeighbors = local_table[ii - 1][jj - 1] + local_table[ii - 1][jj]
          + local_table[ii - 1][jj + 1] + local_table[ii][jj - 1]
          + local_table[ii][jj + 1] + local_table[ii + 1][jj - 1] + local_table[ii + 1][jj]
          + local_table[ii + 1][jj + 1];

        size_t new_index_x = y + j;

        if (x < N + 1 && y < N + 1)			//only threads required do the job
        {
        	if(livingNeighbors == 3 || (livingNeighbors == 2 && local_table[ii][jj] == 1))
				nextTable[new_index_y + new_index_x] = 1;
			else nextTable[new_index_y + new_index_x] = 0;

 		}

      }
    }


}
