#include <stdio.h>
#include <stdlib.h>

void print_table(int **lt,int size)
{
	int i,j;
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
	 	{
			printf("%2d",lt[i][j]);
		}
		printf("\n");
	}
}

void copy(int **t1,int **t2,int size)
{
	int i,j;
	for(i=0;i<size;i++)
	 for(j=0;j<size;j++)
	  t1[i][j] = t2[i][j];
}

int check(int **lt, int **next_gen_table,int size)
{
	int i,j;
	for(i=1;i<size-1;i++)
	 for(j=1;j<size-1;j++)
	   if(lt[i][j] != next_gen_table[i][j]) return 1;
	return 0;
}

int** sub_table_next_generation(int **lt,int size)
{
	int i,j,neighbours;
	int **temp;
	temp = malloc(size*sizeof(int *));
	for(i=0;i<size;i++)	temp[i] = malloc(size*sizeof(int));
	copy(temp,lt,size);
	for(i=2;i<size-2;i++)
	{
		for(j=2;j<size-2;j++)
		{
			neighbours = 0;
			neighbours = lt[i-1][j-1] + 
			lt[i-1][j] + lt[i-1][j+1] +
			lt[i][j-1] + lt[i][j+1] +
			lt[i+1][j-1] + lt[i+1][j] +
			lt[i+1][j+1];
			
			if(lt[i][j] == 1)
			{
				if((neighbours == 0) || (neighbours == 1) || (neighbours>3))  temp[i][j] = 0;
			}
			else if((lt[i][j] == 0) && (neighbours == 3))  temp[i][j] = 1;
		}
	}
	return temp;
}

void round_table_next_generation(int **lt,int **temp,int size)
{
	int i,j,neighbours;
	for(i=1;i<size-1;i+=(size-3))
	{
		for(j=1;j<size-1;j++)
		{
		 	neighbours = 0;
			neighbours = lt[i-1][j-1] + 
			lt[i-1][j] + lt[i-1][j+1] +
			lt[i][j-1] + lt[i][j+1] +
			lt[i+1][j-1] + lt[i+1][j] +
			lt[i+1][j+1];
             
			if(lt[i][j] == 1)
            {
				if((neighbours == 0) || (neighbours == 1) || (neighbours>3))  temp[i][j] = 0;
            }
            else if((lt[i][j] == 0) && (neighbours == 3))  temp[i][j] = 1;
		}
	}
	for(j=1;j<size-1;j+=(size-3))
        {
                for(i=2;i<size-2;i++)
                {
                        neighbours = 0;
						neighbours = lt[i-1][j-1] + 
						lt[i-1][j] + lt[i-1][j+1] +
						lt[i][j-1] + lt[i][j+1] +
						lt[i+1][j-1] + lt[i+1][j] +
						lt[i+1][j+1];
                       
                        if(lt[i][j] == 1)
                        {
                               if((neighbours == 0) || (neighbours == 1) || (neighbours>3))  temp[i][j] = 0;
                        }
                        else if((lt[i][j] == 0) && (neighbours == 3)) temp[i][j] = 1;
                }
        }
}
