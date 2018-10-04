#include "mpi.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define N 32
#define GENERATION_NUM 100

int main(int argc, char **argv)
{
	double local_start, local_finish, local_elapsed, elapsed;		//parameters for time calculation
    int myid,numprocs,proc_dim,rank,i,j;
	int up,down,left,right,up_right,up_left,down_right,down_left;
	int ndims=2,reorder,periods[2],dim_size[2],coords[2];
	int tag=25,*message1,*message2,*message_down,*message_up,end_flag;
	time_t t;
    MPI_Status recv_status, send_status;
    MPI_Request request1,request2,request3,request4,request5,request6,request7,request8,request9,request10,request11,request12,request13,request14,request15,request16;
	MPI_Comm old_comm,new_comm;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	srand((unsigned) time(NULL)+myid);
	proc_dim = N/sqrt(numprocs);

	int **local_table,**temp_table,n,m;
	n = proc_dim+2;
	m = proc_dim+2;
	local_table = malloc(n * sizeof(int *));
  	int *temp = malloc(n * m * sizeof(int));
  	for(i = 0; i < n; ++i)
  	 local_table[i] = temp + i * m;

	/*New Datatype for column*/
	MPI_Datatype col_type;
	MPI_Type_vector(n-2, 1, n, MPI_INT,&col_type);
    	MPI_Type_commit(&col_type);

	message_down = malloc((proc_dim)*sizeof(int));
	message_up = malloc((proc_dim)*sizeof(int));

	/*Virtual Topologie Settings*/
    dim_size[0] = sqrt(numprocs);
    dim_size[1] = dim_size[0];
	periods[0] = 1;
	periods[1] = 1;
	MPI_Cart_create(MPI_COMM_WORLD,ndims,dim_size,periods,reorder,&new_comm);

	/***********Start Timer***********/
    MPI_Barrier(new_comm);		//wait all processes to reach this line
    local_start = MPI_Wtime();
    /*********************************/

	/*Local Table Initialize*/
	for(i=0;i<proc_dim+2;i++)
	for(j=0;j<proc_dim+2;j++)
	{
	 	if((i==0) || (j==0) || (i==proc_dim+1) || (j==proc_dim+1))	local_table[i][j] = -1;
		else	local_table[i][j] = rand() % 2;
	}

	if(myid==0)
	{
		for(i=0;i<dim_size[0];i++)
		{
			for(j=0;j<dim_size[1];j++)
			{
				coords[0]=i;
				coords[1]=j;
				MPI_Cart_rank(new_comm,coords,&rank);
				printf("%d,%d:%d     ",coords[0],coords[1],rank);
			}
			printf("\n");
		}
	}

	/*Find Neighbours*/
	
	MPI_Cart_shift(new_comm,1,1,&left,&right);	//left,right neighbours
	MPI_Cart_shift(new_comm,0,1,&up,&down);		//up,down neighbours
	MPI_Send(&left, 1, MPI_INT, down, tag, new_comm);
	MPI_Recv(&up_left, 1, MPI_INT, up, tag, new_comm, &recv_status);	//get up_right and up_left from up neighbour
	MPI_Send(&right, 1, MPI_INT, down, tag, new_comm);
	MPI_Recv(&up_right, 1, MPI_INT, up, tag, new_comm, &recv_status);
	MPI_Send(&left, 1, MPI_INT, up, tag, new_comm);
    MPI_Recv(&down_left, 1, MPI_INT, down, tag, new_comm, &recv_status);	//get down_right and down_left from down neighbour
	MPI_Send(&right, 1, MPI_INT, up, tag, new_comm);
    MPI_Recv(&down_right, 1, MPI_INT, down, tag, new_comm, &recv_status);
	MPI_Barrier(new_comm);			//ensure that all have recieve neighbors

	printf("myid=%d\n",myid); 

	for(i=0;i<GENERATION_NUM;i++)
	{
	/*Send to neighbours and Receive*/
	int ur,ul,dr,dl;
	ur=local_table[1][proc_dim];
	ul=local_table[1][1];
	dr=local_table[proc_dim][proc_dim];
	dl=local_table[proc_dim][1];
	MPI_Isend(&local_table[1][1], proc_dim, MPI_INT, up, tag,new_comm, &request1);
    MPI_Irecv(&local_table[proc_dim+1][1], proc_dim, MPI_INT, down, tag,new_comm, &request2);
	MPI_Isend(&local_table[proc_dim][1], proc_dim, MPI_INT, down, tag,new_comm, &request3);
    MPI_Irecv(&local_table[0][1], proc_dim, MPI_INT, up, tag,new_comm, &request4);
	MPI_Isend(&local_table[1][1], 1, col_type, left, tag,new_comm, &request5);
    MPI_Irecv(&local_table[1][proc_dim+1], 1, col_type, right, tag,new_comm, &request6);
	MPI_Isend(&local_table[1][proc_dim], 1, col_type, right, tag,new_comm, &request7);
    MPI_Irecv(&local_table[1][0], 1, col_type, left, tag,new_comm, &request8);
	MPI_Isend(&ur, 1, MPI_INT, up_right, tag,new_comm, &request9);
    MPI_Irecv(&local_table[proc_dim+1][0], 1, MPI_INT, down_left, tag,new_comm, &request10);
	MPI_Isend(&ul, 1, MPI_INT, up_left, tag,new_comm, &request11);
    MPI_Irecv(&local_table[proc_dim+1][proc_dim+1], 1, MPI_INT, down_right, tag,new_comm, &request12);
	MPI_Isend(&dr, 1, MPI_INT, down_right, tag,new_comm, &request13);
    MPI_Irecv(&local_table[0][0], 1, MPI_INT, up_left, tag,new_comm, &request14);
	MPI_Isend(&dl, 1, MPI_INT, down_left, tag,new_comm, &request15);
    MPI_Irecv(&local_table[0][proc_dim+1], 1, MPI_INT, up_right, tag,new_comm, &request16);
/*Calculate next generation for inner table*/
//if(myid==4)print_table(local_table,proc_dim+2);
	temp_table = sub_table_next_generation(local_table,proc_dim+2);
//if(myid==4)print_table(local_table,proc_dim+2);
	/*Wait for Isend and Irecv to finish*/
	MPI_Wait(&request1, &send_status);
    MPI_Wait(&request2, &send_status);
	MPI_Wait(&request3, &send_status);
    MPI_Wait(&request4, &send_status);
	MPI_Wait(&request5, &send_status);
    MPI_Wait(&request6, &send_status);
	MPI_Wait(&request7, &send_status);
    MPI_Wait(&request8, &send_status);
	MPI_Wait(&request9, &send_status);
	MPI_Wait(&request10, &send_status);
	MPI_Wait(&request11, &send_status);
	MPI_Wait(&request12, &send_status);
	MPI_Wait(&request13, &send_status);
	MPI_Wait(&request14, &send_status);
	MPI_Wait(&request15, &send_status);
	MPI_Wait(&request16, &send_status);
//	printf("Process %d received\n", myid);
//	printf("Up\n");for(i=0;i<proc_dim;i++)	printf("%d",message_up[i]);printf("\n");
//	printf("Down\n");for(i=0;i<proc_dim;i++) printf("%d",message_down[i]);printf("\n");
	int table_changed;				//flag for next generation table status
	
	round_table_next_generation(local_table,temp_table,proc_dim+2);
	//table_changed = check(local_table,temp_table,proc_dim+2);
	copy(local_table,temp_table);		///PROSOXIIIIII EDWWWW SPATALA XRONO!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//MPI_Allreduce(&table_changed,&end_flag,1,MPI_INT,MPI_SUM,new_comm);
	//if(end_flag == 0)	break;

	/*if(myid==4)
	{
		printf("myid=%d\n",myid); 
		print_table(local_table,proc_dim+2);
		print_table(temp_table,proc_dim+2);
		printf("status = %d\n",table_changed);
		printf("end_flag = %d\n",end_flag);
	}*/
	}
	/***********Finish Timer***********/
	local_finish = MPI_Wtime();
	local_elapsed = local_finish - local_start;
	MPI_Reduce(&local_elapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,new_comm);
	/*********************************/
	if(myid == 0)	printf("Elapsed Time = %e seconds\n", elapsed);





MPI_Finalize();
}
