#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Matrix.h"

#define TRUE  1
#define FALSE 0

#define TEST FALSE

#define RANDOM_NUM ((double)rand()/(double)(RAND_MAX))


#define logistic(x) 	(1/(1+exp(-(x))))
#define Dlogistic(x) 	((x)*(1-(x)))	
// #define tanh(x) 		tanh(x)
#define Dtanh(x) 		(1-((x)*(x)))
#define linear(x) 		(x)
#define Dlinear(x) 		(1)

#if TEST
int main(){

	matrix_t A = new_matrix(2,2);
	matrix_t B = new_matrix(2,2);
	A.data[0][0]=0.555;
	A.data[0][1]=-0.666;
	A.data[1][0]=0.777;
	A.data[1][1]=-0.888;

	B.data[0][0]=0.022;
	B.data[0][1]=0.033;
	B.data[1][0]=-0.044;
	B.data[1][1]=0.055;

	matrix_t C = new_matrix(2,2);
	matrix_t D = new_matrix(2,2);
	C.data[0][0]=2;
	C.data[0][1]=2;
	C.data[1][0]=2;
	C.data[1][1]=2;

	D.data[0][0]=0.5;
	D.data[0][1]=0.5;
	D.data[1][0]=0.5;
	D.data[1][1]=0.5;
	
	print_matrix(C);
	printf("---------------\n");

	save_mat(C, "matrix_c");

	load_mat(&D, "matrix_c");
	print_matrix(D);

	// matrix_t mat1 = mul_mat(C,D);
	// matrix_t mat2 = mul_num_m(C,D);
	// print_matrix(mat1);
	// print_matrix(mat2);

	printf("---------------\n");
	// delete_matrix(&mat);
	// mat = new_matrix(2,2);
	// print_matrix(mat);

	return 0;}
#endif

matrix_t new_matrix(int rows, int cols){

	matrix_t matrix;

	matrix.rows = rows;
	matrix.cols = cols;

	matrix.data = (double **) malloc(rows * sizeof(double*));
	for (int i = 0; i < rows; ++i)
		matrix.data[i] = (double *) calloc(cols, cols*sizeof(double)); 

	return matrix;
}

void delete_matrix(matrix_t* matrix){
	for (int i = 0; i < matrix->rows; ++i)
		free(matrix->data[i]);
	free(matrix->data);
	matrix->data = NULL;
}

void randomize(matrix_t* matrix){
	for (int i = 0; i < matrix->rows; ++i)
		for (int j = 0; j < matrix->cols; ++j)
			matrix->data[i][j] = RANDOM_NUM * 2 - 1;
}

void initialize(matrix_t* matrix, float num){
	for (int i = 0; i < matrix->rows; ++i)
		for (int j = 0; j < matrix->cols; ++j)
			matrix->data[i][j] = num;	
}

void print_matrix(matrix_t matrix){
	printf("\n|");
	for (int i = 0; i < matrix.rows; ++i){
		for (int j = 0; j < matrix.cols; ++j)
			printf("%7.3f", matrix.data[i][j]);
		printf("\n|");
	}	
	printf("\n");
}

void add_num(matrix_t* matrix, float num){
	for (int i = 0; i < matrix->rows; ++i)
		for (int j = 0; j < matrix->cols; ++j)
			matrix->data[i][j] += num;	
}

void add_mat(matrix_t* matrix, matrix_t B){
	if(matrix->cols != B.cols || matrix->rows != B.rows){
		printf("ERROR: These matrices could not be added\n");
		exit(0);
	}
	for (int i = 0; i < matrix->rows; ++i)
		for (int j = 0; j < matrix->cols; ++j)
			matrix->data[i][j] += B.data[i][j];	
}

void sub_num(matrix_t* matrix, float num){
	for (int i = 0; i < matrix->rows; ++i)
		for (int j = 0; j < matrix->cols; ++j)
			matrix->data[i][j] -= num;	
}

void sub_mat_m(matrix_t* matrix, matrix_t B){
	if(matrix->cols != B.cols || matrix->rows != B.rows){
		printf("ERROR: These matrices could not be added\n");
		exit(0);
	}
	for (int i = 0; i < matrix->rows; ++i)
		for (int j = 0; j < matrix->cols; ++j)
			matrix->data[i][j] -= B.data[i][j];	
}

matrix_t sub_mat(matrix_t A, matrix_t B){
	if(A.cols != B.cols || A.rows != B.rows){
		printf("ERROR: These matrices could not be subtracted\n");
		exit(0);
	}

	matrix_t matrix = new_matrix(A.rows, A.cols);

	for (int i = 0; i < matrix.rows; ++i)
		for (int j = 0; j < matrix.cols; ++j)
			matrix.data[i][j] = A.data[i][j] - B.data[i][j];	

	return matrix;
}


void mul_num(matrix_t* matrix, float num){
	for (int i = 0; i < matrix->rows; ++i)
		for (int j = 0; j < matrix->cols; ++j)
			matrix->data[i][j] *= num;	
}

matrix_t mul_num_m(matrix_t A, matrix_t B){
	if(A.cols != B.cols || A.rows != B.rows){
		printf("ERROR: <mul_num_m()> Matrix.c\n");
		exit(0);
	}

	matrix_t matrix = new_matrix(A.rows, A.cols);

	for (int i = 0; i < matrix.rows; ++i)
		for (int j = 0; j < matrix.cols; ++j)
			matrix.data[i][j] = A.data[i][j] * B.data[i][j];	

	return matrix;
}

matrix_t mul_mat(matrix_t A, matrix_t B){

	if(A.cols != B.rows){
		printf("ERROR: Columns of A must match trows of B.\n");
		exit(0);
	}

	matrix_t matrix = new_matrix(A.rows, B.cols);

	for(int i=0; i<A.rows; ++i)
		for(int j=0; j<B.cols; ++j)
			for(int k=0; k<A.cols; ++k)
				matrix.data[i][j] += A.data[i][k] * B.data[k][j];

	return matrix;
}

double sum_mat(matrix_t matrix){
	double sum = 0.0;

	for (int i = 0; i < matrix.rows; ++i)
		for (int j = 0; j < matrix.cols; ++j)
			sum += matrix.data[i][j];	
	return sum;
}

matrix_t transpose(matrix_t matrix){

	matrix_t t_matrix = new_matrix(matrix.cols, matrix.rows);

	for (int i = 0; i < matrix.rows; ++i)
		for (int j = 0; j < matrix.cols; ++j)
			t_matrix.data[j][i] = matrix.data[i][j];

	return t_matrix;
}
int save_mat(matrix_t matrix,char* matrix_name){

	FILE *fp;

	if ((fp = fopen(matrix_name, "w")) == NULL)
		return -1;

	fprintf(fp, "%d\n", matrix.rows);
	fprintf(fp, "%d\n", matrix.cols);

	for (int i = 0; i < matrix.rows; ++i)
		for (int j = 0; j < matrix.cols; ++j)
			fprintf(fp, "%lf\n", matrix.data[i][j]);

	fclose(fp);

	return 0;
}

int load_mat(matrix_t* matrix, char* matrix_name){

	FILE *fp;

	if((fp = fopen(matrix_name, "r")) == NULL)
		return -1;

	delete_matrix(matrix);

	int rows = 0;
	int cols = 0;

	fscanf(fp, "%d", &rows);
	fscanf(fp, "%d", &cols);

	*matrix = new_matrix(rows, cols);

	for (int i = 0; i < matrix->rows; ++i)
		for (int j = 0; j < matrix->cols; ++j)
			fscanf(fp, "%lf\n", &matrix->data[i][j]);

	fclose(fp);

	return 0;
}

void ActivationFunction(matrix_t* matrix, int activationFunction){
	for (int i = 0; i < matrix->rows; ++i)
		for (int j = 0; j < matrix->cols; ++j)
			switch (activationFunction){
				case LOGISTIC:
					matrix->data[i][j] = logistic(matrix->data[i][j]);	
					break;
				case TANH:
					matrix->data[i][j] = tanh(matrix->data[i][j]);
					break;
				case LINEAR:
					matrix->data[i][j] = linear(matrix->data[i][j]);
					break;
				case D_LOGISTIC:
					matrix->data[i][j] = Dlogistic(matrix->data[i][j]);
					break;
				case D_TANH:
					matrix->data[i][j] = Dtanh(matrix->data[i][j]);  
					break;
				case D_LINEAR:
					matrix->data[i][j] = Dlinear(matrix->data[i][j]);
					break;
				default:
					printf("No activation Function found\n");
					exit(0);
			}
			
}
