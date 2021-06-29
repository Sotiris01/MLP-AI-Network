#define LOGISTIC 	0
#define TANH 		1
#define LINEAR  	2
#define D_LOGISTIC 	3
#define D_TANH 		4
#define D_LINEAR  	5

typedef struct matrix_s
{
	int rows;
	int cols;
	double** data;
} matrix_t;

matrix_t new_matrix(int rows, int cols);
matrix_t mul_num_m(matrix_t A, matrix_t B);
matrix_t mul_mat(matrix_t A, matrix_t B);
matrix_t sub_mat(matrix_t A, matrix_t B);
matrix_t transpose(matrix_t matrix);
void delete_matrix(matrix_t* matrix);
void randomize(matrix_t* matrix);
void initialize(matrix_t* matrix, float num);
void print_matrix(matrix_t matrix);
void add_num(matrix_t* matrix, float num);
void add_mat(matrix_t* matrix, matrix_t B);
void sub_num(matrix_t* matrix, float num);
void mul_num(matrix_t* matrix, float num);
void ActivationFunction(matrix_t* matrix, int activationFunction);
double sum_mat(matrix_t matrix);

int save_mat(matrix_t matrix,char* matrix_name);
int load_mat(matrix_t* matrix, char* matrix_name);

void sub_mat_m(matrix_t* matrix, matrix_t B);