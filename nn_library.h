

#define MAXCOM 1000
#define MAXLIST 100

#define LOGISTIC 	0
#define TANH 		1
#define LINEAR  	2
#define D_LOGISTIC 	3
#define D_TANH 		4
#define D_LINEAR  	5

// number of examples
extern int N;
// number of batches
extern int B;

typedef struct net_structure_s
{
	int input_nodes;
	int output_nodes;
	int H1_nodes;
	int H2_nodes;
	int H2_activationF;
} net_structure_t;

typedef struct example_s
{
	float* x;
	char category[12];
}example_t;


// Global 
extern example_t* 	train_examples;	
extern example_t* 	test_examples;
extern matrix_t*	target_vector;

extern matrix_t		weights_i_h1;
extern matrix_t		weights_h1_h2;
extern matrix_t		weights_h2_o;

extern matrix_t 	weights_i_h1_delta;
extern matrix_t 	weights_h1_h2_delta;
extern matrix_t 	weights_h2_o_delta;

extern matrix_t 	bias_h1;
extern matrix_t 	bias_h2;
extern matrix_t 	bias_o;

extern matrix_t 	bias_h1_delta;
extern matrix_t 	bias_h2_delta;
extern matrix_t 	bias_o_delta;

extern double		total_error;
extern float 		learning_rate;
extern float		termination_threshold;

extern net_structure_t net_str;	
#define NUM_of_inputs	net_str.input_nodes
#define NUM_of_outputs	net_str.output_nodes
#define NUM_of_H1		net_str.H1_nodes
#define NUM_of_H2		net_str.H2_nodes

void set_net_structure_to_default();
void takeInput(char* line);
int parseSpace(char* line, char** parsedLine);
void command_exec(int argc, char** parsedLine);
void print_net_ctructure();
void define_target_vectors(matrix_t** categories_mat);
matrix_t to_matrix(float* x, int d);
void set_num_of_examples(int num);
void set_num_of_batches(int num);
void update_total_error(matrix_t error);
void print_weights();
void print_bias();
void print_learning_rate();
void print_termination_threashold();
void print_num_of_batch();
void print_num_of_examples();


