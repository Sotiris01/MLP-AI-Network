#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include "Matrix.h"
#include "nn_library.h"
#include "create_examples.h"
#include "load_examples.h"

#define clear() printf("\033[H\033[J");
#define cleanScanf() char x;scanf("%c", &x);
#define command(i,c) (!strcmp(parsedLine[i],c))
#define MC(x) (int)((x).category[1])-48-1
#define TARGET(x) target_vector[MC(x)]

#define MAX_TIMES 10000

int new_train = 0;
int ctrC = 0;
void ctrChandler(){ctrC = 1;}

// number of examples
int N = 6000;
// number of batches
int B = 1;


// Global 
net_structure_t net_str;

example_t* 	train_examples;	
example_t* 	test_examples;
matrix_t*	target_vector;

matrix_t	weights_i_h1;
matrix_t	weights_h1_h2;
matrix_t	weights_h2_o;

matrix_t 	weights_i_h1_delta;
matrix_t 	weights_h1_h2_delta;
matrix_t 	weights_h2_o_delta;

matrix_t 	bias_h1;
matrix_t 	bias_h2;
matrix_t 	bias_o;

matrix_t 	bias_h1_delta;
matrix_t 	bias_h2_delta;
matrix_t 	bias_o_delta;

double		total_error = 0.0;
float 		learning_rate = 0.001;
float		termination_threshold = 0.01;


void command_exec(int argc, char** parsedLine);
void forward_pass(float* x, int d, matrix_t* output);
void backprop(float* x, int d, matrix_t target);
void init_weights_end_bias();
void update_weights_and_bias();
void delete_weights_end_bias();
void save_weights_end_bias();
void load_weights_end_bias();
void print_weights();
void print_bias();
int predict(example_t example);

int main(){
	srand(time(0));  
	signal(SIGINT, ctrChandler);

	set_num_of_batches(1);
	// set_num_of_batches(N/10);
	// set_num_of_batches(N/100);
	// set_num_of_batches(N);

	clear();

	char inputLine[MAXCOM], *parsedLine[MAXLIST];
	int argc;

	Create_train_examples();
	Create_test_examples();

	set_net_structure_to_default();

	load_train_examples(&train_examples);
	load_test_examples(&test_examples);

	define_target_vectors(&target_vector);

	while(1){

		takeInput(inputLine);
		argc = parseSpace(inputLine, parsedLine);
		command_exec(argc, parsedLine);
	}
	return 0;
}

void forward_pass(float* x, int d, matrix_t* output){
	matrix_t input  = to_matrix(x, d);

	matrix_t H1 = mul_mat(weights_i_h1, input);
	add_mat(&H1,bias_h1);
	ActivationFunction(&H1, TANH);
	delete_matrix(&input);
	
	matrix_t H2 = mul_mat(weights_h1_h2, H1);
	add_mat(&H2,bias_h2);
	ActivationFunction(&H2, net_str.H2_activationF);
	delete_matrix(&H1);

 	*output = mul_mat(weights_h2_o, H2);
	add_mat(output,bias_o);
	ActivationFunction(output, LOGISTIC);
	delete_matrix(&H2);
}

void backprop(float* x, int d, matrix_t target){

	// ~~ forward pass ~~ //

	// INPUT output
	matrix_t input  = to_matrix(x, d);
	// print_matrix(input);

	// HIDEN L1 output
	matrix_t H1 = mul_mat(weights_i_h1, input);
	// print_matrix(H1);
	add_mat(&H1,bias_h1);
	ActivationFunction(&H1, TANH);
	// print_matrix(H1);

	// HIDED L2 output
	matrix_t H2 = mul_mat(weights_h1_h2, H1);
	add_mat(&H2,bias_h2);
	ActivationFunction(&H2, net_str.H2_activationF);

	// total output
	matrix_t output = mul_mat(weights_h2_o, H2);
	add_mat(&output,bias_o);
	ActivationFunction(&output, LOGISTIC);

	// ~~ backward pass ~~ //

	// layer H+1 (output) error
	matrix_t output_error = sub_mat(target, output);
	update_total_error(output_error);

	ActivationFunction(&output, D_LOGISTIC);
	// calculate H+1 layer gradient
	matrix_t output_gradient = mul_num_m(output, output_error);
	delete_matrix(&output_error);

	matrix_t H2_T = transpose(H2);
	// calculate new L2 delta
	matrix_t new_delta = mul_mat(output_gradient, H2_T);
	add_mat(&weights_h2_o_delta, new_delta);
	add_mat(&bias_o_delta, output_gradient);
	
	delete_matrix(&H2_T);
	delete_matrix(&new_delta);

	matrix_t weights_h2_o_T = transpose(weights_h2_o);
	// layer L2 error
	matrix_t H2_error = mul_mat(weights_h2_o_T, output_gradient);

	delete_matrix(&weights_h2_o_T);
	delete_matrix(&output_gradient);

	ActivationFunction(&H2, (net_str.H2_activationF == LOGISTIC)?D_LOGISTIC:D_LINEAR);
	// calculate L2 layer gradient
	matrix_t H2_gradient = mul_num_m(H2, H2_error);
	delete_matrix(&H2_error);

	matrix_t H1_T = transpose(H1);
	// calculate new L1 delta
 	new_delta = mul_mat(H2_gradient, H1_T);
	add_mat(&weights_h1_h2_delta, new_delta);
	add_mat(&bias_h2_delta, H2_gradient);

	delete_matrix(&H1_T);
	delete_matrix(&new_delta);
	
	matrix_t weights_h1_h2_T = transpose(weights_h1_h2);
	// layer L1 error
	matrix_t H1_error = mul_mat(weights_h1_h2_T, H2_gradient);

	delete_matrix(&weights_h1_h2_T);
	delete_matrix(&H2_gradient);

	ActivationFunction(&H1, D_TANH);
	// calculate L1 leyer gradient
	matrix_t H1_gradient = mul_num_m(H1, H1_error);
	delete_matrix(&H1_error);

	matrix_t input_T = transpose(input);
	// calculate new input layer delta
	new_delta = mul_mat(H1_gradient, input_T);
	add_mat(&weights_i_h1_delta, new_delta);
	add_mat(&bias_h1_delta, H1_gradient);

	delete_matrix(&input_T);
	delete_matrix(&new_delta);

	delete_matrix(&H1_gradient);

	/// more deletes
	delete_matrix(&input);
	delete_matrix(&H1);
	delete_matrix(&H2);
	delete_matrix(&output);

}

void test(){
	int sum = 0;
	int prediction;
	double corrects;

	FILE	*Correct_fp;
	FILE	*Incorrect_fp;
	Correct_fp = fopen("Correct.txt", "w");
	Incorrect_fp = fopen("Incorrect.txt", "w");

	if(! new_train)
		load_weights_end_bias();
	new_train = 0;

	for (int i = 0; i < N/2; ++i){
		prediction = predict(test_examples[i]);
		sum += prediction;
		if(prediction)
			fprintf(Correct_fp, "%f\t%f\n", test_examples[i].x[0], test_examples[i].x[1]);
		else
			fprintf(Incorrect_fp, "%f\t%f\n", test_examples[i].x[0], test_examples[i].x[1]);
	}

	corrects = (double)sum / (double)(N/2);
	printf("\n\tCorrects -> %.2f%%\n", corrects*100);

	char * commandsForGnuplot[] = {	"set title \"Results\"", 
									"plot	'Correct.txt' pt 1 lc 2,\
											'Incorrect.txt' pt 2 lc 7"};
	FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
	for (int i=0; i < 2; i++)
		fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); //Send commands to gnuplot one by one.

	fclose(Correct_fp);
	fclose(Incorrect_fp);
	fclose(gnuplotPipe);
}

int predict(example_t example){
	matrix_t output;
	int max = 0;
	double tmax = 0.0;

	forward_pass(example.x, NUM_of_inputs, &output);

	for (int i = 0; i < NUM_of_outputs; ++i)
	{
		if(output.data[i][0] > tmax){
			tmax = output.data[i][0];
			max = i;
		}
	}

	if(TARGET(example).data[max][0])
		return 1; // Correct
	else
		return 0; // Incorrect

}
void train(){
	ctrC=0;
	init_weights_end_bias();
	double p_error = total_error;

	for (int i = 0; i < MAX_TIMES; ++i){
		for (int j = 0; j < N/2; ++j){
			example_t next_example = train_examples[j];	

			backprop(next_example.x, NUM_of_inputs, TARGET(next_example));

			// update weights end biases
			if(((j+1) % B) == 0)
				update_weights_and_bias();
		}

		if(i>500)
		if(fabs(p_error - total_error) < termination_threshold)
			break;

		printf("T = %3d: total error ->  %lf\n", i+1, total_error);
		p_error = total_error;
		total_error = 0.0;	

		if(ctrC)break;	
	} 
	new_train = 1;
	test();
}

void update_total_error(matrix_t error){

	matrix_t error_squared = mul_num_m(error, error);

	total_error += sum_mat(error_squared) / 2;

	delete_matrix(&error_squared);
}

void init_weights_end_bias(){
	weights_i_h1 = new_matrix(NUM_of_H1, NUM_of_inputs);
	randomize(&weights_i_h1);
	weights_h1_h2 = new_matrix(NUM_of_H2, NUM_of_H1);
	randomize(&weights_h1_h2);
	weights_h2_o = new_matrix(NUM_of_outputs, NUM_of_H2);
	randomize(&weights_h2_o);

	weights_i_h1_delta = new_matrix(NUM_of_H1, NUM_of_inputs);
	weights_h1_h2_delta = new_matrix(NUM_of_H2, NUM_of_H1);
	weights_h2_o_delta = new_matrix(NUM_of_outputs, NUM_of_H2);


	bias_h1 = new_matrix(NUM_of_H1, 1);
	randomize(&bias_h1);
	bias_h2 = new_matrix(NUM_of_H2, 1);
	randomize(&bias_h2);
	bias_o = new_matrix(NUM_of_outputs, 1);
	randomize(&bias_o);

	bias_h1_delta = new_matrix(NUM_of_H1, 1);
	bias_h2_delta = new_matrix(NUM_of_H2, 1);
	bias_o_delta = new_matrix(NUM_of_outputs, 1);
}

void update_weights_and_bias(){
	
	mul_num(&weights_h2_o_delta, learning_rate);
	add_mat(&weights_h2_o, weights_h2_o_delta);
	// print_matrix(weights_h2_o);
	mul_num(&bias_o_delta, learning_rate);
	add_mat(&bias_o, bias_o_delta);

	mul_num(&weights_h1_h2_delta, learning_rate);
	add_mat(&weights_h1_h2, weights_h1_h2_delta);
	// print_matrix(weights_h1_h2);
	mul_num(&bias_h2_delta, learning_rate);
	add_mat(&bias_h2, bias_h2_delta);

	mul_num(&weights_i_h1_delta, learning_rate);
	add_mat(&weights_i_h1, weights_i_h1_delta);
	// print_matrix(weights_i_h1);
	mul_num(&bias_h1_delta, learning_rate);
	add_mat(&bias_h1, bias_h1_delta);

	initialize(&weights_h2_o_delta, 0);
	initialize(&weights_h1_h2_delta, 0);
	initialize(&weights_i_h1_delta, 0);

	initialize(&bias_o_delta, 0);
	initialize(&bias_h2_delta, 0);
	initialize(&bias_h1_delta, 0);
}

void delete_weights_end_bias(){

	delete_matrix(&weights_i_h1);
	delete_matrix(&weights_h1_h2);
	delete_matrix(&weights_h2_o);

	delete_matrix(&weights_i_h1_delta);
	delete_matrix(&weights_h1_h2_delta);
	delete_matrix(&weights_h2_o_delta);


	delete_matrix(&bias_h1);
	delete_matrix(&bias_h2);
	delete_matrix(&bias_o);

	delete_matrix(&bias_h1_delta);
	delete_matrix(&bias_h2_delta);
	delete_matrix(&bias_o_delta);
}

void save_weights_end_bias(){
	save_mat(weights_i_h1, "memory/weights_i_h1.matrix");
	save_mat(weights_h1_h2, "memory/weights_h1_h2.matrix");
	save_mat(weights_h2_o, "memory/weights_h2_o.matrix");

	save_mat(bias_h1, "memory/bias_h1.matrix");
	save_mat(bias_h2, "memory/bias_h2.matrix");
	save_mat(bias_o, "memory/bias_o.matrix");
}

void load_weights_end_bias(){
	load_mat(&weights_i_h1, "memory/weights_i_h1.matrix");
	load_mat(&weights_h1_h2, "memory/weights_h1_h2.matrix");
	load_mat(&weights_h2_o, "memory/weights_h2_o.matrix");

	load_mat(&bias_h1, "memory/bias_h1.matrix");
	load_mat(&bias_h2, "memory/bias_h2.matrix");
	load_mat(&bias_o, "memory/bias_o.matrix");
}

void command_exec(int argc, char** parsedLine){

	if(argc > 0){
	if(command(0, "define")){
		if (argc > 1){ 
			if(command(1, "default")){
				set_net_structure_to_default();
				define_target_vectors(&target_vector);
				load_train_examples(&train_examples);
				load_test_examples(&test_examples);
			}	
		}else{
			printf("set num of input nodes: ");
			scanf("%d", &NUM_of_inputs);
			printf("set num of output nodes: ");
			scanf("%d", &NUM_of_outputs);
			printf("set num of first hidden layer nodes: ");
			scanf("%d", &NUM_of_H1);
			printf("set num of second hidden layer nodes: ");
			scanf("%d", &NUM_of_H2);
			char c[12];
			printf("set the activation function for second layer: \n\t press <log> for logistic or <linear> for linear function: ");
			scanf("%s", c);
			net_str.H2_activationF = strcmp(c, "log")? LINEAR: LOGISTIC;

			cleanScanf();

			define_target_vectors(&target_vector);
			load_train_examples(&train_examples);
			load_test_examples(&test_examples);
		}
	}else if(command(0, "set")){
		if(command(1, "b")){
			int b;
			printf("Number of batches: ");
			scanf("%d", &b);
			set_num_of_batches(b);
		}else if(command(1, "lr")){
			printf("Give me the learning rate: ");
			scanf("%f", &learning_rate);
		}		
	}else if(command(0, "print")){
		if (argc > 1){ 
			if(command(1, "str")){
				print_net_ctructure();
			}else if(command(1, "weights")){
				print_weights();
			}else if(command(1, "bias")){
				print_bias();
			}else if(command(1, "rate")){
				print_learning_rate();
			}else if(command(1, "term")){
				print_termination_threashold();
			}else if(command(1, "batch")){
				print_num_of_batch();
			}else if(command(1, "examp")){
				print_num_of_examples();
			}
		}else{
			print_net_ctructure();
			print_learning_rate();
			print_termination_threashold();
			print_num_of_batch();
			print_num_of_examples();
		}
	}else if(command(0, "train") ||
			 command(0, "t")){
		train();
	}else if(command(0, "test") ||
			 command(0, "r")){
		test();
	}else if(command(0, "save") ||
			 command(0, "s")){
		save_weights_end_bias();
		printf("New weights saved\n");
	}else if(command(0, "clear")){
		clear();
	}else if(command(0, "create") ||
			 command(0, "c")){

		Create_train_examples();
		Create_test_examples();

		load_train_examples(&train_examples);
		load_test_examples(&test_examples);
	}else if(command(0, "help") ||
			 command(0, "h")){
		printf("\n");
		printf("<define default>   	to set network structure to default\n");
		printf("<define>          	to DEFINE the neural network structure\n");
		printf("\n");
		printf("<set b>          	to set the number of batches\n");
		printf("<set lr>          	to set the learning rate\n");
		printf("\n");
		printf("<train>          	to TRAIN the neural network\n");
		printf("<test>          	to TEST the neural network\n");
		printf("\n");
		printf("<save>          	to SAVE current weithts end biases\n");
		printf("\n");
		printf("<create>          	to recreate all examples\n");
		printf("\n");
		printf("<print>				to see all neural network info\n");
		printf("<print str>			to see the network structure\n");
		printf("<print weights>		to see the network structure\n");
		printf("<print bias>       	to see the network structure\n");
		printf("<print rate> 		to see the learning rate\n"); 
		printf("<print term> 		to see the termination threshold\n"); 
		printf("<print batch>		to see the number of batches\n"); 
		printf("<print examp>		to see the number of examples\n"); 
		printf("\n");
		printf("<quit>				to leave the program\n"); 
		 
		printf("\n");
	}else if(command(0, "quit") ||
			 command(0, "q")){
		delete_weights_end_bias();
		free(train_examples);
		free(test_examples);
		free(target_vector);

		printf("\n\t<<< Bay Bay >>>\n");

		exit(0);
	}}
}