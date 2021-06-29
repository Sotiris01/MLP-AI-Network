#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "Matrix.h"
#include "create_examples.h"
#include "nn_library.h"
#include "load_examples.h"

void set_net_structure_to_default(){
	NUM_of_inputs = 2;
	NUM_of_outputs = 4;

	// NUM_of_H1 = 7;
	// NUM_of_H2 = 4;

	// NUM_of_H1 = 8;
	// NUM_of_H2 = 5;

	NUM_of_H1 = 10;
	NUM_of_H2 =  6;

	net_str.H2_activationF = LOGISTIC;
	// net_str.H2_activationF = LINEAR;
}

void set_num_of_examples(int num){
	if(num > 0)
	if((num % 2) == 0)
		N = num;
}

void set_num_of_batches(int num){
	if(num > 0){
		if(((N/2) % num) == 0){
			B = num;
			printf("You set %d batches\n", B);
		}else{
			printf("You can't have %d batches\n", num);
		}
	}else{
		printf("batches must be a possitive number!\n");
	}
}
 
void print_learning_rate(){
	printf("Learning rate = %f\n", learning_rate);
}

void print_termination_threashold(){
	printf("Termination treashold = %f\n", termination_threshold);
}

void print_num_of_batch(){
	printf("Number of batces = %d\n", B);
}

void print_num_of_examples(){
	printf("Number of training examples = %d\n", N/2);
	printf("Number of test examples = %d\n", N/2);
}


void print_weights(){
	printf("______________________\n");
	printf("|\n|  weights input to H1:\n|");
	print_matrix(weights_i_h1);
	printf("|\n|  weights H1 to H2:\n|");
	print_matrix(weights_h1_h2);
	printf("|\n|  weights H2 to output:\n|");
	print_matrix(weights_h2_o);
	printf("|______________________\n");
}
void print_bias(){
	printf("______________________\n");
	printf("|\n|  bias of H1:\n|");
	print_matrix(bias_h1);
	printf("|\n|  bias of H2:\n|");
	print_matrix(bias_h2);
	printf("|\n|  bias of output\n|");
	print_matrix(bias_o);
	printf("|______________________\n");
}




void print_net_ctructure(){
	printf("\n");
	printf("input modes -> %d\n", NUM_of_inputs);
	printf("output modes -> %d\n", NUM_of_outputs);
	printf("hidden 1 modes -> %d\n", NUM_of_H1);
	printf("hidden 2 modes -> %d\n", NUM_of_H2);
	printf("hidden 2 activation function -> %s\n",
						(net_str.H2_activationF == LOGISTIC)? "logistic": "linear");
}

void takeInput(char* line){
	int i = 0;

	printf(" >>> ");

	do{
		line[i++] = getchar();

	} while(line[i-1] != '\n');

	line[--i] = '\0';
}

int parseSpace(char* line, char** parsedLine) { 
	int i; 
  
	for (i = 0; i < MAXLIST; i++) { 
		parsedLine[i] = strsep(&line, " "); 
  
		if (parsedLine[i] == NULL) 
			break; 
		if (strlen(parsedLine[i]) == 0) 
		    i--; 
	} 
	return i;
} 

void define_target_vectors(matrix_t** categories_mat){
	
	free(*categories_mat);

	*categories_mat = (matrix_t*) malloc(sizeof(matrix_t)*NUM_of_outputs);

	for (int i = 0; i < NUM_of_outputs; ++i){
		(*categories_mat)[i] = new_matrix(NUM_of_outputs, 1);
		(*categories_mat)[i].data[i][0] = 1;
	}

	// for (int i = 0; i < net_str.output_nodes; ++i)
	// 	print_matrix((*categories_mat)[i]);
}

matrix_t to_matrix(float* x, int d){
	matrix_t matrix = new_matrix(d, 1);

	for (int i = 0; i < d; ++i)
		matrix.data[i][0] = x[i];
	
	return matrix;
}