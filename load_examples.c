#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "Matrix.h"
#include "create_examples.h"
#include "nn_library.h"
#include "load_examples.h"


#define TRUE  1
#define FALSE 0

#define TEST FALSE

#if TEST
int main(){

	example_t* train_examples;
	// example_t* test_examples;


	load_train_examples(&train_examples);
	// load_test_examples(&test_examples);

	printf("Train examples: \n");
	for (int i = 0; i < 10; ++i){
		for (int j = 0; j < net_str.input_nodes; ++j)
		{
			printf("%f ", train_examples[i].x[j]);
		}
		printf("%s\n", train_examples[i].category);
	}
		// printf("%f, %f, %s\n",train_examples[i].x1, train_examples[i].x2, train_examples[i].category);

	// printf("Test examples: \n");
	// for (int i = 0; i < 10; ++i)
	// 	printf("%f, %f, %s\n",test_examples[i].x1, test_examples[i].x2, test_examples[i].category);


	return 0;}
#endif

void parseTab(char* line, char** parsedLine) { 

	for (int i = 0; i < NUM_of_inputs+1; i++)
		parsedLine[i] = strsep(&line, "\t"); 
} 

int load_train_examples(example_t** train_examples){
	FILE *fp;
	int i = 0;
	char line[255];
	char* parsedLine[NUM_of_inputs+1];

	#if TEST
	int num_of_examples = 10;
	#else
	int num_of_examples = N/2;
	#endif

	*train_examples = (example_t*) malloc(sizeof(example_t) * num_of_examples);

	if ((fp = fopen("train_examples.txt", "r")) == NULL){
		printf("Error with loading training examples\n");
		return -1;
	}

	#if TEST
	while(i < num_of_examples){
		fgets(line, 255, fp);
	#else
	while (fgets(line, 255, fp) != NULL){
	#endif
		parseTab(line, parsedLine);
		(*train_examples)[i].x = (float*)malloc(sizeof(float)*NUM_of_inputs);
		for (int j = 0; j < NUM_of_inputs; ++j){
			(*train_examples)[i].x[j] = atof(parsedLine[j]);
		}
		strcpy((*train_examples)[i].category, parsedLine[NUM_of_inputs]);
		i++;
	}

	fclose(fp);

	if(i != num_of_examples){
		printf("Error with loading training examples\n");
		return -1;
	}
	printf("Training examples succesfuly loaded\n");
	return 0;
}

int load_test_examples(example_t** test_examples){
	FILE *fp;
	int i = 0;
	char line[255];
	char* parsedLine[NUM_of_inputs+1];

	#if TEST
	int num_of_examples = 10;
	#else
	int num_of_examples = N/2;
	#endif

	*test_examples = (example_t*) malloc(sizeof(example_t) * num_of_examples);

	if ((fp = fopen("test_examples.txt", "r")) == NULL){
		printf("Error with loadint testing examples\n");
		return -1;
	}

	#if TEST
	while(i < num_of_examples){
		fgets(line, 255, fp);
	#else
	while (fgets(line, 255, fp) != NULL){
	#endif
		parseTab(line, parsedLine);
		(*test_examples)[i].x = (float*)malloc(sizeof(float)*NUM_of_inputs);
		for (int j = 0; j < NUM_of_inputs; ++j){
			(*test_examples)[i].x[j] = atof(parsedLine[j]);
		}
		strcpy((*test_examples)[i].category, parsedLine[NUM_of_inputs]);
		i++;
	}

	fclose(fp);

	if(i != num_of_examples){
		printf("Error with loadint testing examples\n");
		return -1;
	}
	printf("Testing examples succesfuly loaded\n");
	return 0;
}

