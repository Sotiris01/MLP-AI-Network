#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Matrix.h"
#include "nn_library.h"
#include "create_examples.h"

#define TRUE  1
#define FALSE 0

#define TEST FALSE


#define RANDOM_NUM ((float)rand()/(float)(RAND_MAX))
#define RANDOM_POINT {(RANDOM_NUM*2-1),(RANDOM_NUM*2-1)}

#define CATEGORY_1(p) (((p).x1*(p).x1 + (p).x2*(p).x2) < 0.25)
#define CATEGORY_2(p) (	((p).x1 <= -0.4 && (p).x2 <= -0.4) ||\
						((p).x1 >=  0.4 && (p).x2 >=  0.4))
#define CATEGORY_3(p) (	((p).x1 <= -0.4 && (p).x2 >=  0.4) ||\
						((p).x1 >=  0.4 && (p).x2 <= -0.4))
#define CATEGORY_4(p)	(1) // Any other case

#define NOISE(p) ((RANDOM_NUM * 100) <= (p))

// #define PROBABILITY 10 // some -> 1 | too mucth -> 100
int NOISE_PROBABILITY = 0; // 0% by default


typedef struct point_s{
	float x1;
	float x2;
} point_t;

char* category(point_t);

#if TEST

int N = 6000;

void create_test_files(point_t point, char* cat){


	FILE	*C1_fp;
	FILE	*C2_fp;
	FILE	*C3_fp;
	FILE	*C4_fp;

	C1_fp = fopen("C1_examples.txt", "a");
	C2_fp = fopen("C2_examples.txt", "a");
	C3_fp = fopen("C3_examples.txt", "a");
	C4_fp = fopen("C4_examples.txt", "a");

	if 		(!strcmp(cat, "C1"))
		fprintf(C1_fp, "%.3f\t%.3f\n", point.x1, point.x2);
	else if (!strcmp(cat, "C2"))
		fprintf(C2_fp, "%.3f\t%.3f\n", point.x1, point.x2);
	else if (!strcmp(cat, "C3"))
		fprintf(C3_fp, "%.3f\t%.3f\n", point.x1, point.x2);
	else if (!strcmp(cat, "C4"))
		fprintf(C4_fp, "%.3f\t%.3f\n", point.x1, point.x2);
	
	fclose(C1_fp);
	fclose(C2_fp);
	fclose(C3_fp);
	fclose(C4_fp);}
int main(){



	fopen("C1_examples.txt", "w");
	fopen("C2_examples.txt", "w");
	fopen("C3_examples.txt", "w");
	fopen("C4_examples.txt", "w");


	// Create_train_examples();
	Create_test_examples();



	char * commandsForGnuplot[] = {	"set title \"TEST\"", 
									"plot	'C1_examples.txt' pt 1 lc 1,\
											'C2_examples.txt' pt 1 lc 2,\
											'C3_examples.txt' pt 1 lc 3,\
											'C4_examples.txt' pt 6 lc 4 "};
	FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
	for (int i=0; i < 2; i++)
		fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); //Send commands to gnuplot one by one.

	return 0;}
#endif

int Create_train_examples(){

	NOISE_PROBABILITY = 10; //10% noise
	
	if(Create_examples("train_examples.txt", N/2)){
		printf("Error with creating training examples\n");
		return -1;
	}
	printf("Training examples have been created successfully\n");

	return 0;
}

int Create_test_examples(){

	NOISE_PROBABILITY = 0; // 0% noise
	
	if(Create_examples("test_examples.txt", N/2)){
		printf("Error with creating testing examples\n");
		return -1;
	}
	printf("Testing examples have been created successfully\n");

	return 0;
}

int Create_examples(char* file_name , int num_of_examples){

	FILE	*fp;

	if ((fp = fopen(file_name, "w")) == NULL)
		return -1;

	for (int i=0; i<num_of_examples; i++){

		point_t point = RANDOM_POINT;

		char* cat = category(point); 

		fprintf(fp, "%f\t%f\t%s\n", point.x1, point.x2, cat);

		#if TEST
		create_test_files(point, cat);
		#endif
	}

	if(fclose(fp) == EOF)
		return -1;

	return 0;
}

char* category(point_t point){

	if 		(CATEGORY_1(point))
		return (NOISE(NOISE_PROBABILITY) ? "C4" : "C1");

	else if (CATEGORY_2(point))
		return (NOISE(NOISE_PROBABILITY) ? "C4" : "C2");

	else if (CATEGORY_3(point))
		return (NOISE(NOISE_PROBABILITY) ? "C4" : "C3");

	else if (CATEGORY_4(point))
		return "C4";
	
	return "";
}

























