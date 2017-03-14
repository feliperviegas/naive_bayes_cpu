#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "io.hpp"
#include "nb_functions.hpp"

//#include "nb_cuda.h"


int main(int argc, char *argv[])  {

	int cuda;
	int numDocs, numClasses, numTerms;
	int numDocsTest, numTermsTest;
	double alpha, lambda;
	/***************************************/

	if(argc!=21){
		printf("\n\n./nb -cudaDevice [device=1] -nd [NumDocs] -nc [numClasses] -nt [numTerms] -fl [fileTrainning] -ndT [NumDocsTest] -ntT [numTermsTest] -ft [fileTest] -a [alpha] -l [lambda]\n\n");
		exit(0);
	}


	//Parametros
	cuda = atoi(argv[2]);
	numDocs = atoi(argv[4]);
	numClasses = atoi(argv[6]);
	numTerms = atoi(argv[8]);

	numDocsTest = atoi(argv[12]);
	numTermsTest = atoi(argv[14]);

	alpha = atof(argv[18]);
	lambda = atof(argv[20]);

	//Timing

	double *resultado = (double*) malloc(2*sizeof(double));
	resultado[0] =0.0;
	resultado[1] =0.0;

	resultado = nb_gpu(argv[10], argv[16], numDocs, numClasses, numTerms, numDocsTest,numTermsTest, alpha, lambda);

/*============================= DONE =================================*/

	return 0;
}
