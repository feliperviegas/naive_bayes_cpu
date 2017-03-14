#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <ctime>
#include "io.hpp"
#include "evaluate.h"



void learning_kernel(double *probClass, double *probMatrix, int *freqClassVector, double *matrixTermFreq, double *totalFreqClassVector, 
					int numClasses, int numDocs, int numTerms, double totalTerms, double alpha, double *freqTermVector, double totalTermFreq, double lambda) {

	int i;
	double prob;
	for(i = 0; i < numClasses; i++){
		probClass[i] = ((double)(freqClassVector[i]+alpha))/((double)(numDocs+alpha*numClasses));
		for(int idx = 0; idx < numTerms; idx++){
		 	prob = (matrixTermFreq[i*numTerms + idx] + alpha) / (totalFreqClassVector[i] + alpha*totalTerms); 
			prob = lambda*(freqTermVector[idx]/totalTermFreq) + (1.0 - lambda)*prob; 
			probMatrix[i*numTerms + idx] = log(prob);
		}

	}
}

void trainning_kernel(double *probClass, double *probMatrix, int *docTestIndexVector, int *docTestVector, double* docTestFreqVector,
						int *docClasse, int numClasses, int numTerms, int numDocsTest, double *freqTermVector, int totalTerms) {

	int d,c,t;
	int term; 
	double freq;
	double prob, nt;
	double sumProb = 0.0;

	double  maiorProb;// = -999999.0;
	int maiorClasseProb;
	for(int idx = 0; idx < numDocsTest; idx++){
		d = idx;
		for(c=0;c<numClasses;c++){
			sumProb = log(probClass[c]);
			int inicio = docTestIndexVector[d];//+1;
			int fim = docTestIndexVector[d+1];
			for(t=inicio;t<fim;t++){
                		term = docTestVector[t];
				freq = docTestFreqVector[t];
				prob = probMatrix[c*numTerms + term];
				if(freqTermVector[term] != 0)
	   			sumProb += freq*prob;
			}
			if(c == 0){
                maiorProb = sumProb;
                maiorClasseProb = 0;
                        }
			if(sumProb > maiorProb){
				maiorClasseProb = c;
				maiorProb = sumProb;
			}
		}
		docClasse[d] = maiorClasseProb;	
	}

}


double *nb_gpu(const char* filenameTreino, const char* filenameTeste, int numDocs, int numClasses, int numTerms, int numDocsTest, int numTermsTest, double alpha, double lambda){
		/* =========================== TREINO ================================*/

    clock_t begin, end;
	begin = clock();
	int *docTestIndexVector = (int*) malloc((numDocsTest + 1) * sizeof(int));
	int *docTestVector = NULL;
	double *docTestFreqVector = NULL;

	int *freqClassVector = (int*) malloc(numClasses * sizeof(int));
	double *totalFreqClassVector = (double*) malloc(
			numClasses * sizeof(double));
	double *matrixTermFreq = (double*) malloc(
			(numTerms * numClasses) * sizeof(double));
	double *freqTermVector = (double*) malloc((numTerms) * sizeof(double));
	double totalTermFreq = 0.0;
	int totalTerms = 0;


	for (int i = 0; i < numClasses; i++) {
		totalFreqClassVector[i] = 0.0;
		freqClassVector[i] = 0;
		for (int j = 0; j < numTerms; j++) {
			matrixTermFreq[i * numTerms + j] = 0.0;
		}
	}
	for (int j = 0; j < numTerms; j++) {
		freqTermVector[j] = 0.0;
	}
	set<int> vocabulary;

	readTrainData(filenameTreino, freqClassVector, totalFreqClassVector, freqTermVector, &totalTermFreq, 
		numClasses, numTerms, &totalTerms, matrixTermFreq, vocabulary);


	double *probClass = (double*) malloc (numClasses*sizeof(double));
	double *probMatrix = (double*) malloc (numClasses*numTerms*sizeof(double));

    alpha=0.0;
    double alpha_macro = 0.0, alpha_micro = 0.0, lambda_macro = 0.0, lambda_micro = 0.0;
    double macro = 0.0, micro = 0.0;
    while(alpha <= 1.0){
		lambda = 0.1;
		while(lambda <= 1.0){

		// clock_t beginf, endf;
		// beginf = clock();
		learning_kernel(probClass,probMatrix,freqClassVector,matrixTermFreq,totalFreqClassVector,numClasses,numDocs,numTerms,totalTerms, alpha, freqTermVector, totalTermFreq, lambda);
		// endf = clock();
		// cout << "Time Function " << double(endf - beginf) / CLOCKS_PER_SEC << endl;

		/* ============================ TESTE ================================*/
		int *realClass = (int*) malloc ((numDocsTest)*sizeof(int));
		///*-- Passo 4: Leitura do teste --*/
		docTestVector = readTestData(filenameTeste, docTestIndexVector, realClass,
				&docTestFreqVector);

		int *docClasse = (int*) malloc ((numDocsTest)*sizeof(int));

		// beginf = clock();
		trainning_kernel(probClass, probMatrix, docTestIndexVector, docTestVector, docTestFreqVector, docClasse, numClasses,numTerms,numDocsTest,freqTermVector, totalTerms);
		// endf = clock();
		// cout << "Time Function " << double(endf - beginf) / CLOCKS_PER_SEC << endl;
		double *valor = (double*) malloc(2*sizeof(double));
		valor[0] = evaluate(realClass, docClasse, numDocsTest, 1); //macroF1
		valor[1] = evaluate(realClass, docClasse, numDocsTest, 0); //microF1
		// std::cerr << "Naive Bayes :" << valor[0]*100 << " " << valor[1]*100 << std::endl;
		// std::cout << valor[0]*100 << " " << valor[1]*100 << " ";

        // std::cout << alpha << " " << lambda << " " << valor[0]*100 << std::endl;
        // std::cerr << alpha << " " << lambda << " " << valor[1]*100 << std::endl;

        if(valor[0]*100 > macro){
        	alpha_macro = alpha;
        	lambda_macro = lambda;
        	macro = valor[0]*100;
        }
        if(valor[1]*100 > micro){
        	alpha_micro = alpha;
        	lambda_micro = lambda;
        	micro = valor[1]*100;
        }

        lambda += 0.01;

      }

      alpha += 0.1;
    }

	std::cout << alpha_macro << " " << lambda_macro << " " << macro << std::endl;
    std::cerr << alpha_micro << " " << lambda_micro << " " << micro << std::endl;

	/*-- Liberando Memoria Device --*/
    free(docTestIndexVector);
    free(docTestVector);
    free(freqTermVector);
    free(probMatrix);
    free(probClass);
    free(docClasse);
    free(realClass);

    free(freqClassVector);
    free(totalFreqClassVector);
    free(matrixTermFreq);
    // end = clock();
    // double dif = double(end - begin) / CLOCKS_PER_SEC;
    // cerr << "Tempo de classificacao " <<  dif;
    // cout << "Time " << dif << endl;

    return valor;

		/*============================= DONE =================================*/
}
