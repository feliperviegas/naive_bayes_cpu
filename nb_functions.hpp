#include "stdio.h"

#ifndef COMPONENTES_H
#define COMPONENTES_H

void learning_kernel(double *probClass, double *probMatrix, int *freqClassVector, double *matrixTermFreq, double *totalFreqClassVector, 
					int numClasses, int numDocs, int numTerms, double totalTerms, double alpha, double *freqTermVector, double totalTermFreq, double lambda);
void trainning_kernel(double *probClass, double *probMatrix, int *docTestIndexVector, int *docTestVector, double* docTestFreqVector,
						int *docClasse, int numClasses, int numTerms, int numDocsTest, double *freqTermVector, int totalTerms);
double *nb_gpu(const char* filenameTreino, const char* filenameTeste, int numDocs, int numClasses, int numTerms, int numDocsTest, int numTermsTest, double alpha, double lambda);


void super_parent_freq(int *docIndexVector, int *docVector, int *totalTermClassSp,int numTerms, int numDocs, int totalTerms);


#endif
