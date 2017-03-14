#include "nb.hpp"
#include "io.hpp"
//#include "nb_cuda.h"

struct rusage resources;
struct rusage ru;
struct timeval tim;
struct timeval tv;

/*=============================================================================================*/
void temposExecucao(double *utime, double *stime, double *total_time)
{
    int rc;
    
    if((rc = getrusage(RUSAGE_SELF, &resources)) != 0)
       perror("getrusage Falhou");

    *utime = (double) resources.ru_utime.tv_sec + (double) resources.ru_utime.tv_usec * 1.e-6 ;
    *stime = (double) resources.ru_stime.tv_sec + (double) resources.ru_stime.tv_usec * 1.e-6 ;
    *total_time = *utime + *stime;

}

double tempoAtual()
{

 gettimeofday(&tv,0);

 return tv.tv_sec + tv.tv_usec/1.e6;     
}
/*=============================================================================================*/

void calcFreq2(int *termVector, int *termIndexVector, int *matrixTermFreq, int numDocs, int numTerms, int numClasses){
	
	int classe;
	int term,freq,doc;
	int i,j;
	int inicio,final,k;
	k = 1;
	
	for(i=0; i<numTerms; i++){	
		
		if(	termIndexVector[i] != -1){
			
			inicio = termIndexVector[i];
			while(termIndexVector[i+k] == -1) k++;
			final = termIndexVector[i+k];
			
			for(j=inicio;j<final;j+=3){
				doc = termVector[j];
				classe = termVector[j+1];
				freq = termVector[j+2];
				
				matrixTermFreq[classe*numTerms + i] += freq;			
			}	
		}
		k = 1;	
	}		
}

void learning(int *matrixTermFreq, int *totalFreqClassVector, int *freqClassVector, double *probClass, double *probMatrix, int numClasses, int numTerms, int numDocs){
	
	int i,j;
	
	for(i = 0; i < numClasses; i++){
		probClass[i] = ((double)freqClassVector[i]+1.0)/((double)numDocs+(double)numClasses);
		//probClass[i] = log((double)freqClassVector[i] / (double)numDocs);
		for(j = 0; j < numTerms; j++){
			//probMatrix[i*numTerms + j] =  (double)matrixTermFreq[i*numTerms + j] /  (double)totalFreqClassVector[i];
            probMatrix[i*numTerms + j] =  ((double)matrixTermFreq[i*numTerms + j] + 1.0) / ( (double)totalFreqClassVector[i] + (double)numTerms);
        }
    }
	
    //for(i = 0; i < numClasses; i++){
        //printf("%d %d\n", i, totalFreqClassVector[i]);
		//for(j = 0; j < numTerms; j++){
            //printf("%d - %d = %.30f\n", j, i, probMatrix[i*numTerms + j]);
        //}
    //}
}

double *trainning(int *docTestVector, int *docTestIndexVector, double *probClass, double *probMatrix, int numDocsTest, int numClasses, int numTermsTest, int numTerms, double* freqTermVector, int totalTermFreq){
	long double norm = 1.0/255102.00;//(double)numTerms;
	int d,c,t,f;
	double probClasse, probClasseNorm = 0.0;
    double probAux, probNorm;
	
	double maiorProb = 0.0;
	int maiorClasseProb = 0;
	int term, freq;
	
	double* docClasse = (double*) malloc (numDocsTest*3*sizeof(double));
	for(d=1;d<=numDocsTest+1;d++){
		for(c=0;c<numClasses;c++){
			probClasse = log(probClass[c]);
            //probClasseNorm = log(probClass[0]);
			for(t=docTestIndexVector[d]+1;t<docTestIndexVector[d+1];t+=2){
				term = docTestVector[t];
				freq = docTestVector[t+1];
				if(probMatrix[c*numTermsTest + term] > 0.0 && freq > 0){
                    probAux =  3.0*(freqTermVector[term]/totalTermFreq) + 7.0*(probMatrix[c*numTermsTest + term]/norm);
                    //probNorm =  3.0*(freqTermVector[term]/totalTermFreq) + 7.0*(probMatrix[0*numTermsTest + term]/norm);
				}
                else{
                    probAux = 7.0*(1.0/(double)numTerms)/norm;
                   //probNorm = 7.0*(1.0/(double)numTerms)/norm;
                }
                for(f=0;f<freq;f++){
                    probClasse += log(probAux);
                    //probClasseNorm += log(probNorm);
                }
			}
            //probClasse = probClasse/probClasseNorm;
			if(probClasse > maiorProb){
					maiorClasseProb = c;
					maiorProb = probClasse;
			}
		}
		
		docClasse[d*3 + 0] = docTestVector[docTestIndexVector[d]];
		docClasse[d*3 + 1] = maiorClasseProb;
		docClasse[d*3 + 2] = maiorProb;
		
		probClasse = 0.0;
		maiorProb = 0.0;
		maiorClasseProb = 0;
	}
		
	return docClasse;
}


double *nb_cpu2(const char* filenameTreino, const char* filenameTeste, int numDocs, int numClasses, int numTerms, int numDocsTest, int numTermsTest){

	double iTreino, fTreino, iTeste, fTeste;
	
/* =========================== TREINO ================================*/
	
	/*-- Passo 1: Lendo arquivo de treino --*/
	int *termVector = NULL;
	int *termIndexVector = (int*) malloc ((numTerms+2)*sizeof(int));
	int *freqClassVector = (int*) malloc (numClasses*sizeof(int));
	int *totalFreqClassVector = (int*) malloc (numClasses*sizeof(int));
	int *matrixTermFreq = (int*) malloc ((numTerms*numClasses)*sizeof(int));
	double *freqTermVector = (double*) malloc ((numTerms)*sizeof(double));
    int totalTermFreq;
	int totalTerms;
	int i,j;
	for(i=0;i<numClasses;i++){
		totalFreqClassVector[i] = 0;
		for(j=0;j<numTerms;j++){
				matrixTermFreq[i*numTerms +j] = 0;
		}
	}
	
	readTrainingData2(filenameTreino, totalFreqClassVector, freqClassVector, freqTermVector, &totalTermFreq, numClasses, numTerms, &totalTerms, matrixTermFreq);
	//for(i=0;i<numTerms;i++) printf(" %d\n", termIndexVector[i]);
	//printf("----------2----------\n");
	//printf("---------\n");
	//int inicio,final,k;
	//k = 1;
	//for(i=0; i<numTerms; i++){	
		//if(	termIndexVector[i] != -1){
			//inicio = termIndexVector[i];
			//while(termIndexVector[i+k] == -1) k++;
			//final = termIndexVector[i+k];
			//printf("--> term %d: ",i);
			//for(j=inicio;j<final;j+=3){
				//printf(" %d:%d:%d/",termVector[j],termVector[j+1],termVector[j+2]);
			//}
		//}
		//printf("\n");
	//}
	
	/*-- Passo 2: Calculo das frequencias --*/	
	iTreino = tempoAtual();
	calcFreq2(termVector,termIndexVector,matrixTermFreq,numDocs,numTerms,numClasses);
	
	free(termVector);
	free(termIndexVector);
		
	//printf("----------FreqClassVector----------\n");
	//for(i=0;i<numClasses;i++) printf(" %d\n",freqClassVector[i]);
	//printf("-------totalFreqClassVector---------\n");
	//for(i=0;i<numClasses;i++) printf(" %d\n",totalFreqClassVector[i]);
	//printf("---------\n");
	//for(i=0;i<numClasses;i++){
		//for(j=0;j<numTerms;j++){
			//printf("%d %d %d\n",i,j,matrixTermFreq[i*numTerms + j]);
		//}
	//}
	
	/*-- Passo 3: Calculo das Probabilidades --*/	
	double *probClass = (double*) malloc (numClasses*sizeof(double));
	double *probMatrix = (double*) malloc (numClasses*numTerms*sizeof(double));
	
	learning(matrixTermFreq,totalFreqClassVector,freqClassVector,probClass,probMatrix,numClasses,numTerms,numDocs);

	//printf("----------probClass----------\n");
	//for(i=0;i<numClasses;i++) printf(" %lf\n",probClass[i]);
	//printf("---------probTermMatrix--------\n");
	//for(i=0;i<numClasses;i++){
		//for(j=0;j<numTerms;j++){
			//printf("%d %d %d %lf\n",i,j,matrixTermFreq[i*numTerms + j],probMatrix[i*numTerms + j]);
		//}
		//printf("\n");
	//}
	free(matrixTermFreq);
	free(totalFreqClassVector);
	free(freqClassVector);
	
	fTreino = tempoAtual();
	//printf("#1\n");
/* ============================ TESTE ================================*/
	
	/*-- Passo 4: Leitura do teste --*/
	int *docTestIndexVector = (int*) malloc ((numDocs+1)*sizeof(int));
	int *docTestVector = NULL;
	int *realClass = (int*) malloc ((numDocsTest+1)*sizeof(int));
	docTestVector = readTestData(filenameTeste,docTestIndexVector, realClass);
	
	iTeste = tempoAtual();
	/*-- Passo 5: Teste! --*/
	double *docClasses = NULL;

	//for(i=1;i<=numDocsTest;i++) printf(" %d\n", docTestIndexVector[i]);
	//printf("----------2----------\n");
	//printf("---------\n");
	//for(i=1;i<=numDocsTest;i++){
		//printf("--> doc %d: ",i);
		//for(j=docTestIndexVector[i]+1;j<docTestIndexVector[i+1];j+=2){
			//printf(" %d:%d",docTestVector[j],docTestVector[j+1]);
		//}
		//printf("\n");
	//}
	
	docClasses = trainning(docTestVector,docTestIndexVector,probClass,probMatrix,numDocsTest,numClasses,numTermsTest, numTerms, freqTermVector, totalTermFreq);
	
	fTeste = tempoAtual();
	
	//for(i=1;i<=numDocsTest;i++){
	//		printf("%d: %lf %lf %lf\n",i,docClasses[i*3 + 0],docClasses[i*3 +1],docClasses[i*3 +2]);		
	//}
	
	//printf("%lf %lf\n",fTreino-iTreino,fTeste-iTeste);
	
	//free(docTestIndexVector);
	//free(docTestVector);
	//free(probClass);
	//free(probMatrix);
	return docClasses;
/*============================= DONE =================================*/

}
