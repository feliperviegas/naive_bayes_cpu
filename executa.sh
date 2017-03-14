train=$1;
test=$2;
alpha=$3;
lambda=$4;

numAttributes=`awk -f attributes.awk $train $test`;
numClasses=`awk -F";" '{vet[$3]++;}END{cont=0;for(i in vet) cont++; print cont;}' $train $test`;

numDocTreino=`awk -F";" 'END{print NR;}' treino.dat`;
numDocTeste=`awk -F";" 'END{print NR;}' teste.dat`;
echo "$numDocTreino - $numDocTeste";

time ./nb -c 0 -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl treino.dat -ndT $numDocTeste -ntT $numAttributes -ft teste.dat -a $alpha -l $lambda > "res_nb.dat"


