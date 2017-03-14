nb(){
i=$1;
alpha=$2;
lambda=$3;
k=$4;
file1=$5;
file2=$6;
./nb -d $k -nd ${numDocTreino} -nc ${numClasses} -nt ${numAttributes} -fl "treino_weighted.dat" -ndV ${numDocTeste} -ntV ${numAttributes} -ft "teste_weighted.dat" -a ${alpha} -l ${lambda} >> "${file1}.dat" 2>> "${file2}.dat";
}
dataset=$1;
n=$2;
k=$3;
n1=`expr $n - 1`;
numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{print $3;}' $dataset | sort | uniq -c | wc -l`;
echo "# Attributes: ${numAttributes}";
echo "# Classes: ${numClasses}";


./crossValidation -d $dataset -p $n -i 1;

for i in `seq 0 1 $n1`
do
	mv "dados${i}.dat" "dados_t${i}.dat"
done

rm -f "res_micro.dat" "res_macro.dat";

for i in `seq 0 1 $n1`
do
  rm -f treino.dat
  echo "Iteration $i";
  for j in `seq 0 1 $n1`
  do
    if [ "${i}" -eq "${j}" ]; then
      cat "dados_t${j}.dat" > teste.dat;
    else
      cat "dados_t${j}.dat" >> treino.dat;
    fi
  done;

  awk -F";" '{if(NF > 3) print $0;}' "treino.dat" > tmp; mv tmp "treino.dat";
  awk -F";" '{if(NF > 3) print $0;}' "teste.dat" > tmp; mv tmp "teste.dat";

  wc -l "treino.dat"
  bash cross.nb.sh "treino.dat" ${numClasses} ${numAttributes} ${n} ${k};

  #tf-idf
  ./tf-idf -d "treino.dat" -t 0 > "treino_weighted.dat"
  awk -F";" '{if(NF > 3) print $0;}' "treino_weighted.dat" > tmp; mv tmp "treino_weighted.dat";
  #cat "treino.dat" > "treino_weighted.dat"
  cat "teste.dat" > "teste_weighted.dat"

  numDocTreino=`awk -F";" 'END{print NR;}' "treino_weighted.dat"`;
  numDocTeste=`awk -F";" 'END{print NR;}' "teste_weighted.dat"`;

  echo "# Training Docs: ${numDocTreino}"; 
  echo "# Test Docs: ${numDocTeste}";

  echo "Selecting the best parameters for MacF1..."
  #selecionar melhor parametro lambda e alpha
  sort -k3 -g -r "mean_macro.dat" | head -1 > "best_macro${i}.dat";
  alpha=`awk '{print $1}' "best_macro${i}.dat"`;
  lambda=`awk '{print $2}' "best_macro${i}.dat"`;
  file1="res_macro";
  file2="trash";
  echo "alpha: ${alpha}"; 
  echo "lambda: ${lambda}";
  nb ${i} ${alpha} ${lambda} ${k} ${file1} ${file2};

  echo "Selecting the best parameters for MicF1..."
  sort -k3 -g -r "mean_micro.dat" | head -1 > "best_micro${i}.dat";

  alpha=`awk '{print $1}' "best_micro${i}.dat"`;
  lambda=`awk '{print $2}' "best_micro${i}.dat"`;
  file1="trash";
  file2="res_micro";
  echo "alpha: ${alpha}"; 
  echo "lambda: ${lambda}";
  nb ${i} ${alpha} ${lambda} ${k} ${file1} ${file2};

done;

awk -f mean_sd.awk "res_macro.dat" | awk '{print "MacroF1: "$3" +- "$4;}' > res_nb.dat
awk -f mean_sd.awk "res_micro.dat" | awk '{print "MicroF1: "$3" +- "$4;}' >> res_nb.dat

rm -f dados* treino* teste*
