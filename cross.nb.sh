nb(){
it=$1;
alpha=$2;
lambda=$3;
./nb -d $k -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl "treino_t.dat" -ndV $numDocValidacao -ntV $numAttributes -ft "validacao.dat" -a $alpha -l $lambda >> "macro_${alpha}_${lambda}.dat" 2>> "micro_${alpha}_${lambda}.dat";
}

train=$1;
numClasses=$2;
numAttributes=$3;
n=$4;
n1=`expr $n - 1`;
k=$5;

./crossValidation -d ${train} -p $n -i 1;

rm -f micro_*.dat macro_*.dat;

for it in `seq 0 1 $n1`
do
  rm -f "treino_t.dat"
  echo "Iteration ${it}";
  for j in `seq 0 1 $n1`
  do
    if [ "${it}" -eq "${j}" ]; then
      cat "dados${j}.dat" > "validacao.dat";
    else
      cat "dados${j}.dat" >> "treino_t.dat";
    fi
  done;

  ./tf-idf -d "treino_t.dat" -t 0 > tmp; mv tmp "treino_t.dat";
  awk -F";" '{if(NF > 3) print $0;}' "treino_t.dat" > tmp; mv tmp "treino_t.dat";
  
  numDocTreino=`awk -F";" 'END{print NR;}' "treino_t.dat"`;
  numDocValidacao=`awk -F";" 'END{print NR;}' "validacao.dat"`;
  echo "# Training Docs: $numDocTreino"; 
  echo "# Validation Docs: $numDocValidacao";
	for alpha in `seq 0.0 0.1 1.0 | sed s/','/'.'/g`
	do
	  echo "${it} - ${alpha}"
	  for lambda in `seq 0.00 0.01 1.0 | sed s/','/'.'/g`
	  do
	    echo -n "."
	    nb ${it} ${alpha} ${lambda} ${k};
	  done
	  echo "";
	done
done

rm -f "mean_micro.dat" "mean_macro.dat"

echo "Taking the average of the results..."
for alpha in `seq 0.0 0.1 1.0 | sed s/','/'.'/g`
do
  for lambda in `seq 0.00 0.01 1.0 | sed s/','/'.'/g`
  do
    #Media das execucoes
      awk -f mean_sd.awk macro_${alpha}_${lambda}.dat >> "mean_macro.dat";
      awk -f mean_sd.awk micro_${alpha}_${lambda}.dat >> "mean_micro.dat";
  done
done
