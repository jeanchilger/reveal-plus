#bin/bash

FOLDER=$1 # diretorio a ser criado dentro de cada Fold
numfeatures=$2 # numero de features no treino / teste
partitions=5 # numero de particoes a ser usado
binsfrom=10 # numero de bins (de) 
binsto=10 # numero de bins (at


suffix=B10

# roda
# gera arquivos contendo os limites dos BINS determinados pelo TUBE
../.././gera_bins_TUBE.sh treina.arff  $numfeatures $binsfrom $binsto

 # remove os headers dos arquivos weka
if [ ! -f ../train_nohead.arff ]; then
    grep -v @ ../train.txt.arff | grep -v ^$ > train_nohead.arff
    grep -v @ ../test.txt.arff | grep -v ^$ > test_nohead.arff
fi 

 # discretiza usando os bins definidos pelo TUBE; os arquivos jah sao gerados no formato LAC
echo "Discretizando treino e teste de acordo com os bins TUBE"
../../discretize_TUBE.pl train-$suffix .train_nohead.arff $numfeatures lac_train_TUBE$suffix.txt
../../discretize_TUBE.pl train-$suffix test_nohead.arff $numfeatures lac_test_TUBE$suffix.txt

 # roda o ALAC
 ../.././discretize_TUBE.pl train-10 train_nohead.arff 10 lac_train_TUBE10.txt

  ./reconstroi_arff_sementes.pl > sementes_arff

  grep -v [0-1]$  ../treina.arff > final_treina.arff
 cat sementes_arff >> final_treina.arff