#bin/bash

FOLDER="XXX" # diretorio a ser criado dentro de cada Fold
numfeatures=$3 # numero de features no treino / teste

echo "xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX          "$3;

treina_arff=$1".arff"
#rm -r train-B*
 rm train-B*
echo "VALORRRRRRRRRR wc -l $treina_arff"
../.././gera_bins_TUBE.sh $treina_arff  $numfeatures  
echo "bins geradas com sucesso"
