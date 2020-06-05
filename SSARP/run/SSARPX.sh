#bin/bash

FOLDER="XXX" # diretorio a ser criado dentro de cada Fold
numfeatures=10 # numero de features no treino / teste
partitions=3 # numero de particoes a ser usado
binsfrom=10 # numero de bins (de)
binsto=10 # numero de bins (at

treina_arff=$1".arff"

file_original=$2
suffix=B$3
numfeatures=$3
round=$4
seed_ssarp=$5
topic=$6
time=$7
rules=$8


echo "execution number "$round" the file have "$numfeatures" features and size `wc -l < $file_original` treina arff `wc -l < $treina_arff` ";


rm -r result_temp_lac_train_TUBE*

#
# remove os headers dos arquivos weka

grep -v @ $treina_arff | grep -v ^$ > train_nohead.arff




if [ ! -s train_nohead.arff ]; then
    echo "empty file"
    exit;
fi


x=$(cat train_nohead.arff | wc -l)

if [ $x -le 1 ]; then
    echo "file almost empty"
    cp $file_original  /tmp/final_treina.arff.$topic
    exit;
fi

if [ -f lac_train_TUBE.txt ]; then rm -f lac_train_TUBE.txt.$topic; fi
if [ -f lac_train_TUBE.txt ]; then rm -f lac_train_TUBEfinal.txt.$topic; fi


#if [ -f alac_lac_train_TUBEfinal.txt* ]; then rm -f alac_lac_train_TUBEfinal.txt*; fi


 # discretiza usando os bins definidos pelo TUBE; os arquivos jah sao gerados no formato LAC
echo "Discretizando treino e teste de acordo com os bins TUBE"
echo ../discretize_TUBE.pl train-$suffix train_nohead.arff $numfeatures  lac_train_TUBE.txt.$topic
../discretize_TUBE.pl train-$suffix train_nohead.arff $numfeatures  lac_train_TUBE.txt.$topic




# echo  ./updateRows.pl lac_train_TUBE.txt lac_train_TUBEfinal.txt $numfeatures
./updateRows.pl lac_train_TUBE.txt.$topic lac_train_TUBEfinal.txt.$topic $round

#tail -n 1 alac_lac_train_TUBEfinal.txt.$topic | grep "CLASS=0"  >> lac_train_TUBEfinal.txt.$topic

if [ $time -eq 1 ];
then
    echo "removendo o arquivo bkp anterior "
    rm bkppairs
    rm instanceFile
    rm alac_full_lac_train_TUBEfinal.txt.$topic
    rm alac_lac_train_TUBEfinal.txt.$topic
    echo "iniciando o seed"
    
    echo "discretizando seed ssarp"

../discretize_TUBE.pl train-$suffix $seed_ssarp $numfeatures  lac_train_TUBE_seed.txt.$topic
echo "seed_ssarp----------"
cat $seed_ssarp
./updateRows.pl lac_train_TUBE_seed.txt.$topic  lac_train_TUBEfinal_seed.txt.$topic   0

    cat lac_train_TUBEfinal_seed.txt.$topic  >> alac_full_lac_train_TUBEfinal.txt.$topic

    cat lac_train_TUBEfinal_seed.txt.$topic | grep "CLASS=0"  > bkpseed
    cat alac_full_lac_train_TUBEfinal.txt.$topic | grep "CLASS=1" | head -n 1 >> bkpseed
    cat bkpseed >> alac_lac_train_TUBEfinal.txt.$topic

    echo "bkp seed is "
    cat bkpseed 
    cp alac_lac_train_TUBEfinal.txt.$topic  bkppairs
   # python3 testeActive.py lac_train_TUBE.txt.$topic 5 /tmp/fullAllacfile.$topic lac_train_TUBE_seed.txt.$topic 1

fi

#python3 testeActive.py lac_train_TUBE.txt.$topic 5 /tmp/fullAllacfile.$topic lac_train_TUBE_seed.txt.$topic 0

i=1
while [[ $i -le 1 ]]; do
  # rm alac_lac_train_TUBEfinal.txt
   #rm alac_full_lac_train_TUBEfinal.txt
    echo " roda o ALAC ....     `wc -l < alac_lac_train_TUBEfinal.txt.$topic`"
  ../run_alac_repeated.sh lac_train_TUBEfinal.txt.$topic  $round $rules
 i=$(($i+1))
done
#junta as instancias selecionadas em cada particao em um arquivo unico contendo todas as features
echo "Gerando o treino a partir das instancias selecionadas em cada particao..ssss"

cp alac_lac_train_TUBEfinal.txt.$topic   bkppairs.$topic




cat lac_train_TUBEfinal_seed.txt.$topic  |   awk '{ print $1 }' |  while read instance; do  sed -i  "/^$instance /d"  bkppairs.$topic ;  done




./scriptRemoveRows.pl alac_full_lac_train_TUBEfinal.txt.$topic  composite_train_uniqB$numfeatures composite_train_uniqBold$numfeatures $round




cat alac_full_lac_train_TUBEfinal.txt.$topic >> pairsStore
cat lac_train_TUBEfinal.txt.$topic >> pairsStore
sort pairsStore -k1 |  uniq > pairsStoreB
mv pairsStoreB pairsStore
# ###################################
# #cat alac_lac_train_TUBE.txt* | awk '{ print $1 }' |  while read instance; do  sed -i  "/^$instance /d" $file_original  ;  done
#


#cat composite_train_uniqB$numfeatures* >> bkppairs
#
#
# cat alac_lac_train_TUBEfinal.txt*  > composite_train$suffix.txt

# cat alac_lac_train_TUBEfinal.txt* | while read instance; do echo "$instance"; done

#cat composite_train_uniqB$numfeatures | awk '{ print $1 }'  | while read instance; do  sed  -n  "$instance"p  $treina_txt; done >> /tmp/final_treina.txt;

cat composite_train_uniqB$numfeatures  | awk '{ print $1 }'  | while read instance; do  sed  -n  "$instance"p  $file_original; done > final_treina.arff.$round.$topic;

cut -d' ' -f1 final_treina.arff.$round.$topic > label.$round.$topic

cut -d' ' -f2- final_treina.arff.$round.$topic >> all_treina.arff
echo "gerando arquivo SVM treino.."
#cut -d' ' -f1- final_treina.arff.$round.$topic >  /tmp/outSSARP.$round.$topic
