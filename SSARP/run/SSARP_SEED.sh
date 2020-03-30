#bin/bash

FOLDER="XXX" # diretorio a ser criado dentro de cada Fold
numfeatures=10 # numero de features no treino / teste
partitions=5 # numero de particoes a ser usado
binsfrom=10 # numero de bins (de) 
binsto=10 # numero de bins (at

treina_arff=$1".arff"

file_original=$2
suffix=B$3
numfeatures=$3
vez=$4
seed_ssarp=$5
topic=$6
#sleep 100000000

echo "the file have "$numfeatures;

# gera arquivos contendo os limites dos BINS determinados pelo TUBE
# if [[ $flag -le "7" ]]; then
   echo "******************************************remove training set"
   #../.././gera_bins_TUBE.sh $treina_arff  $numfeatures $binsfrom $binsto

  
    
#     rm composite_train_un*
#     rm saida_SSARP
#     rm /tmp/final_treina_arff.txt
  #  echo "ONLY Produce bins"
    rm -r result_temp_lac_train_TUBEfinal.txt/ 
    rm -r result_temp_lac_train_TUBE*
#     rm composite_train_uniq*
    
 rm train_nohead.arff  
     ../.././gera_bins_TUBE.sh $treina_arff  $numfeatures $binsfrom $binsto   
# fi
 
# rm alac_lac_train_TUBEfinal.txt*


# 
# remove os headers dos arquivos weka
if [ ! -f train_nohead.arff ]; then
    # grep  @ $treina_arff  > /tmp/final_treina.arff
    grep -v @ $treina_arff | grep -v ^$ > train_nohead.arff

#     grep -v @ $test_arff | grep -v ^$ > test_nohead.arff
fi 



if [ ! -s train_nohead.arff ]; then
    echo "empty file"
    exit;
fi


x=$(cat train_nohead.arff | wc -l)

if [ $x -le 2 ]; then
    echo "file almost empty"
    cp $file_original  /tmp/final_treina.arff.$topic
    exit;
fi

if [ -f lac_train_TUBE.txt ]; then rm -f lac_train_TUBE.txt.$topic; fi
if [ -f lac_train_TUBE.txt ]; then rm -f lac_train_TUBEfinal.txt.$topic; fi


#if [ -f alac_lac_train_TUBEfinal.txt* ]; then rm -f alac_lac_train_TUBEfinal.txt*; fi


 # discretiza usando os bins definidos pelo TUBE; os arquivos jah sao gerados no formato LAC
echo "Discretizando treino e teste de acordo com os bins TUBE"
echo ../../discretize_TUBE.pl train-$suffix train_nohead.arff $numfeatures  lac_train_TUBE.txt.$topic
../../discretize_TUBE.pl train-$suffix train_nohead.arff $numfeatures  lac_train_TUBE.txt.$topic




# echo  ./updateRows.pl lac_train_TUBE.txt lac_train_TUBEfinal.txt $numfeatures
./updateRows.pl lac_train_TUBE.txt.$topic lac_train_TUBEfinal.txt.$topic $vez





if [ $vez -eq 5 ]; 
then
    echo "removendo o arquivo bkp anterior "
    rm  bkppairs.$topic 
    rm alac_full_lac_train_TUBEfinal.txt.$topic 
    rm alac_lac_train_TUBEfinal.txt.$topic
    echo "iniciando o seed"
    #cat lac_train_TUBEfinal_seed.txt >> lac_train_TUBEfinal.txt
    echo "discretizando seed ssarp"
../../discretize_TUBE.pl train-$suffix $seed_ssarp $numfeatures  lac_train_TUBE_seed.txt.$topic 
./updateRows.pl lac_train_TUBE_seed.txt.$topic  lac_train_TUBEfinal_seed.txt.$topic   $(($vez-1))
    
    cat lac_train_TUBEfinal_seed.txt.$topic  >> alac_full_lac_train_TUBEfinal.txt.$topic 
    cat lac_train_TUBEfinal_seed.txt.$topic  >> alac_lac_train_TUBEfinal.txt.$topic 
    
#                             ../../lac_to_weka.pl lac_train_TUBE.txt lac_train_TUBE.tmp lac_train_header_TUBE.arff $numfeatures
#                             sed 's/\,'\''\\'\''$/}/' lac_train_TUBE.tmp > lac_train_TUBE.arff
#                              echo -n "Calculando ChiSquared e ordenando features..."
# 			      echo "../../create_ChiSquared_train_TUBE.sh lac_train_TUBE.arff ChiSquared_Class_train_TUBE$suffix.txt $numfeatures $suffix"
# 			      ../../create_ChiSquared_train_TUBE.sh lac_train_TUBE.arff ChiSquared_Class_train_TUBE$suffix.txt $numfeatures $suffix
# 			      echo "../../order_chi_TUBE.pl ../ChiSquared_Class_train_TUBE$suffix.txt | tail -n 1 > ../features_train_TUBE$suffix.txt"
# 			      ../../order_chi_TUBE.pl ChiSquared_Class_train_TUBE$suffix.txt | tail -n 1 > ../features_train_TUBE$suffix.txt
# 			      echo " ordenacao das features finalizada."


#else
#     cat bkppairs >> lac_train_TUBEfinal.txt
fi
echo " roda o ALAC $numfeatures" 





########################


		    

			
# 			rm -f features_*_train.txt
# 			rm -f alac_features_*_train.txt
# 			rm -rf result_temp_features_*_train.txt
# 			
# 			# cria $partitions particoes usando a lista de features e o treino (gerando $partitions arquivos de treino)
# 			echo "Criando arquivos de particoes... "
# 			echo "../../create_partition_files.pl ../lac_train_TUBE.txt ../features_train_TUBE$suffix.txt $partitions"
# 			../../create_partition_files.pl lac_train_TUBEfinal.txt features_train_TUBE$suffix.txt 
# 			
# 			
# cp ../../ppss .
# 			ls features_* | while read FILE; do echo $FILE; done > ppss_input.txt
# 			rm -rf ppss_dir
# 			echo "Rodando o ALAC nas $partitions particoes..."
# 			./ppss -f ppss_input.txt -c '../../run_alac_repeated.sh "$ITEM"' -p $partitions
# 			
# 			ls alac_features_*_train.txt | while read FILE; do 
# 			      echo "`wc $FILE | awk '{ print $1 }'` instancias selecionadas no arquivo $FILE"
# 			done











############################

i=1 
while [[ $i -le 1 ]]; do
  # rm alac_lac_train_TUBEfinal.txt
   #rm alac_full_lac_train_TUBEfinal.txt
    echo " roda o ALAC ...."
  ../../run_alac_repeated_seed.sh lac_train_TUBEfinal.txt.$topic  $vez
#    cat alac_lac_train_TUBEfinal.txt | grep "CLASS=1" | awk '{ print $1 }' |  while read instance; do  sed -i  "/^$instance /d" lac_train_TUBEfinal.txt  ;  done
#   
#   
#     x=$(cat lac_train_TUBEfinal.txt | wc -l)
# 
#     if [ $x -le 1 ]; then
#         echo "file almost empty" 
#         if [ $x -eq 1 ]; then
#             cat lac_train_TUBEfinal.txt >> alac_lac_train_TUBEfinal.txt
#         fi
#         break;
#     fi

 i=$(($i+1))
done
#junta as instancias selecionadas em cada particao em um arquivo unico contendo todas as features
echo "Gerando o treino a partir das instancias selecionadas em cada particao..ssss"

cp alac_lac_train_TUBEfinal.txt.$topic   bkppairs.$topic 
cat lac_train_TUBEfinal_seed.txt.$topic  |   awk '{ print $1 }' |  while read instance; do  sed -i  "/^$instance /d"  bkppairs.$topic ;  done

#cat alac_lac_train_TUBE.txt* | awk '{ print $1 }' |  while read instance; do  sed -i  "/^$instance /d" $file_original  ;  done

./scriptRemoveRows.pl alac_full_lac_train_TUBEfinal.txt.$topic  composite_train_uniqB$numfeatures composite_train_uniqBold$numfeatures $vez

#cat composite_train_uniqB$numfeatures* >> bkppairs
# 
# 
# cat alac_lac_train_TUBEfinal.txt*  > composite_train$suffix.txt

# cat alac_lac_train_TUBEfinal.txt* | while read instance; do echo "$instance"; done

#cat composite_train_uniqB$numfeatures | awk '{ print $1 }'  | while read instance; do  sed  -n  "$instance"p  $treina_txt; done >> /tmp/final_treina.txt;

cat composite_train_uniqB$numfeatures  | awk '{ print $1 }'  | while read instance; do  sed  -n  "$instance"p  $file_original; done > final_treina.arff.$vez.$topic;

cut -d' ' -f1 final_treina.arff.$vez.$topic > /tmp/label_ssarp.$topic
