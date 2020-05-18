#bin/bash
#input example ./SSARPX.sh inputSSARP.08.Rad label.08.Rad 50 08 seed_out.08.Rad.arff Rad 6 5

numfeatures=10 # numero de features no treino / teste
binsfrom=10 # numero de bins (de) 
binsto=10 # numero de bins (at
source "${ABS_PATH}/handle_errors"
treina_arff=$1".arff"

file_original=$2
suffix=B$3
numfeatures=$3
vez=$4
seed_ssarp=$5
topic=$6
rules=$8
#sleep 100000000

echo "the file have "$numfeatures;

#try
#(
    rm -r result_temp_lac_train_TUBEfinal.txt/ 
    rm -r result_temp_lac_train_TUBE*
        
    rm train_nohead.arff  
    echo "gerando bins ../.././gera_bins_TUBE.sh $treina_arff  $numfeatures $binsfrom $binsto  " 
    ../.././gera_bins_TUBE.sh $treina_arff  $numfeatures $binsfrom $binsto
#) 2> $STD_ERROR_OUT

#catch || {
#    kill  -SIGUSR2 -SIGUSR1 `ps --pid $$ -oppid=`; exit_on_error
#}   


# 
# remove os headers dos arquivos weka
if [ ! -f train_nohead.arff ]; then
    grep -v @ $treina_arff | grep -v ^$ > train_nohead.arff
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


echo "Discretizando treino e teste de acordo com os bins TUBE"
echo ../../discretize_TUBE.pl train-$suffix train_nohead.arff $numfeatures  lac_train_TUBE.txt.$topic
../../discretize_TUBE.pl train-$suffix train_nohead.arff $numfeatures  lac_train_TUBE.txt.$topic




./updateRows.pl lac_train_TUBE.txt.$topic lac_train_TUBEfinal.txt.$topic $vez





if [ $vez -eq 10 ]; 
then
    #try
    #(
        echo "removendo o arquivo bkp anterior "
        rm  bkppairs.$topic 
        rm alac_full_lac_train_TUBEfinal.txt.$topic 
        rm alac_lac_train_TUBEfinal.txt.$topic
        echo "iniciando o seed"

        echo "discretizando seed ssarp"
        ../../discretize_TUBE.pl train-$suffix $seed_ssarp $numfeatures  lac_train_TUBE_seed.txt.$topic 
        ./updateRows.pl lac_train_TUBE_seed.txt.$topic  lac_train_TUBEfinal_seed.txt.$topic   $(($vez-1))
        
        cat lac_train_TUBEfinal_seed.txt.$topic  >> alac_full_lac_train_TUBEfinal.txt.$topic 
        cat lac_train_TUBEfinal_seed.txt.$topic  >> alac_lac_train_TUBEfinal.txt.$topic 
    #) 2> $STD_ERROR_OUT

    #catch || {
    #    kill -SIGUSR2 -SIGUSR1 `ps --pid $$ -oppid=`; exit_on_error
    #}
fi

echo " roda o ALAC $numfeatures" 
i=1 
while [[ $i -le 1 ]]; do
  
    echo " roda o ALAC ....with $rules"
    ../../run_alac_repeated.sh lac_train_TUBEfinal.txt.$topic  $vez $rules


    i=$(($i+1))
done

#junta as instancias selecionadas em cada particao em um arquivo unico contendo todas as features
echo "Gerando o treino a partir das instancias selecionadas em cada particao.."
#try
#(
    cp alac_lac_train_TUBEfinal.txt.$topic   bkppairs.$topic 
    cat lac_train_TUBEfinal_seed.txt.$topic  |   awk '{ print $1 }' |  while read instance; do  sed -i  "/^$instance /d"  bkppairs.$topic ;  done

    ./scriptRemoveRows.pl alac_full_lac_train_TUBEfinal.txt.$topic  composite_train_uniqB$numfeatures composite_train_uniqBold$numfeatures $vez

    cat composite_train_uniqB$numfeatures  | awk '{ print $1 }'  | while read instance; do  sed  -n  "$instance"p  $file_original; done > /tmp/final_treina.arff.$vez.$topic;

    cut -d' ' -f1 /tmp/final_treina.arff.$vez.$topic > /tmp/label.$vez.$topic
    echo "gerando arquivo SVM treino.."
    cut -d' ' -f1 /tmp/final_treina.arff.$vez.$topic >  /tmp/outSSARP.$vez.$topic
#) 2> $STD_ERROR_OUT

#catch || {
 #   kill -SIGUSR2 -SIGUSR1 `ps --pid $$ -oppid=`; exit_on_error
#}
 