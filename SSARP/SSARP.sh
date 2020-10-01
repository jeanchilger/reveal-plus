#bin/bash

FOLDER=$1 # diretorio a ser criado dentro de cada Fold
numfeatures=$2 # numero de features no treino / teste
partitions=5 # numero de particoes a ser usado
binsfrom=10 # numero de bins (de) 
binsto=10 # numero de bins (ate)


testeFile=../test.txt
# roda do Fold1 ao Fold5
i=1;
while [ $i -le 1 ]; do
	
	cd Fold$i
		
	if [ ! -d $FOLDER ]; then # soh executa se o diretorio nao existe ainda
		mkdir $FOLDER
		cd $FOLDER
		echo "Processando Fold$i..."
		rm -r ../train.txt.arff 
		if [ ! -f  ../train.txt.arff ]; then 
		    # converte train.txt, test.txt e vali.txt para o formato weka (sem discretizar)
		    echo "Convertendo arquivos para o formato weka..."
		    ../../convert_letor_to_weka ../train.txt $numfeatures > ../train.txt.arff
		    ../../convert_letor_to_weka $testeFile $numfeatures >  ../test.txt.arff
		    ../../convert_letor_to_weka ../validation.txt $numfeatures >  ../vali.txt.arff
		fi
		
		# gera arquivos contendo os limites dos BINS determinados pelo TUBE
		echo "Gerando os bins TUBE..."
		../../gera_bins_TUBE.sh ../train.txt.arff $numfeatures $binsfrom $binsto
		
		if [ -f ../train_nohead.arff ]; then rm -f ../train_nohead.arff; fi
		# roda para cada numero de bins
		k=$binsfrom
		while [ $k -le $binsto ]; do

		  suffix=B$k # sufixo indicando o numero de bins
		  if [ ! -f ssarp_eval$suffix.txt ]; then
		
			# remove os headers dos arquivos weka
			if [ ! -f ../train_nohead.arff ]; then
			      grep -v @ ../train.txt.arff | grep -v ^$ > ../train_nohead.arff
			      grep -v @ ../test.txt.arff | grep -v ^$ > ../test_nohead.arff
			fi 
		
			if [ -f ../lac_train_TUBE$suffix.txt ]; then rm -f ../lac_train_TUBE$suffix.txt; fi
			if [ -f ../lac_test_TUBE$suffix.txt ]; then rm -f ../lac_test_TUBE$suffix.txt; fi
		
			# discretiza usando os bins definidos pelo TUBE; os arquivos jah sao gerados no formato LAC
			echo "Discretizando treino e teste de acordo com os bins TUBE"
			echo ../../discretize_TUBE.pl train-$suffix ../train_nohead.arff $numfeatures ../lac_train_TUBE$suffix.txt 
			../../discretize_TUBE.pl train-$suffix ../train_nohead.arff $numfeatures ../lac_train_TUBE$suffix.txt
			../../discretize_TUBE.pl train-$suffix ../test_nohead.arff $numfeatures ../lac_test_TUBE$suffix.txt
			
			if [ -f ../lac_train_TUBE.tmp ]; then rm -f ../lac_train_TUBE.tmp; fi
			if [ -f ../lac_train_header_TUBE.arff ]; then rm -f ../lac_train_header_TUBE.arff; fi
			
			# gera o treino no formato weka para calculo do ChiSquared das features
			../../lac_to_weka.pl ../lac_train_TUBE$suffix.txt ../lac_train_TUBE.tmp ../lac_train_header_TUBE.arff $numfeatures
			sed 's/\,'\''\\'\''$/}/' ../lac_train_TUBE.tmp > ../lac_train_TUBE$suffix.arff
			
			# calcula o ChiSquared de cada feature com relacao as demais e gera uma lista ordenada das features
# 			rm -r ../features_train_TUBE$suffix.txt
# 			if [ ! -f ../features_train_TUBE$suffix.txt ]; then
# 			      echo -n "Calculando ChiSquared e ordenando features..."
# 			      ../../create_ChiSquared_train_TUBE.sh ../lac_train_TUBE$suffix.arff ../ChiSquared_Class_train_TUBE$suffix.txt $numfeatures $suffix
# 			      ../../order_chi_TUBE.pl ../ChiSquared_Class_train_TUBE$suffix.txt | tail -n 1 > ../features_train_TUBE$suffix.txt
# 			      echo " ordenacao das features finalizada."
# 			fi 
# 			
# 			rm -f features_*_train.txt
# 			rm -f alac_features_*_train.txt
# 			rm -rf result_temp_features_*_train.txt
# 			
# 			# cria $partitions particoes usando a lista de features e o treino (gerando $partitions arquivos de treino)
# 			echo "Criando arquivos de particoes... "
# 			echo "../../create_partition_files.pl ../lac_train_TUBE$suffix.txt ../features_train_TUBE$suffix.txt $partitions"
# 			../../create_partition_files.pl ../lac_train_TUBE$suffix.txt ../features_train_TUBE$suffix.txt $partitions
# 			
# 			#roda o ALAC em cada particao em paralelo usando o ppss (roda um processo para cada particao; dependendo da quantidade de 
# 			#cores da maquina, pode ser mais rapido modificar o parametro -p para um numero compativel: esse parametro determinados
# 			#quantos processos sao executados em paralelo; caso seja menor que o numero de particoes, entao roda em batch)
# 			cp ../../ppss .
# 			ls features_* | while read FILE; do echo $FILE; done > ppss_input.txt
# 			rm -rf ppss_dir
# 			echo "Rodando o ALAC nas $partitions particoes..."
# 			echo "./ppss -f ppss_input.txt -c '../../run_alac_repeated.sh "$ITEM"' -p $partitions"
# 			./ppss -f ppss_input.txt -c '../../run_alac_repeated.sh "$ITEM"' -p $partitions
#                         #cp ../lac_train_TUBE$suffix.txt lac_train_TUBE$suffix.txt
# 			#../../run_alac_repeated.sh lac_train_TUBE$suffix.txt 1
# 			#ls alac_features_*_train.txt | while read FILE; do   echo "`wc $FILE | awk '{ print $1 }'` instancias selecionadas no arquivo $FILE" done
# 			
# 			# junta as instancias selecionadas em cada particao em um arquivo unico contendo todas as features
# 			echo "Gerando o treino a partir das instancias selecionadas em cada particao.."
# 			cat alac_features_*_train.txt | awk '{ print $1 }' | while read instance; do grep "^$instance\ " ../lac_train_TUBE$suffix.txt >> composite_train$suffix.txt; done
#                         
#                         #cat alac_lac_train_TUBEB10.txt | awk '{ print $1 }' | while read instance; do grep "^$instance\ " ../lac_train_TUBE$suffix.txt >> composite_train$suffix.txt; done
# 			# elimina instancias repetidas selecionadas em mais de uma particao
# 			sort composite_train$suffix.txt | uniq > composite_train_uniq$suffix.txt
# 			echo "`wc composite_train_uniq$suffix.txt | awk '{ print $1 }'` instancias distintas selecionadas"
# 
# 			# roda o LAC para rankear usando o treino ativo criado acima para ordenar o teste
# 			echo "Executando o LAC para ordenar o teste usando o treino ativo..."
# 			../../alac -i composite_train_uniq$suffix.txt -t ../lac_test_TUBE$suffix.txt -s 1 -m 3 -e 1000000000 -c 0.001 > lac_resultado$suffix.txt 
# 			grep "^[0-9]*\-[0-9]*\ " lac_resultado$suffix.txt | awk '{ print $12 }' > lac_resultado_ranking$suffix.txt
# 			
# 			# Avaliacao final do resultado usando o script do LETOR (calcula MAP e NDCG). Resultado gravado em alac_avaliacao$suffix.txt
# 			echo "Calculando resultados finais para o Fold$i..."
# 			perl ../../Eval-Score-3.0.pl ../test.txt lac_resultado_ranking$suffix.txt ssarp_eval$suffix.txt 0
# 			echo "MAP obtido para o Fold$i $suffix: `cat ssarp_eval$suffix.txt | grep MAP | awk '{ print $2 }'`"

		  fi
		  k=$(($k+1))
		done
	  
		cd ../../
	else
		cd ../	
	fi
	i=$(($i+1))
done


cat Fold1/$FOLDER/composite_train_uniq$suffix.txt | awk '{ print $1 }'  | while read instance; do  sed  -n  "$instance"p  Fold1/testfile.txt; done >> /tmp/final_treina.arff;

# imprime MAP para os 5 Folds.
j=$binsfrom; 
while [ $j -le $binsto ]; do 
  SUM=0; 
  SUMMAP=0; 
  TOTINST=0;
  i=1; 
  while [ $i -le 5 ]; do 
    NINST=`wc Fold$i/$FOLDER/composite_train_uniqB$j.txt | awk '{ print $1 }'`; 
    SUM=$(($SUM+$NINST)); 
    MAP=`cat Fold$i/$FOLDER/ssarp_evalB$j.txt | grep MAP | awk '{ print $2 }'`; 
    SUMMAP=`echo "scale=8; $SUMMAP+$MAP" | bc`;
    i=$(($i+1)); 
  done; 
  echo "Resultado para $j bins:"
  echo -n "Numero de instancias selecionadas: "; 
  echo "scale=0; $SUM/5" | bc;
  echo -n "Valor final do MAP: "; 
  echo "scale=10; $SUMMAP/5" | bc;
  j=$(($j+1)); 
done 

