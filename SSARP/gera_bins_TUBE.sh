#bin/bash
# Gera arquivos de nome LLK$i.txt contendo os bins criados pelo TUBE (aonde $i eh o numero de bins)
# Recebe como parametro: $1: arquivo a ser discretizado no formato weka (.arff)
# $2 numero de features no arquivo
# $3 numero inicial de bins (permite criar varias discretizacoes de $3 a $4 bins)
# $4 numero final de bins

trainfile=$1
numfeatures=$2
TOPIC=$5

# se o numero de bins nao for especificado, assume que sao 10 bins
if [ "$3" == "" -o "$4" == "" ]; then
      i=10
      f=10
else
      i=$3
      f=$4
fi

while [ $i -le $f ]; do
		suffix="B"$2$TOPIC
		j=1
		while [ $j -le $numfeatures ]; do
		  if [ ! -f train-$suffix-$j-0LL.hist ]; then
		    echo "Creating TUBE file train-$suffix-$j-0LL.hist......."
		    echo "@relation documents" > attr$j.arff
		    echo "" >> attr$j.arff
		    echo "@attribute F$j numeric" >> attr$j.arff
		    echo "" >> attr$j.arff
		    echo "@data" >> attr$j.arff
		    grep -v @ $trainfile | grep -v ^$ | awk -F "," '{ print $'"$j"' }' | sed 's/\,//' >> attr$j.arff
		    java -Xmx1024m -classpath ../TUBE/src/ weka.estimators.TUBEstimator -i attr$j.arff -V 8  -B $i -X train-$suffix-$j >> LLK$i.txt
		    #rm -f attr$j.arff
		  fi
                    j=$(($j+1))
		done
		i=$(($i+1))
done
for i in `seq 1 50`; do  mv "train-B50$TOPIC-`echo $i`-0null.hist" "train-B50$TOPIC-`echo $i`-0LL.hist"; done
