source "${UTIL_PATH}/colors"

export TOPIC=$1
export file=$2
export Rel=$3
export rules=$4
export sliding_windows=$5

echo "Starting script stopping point with sliding windows of $sliding_windows..."

#----------------------------------------------------------------------
# splits dataset into files of 30 docs
#----------------------------------------------------------------------
sort -k1 goldendb > temp
mv temp goldendb
rm -r data.$TOPIC
mkdir data.$TOPIC
pushd data.$TOPIC

split -l 30 -d ../final_ranking.$TOPIC xxx --suffix-length=6
len=`ls | wc -l`
len=$(($len))

popd

mkdir data_plus.$TOPIC
pushd data_plus.$TOPIC
split -l 30 -d ../result_plus.$TOPIC xxx --suffix-length=6
len_plus=`ls | wc -l`
len_plus=$(($len_plus-1))
popd

mem=0
flag=0
totalPos=0
totalNeg=0
totalPairsInput=0
mem_i=0

exitstatus=0
j=100
rm out_after_ssarp.$TOPIC
run=0
run_memory=0
run_attempy=0

echo "Len size $len"
cat x_posit.* \
    | sort \
    | uniq >> posit_file
sort -k2 posit_file > temp
mv temp posit_file
lo=1
hi=$len
inicialize_allac=2
count=0
loss_ac=0

#----------------------------------------------------------------------
# if number of docs is 0, ends execution
#----------------------------------------------------------------------
if [ $len -le 0 ]; then
    cat x_posit_ssarp_end.* x_negat_ssarp_end.* \
        | cut -d' ' -f2  >> out_after_ssarp.$TOPIC

    cat x_posit.* x_negat.* \
        | cut -d' ' -f2  >> out_after_ssarp.$TOPIC

    cat out_after_ssarp.$TOPIC \
        | sort \
        | uniq > temp

    cp temp out_after_ssarp.$TOPIC

    echo "positivos `cat out_after_ssarp.$TOPIC \
        | cut -d' ' -f1 \
        | sort -k1 \
        | uniq \
        | join - goldendb \
        |  wc -l` total `wc -l < out_after_ssarp.$TOPIC` input $totalPairsInput loss $loss_ac " >> runs.log

    exit
fi

cat x_posit_ssarp_end.* x_negat_ssarp_end.* \
    | cut -d' ' -f2  >> out_after_ssarp.$TOPIC

cat x_posit.* x_negat.* \
    | cut -d' ' -f2  >> out_after_ssarp.$TOPIC

cat out_after_ssarp.$TOPIC \
    | sort \
    | uniq > temp

cp temp out_after_ssarp.$TOPIC

r=`cat out_after_ssarp.$TOPIC \
    | cut -d' ' -f1 \
    | sort -k1 \
    | uniq \
    | join - goldendb \
    |  wc -l`

final_pairs=`wc -l < out_after_ssarp.$TOPIC`
total=$Rel
recall=`echo "scale=6; ($r / $total)" | bc`
precision=`echo "scale=6; ($r / $final_pairs)" | bc`

echo "Cleaning from $len until 0 stopping when $pos equal zero"
flag=0
temp=0
mid=$len
new_mid=0
mid=$(($mid-1)) # start from zero

for i in $(seq $mid -1 0 ); do
	j=$(($j+1))
    temp=$(($temp+1))

    name=`printf "data.$TOPIC/xxx%06d" $i` # create name of datafile
    echo "Cleaning 2 $name ---- $i"

    cat $name | sort > temp
    cat x_negat.* x_posit.* \
        | cut -d' ' -f2 \
        | sort \
        | join - temp -v2 \
        | sort > clean_data # data without already labeled docs

    echo "Script prune pairs..."

    bash "${UTIL_PATH}/script_prune_pairs.sh" clean_data $j $TOPIC $inicialize_allac $rules $sliding_windows > /tmp/lixo.$TOPIC

    inicialize_allac=3

    cat /tmp/lixo.$TOPIC

    pos=`grep "docs positivos coletados" /tmp/lixo.$TOPIC \
        | cut -d' ' -f4`

    neg=`grep "docs negativos coletados" /tmp/lixo.$TOPIC \
        | cut -d' ' -f11`

    real_posit=`cat clean_data \
        | cut -d' ' -f1 \
        | sort -k1 \
        | uniq \
        | join - goldendb \
        | wc -l`

    loss=$(($real_posit-$pos))
    loss_ac=$(($loss+$loss_ac))

    echo "$name pos $pos neg $neg real posit $real_posit loss $loss_ac" >> runs.log

    cat x_posit_ssarp_end.$j x_negat_ssarp_end.$j \
        | cut -d' ' -f2  >> out_after_ssarp.$TOPIC

    r=`cat out_after_ssarp.$TOPIC \
        | cut -d' ' -f1 \
        | sort -k1 \
        | uniq \
        | join - goldendb \
        | wc -l`

    final_pairs=`wc -l < out_after_ssarp.$TOPIC`
    total=$Rel

    recall=`echo "scale=6; ($r / $total)" \
        | bc`

    precision=`echo "scale=6; ($r / $final_pairs)" \
        | bc`

    echo "****final result AM is $TOPIC $r ------$final_pairs Recall $recall Precision $precision posit $pos $net labellingEffort `wc -l < out_after_ssarp.$TOPIC` loss_ac $loss_ac " >> runs.log
    echo "flag $flag" >> runs.log

    if [ $pos -le 0 ]; then
        echo "increase flag value" >> runs.log
        flag=$(($flag+1))
    else
        echo "new flag equal zero" >> runs.log
        flag=0
    fi

    if [ $flag -eq $sliding_windows ]; then
        new_mid=$i;
        echo "break because flag sliding windows" >> runs.log
        break;
    fi
done
flag=0

temp=0
if [ $new_mid -le 0 ]; then
    echo "######inside active plus $len_plus" >> runs.log
    for i in $(seq $len_plus -1 0 ); do
        name=`printf "data_plus.$TOPIC/xxx%06d" $i`
        echo "inside active plus $name ---- $i" >> runs.log
        j=$(($j+1))
        temp=$(($temp+1))

        cat $name \
            | sort > temp

        cat x_negat.* x_posit.* \
            | cut -d' ' -f2 \
            | sort \
            | join - temp -v2 \
            | sort > new_data

        bash "${UTIL_PATH}/script_prune_pairs.sh" new_data $j $TOPIC $inicialize_allac $rules $sliding_windows &> /tmp/lixo.$TOPIC

        cat /tmp/lixo.$TOPIC

        pos=`grep "docs positivos coletados" /tmp/lixo.$TOPIC \
            | cut -d' ' -f4`

        neg=`grep "docs positivos coletados" /tmp/lixo.$TOPIC \
            | cut -d' ' -f11`

        real_posit=`cat new_data \
            | cut -d' ' -f1 \
            | sort -k1 \
            | uniq \
            | join - goldendb \
            |  wc -l`

        loss=$(($real_posit-$pos))
        loss_ac=$(($loss+$loss_ac))

        echo "$name pos $pos neg $neg real posit $real_posit loss $loss_ac" >> runs.log
        cat x_posit_ssarp_end.$j x_negat_ssarp_end.$j \
            | cut -d' ' -f2 >> out_after_ssarp.$TOPIC

        r=`cat out_after_ssarp.$TOPIC \
            | cut -d' ' -f1 \
            | sort -k1 \
            | uniq \
            | join - goldendb \
            | wc -l`

        final_pairs=`wc -l < out_after_ssarp.$TOPIC`
        total=$Rel
        recall=`echo "scale=6; ($r / $total)" | bc`
        precision=`echo "scale=6; ($r / $final_pairs)" | bc`
        echo "****final result AM is $TOPIC $r ------$final_pairs  Recall $recall  Precision $precision  posit  $pos $net labellingEffort `wc -l < out_after_ssarp.$TOPIC` loss_ac $loss_ac " >> runs.log
        echo "flag $flag" >> runs.log

        if [ $pos -le 0 ]; then
            flag=$(($flag+1))
        else
            flag=0
        fi

        if [ $flag -eq $sliding_windows ]; then
            new_mid=$i;
            break;
        fi

        if [ $j -ge 2000 ]; then
            echo "##break because of stopping $i" >> runs.log
            break;
        fi
    done
fi

cat x_posit_ssarp_end.* x_negat_ssarp_end.* \
    | cut -d' ' -f2  > training_set.$TOPIC

cat only_levels_ssarp.$TOPIC training_set.$TOPIC \
    | sort \
    | uniq > ssarp_labelling_full.$TOPIC

cat x_posit.* x_negat.* \
    | cut -d' ' -f2 >> training_set.$TOPIC

sort training_set.$TOPIC \
    | sort \
    | uniq > temp

mv temp training_set.$TOPIC

cat x_posit_ssarp_end.* x_negat_ssarp_end.* \
    | cut -d' ' -f2 >> out_after_ssarp.$TOPIC

cat out_after_ssarp.$TOPIC \
    | sort \
    | uniq > temp

cp temp out_after_ssarp.$TOPIC

r=`cat out_after_ssarp.$TOPIC \
    | cut -d' ' -f1 \
    | sort -k1 \
    | uniq \
    | join - goldendb \
    | wc -l`

final_pairs=`wc -l < out_after_ssarp.$TOPIC`
total=`wc -l < goldendb`
echo "total de relevantes $total"
recall=`echo "scale=6; ($r / $total)" | bc`
precision=`echo "scale=6; ($r / $final_pairs)" | bc`

echo "final REVEAL is $TOPIC $r ------$final_pairs  Recall $recall  Precision $precision  posit  $r  labellingEffort `wc -l < training_set.$TOPIC`  onlyssarp  `wc -l < ssarp_labelling_full.$TOPIC` loss_ac $loss_ac sliding_windows $sliding_windows" >> runs.log

e_primary "-------------------------------------------------------------"
e_secondary "Final result REVEAL ${YELLOW}Topic:${END} $TOPIC"
echo -e "\tRelevants Found ..... $r"
echo -e "\tTotal Relevants ..... $total"
echo -e "\tLoss ................ $loss_ac"
echo -e "\tLabeled Documents ... `wc -l < training_set.$TOPIC`"
echo -e "\tRecall .............. $recall"
echo -e "\tPrecision ........... $precision"
e_primary "-------------------------------------------------------------"

echo "$recall $precision `wc -l < training_set.$TOPIC`" > reveal.final
